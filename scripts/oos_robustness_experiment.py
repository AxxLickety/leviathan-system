"""
OOS Model Robustness Experiment — Leviathan Phase OOS
======================================================
Compares four training specifications on the same train/test split.

Specs:
  1. interaction_logit  — current baseline (dti, regime, dti×regime)
  2. main_effects       — logit without interaction term
  3. l1_regularized     — L1-penalized logit (alpha=0.5), same features as spec 1
  4. rule_only_wf       — walk-forward DTI cutoff; no logit at all

For each spec:
  - Convergence status and pseudo R²
  - Fitted model-suggested DTI cutoff (analytical at regime=0, p=0.10)
  - OOS overlay performance (p05, Sharpe, maxdd) using the spec's own cutoff
  - OOS overlay performance using the shared walk-forward cutoff
  - always_in baseline for comparison

Does NOT modify oos_train.py / oos_eval.py or frozen outputs.

Run:
    python scripts/oos_robustness_experiment.py

Output:
    outputs/oos/robustness/model_comparison.csv
    outputs/oos/robustness/model_comparison_print.txt  (human-readable)
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.research.path_a.build_dataset import build_master_df
from src.research.path_a.label_correction import add_correction_label
from src.research.path_a.fit_logit import fit_interaction_logit
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime
from src.backtests.evaluation import summarize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_END  = "2007-12-31"
TEST_START = "2008-03-31"
OUT_DIR    = Path("outputs/oos/robustness")

_WF_WINDOW_Q    = 20
_WF_QGRID       = np.arange(0.60, 0.96, 0.05)
_WF_FALLBACK_Q  = 0.80
_WF_MIN_REGIME0 = 10
_LOGIT_PROB     = 0.10   # probability level used for analytical threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _walk_forward_cutoff(df_train: pd.DataFrame) -> float:
    """Replicate oos_train.py walk-forward quantile search."""
    df_wf = (
        df_train
        .dropna(subset=["fwd_return", "dti", "regime"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    cutoff: float | None = None
    for t in range(_WF_WINDOW_Q, len(df_wf)):
        train_wf = df_wf.iloc[:t]
        r0 = train_wf.loc[train_wf["regime"] == 0, "dti"].dropna()
        if len(r0) < _WF_MIN_REGIME0:
            thr = float(train_wf["dti"].quantile(_WF_FALLBACK_Q))
        else:
            best_thr = float(r0.quantile(_WF_FALLBACK_Q))
            best_val = -np.inf
            for q in _WF_QGRID:
                cand = float(r0.quantile(q))
                hold = ~((train_wf["regime"] == 0) & (train_wf["dti"] > cand))
                val  = float((train_wf["fwd_return"] * hold.astype(float)).quantile(0.05))
                if val > best_val:
                    best_val, best_thr = val, cand
            thr = best_thr
        cutoff = thr
    assert cutoff is not None
    return cutoff


def _oos_overlay(df_test: pd.DataFrame, cutoff: float, name: str) -> dict:
    """Apply overlay filter with given cutoff; return summarize() dict."""
    hold = ~((df_test["regime"] == 0) & (df_test["dti"] > cutoff))
    ret  = (df_test["fwd_return"] * hold.astype(float)).rename(name)
    s    = summarize(ret, name=name)
    return s


def _pseudo_r2_manual(params: np.ndarray, X: pd.DataFrame, y: pd.Series) -> float:
    """McFadden pseudo R² for any parameter vector."""
    logit_vals = X.values @ params
    pred = 1.0 / (1.0 + np.exp(-logit_vals))
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    y_   = y.values.astype(float)
    llf  = float(np.sum(y_ * np.log(pred) + (1 - y_) * np.log(1 - pred)))
    p_mn = float(y_.mean())
    llnull = float(np.sum(y_ * np.log(p_mn) + (1 - y_) * np.log(1 - p_mn)))
    if llnull == 0:
        return np.nan
    return 1.0 - llf / llnull


def _logit_dti_threshold(params: pd.Series, regime: int = 0, prob: float = 0.10) -> float:
    """
    Analytical DTI threshold from logit params.
    Works for both interaction (4-param) and main-effects (3-param) models.
    """
    logit_p = np.log(prob / (1.0 - prob))
    b0 = float(params.get("const",        params.iloc[0]))
    b1 = float(params.get("dti",          params.iloc[1]))
    b2 = float(params.get("regime",       0.0))
    b3 = float(params.get("dti_x_regime", 0.0))
    denom = b1 + b3 * regime
    if abs(denom) < 1e-12:
        return np.nan
    return (logit_p - b0 - b2 * regime) / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load panel (diagnostics printed once)
    # ------------------------------------------------------------------
    df = build_master_df()
    df = compute_forward_return(df)
    df = assign_fragility_regime(df)

    df_train_full = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()
    df_test_full  = df[df["date"] >= pd.Timestamp(TEST_START)].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_train = add_correction_label(df_train_full)
        df_test  = add_correction_label(df_test_full)

    n_y1 = int(df_train["y"].sum())
    n_y0 = int((df_train["y"] == 0).sum())

    print(f"\n[ROBUSTNESS] Training: {len(df_train)} labeled rows  "
          f"y=1:{n_y1}  y=0:{n_y0}  prev={n_y1/len(df_train):.1%}")
    print(f"[ROBUSTNESS] Test    : {len(df_test)} labeled rows  "
          f"y=1:{df_test['y'].sum()}  y=0:{(df_test['y']==0).sum()}")

    # ------------------------------------------------------------------
    # Shared walk-forward cutoff (same for all specs; derived from training only)
    # ------------------------------------------------------------------
    wf_cutoff = _walk_forward_cutoff(df_train_full)
    print(f"[ROBUSTNESS] Walk-forward cutoff: {wf_cutoff:.4f}")

    # always_in baseline
    ai_summary = summarize(df_test["fwd_return"], name="always_in")

    rows: list[dict] = []

    # ==================================================================
    # Spec 1: Interaction logit (baseline)
    # ==================================================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_train_inter = df_train.copy()
        df_train_inter["dti_x_regime"] = df_train_inter["dti"] * df_train_inter["regime"]
        X1 = sm.add_constant(df_train_inter[["dti", "regime", "dti_x_regime"]])
        y1 = df_train_inter["y"]
        model1 = sm.Logit(y1, X1)
        res1   = model1.fit(disp=False)

    converged1  = bool(res1.mle_retvals.get("converged", False))
    pseudo_r2_1 = float(res1.prsquared) if not np.isinf(res1.prsquared) else np.nan
    cutoff1     = _logit_dti_threshold(res1.params, regime=0, prob=_LOGIT_PROB)

    s1_model = _oos_overlay(df_test, cutoff1,   "overlay_spec1_model_cutoff")
    s1_wf    = _oos_overlay(df_test, wf_cutoff, "overlay_spec1_wf_cutoff")
    rows.append({
        "spec":           "interaction_logit",
        "n_y1":           n_y1,
        "converged":      converged1,
        "pseudo_r2":      round(pseudo_r2_1, 4),
        "coef_const":     round(float(res1.params["const"]), 4),
        "coef_dti":       round(float(res1.params["dti"]), 4),
        "coef_regime":    round(float(res1.params["regime"]), 4),
        "coef_dti_x_r":   round(float(res1.params["dti_x_regime"]), 4),
        "model_cutoff":   round(cutoff1, 4),
        "wf_cutoff":      round(wf_cutoff, 4),
        "oos_p05_model":  round(s1_model["p05"], 6),
        "oos_sharpe_model": round(s1_model["sharpe"], 4),
        "oos_maxdd_model":  round(s1_model["maxdd"], 4),
        "oos_p05_wf":     round(s1_wf["p05"], 6),
        "oos_sharpe_wf":  round(s1_wf["sharpe"], 4),
        "oos_maxdd_wf":   round(s1_wf["maxdd"], 4),
    })

    # ==================================================================
    # Spec 2: Main-effects logit (no interaction term)
    # ==================================================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X2 = sm.add_constant(df_train[["dti", "regime"]])
        y2 = df_train["y"]
        model2 = sm.Logit(y2, X2)
        res2   = model2.fit(disp=False)

    converged2  = bool(res2.mle_retvals.get("converged", False))
    pseudo_r2_2 = float(res2.prsquared) if not np.isinf(res2.prsquared) else np.nan
    cutoff2     = _logit_dti_threshold(res2.params, regime=0, prob=_LOGIT_PROB)

    s2_model = _oos_overlay(df_test, cutoff2,   "overlay_spec2_model_cutoff")
    s2_wf    = _oos_overlay(df_test, wf_cutoff, "overlay_spec2_wf_cutoff")
    rows.append({
        "spec":           "main_effects_logit",
        "n_y1":           n_y1,
        "converged":      converged2,
        "pseudo_r2":      round(pseudo_r2_2, 4),
        "coef_const":     round(float(res2.params["const"]), 4),
        "coef_dti":       round(float(res2.params["dti"]), 4),
        "coef_regime":    round(float(res2.params.get("regime", np.nan)), 4),
        "coef_dti_x_r":   np.nan,
        "model_cutoff":   round(cutoff2, 4),
        "wf_cutoff":      round(wf_cutoff, 4),
        "oos_p05_model":  round(s2_model["p05"], 6),
        "oos_sharpe_model": round(s2_model["sharpe"], 4),
        "oos_maxdd_model":  round(s2_model["maxdd"], 4),
        "oos_p05_wf":     round(s2_wf["p05"], 6),
        "oos_sharpe_wf":  round(s2_wf["sharpe"], 4),
        "oos_maxdd_wf":   round(s2_wf["maxdd"], 4),
    })

    # ==================================================================
    # Spec 3: L1-regularized logit (LASSO, alpha=0.5)
    # Uses the same interaction feature set as spec 1.
    # fit_regularized avoids perfect-separation failures at cost of
    # biased coefficients; better suited for small-N regimes.
    # ==================================================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res3 = model1.fit_regularized(method="l1", alpha=0.5, disp=False)

    # fit_regularized returns RegularizedResults (limited attributes)
    try:
        converged3 = bool(res3.mle_retvals.get("converged", True))
    except AttributeError:
        converged3 = True   # L1 optimization typically converges

    pseudo_r2_3 = _pseudo_r2_manual(res3.params.values, X1, y1)
    params3_series = pd.Series(res3.params.values, index=X1.columns)
    cutoff3 = _logit_dti_threshold(params3_series, regime=0, prob=_LOGIT_PROB)

    s3_model = _oos_overlay(df_test, cutoff3,   "overlay_spec3_model_cutoff")
    s3_wf    = _oos_overlay(df_test, wf_cutoff, "overlay_spec3_wf_cutoff")
    rows.append({
        "spec":           "l1_regularized_logit",
        "n_y1":           n_y1,
        "converged":      converged3,
        "pseudo_r2":      round(pseudo_r2_3, 4),
        "coef_const":     round(float(params3_series.get("const", np.nan)), 4),
        "coef_dti":       round(float(params3_series.get("dti", np.nan)), 4),
        "coef_regime":    round(float(params3_series.get("regime", np.nan)), 4),
        "coef_dti_x_r":   round(float(params3_series.get("dti_x_regime", np.nan)), 4),
        "model_cutoff":   round(cutoff3, 4) if not np.isnan(cutoff3) else np.nan,
        "wf_cutoff":      round(wf_cutoff, 4),
        "oos_p05_model":  round(s3_model["p05"], 6) if not np.isnan(cutoff3) else np.nan,
        "oos_sharpe_model": round(s3_model["sharpe"], 4) if not np.isnan(cutoff3) else np.nan,
        "oos_maxdd_model":  round(s3_model["maxdd"], 4) if not np.isnan(cutoff3) else np.nan,
        "oos_p05_wf":     round(s3_wf["p05"], 6),
        "oos_sharpe_wf":  round(s3_wf["sharpe"], 4),
        "oos_maxdd_wf":   round(s3_wf["maxdd"], 4),
    })

    # ==================================================================
    # Spec 4: Rule-only (walk-forward cutoff, no logit)
    # Treats the walk-forward result as the sole decision rule.
    # No model fitting; no threshold extrapolation.
    # ==================================================================
    s4 = _oos_overlay(df_test, wf_cutoff, "overlay_spec4_rule_only")
    rows.append({
        "spec":           "rule_only_wf",
        "n_y1":           n_y1,
        "converged":      "N/A",
        "pseudo_r2":      np.nan,
        "coef_const":     np.nan,
        "coef_dti":       np.nan,
        "coef_regime":    np.nan,
        "coef_dti_x_r":   np.nan,
        "model_cutoff":   round(wf_cutoff, 4),
        "wf_cutoff":      round(wf_cutoff, 4),
        "oos_p05_model":  round(s4["p05"], 6),
        "oos_sharpe_model": round(s4["sharpe"], 4),
        "oos_maxdd_model":  round(s4["maxdd"], 4),
        "oos_p05_wf":     round(s4["p05"], 6),
        "oos_sharpe_wf":  round(s4["sharpe"], 4),
        "oos_maxdd_wf":   round(s4["maxdd"], 4),
    })

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    buf = io.StringIO()
    sep = "=" * 76

    lines = [
        sep,
        "OOS MODEL ROBUSTNESS EXPERIMENT",
        sep,
        f"  Training : {TRAIN_END}  |  n_labeled={len(df_train)}  "
        f"y1={n_y1}  y0={n_y0}  prev={n_y1/len(df_train):.1%}",
        f"  Test     : {TEST_START} →  |  n_labeled={len(df_test)}",
        f"  Walk-fwd cutoff (shared) : {wf_cutoff:.4f}",
        f"  always_in  p05={ai_summary['p05']:.4f}  "
        f"sharpe={ai_summary['sharpe']:.3f}  maxdd={ai_summary['maxdd']:.4f}",
        "",
        f"  {'Spec':<22} {'Conv':<6} {'PsR2':<7} {'ModelCut':<10} "
        f"{'p05(mdl)':<10} {'SR(mdl)':<9} {'MDD(mdl)':<10} "
        f"{'p05(wf)':<10} {'SR(wf)':<8} {'MDD(wf)'}",
        "-" * 76,
    ]

    fmt = "  {spec:<22} {conv:<6} {pr2:<7} {mc:<10} {p05m:<10} {srm:<9} {mddm:<10} {p05w:<10} {srw:<8} {mddw}"
    for r in rows:
        lines.append(fmt.format(
            spec  = r["spec"],
            conv  = str(r["converged"])[:5],
            pr2   = f"{r['pseudo_r2']:.4f}" if not (isinstance(r['pseudo_r2'], float) and np.isnan(r['pseudo_r2'])) else "N/A  ",
            mc    = f"{r['model_cutoff']:.4f}" if not (isinstance(r['model_cutoff'], float) and np.isnan(r['model_cutoff'])) else "N/A",
            p05m  = f"{r['oos_p05_model']:.4f}" if not (isinstance(r['oos_p05_model'], float) and np.isnan(r['oos_p05_model'])) else "N/A",
            srm   = f"{r['oos_sharpe_model']:.3f}" if not (isinstance(r['oos_sharpe_model'], float) and np.isnan(r['oos_sharpe_model'])) else "N/A",
            mddm  = f"{r['oos_maxdd_model']:.4f}" if not (isinstance(r['oos_maxdd_model'], float) and np.isnan(r['oos_maxdd_model'])) else "N/A",
            p05w  = f"{r['oos_p05_wf']:.4f}",
            srw   = f"{r['oos_sharpe_wf']:.3f}",
            mddw  = f"{r['oos_maxdd_wf']:.4f}",
        ))

    lines += ["", sep, "Coefficients:", "-" * 40]
    for r in rows:
        if r["spec"] == "rule_only_wf":
            continue
        coef_str = (
            f"  {r['spec']:<22}  const={r['coef_const']:>9.3f}  "
            f"dti={r['coef_dti']:>8.4f}  regime={r['coef_regime']:>9.3f}  "
            f"dti_x_r={r['coef_dti_x_r']:>8.3f}"
            if not np.isnan(r['coef_dti_x_r']) else
            f"  {r['spec']:<22}  const={r['coef_const']:>9.3f}  "
            f"dti={r['coef_dti']:>8.4f}  regime={r['coef_regime']:>9.3f}  dti_x_r=   N/A"
        )
        lines.append(coef_str)

    lines += ["", sep]
    output_text = "\n".join(lines) + "\n"

    print("\n" + output_text)

    txt_path = OUT_DIR / "model_comparison_print.txt"
    txt_path.write_text(output_text)

    print(f"[ROBUSTNESS] Wrote {OUT_DIR / 'model_comparison.csv'}")
    print(f"[ROBUSTNESS] Wrote {txt_path}")


if __name__ == "__main__":
    main()
