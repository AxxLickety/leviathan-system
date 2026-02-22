from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Config (edit if needed)
# -----------------------------
IN_PATH = Path("data/processed/phase3_panel_wret_regime_alt.csv")
P2_PATH = Path("outputs/phase2/phase2_strategy_series.csv")  # for baseline strategies if needed

OUT_SERIES = Path("outputs/phase4/series/oos_series.csv")
OUT_SUMMARY = Path("outputs/phase4/tables/oos_summary.csv")
OUT_SPECS = Path("outputs/phase4/tables/exposure_specs.csv")

DATE_COL = "date"
RET_COL = "ret_1q_fwd"
DTI_COL = "dti"
MS_COL = "months_supply"
REGIME_COL = "regime"

FIRST_TEST = "2011-03-31"   # same as your phase3.yaml default
ROLL_WIN = 20              # 5 years of quarters for rolling stats (tunable)

# exposure family parameters (grid light)
SPECS = [
    {"name":"linear_clip_k0.30", "kind":"linear",  "k":0.30, "clip_min":0.0, "clip_max":1.0},
    {"name":"linear_clip_k0.50", "kind":"linear",  "k":0.50, "clip_min":0.0, "clip_max":1.0},
    {"name":"logistic_a0_b1.2",  "kind":"logistic","a":0.0,  "b":1.2,  "clip_min":0.0, "clip_max":1.0},
    {"name":"logistic_a0_b1.8",  "kind":"logistic","a":0.0,  "b":1.8,  "clip_min":0.0, "clip_max":1.0},
]

# -----------------------------
# Helpers
# -----------------------------
def zscore_series(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - mu) / sd

def clip01(x: np.ndarray | pd.Series, lo: float = 0.0, hi: float = 1.0):
    return np.clip(x, lo, hi)

def exposure_from_score(score: pd.Series, spec: dict) -> pd.Series:
    # Higher score = more fragile => lower exposure
    if spec["kind"] == "linear":
        ex = 1.0 - spec["k"] * score
        ex = clip01(ex, spec["clip_min"], spec["clip_max"])
        return pd.Series(ex, index=score.index)
    if spec["kind"] == "logistic":
        # exposure in (0,1): 1/(1+exp(a + b*score))
        ex = 1.0 / (1.0 + np.exp(spec["a"] + spec["b"] * score))
        ex = clip01(ex, spec["clip_min"], spec["clip_max"])
        return pd.Series(ex, index=score.index)
    raise ValueError(f"Unknown kind: {spec['kind']}")

def summarize_strategy(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"n":0, "mean":np.nan, "vol":np.nan, "p05":np.nan, "min":np.nan}
    return {
        "n": int(r.count()),
        "mean": float(r.mean()),
        "vol": float(r.std()),
        "p05": float(np.percentile(r, 5)),
        "min": float(r.min()),
    }

# -----------------------------
# Load + build score
# -----------------------------
def main():
    assert IN_PATH.exists(), f"Missing {IN_PATH}"
    df = pd.read_csv(IN_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    # drop the last quarter with missing forward return
    df = df.dropna(subset=[RET_COL]).copy()

    # Build a continuous fragility score using rolling z-scores.
    # Simple, interpretable, and avoids re-fitting each step.
    df["dti_z"] = zscore_series(df[DTI_COL], ROLL_WIN)
    df["ms_z"]  = zscore_series(df[MS_COL],  ROLL_WIN)

    # Fragility score: higher when affordability stress is high AND supply pressure is high.
    # This mirrors Phase 3 logic: interaction matters.
    # You can tune weights; start simple.
    df["fragility_score"] = (df["dti_z"].fillna(0) * df["ms_z"].fillna(0))

    # OOS split
    first_test = pd.to_datetime(FIRST_TEST)
    df["is_oos"] = df[DATE_COL] >= first_test

    # Baseline: fully invested
    df["exposure_baseline"] = 1.0
    df["strat_baseline"] = df["exposure_baseline"] * df[RET_COL]

    # Phase2 reference (optional): merge invested_p2 if you want direct comparison
    if P2_PATH.exists():
        p2 = pd.read_csv(P2_PATH, usecols=["date","invested_p2","strat_p2"])
        p2["date"] = pd.to_datetime(p2["date"])
        df = df.merge(p2, on="date", how="left")
    else:
        df["invested_p2"] = np.nan
        df["strat_p2"] = np.nan

    # Run each exposure spec
    all_summ = []
    specs_out = []

    for spec in SPECS:
        name = spec["name"]
        ex = exposure_from_score(df["fragility_score"], spec)
        # gate+scaling: only scale when Phase2 gate says invested
        gate = df["invested_p2"]
        if gate.notna().any():
            ex2 = ex * gate.fillna(1.0)
        else:
            ex2 = ex

        df[f"exposure_{name}"] = ex2
        df[f"strat_{name}"] = ex2 * df[RET_COL]

        # summarize OOS only
        oos = df.loc[df["is_oos"], f"strat_{name}"]
        summ = summarize_strategy(oos)
        summ.update({"strategy": name, "window": ROLL_WIN, "first_test": FIRST_TEST})
        all_summ.append(summ)

        specs_out.append({"strategy": name, **spec, "roll_win": ROLL_WIN, "first_test": FIRST_TEST})

    # Add baseline + phase2 summaries (OOS)
    base_oos = df.loc[df["is_oos"], "strat_baseline"]
    s0 = summarize_strategy(base_oos)
    s0.update({"strategy":"baseline_full", "window": ROLL_WIN, "first_test": FIRST_TEST})
    all_summ = [s0] + all_summ

    if df["strat_p2"].notna().any():
        p2_oos = df.loc[df["is_oos"], "strat_p2"]
        s2 = summarize_strategy(p2_oos)
        s2.update({"strategy":"phase2_gate", "window": ROLL_WIN, "first_test": FIRST_TEST})
        all_summ = [s0, s2] + all_summ[1:]

    # Save
    OUT_SERIES.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SPECS.parent.mkdir(parents=True, exist_ok=True)

    # Keep only a clean time series set
    keep_cols = [DATE_COL, RET_COL, "is_oos", "fragility_score", "dti_z", "ms_z", "exposure_baseline", "strat_baseline"]
    if "invested_p2" in df.columns:
        keep_cols += ["invested_p2","strat_p2"]
    # add all exposure/strat cols
    keep_cols += [c for c in df.columns if c.startswith("exposure_") and c not in keep_cols]
    keep_cols += [c for c in df.columns if c.startswith("strat_") and c not in keep_cols]

    df_out = df[keep_cols].copy()
    df_out.to_csv(OUT_SERIES, index=False)

    pd.DataFrame(all_summ).to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(specs_out).to_csv(OUT_SPECS, index=False)

    print("wrote:", OUT_SERIES)
    print("wrote:", OUT_SUMMARY)
    print("wrote:", OUT_SPECS)
    print("\nOOS summary preview:\n", pd.DataFrame(all_summ).sort_values("p05"))

if __name__ == "__main__":
    main()
