from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Labeling horizon — must match add_correction_label default (horizon_max_q=20)
# ---------------------------------------------------------------------------
_LABEL_HORIZON = 20

# ---------------------------------------------------------------------------
# Joint-trigger crash parameters
# ---------------------------------------------------------------------------
_ROLL_W        = 20      # quarters for rolling percentile rank (inline, no import)
_JOINT_THRESH  = 0.65    # dti_pct_roll > this qualifies for joint trigger
_JOINT_PROB    = 0.45    # per-eligible-row trigger probability
_JOINT_DUR_MIN = 3
_JOINT_DUR_MAX = 4
_JOINT_SHOCK_LO = 0.070
_JOINT_SHOCK_HI = 0.090

# Background noise crash params
_BG_PROB      = 0.03
_BG_DUR_MIN   = 2
_BG_DUR_MAX   = 3
_BG_SHOCK_LO  = 0.020
_BG_SHOCK_HI  = 0.040

# GFC structural crash params
_GFC_SHOCK    = 0.065


def build_master_df(
    *,
    start: str = "1992-01-01",   # extended backward from 1999 for statistical power
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="QE")
    n = len(dates)

    # ------------------------------------------------------------------
    # Real rate and regime (unchanged)
    # ------------------------------------------------------------------
    cycle = 1.5 * np.sin(np.linspace(0, 8 * np.pi, n))
    shocks = rng.normal(0, 0.6, size=n)
    real_rate = cycle + shocks
    # Fragility regime: matches assign_fragility_regime() in src/evaluation/regime.py
    # (default threshold=0.0). Kept inline to avoid dependency coupling in Path A.
    regime = (real_rate < 0).astype(int)

    # ------------------------------------------------------------------
    # DTI (unchanged)
    # ------------------------------------------------------------------
    trend = np.linspace(85, 135, n)
    dti = trend + regime * 6 + rng.normal(0, 2.0, size=n)
    dti = np.clip(dti, 60, 180)

    # ------------------------------------------------------------------
    # Base returns (unchanged)
    # ------------------------------------------------------------------
    base_growth = 0.008 + regime * 0.004
    noise = rng.normal(0, 0.01, size=n)

    # ==================================================================
    # Inline rolling percentile rank of DTI
    # (Do NOT import from src/evaluation/transforms.py — keep Path A
    #  dependency-free per CLAUDE.md)
    # ==================================================================
    dti_arr = dti  # numpy array
    dti_pct_roll = np.empty(n)
    for t in range(n):
        start_w = max(0, t - _ROLL_W + 1)
        seg = dti_arr[start_w : t + 1]
        dti_pct_roll[t] = np.sum(seg <= dti_arr[t]) / len(seg)

    # ==================================================================
    # Crash mechanism — joint-trigger design
    # ==================================================================
    #
    # Economic motivation
    # -------------------
    # The primary crash mechanism is triggered when BOTH:
    #   (a) monetary policy is restrictive (regime==0, positive real rates)
    #   (b) affordability is stretched relative to recent history
    #       (dti_pct_roll > _JOINT_THRESH)
    #
    # This joint condition captures the scenario where stretched borrowers
    # face refinancing stress as monetary conditions tighten, triggering
    # forced deleveraging and price corrections.
    #
    # Layers
    # ------
    # Layer A — Structural GFC crash (secondary, calendar-anchored)
    #   Fixed at 2007-Q1 through 2009-Q1.  Represents the 2008 financial
    #   crisis.  Straddles the train/test boundary.
    #   Depth: -0.065/qtr × ~8q = −0.52 total (~40% peak-to-trough).
    #
    # Layer B — Background noise crashes (minor, any period)
    #   Per-quarter probability: 3%.  Duration 2-3q, depth −0.02–0.04/qtr.
    #   Mostly too small to generate y=1 labels.
    #
    # Layer C — Joint trigger crashes (PRIMARY)
    #   Eligibility: regime==0 AND dti_pct_roll > _JOINT_THRESH
    #   Per-eligible-row trigger probability: _JOINT_PROB (0.45)
    #   Duration 3-4q, depth −0.070–0.090/qtr.
    #   These are the main crash generator for the research signal.
    # ==================================================================

    crash = np.zeros(n)

    # --- Training-window index boundary ---
    train_end_idx = int(
        np.searchsorted(dates, pd.Timestamp("2007-12-31"), side="right") - 1
    )
    # Last labeled training row index (add_correction_label drops last _LABEL_HORIZON rows)
    _last_labeled = train_end_idx - _LABEL_HORIZON

    def _row(date_str: str) -> int:
        """Convert a date string to its row index in the dates array."""
        idx = int(np.searchsorted(dates, pd.Timestamp(date_str), side="left"))
        assert 0 <= idx < n, f"Date {date_str} out of panel range"
        return idx

    # --- Layer A: Structural GFC crash (calendar-anchored) ---
    _GFC_START = _row("2007-03-31")
    _GFC_END   = _row("2009-03-31")    # exclusive upper bound for slice
    assert _GFC_START <= train_end_idx < _GFC_END, (
        "GFC crash must straddle the training/test boundary"
    )
    crash[_GFC_START:_GFC_END] = -_GFC_SHOCK
    gfc_event = (_GFC_START, _GFC_END - _GFC_START, _GFC_SHOCK, "structural_gfc")

    # --- GFC suppress set: ±2 buffer around GFC crash ---
    gfc_suppress: set[int] = set(range(_GFC_START - 2, _GFC_END + 2 + 1))

    # --- Layers B & C: probabilistic crash loop ---
    # Joint trigger eligibility
    joint_eligible = (regime == 0) & (dti_pct_roll > _JOINT_THRESH)

    active_until = -1
    prob_events: list[tuple[int, int, float, str]] = []

    for t in range(n):
        if t <= active_until:
            continue
        if t in gfc_suppress:
            continue

        # Layer C — Joint trigger (primary): checked first
        if joint_eligible[t] and rng.random() < _JOINT_PROB:
            dur   = int(rng.integers(_JOINT_DUR_MIN, _JOINT_DUR_MAX + 1))
            shock = float(rng.uniform(_JOINT_SHOCK_LO, _JOINT_SHOCK_HI))
            for q in range(dur):
                if t + q < n:
                    crash[t + q] -= shock
            active_until = t + dur - 1
            prob_events.append((t, dur, shock, "joint_trigger"))
            continue

        # Layer B — Background noise (minor)
        if rng.random() < _BG_PROB:
            dur   = int(rng.integers(_BG_DUR_MIN, _BG_DUR_MAX + 1))
            shock = float(rng.uniform(_BG_SHOCK_LO, _BG_SHOCK_HI))
            for q in range(dur):
                if t + q < n:
                    crash[t + q] -= shock
            active_until = t + dur - 1
            prob_events.append((t, dur, shock, "background"))

    # ------------------------------------------------------------------
    # Assemble returns and price index
    # ------------------------------------------------------------------
    returns = base_growth + noise + crash
    real_price_index = 100 * np.exp(np.cumsum(returns))
    real_price_index = real_price_index / real_price_index[0] * 100

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    all_events = [gfc_event] + prob_events
    _print_crash_diagnostics(
        dates, all_events, train_end_idx, dti, dti_pct_roll, regime, real_price_index
    )

    return pd.DataFrame(
        {
            "date":             dates,
            "dti":              dti,
            "real_rate":        real_rate,
            "regime":           regime,
            "real_price_index": real_price_index,
        }
    )


def _print_crash_diagnostics(
    dates: pd.DatetimeIndex,
    events: list[tuple[int, int, float, str]],
    train_end_idx: int,
    dti: np.ndarray,
    dti_pct_roll: np.ndarray,
    regime: np.ndarray,
    real_price_index: np.ndarray,
) -> None:
    """Print crash generation diagnostics at dataset build time."""
    n_total = len(events)

    # Breakdown by layer
    layer_counts: dict[str, int] = {}
    for _, _, _, layer in events:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    n_train = sum(1 for (c, _, _, _) in events if c <= train_end_idx)
    n_test  = n_total - n_train

    dti_pct_at_starts = [dti_pct_roll[c] for (c, _, _, _) in events if c < len(dti_pct_roll)]

    # Joint trigger diagnostics
    jt_events = [(c, d, s, l) for (c, d, s, l) in events if l == "joint_trigger"]
    jt_regime0_frac = (
        sum(1 for (c, _, _, _) in jt_events if regime[c] == 0) / len(jt_events)
        if jt_events else float("nan")
    )
    jt_pct_frac = (
        sum(1 for (c, _, _, _) in jt_events if dti_pct_roll[c] > 0.65) / len(jt_events)
        if jt_events else float("nan")
    )

    print("[build_master_df] Crash diagnostics:")
    print(f"  Panel rows           : {len(dates)}  "
          f"({dates[0].date()} – {dates[-1].date()})")
    print(f"  Total crash events   : {n_total}")
    print(f"  By layer             : "
          f"structural_gfc={layer_counts.get('structural_gfc', 0)}  "
          f"joint_trigger={layer_counts.get('joint_trigger', 0)}  "
          f"background={layer_counts.get('background', 0)}")
    print(f"  In training window   : {n_train}  (start <= {dates[train_end_idx].date()})")
    print(f"  In test window       : {n_test}")

    if jt_events:
        print(f"  Joint trigger stats  : "
              f"regime==0 frac={jt_regime0_frac:.2f}  "
              f"pct>0.65 frac={jt_pct_frac:.2f}")

    if dti_pct_at_starts:
        dti_sorted = sorted(dti_pct_at_starts)
        print(f"  dti_pct_roll at crash starts: "
              f"min={dti_sorted[0]:.3f}  "
              f"median={dti_sorted[len(dti_sorted)//2]:.3f}  "
              f"max={dti_sorted[-1]:.3f}")

    print("  Note: Primary crash trigger = regime==0 AND dti_pct_roll > 0.65")
    print("  Event detail (start_row, date, dur, shock/qtr, total, layer, split):")
    for c, dur, shock, layer in sorted(events, key=lambda e: e[0]):
        split    = "TRAIN" if c <= train_end_idx else "TEST"
        date_str = str(dates[c].date()) if c < len(dates) else "OOB"
        print(f"    row {c:3d}  {date_str}  dur={dur}q  "
              f"shock={shock:.3f}/qtr  total={dur*shock:.3f}  "
              f"[{layer}]  [{split}]")
