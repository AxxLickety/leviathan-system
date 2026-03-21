from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Labeling horizon — must match add_correction_label default (horizon_max_q=20)
# ---------------------------------------------------------------------------
_LABEL_HORIZON = 20


def build_master_df(
    *,
    start: str = "1999-01-01",
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
    # Fragility-driven crash mechanism
    # ==================================================================
    #
    # Economic motivation
    # -------------------
    # Periods of accommodative monetary policy (real_rate < 0, regime=1)
    # push DTI toward stressed levels as borrowers over-extend.  When
    # rate conditions normalize (regime=0, positive real rates), those
    # stretched borrowers face refinancing pressure and reduced purchasing
    # power.  If DTI is simultaneously in the top quartile of its
    # historical distribution, the market is in a "fragile" state where a
    # sentiment shift or credit tightening can trigger forced selling and
    # a self-reinforcing price correction.
    #
    # Implementation layers
    # ---------------------
    # Layer 1 — Training crash (deterministic, row ~24 ≈ 2005)
    #   Ensures the training window contains a crash fully observable by
    #   add_correction_label.  Placement: trough at row 28 is within the
    #   20-quarter lookahead of training rows 8-15, producing ~7 y=1 labels
    #   at ≈ 44% prevalence.
    #
    # Layer 2 — GFC-proxy crash (structural, rows ~31-39 ≈ 2006-2009)
    #   Represents a large, infrequent systemic event matching the scale of
    #   the 2008 housing crisis.  Enhanced from original -0.04/qtr to
    #   -0.065/qtr for a more realistic 35-40% peak-to-trough decline.
    #
    # Layer 3 — Post-GFC test crash (deterministic, row ~68 ≈ 2016)
    #   Ensures the test period also contains a crash visible to
    #   add_correction_label, so crash_frequency_table has meaningful cell
    #   counts.  Without this, the test period would be entirely y=0.
    #
    # Layer 4 — Probabilistic fragility-driven crashes (full sample)
    #   Eligible rows (regime==0 AND DTI > p75) trigger additional crashes
    #   with 30% probability, using an expanding suppression window to
    #   prevent overlapping events.  Produces scattered crisis episodes
    #   without obvious periodicity.
    #
    # Crash shape
    # -----------
    # Each episode applies a per-quarter log-return shock over 3-4 quarters,
    # creating a short downward path rather than a single jump.  Consecutive
    # crash quarters produce the clustering effect (2-4 quarter episodes).
    # ==================================================================

    crash = np.zeros(n)

    # --- Training-window index boundary ---
    train_end_idx = int(
        np.searchsorted(dates, pd.Timestamp("2007-12-31"), side="right") - 1
    )
    # Last training row that receives a y-label from add_correction_label.
    _last_labeled = train_end_idx - _LABEL_HORIZON   # = 35 - 20 = 15

    # --- Layer 1: Deterministic training crash ---
    # Trough at _last_labeled + _LABEL_HORIZON - 7 = row 28; dur=4 so start=24.
    # Row 24 ≈ 2005-Q1 in synthetic calendar.
    _TC_START = _last_labeled + 9   # = 24
    _TC_DUR   = 4
    _TC_SHOCK = 0.075               # per quarter; total = 0.30 log return drop
    assert _TC_START + _TC_DUR <= train_end_idx, (
        "Training crash must complete within training window"
    )
    for q in range(_TC_DUR):
        crash[_TC_START + q] -= _TC_SHOCK
    train_event = (_TC_START, _TC_DUR, _TC_SHOCK)

    # --- Layer 2: GFC-proxy crash ---
    # Enhanced from original -0.04 to -0.065/qtr for realistic crisis depth.
    _GFC_START = int(n * 0.30)   # ≈ row 31, ≈ late 2006
    _GFC_END   = int(n * 0.38)   # ≈ row 39, ≈ mid 2009
    _GFC_SHOCK = 0.065
    crash[_GFC_START:_GFC_END] = -_GFC_SHOCK
    gfc_event = (_GFC_START, _GFC_END - _GFC_START, _GFC_SHOCK)

    # --- Layer 3: Deterministic post-GFC test crash ---
    # Row 68 ≈ 2016-Q1.  Test sub-row = 68 - 36 = 32; trough at sub-row 36.
    # Visible to test sub-rows 17-47 → ~20 positive test labels at ~42% prevalence.
    _PCT_START = 68   # full-panel row index
    _PCT_DUR   = 4
    _PCT_SHOCK = 0.075
    assert _PCT_START > train_end_idx, "Post-GFC test crash must be in test period"
    for q in range(_PCT_DUR):
        if _PCT_START + q < n:
            crash[_PCT_START + q] -= _PCT_SHOCK
    test_event = (_PCT_START, _PCT_DUR, _PCT_SHOCK)

    # --- Layer 4: Probabilistic fragility-driven crashes ---
    _FRAG_PROB    = 0.30   # per-quarter trigger probability for eligible rows
    _PROB_DUR_MIN = 3
    _PROB_DUR_MAX = 4
    _SHOCK_LO     = 0.065
    _SHOCK_HI     = 0.085

    dti_p75 = float(np.percentile(dti, 75))
    fragile = (regime == 0) & (dti > dti_p75)

    # Build suppression mask from deterministic events to avoid overlap
    _det_suppress = set()
    for start_row, dur, _ in [train_event, gfc_event, test_event]:
        for q in range(dur + 2):      # +2 quarter buffer around each event
            _det_suppress.add(start_row - 1 + q)
            _det_suppress.add(start_row + q)

    active_until = -1
    prob_events: list[tuple[int, int, float]] = []

    for t in range(n):
        if t <= active_until or t in _det_suppress:
            continue
        if fragile[t] and rng.random() < _FRAG_PROB:
            dur   = int(rng.integers(_PROB_DUR_MIN, _PROB_DUR_MAX + 1))
            shock = float(rng.uniform(_SHOCK_LO, _SHOCK_HI))
            for q in range(dur):
                if t + q < n:
                    crash[t + q] -= shock
            active_until = t + dur - 1
            prob_events.append((t, dur, shock))

    # ------------------------------------------------------------------
    # Assemble returns and price index
    # ------------------------------------------------------------------
    returns = base_growth + noise + crash
    real_price_index = 100 * np.exp(np.cumsum(returns))
    real_price_index = real_price_index / real_price_index[0] * 100

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    all_events = [train_event, gfc_event, test_event] + prob_events
    _print_crash_diagnostics(dates, all_events, train_end_idx, dti, real_price_index)

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
    events: list[tuple[int, int, float]],
    train_end_idx: int,
    dti: np.ndarray,
    real_price_index: np.ndarray,
) -> None:
    """Print crash generation diagnostics at dataset build time."""
    n_total = len(events)
    n_train = sum(1 for (c, _, _) in events if c <= train_end_idx)
    n_test  = n_total - n_train

    dti_at_starts = [dti[c] for (c, _, _) in events if c < len(dti)]

    print("[build_master_df] Crash diagnostics:")
    print(f"  Total crash events   : {n_total}")
    print(f"  In training window   : {n_train}  (start <= {dates[train_end_idx].date()})")
    print(f"  In test window       : {n_test}")
    if dti_at_starts:
        dti_sorted = sorted(dti_at_starts)
        print(f"  DTI at crash starts  : "
              f"min={dti_sorted[0]:.1f}  "
              f"median={dti_sorted[len(dti_sorted)//2]:.1f}  "
              f"max={dti_sorted[-1]:.1f}")
    print("  Event detail (start_row, date, dur, shock/qtr, total):")
    for c, dur, shock in sorted(events):
        split    = "TRAIN" if c <= train_end_idx else "TEST"
        date_str = str(dates[c].date()) if c < len(dates) else "OOB"
        print(f"    row {c:3d}  {date_str}  dur={dur}q  "
              f"shock={shock:.3f}/qtr  total={dur*shock:.3f}  [{split}]")
