from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Labeling horizon — must match add_correction_label default (horizon_max_q=20)
# ---------------------------------------------------------------------------
_LABEL_HORIZON = 20


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
    # Fragility-driven crash mechanism
    # ==================================================================
    #
    # Economic motivation
    # -------------------
    # Accommodative monetary policy (regime=1, negative real rates) enables
    # borrowers to over-extend (high DTI).  When conditions normalize to
    # positive real rates (regime=0), stretched borrowers face refinancing
    # stress and forced deleveraging, triggering price corrections.
    #
    # Crash placement: date-anchored (not proportional to dataset length)
    # -------------------------------------------------------------------
    # All deterministic crashes are placed at fixed calendar dates so that
    # extending the dataset backward does not shift the crash calendar.
    # This preserves the train/test boundary semantics and keeps crash
    # density stable regardless of start date.
    #
    # Layers
    # ------
    # Layer 1a — Early training crash (~1996-Q2)
    #   Represents a mid-1990s housing stress episode.  Creates ~12 positive
    #   labels in the early part of the training window (rows ~7-18).
    #   Depth: -0.055/qtr × 4q = −0.22 total (moderate; ~20% price drop).
    #
    # Layer 1b — Pre-GFC training crash (~2005-Q1)
    #   Represents the 2004-2005 affordability stress event.  Creates ~11
    #   positive labels in the late training window (rows ~33-43).
    #   Depth: -0.075/qtr × 4q = −0.30 total (~26% price drop).
    #
    # Layer 2 — GFC-proxy crash (~2007-Q1 → ~2009-Q1)
    #   Straddles the training/test boundary.  Provides a structural 2-year
    #   downward episode mirroring the 2008 financial crisis.
    #   Depth: -0.065/qtr × 8q = −0.52 total (~40% peak-to-trough).
    #
    # Layer 3 — Post-GFC test crash (~2016-Q1)
    #   Ensures the test period contains a crash episode visible to
    #   add_correction_label, so crash_frequency_table has nonzero cells.
    #   Depth: -0.075/qtr × 4q = −0.30 total (~26% price drop).
    #
    # Layer 4 — Probabilistic fragility-driven crashes (full sample)
    #   Eligible: regime==0 AND dti > p75(dti).  Trigger probability: 30%.
    #   Duration 3-4q, depth −0.065 to −0.085/qtr.  Expanding suppression
    #   window prevents overlapping with deterministic events.
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

    # --- Layer 1a: Early training crash (~1996-Q2) ---
    # row 17; trough at row 21; y=1 for training rows ~7-18
    _L1A_START = _row("1996-06-30")
    _L1A_DUR   = 4
    _L1A_SHOCK = 0.055
    assert _L1A_START + _L1A_DUR <= train_end_idx, (
        "Layer 1a crash must complete within training window"
    )
    for q in range(_L1A_DUR):
        crash[_L1A_START + q] -= _L1A_SHOCK
    l1a_event = (_L1A_START, _L1A_DUR, _L1A_SHOCK)

    # --- Layer 1b: Pre-GFC training crash (~2005-Q1) ---
    # row 52; trough at row 56; y=1 for training rows ~33-43
    _L1B_START = _row("2005-03-31")
    _L1B_DUR   = 4
    _L1B_SHOCK = 0.090   # 0.090 vs 0.075: extra margin overcomes regime-1 appreciation
                         # in rows 37-56 (7 quarters at base=0.012 vs 0.008)
    assert _L1B_START + _L1B_DUR <= train_end_idx, (
        "Layer 1b crash must complete within training window"
    )
    assert _L1B_START + _L1B_DUR <= _last_labeled + _LABEL_HORIZON, (
        "Layer 1b trough must be visible from at least one labeled training row"
    )
    for q in range(_L1B_DUR):
        crash[_L1B_START + q] -= _L1B_SHOCK
    l1b_event = (_L1B_START, _L1B_DUR, _L1B_SHOCK)

    # --- Layer 2: GFC-proxy crash (straddles train/test boundary) ---
    # rows 60-67 (2007-Q1 through 2008-Q4); spans train_end_idx=63
    _L2_START = _row("2007-03-31")
    _L2_END   = _row("2009-03-31")    # exclusive upper bound for slice
    _L2_SHOCK = 0.065
    assert _L2_START <= train_end_idx < _L2_END, (
        "GFC crash must straddle the training/test boundary"
    )
    crash[_L2_START:_L2_END] = -_L2_SHOCK
    l2_event = (_L2_START, _L2_END - _L2_START, _L2_SHOCK)

    # --- Layer 3: Post-GFC test crash (~2016-Q1) ---
    # row 96; trough at row 100; produces ~20 positive test labels
    _L3_START = _row("2016-03-31")
    _L3_DUR   = 4
    _L3_SHOCK = 0.075
    assert _L3_START > train_end_idx, (
        "Post-GFC test crash must be in test period"
    )
    for q in range(_L3_DUR):
        if _L3_START + q < n:
            crash[_L3_START + q] -= _L3_SHOCK
    l3_event = (_L3_START, _L3_DUR, _L3_SHOCK)

    # --- Layer 4: Probabilistic fragility-driven crashes ---
    _FRAG_PROB    = 0.30
    _PROB_DUR_MIN = 3
    _PROB_DUR_MAX = 4
    _SHOCK_LO     = 0.065
    _SHOCK_HI     = 0.085

    dti_p75 = float(np.percentile(dti, 75))
    fragile = (regime == 0) & (dti > dti_p75)

    # Suppress probabilistic triggers near deterministic events (±1 buffer)
    _det_suppress: set[int] = set()
    for s, d, _ in [l1a_event, l1b_event, l2_event, l3_event]:
        for q in range(d + 2):
            _det_suppress.add(s - 1 + q)
            _det_suppress.add(s + q)

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
    all_events = [l1a_event, l1b_event, l2_event, l3_event] + prob_events
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
    print(f"  Panel rows           : {len(dates)}  "
          f"({dates[0].date()} – {dates[-1].date()})")
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
