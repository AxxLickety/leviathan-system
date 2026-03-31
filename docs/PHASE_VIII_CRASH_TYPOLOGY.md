# PHASE_VIII_CRASH_TYPOLOGY.md
# Phase VIII — Crash Regime Typology

**Status**: Research memo. Synthesis of Phase V–VII OOS results. No new models fitted.

---

## Motivation

The cross-market OOS validation (Phase VII) confirmed that the `real_rate < 0` regime
filter correctly identifies elevated crash risk in the test window: adverse-regime crash
frequency is materially higher than accommodative-regime frequency, and the regime overlay
avoids peak drawdowns of –86% (pooled, GFC-driven).

The 2022 rate-rise analysis (scripts/oos_2022_raterise.py) revealed a complication: the
regime signal fired correctly at the 2021Q4 transition back to adverse, but the subsequent
price correction was slow and shallow, and no quarter triggered the standard 4-quarter
crash criterion (>5% price drop within 4Q). This discrepancy prompted the typology
described here.

---

## Crash Type Definitions

### Type A — Systemic Crash

**Definition**: A rapid, deep, cross-market price collapse coinciding with or immediately
following a systemic credit or liquidity shock. The regime signal (real_rate < 0) does not
need to transition at crash onset because the adverse regime is already in place. The crash
develops within the adverse-regime window; the regime filter keeps the overlay sidelined
throughout.

**Empirical profile (GFC, 2007–2012)**:
- Peak-to-trough magnitude: –51% (Phoenix), –61% (Las Vegas), –47% (Miami)
- Crash duration (peak to trough): 16–21 quarters
- Regime at peak: adverse (regime=0) — real rates positive throughout the crash build-up and collapse (2006–2011); briefly dipped negative in 2012Q3–Q4 (QE3 era, after troughs had already been reached)
- The adverse regime was effectively continuous through the crash itself
- Cross-market return correlation during crash: 0.84–0.92 (Phoenix/LV/Miami)
- Austin exception: effectively immune (–4.5%), consistent with low prior DTI fragility

**Mechanism**: Systemic credit contraction forces simultaneous deleveraging across
correlated markets. The depth and speed of the crash means the overlay avoids the majority
of the drawdown even if the signal provides no additional advance warning beyond the
already-adverse regime state.

### Type B — Cyclical Correction

**Definition**: A slow, shallow, city-specific price correction following a return to the
adverse regime after a period of accommodative monetary policy. The regime signal
transitions (1→0) correctly at cycle turn, but the actual price peak occurs 2–4 quarters
after the transition, and the drawdown is gradual enough that no short-horizon crash
criterion fires.

**Empirical profile (2022 rate-rise, 2021Q4–present)**:
- Last accommodative quarter: 2021Q3 (real rate –0.14%, regime=1)
- Regime transition to adverse: 2021Q4 (real rate +0.11%)
- Peak-to-trough magnitude: –11.8% (Austin), –4.3% (Las Vegas), –2.6% (Miami), 0% (Phoenix — still making highs as of 2025Q4)
- Quarters from regime transition to price peak: Austin 2Q, Las Vegas 13Q, Miami 14Q
- Quarters from regime transition to trough: Austin 12Q, Las Vegas 14Q
- Cross-market correlation 2022–present: 0.43–0.91 (mixed; Miami diverges)
- 4Q crash criterion (>5% drop): fired for Austin only (3/15 quarters); Phoenix, LV, Miami: 0/15

**Mechanism**: Monetary tightening compresses affordability and slows demand, but does not
trigger forced selling or credit withdrawal. Price adjustment is demand-led and gradual.
Markets with strong underlying demand (Miami, Las Vegas) continue appreciating even in
adverse regime.

---

## Evidence Summary

| Metric | Type A (GFC) | Type B (2022) |
|---|---|---|
| Regime at crash onset | Already adverse (no transition) | Transition 1→0 at 2021Q4 |
| Peak-to-trough (median) | –49% (Sunbelt) | –8% (Austin only) |
| Crash duration | 16–21 quarters | 10–14Q (Austin); ongoing/none (others) |
| Cross-market correlation | 0.84–0.92 | 0.43–0.91 |
| 4Q crash signal (>5%) | Would fire within 4Q of peak | Fires for Austin; not for LV/Miami/Phoenix |
| Regime overlay max drawdown | 0.0% (avoided entirely) | 0.0% (fully out-of-market 2022Q1+) |
| Always-in max drawdown | –86% pooled (GFC-dominated) | –15% pooled |
| Overlay cost (foregone return) | Zero (regime stayed adverse) | –4.9% ann. (2022Q1–present) |

---

## Implication for the Regime Filter

The `real_rate < 0` filter functions as a **systemic risk gate**, not a **cyclical timing
tool**.

In Type A environments, the gate is open at crash onset and the overlay is correctly
sidelined. In Type B environments, the gate closes (transitions to adverse) near the cycle
turn, but the risk materialises slowly. Both uses are directionally correct: the adverse
regime does precede or coincide with elevated risk in both cases. The distinction is in
the severity and speed of the correction, not in the signal's accuracy.

The 4-quarter crash label used in the OOS evaluation is calibrated to Type A dynamics
(fast, deep crashes). It understates the signal's value in Type B environments because
the correction unfolds outside the measurement window.

**What the filter does**: Gates out adverse macro regimes in which systemic housing stress
is possible. Protects capital against rare but catastrophic drawdowns.

**What the filter does not do**: Time cyclical corrections or predict the quarter of peak
prices. Does not distinguish between markets that will correct sharply (Austin 2022) and
those that will continue appreciating (Miami 2022–2025) within the same adverse regime.

---

## Extensions Required for Type B Coverage

To extend the regime filter to cover Type B cyclical corrections, the following additions
would be required:

1. **Longer lag window**: Replace the 4-quarter crash look-ahead with a 12–16 quarter
   window. Based on the 2022 data, Type B corrections take 10–14 quarters from the regime
   transition to trough.

2. **City-level supply data**: Miami continued appreciating through 2025 under the same
   adverse regime as Austin. Cross-market divergence in Type B episodes suggests supply
   conditions are a second-order discriminant. Constrained supply markets (Miami, Las Vegas
   2023+) absorb demand shocks without price declines.

3. **DTI fragility at cycle peak**: The original Path A hypothesis (DTI as a fragility
   filter) was tested in the GFC context. Applying the DTI fragility cutoff to 2021 peak
   quarters (high DTI + adverse regime) may recover predictive power for Type B corrections
   that the regime filter alone cannot distinguish.

4. **Real-time vintage data**: Type B corrections are slow enough that data revisions to
   HPI inputs could affect signal quality. FRED real-time vintage data (not used in current
   analysis) would be required for a production-grade Type B filter.

---

## Data Sources

- FRED FHFA All-Transactions HPI (ATNHPIUS{MSA}Q): Austin (12420), Phoenix (38060),
  Las Vegas (29820), Miami-Miami Beach-Kendall (33124)
- FRED Real Interest Rate: REAINTRATREARAT10Y (10-year TIPS-implied, monthly → quarterly)
- Analysis notebooks: notebooks/phase_oos/phase_oos_crossmarket_regime_only.ipynb,
  scripts/oos_2022_raterise.py
- No new models or thresholds fitted in Phase VIII

---

*Phase VIII is a synthesis phase. All quantitative claims are derived from prior-phase
results. The typology is descriptive, not predictive — no new signal is proposed.*
