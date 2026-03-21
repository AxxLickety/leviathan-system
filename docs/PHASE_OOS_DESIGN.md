# PHASE_OOS_DESIGN.md
# Out-of-Sample Evaluation: Pre-Specified Research Design

**Status**: Pre-registered design. Must not be modified after OOS evaluation begins.
**Last updated**: 2026-03-21

---

## 1. Objective

This phase tests whether the regime-conditional DTI filter — as specified entirely on
training data — retains its downside-protection properties on data it has never seen.

The central design requirement is **separation of threshold discovery from evaluation**.
Every threshold, coefficient, and decision rule used in the test phase must be derived
exclusively from training data and frozen before the test window is opened. Opening
the test window, inspecting test outcomes, and then adjusting thresholds constitutes
look-ahead bias and invalidates the evaluation.

This document is the binding specification. Any deviation must be recorded explicitly
with justification.

---

## 2. Data Split

### Training window
**1990Q1 – 2007Q4** (inclusive)

### Test window
**2008Q1 – latest available quarter** (inclusive)

### Rationale
The 2007Q4 / 2008Q1 boundary is chosen so that the 2008 global financial crisis and
its housing market consequences fall entirely in the test set. This is the primary
stress event the system is designed to detect. If the crisis were included in training,
the logit model and DTI thresholds would be fitted with direct knowledge of the most
severe correction in the sample — producing thresholds optimised on the very event they
are supposed to predict out-of-sample.

The training window (1990–2007) provides approximately 72 quarters of history,
which is modest for the interaction logit and rolling IC analysis. It is used because
it is the longest pre-crisis window available, not because it is considered ample.
The test window captures the 2008–2009 correction, the post-crisis recovery,
the low-rate regime of 2010–2021, and the rate-rise cycle of 2022 onwards — covering
multiple distinct macro regimes.

No data from the test window may be used in any fitting or threshold derivation step.

---

## 3. Threshold Classification

Each decision input must be classified as either a fixed prior or a learned value.
Learned values must be derived from training data only and frozen before test evaluation.

### 3.1 Real rate threshold
- **Type**: Fixed prior (not learned)
- **Value**: 0.0 (zero nominal real rate boundary)
- **Justification**: This is a structural economic boundary, not a data-fitted parameter.
  Negative real rates represent financial repression; positive real rates represent
  conventional monetary conditions. The threshold does not require estimation.
- **Action**: Hard-code as a constant. Do not fit or optimise on any data window.

### 3.2 DTI cutoff
- **Type**: Learned from training data only
- **Derivation**: Derived from the training-window distribution of DTI within
  `regime == 0` observations, using a quantile-search procedure that maximises a
  tail-protection objective on the training set. The same logic currently implemented
  in the strategy filter scripts is used here; it is not treated as a canonical
  authority but as the operationalisation of that search. The final cutoff is the
  value selected at the end of the training window (2007Q4).
- **Freeze requirement**: The cutoff value must be recorded as a scalar constant
  before the test window is opened. It must not be re-estimated on test data.

### 3.3 Path A logit coefficients
- **Type**: Learned from training data only
- **Derivation**: `fit_interaction_logit()` fitted on the labeled training-window
  dataset (`add_correction_label()` applied to 1990Q1–2007Q4 only).
- **Outputs frozen**: `coef.csv` and `thresholds.csv` written from the training fit.
  These files are the sole source of logit-derived thresholds in the test phase.
- **Freeze requirement**: Coefficients and derived DTI thresholds must be written to
  disk before any test-window data is loaded. The test phase reads these files as
  read-only constants.

### 3.4 Freeze assertion requirement
Before any test-window row is evaluated, the code must assert that all learned values
are loaded from pre-written files and that no fitting function has been called on
test-window data. See Section 4.

---

## 4. Data Firewall Rules

The following rules are binding. Violations constitute look-ahead bias.

1. **No test data in training steps.** All fitting functions (`fit_interaction_logit`,
   quantile threshold search, rolling IC) must receive DataFrames filtered to
   `date <= 2007-12-31` before being called.

2. **Thresholds are constants in the test phase.** The test evaluation script must
   load thresholds from frozen files (`outputs/path_a/thresholds.csv`,
   `outputs/path_a/coef.csv`). It must not call any fitting or optimisation function.

3. **Canonical column name.** This phase uses `"fwd_return"` as the single column
   name for the 4-quarter log return throughout. Any source file that uses
   `"fwd_ret_4q"` must rename the column exactly once at load time, immediately
   after reading, with an assertion confirming equivalence:

   ```python
   assert "fwd_ret_4q" in df.columns, "Expected fwd_ret_4q from source"
   df = df.rename(columns={"fwd_ret_4q": "fwd_return"})
   ```

   No code in this phase may reference `"fwd_ret_4q"` after the load step.

4. **Explicit assertion required.** The test evaluation script must include an
   assertion block of the following form before processing any test-window row:

   ```python
   assert train_end == pd.Timestamp("2007-12-31"), "Training cutoff violated"
   assert test_start == pd.Timestamp("2008-03-31"), "Test start violated"
   assert dti_cutoff is not None, "DTI cutoff not loaded from frozen file"
   assert "fwd_return" in df.columns, "Column rename not applied at load"
   # No fitting functions called below this line
   ```

5. **No iteration after viewing results.** Once the test window has been evaluated
   and outputs inspected, thresholds must not be adjusted. Any adjustment requires
   discarding the current test evaluation entirely and pre-registering a new design.

---

## 5. Required Outputs

All outputs must be produced for both the training window and the test window unless
stated otherwise.

### 5.1 Regime × supply count table
A cross-tabulation of regime (0 / 1) × supply condition (high / normal) for both
windows, showing observation counts and crash event counts in each cell.
Required to confirm that the test window contains sufficient observations in each
regime cell to support inference. Cells with very small sample sizes (as a rough
guide, fewer than 10 observations) should be flagged in all tables and interpreted
with explicit caution. They are not automatically excluded — their data is reported —
but conclusions drawn from them must acknowledge the heightened uncertainty.

### 5.2 Conditional crash frequency table
For each regime × supply cell: crash frequency (events / observations) with
**Wilson 95% confidence interval** as the primary uncertainty estimate.
Report the CI explicitly; do not summarise it as significant / not significant.
Bootstrap resampling (B ≥ 1000) may be run as a robustness check and reported
in an appendix, but the Wilson CI is the primary reported interval.
Cells with very small sample sizes (as a rough guide, fewer than 10 observations)
should be flagged in all tables and interpreted with explicit caution. They are not
automatically excluded — their data is reported — but conclusions drawn from them
must acknowledge the heightened uncertainty.

### 5.3 Overlay performance table
Three strategies evaluated on the test window:

| Strategy | Definition |
|---|---|
| Always-in | Fully invested every quarter |
| Overlay | Invested unless regime == 0 and DTI > frozen cutoff |
| Always-out | Never invested (zero return every quarter) |

Report for each: n, mean quarterly log return, vol (ddof=1), Sharpe (ddof=1), p05,
max drawdown. Use `summarize()` from `src/backtests/evaluation.py`.

### 5.4 Equity curve plot
Cumulative log return over the test window for all three strategies on a single chart.
Mark the 2008–2009 crisis window and the 2022 rate-rise onset with vertical lines.
Save to `outputs/oos/equity_curve.png`.

### 5.5 Threshold sensitivity analysis
Re-evaluate the overlay strategy on the test window using DTI cutoffs at ±10% and
±20% of the frozen value. Report the performance table (5.3) for each. Purpose: confirm
that results are not knife-edge sensitive to the exact cutoff. Do not select a
different cutoff based on these results.

---

## 6. Benchmark Definition

### Always-in
Fully invested in the housing price index proxy every quarter. Return equals
`fwd_return` (4-quarter log return on `real_price_index`) with no filtering.
This is the passive buy-and-hold baseline.

### Always-out
Zero return every quarter. Represents the cost of permanent caution. Included to
confirm that the overlay does not achieve better risk metrics merely by being out
of the market more often regardless of regime.

### Optional naive rule benchmark
A regime-blind DTI filter: exit whenever DTI exceeds the frozen cutoff, regardless
of regime. Included if sample size permits. Purpose: isolate the contribution of
the regime interaction from the DTI level effect alone.

---

## 7. Evaluation Criteria

### 7.1 Primary criteria
These are the core questions. Results must be reported in full regardless of direction.

1. **Tail protection**: Is the overlay p05 less negative than always-in p05 on the
   test window? Report the difference and its magnitude.
2. **Drawdown reduction**: Is the overlay max drawdown smaller in magnitude than
   always-in max drawdown on the test window? Report both values.
3. **Crash concentration**: Is crash frequency higher in the `regime == 0,
   DTI > cutoff` cell than in the complement cell? Report Wilson 95% CIs for both
   cells and describe whether they overlap.

### 7.2 Secondary criteria
These provide context but do not determine the primary finding.

1. **Mean return cost**: What is the difference in mean quarterly log return between
   overlay and always-in? A large negative difference weakens the case for the overlay
   even if tail metrics improve.
2. **Sharpe comparison**: Report overlay vs always-in Sharpe. Given the small test
   sample, treat this as directional evidence only.
3. **Cross-market consistency**: Do Austin and Toronto show results in the same
   direction on primary criteria 1 and 2? Pooled results that mask city-level
   disagreement should be noted.
4. **Sensitivity stability**: Do the ±10% and ±20% cutoff variants (Section 5.5)
   show the same directional result on primary criteria?

### 7.3 Interpretation rules

Results must be summarised using one of these three verdicts, stated explicitly at
the top of the results section:

- **SUPPORTS HYPOTHESIS**: All three primary criteria show directional results
  consistent with the hypothesis, and the evidence is substantive enough to warrant
  a qualified positive interpretation. The strength of evidence must be described
  in plain language — a directional result with wide CIs is weaker evidence than
  one with narrow CIs, and both must be characterised as such.
- **DOES NOT SUPPORT HYPOTHESIS**: One or more primary criteria show directional
  results inconsistent with the hypothesis.
- **INCONCLUSIVE**: Primary criteria are directionally consistent but the
  evidence is too weak to distinguish signal from noise given the sample
  constraints, or the results are mixed across cities or regime cells in a way
  that prevents a coherent interpretation.

No numerical threshold determines which verdict applies. The researcher must
describe the weight of evidence across all primary criteria and reach a verdict
that honestly reflects it. The inconclusive verdict is a legitimate and expected
possible outcome given the sample size constraints documented in Section 8.

---

## 8. Limitations

The following limitations apply regardless of outcome and must be acknowledged in
any write-up or presentation of results.

### 8.1 Small sample size
The test window contains approximately 64–72 quarters depending on the latest
available data. Crash events are rare; the test window may contain only 1–2 distinct
correction episodes. Confidence intervals will be wide. Point estimates should not
be over-interpreted, and the inconclusive verdict (Section 7.3) is a plausible
outcome on this basis alone.

### 8.2 Sparse crash events
The correction label (`y = 1`) is defined as a >10% drawdown within 20 quarters.
In a 70-quarter test window, the number of positive labels is small. The Wilson CI
is used as the primary interval precisely because the normal approximation is
unreliable at these sample sizes. Any cell with very few observations should be
flagged and its contribution to the primary evaluation treated with caution rather
than excluded outright — absence of a result in a small cell is not the same as a
null result.

### 8.3 Geographic limitation
The evaluation universe is currently limited to Austin and Toronto. These two markets
share structural similarities (English-speaking, Anglo-Saxon mortgage market, high
immigration inflows) that may make the regime filter appear more generalisable than
it is. Results should not be extrapolated to Continental European, Asian, or
emerging-market housing systems without separate validation.

### 8.4 Data revision caveat
Real interest rate and house price index series are subject to revision. The values
used at decision time (real-time data) may differ from the revised values available
in hindsight. This analysis uses the current vintage of revised data throughout,
which introduces a mild look-ahead bias that cannot be corrected without real-time
data archives. This caveat applies equally to training and test windows and is noted
for completeness.
