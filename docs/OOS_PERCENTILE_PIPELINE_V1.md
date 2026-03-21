# OOS Percentile-DTI Pipeline V1 — Research Memo

**Date:** 2026-03-21
**Depends on:** `docs/OOS_OVERLAY_MULTIBASELINE_PERCENTILE_V1.md`
**Scripts:** `scripts/oos_train_pct.py`, `scripts/oos_eval_pct.py`
**Utility:** `src/evaluation/transforms.py`
**Artifacts:** `outputs/oos_pct/`

---

## Motivation

The percentile multibaseline experiment revealed that the original Leviathan gate was not
what it appeared to be. The absolute walk-forward cutoff (98.81 DTI) was exceeded by all
64 test-period observations, so the gate reduced to:

> *Exit when `regime == 0`*

The DTI condition contributed nothing. This experiment redesigns the pipeline from the
ground up to use rolling-20q percentile-ranked DTI natively — so the DTI dimension is
active and genuinely tested from the start.

---

## Changes from the Original Pipeline

| Aspect | Original (oos_train.py) | Percentile (oos_train_pct.py) |
|---|---|---|
| DTI feature | Raw absolute DTI | `dti_pct_roll20` (rolling 20q causal rank) |
| Walk-forward grid | `r0_dti.quantile(q)` for q in 0.60–0.95 | Direct thresholds 0.50–0.95 |
| Frozen cutoff units | Raw DTI level (~99) | Percentile in (0, 1] |
| Logit features | `const, dti, regime, dti×regime` | `const, dti_pct_roll, regime, dti_pct×regime` |
| Fit method | Newton | BFGS (Newton fails due to near-separation) |
| Output dir | `outputs/oos/frozen/` | `outputs/oss_pct/frozen/` |

**Rolling-percentile definition:** at time t, `dti_pct_roll[t]` = fraction of DTI values in
`[t−19, t]` that are ≤ `DTI[t]`. Bounded in (0, 1]. Causal by construction.

---

## Training Results

```
Labeled rows       : 44  (y=1: 18, y=0: 26)
Regime-0 rows      : 26  (40.9% prevalence)
DTI feature        : dti_pct_roll20
Logit converged    : True (BFGS)
Near-separation    : True (|regime coef| = 244.4)
Pseudo R²          : 0.368
```

### Logit Coefficients

| Parameter | Coef | p-value |
|---|---|---|
| const | +3.060 | 0.020 |
| dti_pct_roll | **−4.397** | **0.030** |
| regime | −244.4 | 0.980 |
| dti_pct_roll × regime | +244.7 | 0.980 |

**dti_pct_roll is significant and negative (p=0.030)**: lower rolling-percentile rank
predicts more crashes. This is the opposite of the raw-DTI hypothesis.

The regime interaction term is near-separated (|coef|=244): the model assigns essentially
all predictive power to the `regime` column (only 2 of 18 regime-1 rows are y=1 vs
16 of 26 regime-0 rows), leaving the interaction term numerically unstable.

### Analytical Thresholds

| Regime | p=0.10 threshold | p=0.20 threshold |
|---|---|---|
| 0 | 1.196 (out of range) | 1.011 (out of range) |
| 1 | 0.995 (boundary) | 0.999 (boundary) |

**For regime=0, no valid threshold exists**: the logit assigns P(y=1 \| regime=0) > 10%
for every possible dti_pct_roll value in [0, 1]. This means the model says all
regime-0 quarters are high-risk, regardless of DTI percentile rank.

### Walk-Forward Result

**Frozen percentile cutoff: 0.5000** (boundary of the search grid 0.50–0.95).

The walk-forward selected the minimum available threshold. At pct_cutoff=0.50, filtering
`regime==0 AND dti_pct_roll > 0.50` in the training window captures all regime-0 rows with
elevated relative DTI — which in practice coincides with almost every regime-0 row in
training (regime-0 training rows have mean `dti_pct_roll` = 0.77). Any higher threshold
(0.55, 0.60, ...) starts admitting regime-0 rows with moderate pct_roll, which have negative
4-quarter returns and worsen p05. The optimizer converged to the pure-regime limit.

---

## OOS Evaluation Results

Test period: 2008-Q1 → 2024-Q4 (48 labeled rows).

| Strategy | mean | sharpe | p05 | maxdd | pct_invested |
|---|---|---|---|---|---|
| always_in | −0.026 | −0.277 | −0.190 | −0.862 | 100% |
| overlay (pct gate) | −0.028 | −0.310 | −0.190 | −0.843 | 81.3% |
| **regime_only** | **−0.004** | **−0.069** | **−0.149** | **−0.574** | **39.6%** |
| always_out | 0.000 | NaN | 0.000 | 0.000 | 0% |

**Leviathan gate (regime==0 AND dti_pct_roll > 0.50) fires on 12 of 32 regime-0 test rows
(37.5% of regime-0, 17.6% of all test rows).**

---

## Key Findings

### 1. The overlay underperforms regime-only

| Comparison | Δsharpe | Δp05 | Δmaxdd |
|---|---|---|---|
| overlay vs regime_only | −0.241 | **−0.041** | **−0.269** |

The pct_cutoff=0.50 gate is **less effective** than simply exiting all regime-0 quarters.
The 20 regime-0 test rows with dti_pct_roll ≤ 0.50 (which the overlay keeps invested) have
worse outcomes than assumed: keeping them in the portfolio worsens p05 by 4.1pp and maxdd
by 26.9pp relative to the full regime exit.

### 2. The DTI percentile dimension is inverted from the hypothesis

The logit coefficient for `dti_pct_roll` is **−4.40**: crash risk is higher when DTI is
**lower** relative to its recent rolling history. This is structurally opposite to the
Leviathan hypothesis ("high DTI = high risk").

Why does this happen in the synthetic data? The y=1 crash labels are generated from
add_correction_label, which looks 20 quarters forward. The rows labeled y=1 are often
"pre-crash buildup" quarters — periods before a crash where DTI is at a moderate rolling
level (the crash is still 1–4 years away). The 4-quarter forward return at these rows is
often positive. The rows WITH the crash (negative 4q fwd_return) have HIGH rolling-pct DTI
because DTI has been rising into the crash.

This creates a situation where:
- High rolling-pct regime-0 rows → negative 4q returns → should be filtered (CORRECT for strategy)
- Low rolling-pct regime-0 rows → positive 4q returns → should be kept (also CORRECT)
- But these same low-pct rows are labeled y=1 (long-horizon crash incoming) → misleads the logit

The strategy-relevant signal (4q fwd_return) says "filter high-pct regime-0"; the
classification signal (y=1/0) says "flag low-pct regime-0." These point in opposite
directions, creating the inverted coefficient.

### 3. The walk-forward correctly diagnoses the signal direction

The walk-forward objective is p05 of 4-quarter filtered returns — aligned with strategy
performance, not crash classification. It correctly found that filtering high-pct regime-0
rows improves p05, converging to the pure-regime limit (threshold=0.50 removes all
regime-0 rows with elevated pct).

However, **in test the signal breaks down**: only 12 of 32 regime-0 rows have
dti_pct_roll > 0.50. The other 20 regime-0 rows (pct_roll ≤ 0.50) stay invested, and they
do not have better outcomes than the gated-out rows — the regime-0 test period is
uniformly negative regardless of rolling DTI level.

---

## Cross-Pipeline Comparison

| Pipeline | Gate | Cutoff | OOS overlay mean | sharpe | p05 | maxdd |
|---|---|---|---|---|---|---|
| Original (oos_eval.py) | regime==0 AND dti>98.81 | 98.81 (abs) | −0.004 | −0.069 | −0.149 | −0.574 |
| Percentile (oos_eval_pct.py) | regime==0 AND pct>0.50 | 0.50 (pct) | −0.028 | −0.310 | −0.190 | −0.843 |
| regime_only | regime==0 (all) | n/a | −0.004 | −0.069 | −0.149 | −0.574 |

**The original absolute-DTI overlay was accidentally identical to regime_only** — the cutoff
98.81 was below all test DTI values, so the DTI condition always fired. The percentile
pipeline introduces a genuine DTI condition, and performance degrades: the pct gate
misses 20/32 regime-0 test periods that should be filtered.

---

## Answers to Decision Questions

### 1. Does DTI add value beyond the regime filter once defined fairly?

**No.** When DTI is expressed as a causal rolling-percentile rank:

- The logit coefficient is −4.40 (inverted from expected direction)
- Analytical thresholds for regime-0 fall outside the valid range (P(y=1)>10% always)
- Walk-forward collapses to a pure regime filter (threshold=0.50, removing all high-pct regime-0 rows)
- OOS performance is **worse** than pure regime filter (Δsharpe=−0.24, Δp05=−0.04)

The DTI percentile rank does not add stable, directionally consistent signal beyond the
regime indicator in this synthetic dataset.

### 2. Does the new pipeline improve downside protection in a believable way?

**No.** The percentile overlay (overlay, 81.3% invested) is strictly worse than regime-only
(39.6% invested) on every risk metric. The downside protection from the original result was
entirely attributable to the regime dimension. The DTI dimension, when activated, hurts.

### 3. Is Leviathan best described as:

**→ (a) Regime-aware risk overlay — this is the correct characterization.**

The only defensible, stable result in this entire pipeline is:
> *Exit when positive real rates (regime==0) regardless of DTI level.*

This is a pure macro-regime filter. The DTI hypothesis — that joint stress of high
affordability burden AND unfavorable rates creates elevated crash risk — is not
confirmed in this synthetic dataset under causal, non-drifting DTI measurement.

---

## Implications for Research Direction

The regime signal is real and consistent:
- In training (oos_train.py, oos_train_pct.py): all crash episodes are in regime-0 periods
- In test: the regime filter alone gives the best risk-adjusted outcome
- Regime = (real_rate ≥ 0) is a clean, interpretable macro indicator

The DTI hypothesis remains an open question:
- In raw absolute DTI: tested a non-stationary level; cutoff was structurally inactive
- In rolling-percentile DTI: coefficient inverted; logit diagnoses "all regime-0 is risky"
  regardless of relative DTI level; walk-forward finds no incremental threshold signal

**Next step:** To test the DTI hypothesis properly, the synthetic crash mechanism needs to
be redesigned so crashes are triggered by `regime==0 AND dti_pct_roll > threshold` jointly.
Currently, L1a and L1b crash placements are anchored to fixed calendar dates independent of
DTI percentile level. The probabilistic Layer 4 uses `dti > p75(dti)` with absolute DTI,
not rolling percentile — and fires too infrequently in training to create learnable signal.
