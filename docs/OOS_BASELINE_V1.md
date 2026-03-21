# OOS Baseline V1 — Research Note

**Date:** 2026-03-21
**Status:** First meaningful OOS run. Synthetic data only. Not a validated empirical result.

---

## 1. Synthetic Crash Design

The baseline uses a four-layer crash mechanism added to `src/research/path_a/build_dataset.py`.
All other columns and the random seed (42) are unchanged.

| Layer | Type | Row (date) | Dur | Shock/qtr | Total drop | Window |
|---|---|---|---|---|---|---|
| 1 | Training crash (forced) | 24 (2005-Q1) | 4q | −0.075 | −0.300 | TRAIN |
| 2 | GFC-proxy (structural) | 31 (2006-Q4) | 8q | −0.065 | −0.520 | TRAIN start, TEST tail |
| 3 | Post-GFC test crash (forced) | 68 (2016-Q1) | 4q | −0.075 | −0.300 | TEST |
| 4 | Probabilistic (fragile rows) | 89 (2021-Q2) | 3q | −0.069 | −0.207 | TEST |

**Fragility eligibility (Layer 4):** `regime == 0` AND `dti > p75(dti)`, trigger probability 30%.

**Economic motivation:** Accommodative monetary policy (regime=1) drives DTI to stressed levels.
Reversal to regime=0 creates refinancing pressure for top-quartile borrowers, triggering corrections.

---

## 2. Label Distribution

| Split | Rows | y=1 | y=0 | Prevalence |
|---|---|---|---|---|
| Training (labeled) | 16 | 7 | 9 | 43.8% |
| Test (labeled) | 48 | 24 | 24 | 50.0% |

*add_correction_label drops the last 20 rows from each window (horizon_max_q=20).
Training has 36 pre-label rows; test has 68.*

---

## 3. Frozen Training Outputs

**Logit:** `fit_interaction_logit` on 16 training rows with features `const, dti, regime, dti×regime`.

| Parameter | Coef | p-value |
|---|---|---|
| const | −70.94 | 0.091 |
| dti | 0.798 | 0.091 |
| regime | −279.65 | 0.978 |
| dti×regime | 2.915 | 0.978 |

Pseudo R²: 0.488. **Convergence: False** (near-separation; small-N artifact with 16 rows, 4 params).

**Regime-conditional DTI thresholds (p=0.10 / p=0.20):**

| Regime | p=0.10 | p=0.20 |
|---|---|---|
| 0 (stressed) | 86.13 | 87.15 |
| 1 (accommodative) | 93.84 | 94.06 |

**Walk-forward DTI cutoff:** 93.75 (16 walk-forward steps, objective = maximize p05 of filtered returns).

---

## 4. OOS Strategy Performance

Test period: 2008-Q1 → 2024-Q4, 48 labeled rows. Overlay uses `dti_cutoff = 93.75`.

| Strategy | n | mean | sharpe | p05 | maxdd |
|---|---|---|---|---|---|
| always_in | 48 | 0.0143 | 0.210 | −0.145 | −0.566 |
| overlay | 48 | 0.0011 | 0.019 | −0.145 | −0.519 |
| always_out | 48 | 0.000 | NaN | 0.000 | 0.000 |
| naive_benchmark | 48 | 0.0011 | 0.019 | −0.145 | −0.519 |

The overlay reduces maxdd by 4.7pp (−0.519 vs −0.566) but at substantial cost to mean return and
Sharpe. The walk-forward cutoff of 93.75 is loose (filters few periods); the logit-derived threshold
of 86.13 for regime=0 would be more aggressive.

---

## 5. Caveats

1. **Synthetic data only.** All series are generated; no real housing prices, DTI, or interest rates.
   Economic relationships are imposed mechanically.

2. **Small training set.** 16 labeled training rows with 4 logit parameters → near-perfect
   separation → convergence failure. Coefficients are numerically unstable; interaction term
   p-values are ~0.98.

3. **Walk-forward cutoff (93.75) vs logit threshold (86.13).** These differ by ~8 DTI points.
   The strategy performance depends heavily on which is used. The walk-forward cutoff was
   optimized for p05 on training returns; the logit threshold is an extrapolation from a
   degenerate fit.

4. **Crash frequency table.** Both regime-0 and regime-1 test cells show identical crash
   frequency (50%), which reflects the symmetric crash placement in the synthetic design,
   not empirical evidence of regime-conditional crash risk.

5. **No look-ahead bias in pipeline.** Firewall between oos_train.py and oos_eval.py is intact.
   All frozen parameters were computed from pre-2008 data only.

---

## 6. Why This Is the First Meaningful OOS Baseline

Prior to this version, the training window produced zero positive labels (the crash mechanism did
not create price drops large enough for `add_correction_label` to detect from labeled rows).
The logit fit was entirely degenerate.

This version establishes:
- A functioning crash mechanism that produces realistic label distributions
- A complete frozen-parameter workflow (train → freeze → eval)
- End-to-end reproducibility (fixed seed, explicit crash placement logic)
- A quantitative baseline against which model and data variations can be compared

The next step is robustness analysis across model specifications and crash design parameters,
documented in `docs/OOS_ROBUSTNESS_V1.md`.
