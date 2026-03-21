# OOS Robustness V1 — Research Memo

**Date:** 2026-03-21
**Depends on:** `docs/OOS_BASELINE_V1.md`
**Artifacts:** `outputs/oos/robustness/model_comparison.csv`, `outputs/oos/robustness/sweep_results.csv`

---

## Phase B — Model Specification Comparison

Four training specifications were compared on the same fixed train/test split
(training ≤ 2007-12-31, 16 labeled rows, 7 y=1).

| Spec | Conv | Pseudo R² | Model cutoff | OOS p05 | OOS Sharpe | OOS maxdd |
|---|---|---|---|---|---|---|
| interaction_logit | **False** | 0.488 | 86.13 | −0.145 | 0.019 | −0.520 |
| main_effects_logit | **True** | 0.478 | 86.39 | −0.145 | 0.019 | −0.520 |
| l1_regularized (α=0.5) | True | 0.035 | 449.6 | −0.145 | 0.210 | −0.566 |
| rule_only_wf | N/A | N/A | 93.75 | −0.145 | 0.019 | −0.520 |
| **always_in** (baseline) | — | — | — | −0.145 | **0.210** | −0.566 |

*OOS metrics use each spec's model-derived cutoff. Walk-forward cutoff produces identical
results across specs 1, 2, and 4 since wf_cutoff=93.75 is the same for all.*

### Key findings

**1. No spec fixes the small-N convergence problem.**
Interaction logit fails to converge (16 rows, 4 parameters, near-perfect separation by DTI).
Main-effects logit (3 parameters) does converge, with marginally lower pseudo R² (0.478 vs 0.488).
This makes it the operationally safer choice for the current training window.

**2. L1 regularization backfires at n=16.**
With alpha=0.5, all four coefficients are shrunk near zero (dti coef = −0.005, others = 0).
The resulting model cutoff is 449.6 DTI — far outside the data range — making it equivalent
to always_in. Pseudo R² drops to 0.035. The training set is too small for LASSO to be useful;
it needs enough signal to resist the penalty.

**3. Logit-derived cutoff (86.13) and walk-forward cutoff (93.75) produce the same OOS outcome.**
Both yield sharpe=0.019 and maxdd=−0.520. This means the strategy outcome is not sensitive
to the ~8 DTI-point gap between these cutoffs. The DTI filter fires at the same set of
test periods regardless of whether the threshold is 86 or 94.

**4. The overlay consistently underperforms always_in on Sharpe (0.019 vs 0.210).**
The overlay is flat (returns=0) during filtered periods, which drags the mean down significantly.
It offers a small maxdd improvement (−0.520 vs −0.566, +4.6pp). This pattern is expected:
a downside-protection strategy sacrifices average return for tail protection, but with the
current DTI cutoff it is filtering aggressively during high-return periods.

**Most stable spec: `main_effects_logit`** — converges, interpretable coefficients, similar pseudo R²
to the interaction model, no degenerate interaction term that absorbs all regime variation.

---

## Phase C — Synthetic Sensitivity Sweep (27 configurations)

### Configuration space

| Parameter | Values swept | Baseline |
|---|---|---|
| crash_shock (per quarter) | 0.05, 0.075, 0.10 | 0.075 |
| frag_prob | 0.10, 0.20, 0.30 | 0.30 |
| dti_percentile | 70, 75, 80 | 75 |

Total: 3 × 3 × 3 = 27 runs. All with seed=42, same structural crash placement.

### Training label sensitivity

| crash_shock | train_y1 | pseudo_R² | Logit status |
|---|---|---|---|
| 0.050 | 4 | 1.000 | Near-perfect separation — degenerate |
| 0.075 | 7 | 0.488 | Near-separation — does not converge |
| 0.100 | 9 | 0.501 | Near-separation — does not converge |

Training labels are **entirely determined by crash_shock**. `frag_prob` and `dti_pct` have
zero effect on training y=1 counts because the training crash (Layer 1, deterministic at
row 24) uses the same placement regardless — only Layer 4 (probabilistic) varies, and
Layer 4 fires only in the test period given the current suppression windows.

**Walk-forward cutoff is identical (93.7538) across all 27 configurations.**
The walk-forward searches DTI quantiles of the training window, and DTI values are
the same regardless of crash_shock. Different crash depths change fwd_return but the
optimizer always selects the same optimal DTI threshold from the regime-0 training rows.

### OOS performance by crash severity

| crash_shock | ov_sharpe | ov_p05 | ov_maxdd | ai_sharpe | diagnosis |
|---|---|---|---|---|---|
| 0.050 | 0.190 | −0.079 | −0.335 | ~0.20 | Mild crashes; overlay barely different from always_in |
| 0.075 | 0.019 | −0.145 | −0.520 | 0.210 | Baseline; overlay hurts Sharpe but reduces maxdd |
| 0.100 | −0.069 | −0.211 | −0.653 | ~0.07 | Deep crashes overwhelm the filter; overlay negative Sharpe |

### Effect of frag_prob and dti_percentile

- `frag_prob` primarily increases **test_y1** (more test-period crash events from Layer 4).
  At frag_prob=0.30 vs 0.10, test_y1 increases by +0-11 rows depending on which fragile
  rows trigger; OOS Sharpe is unaffected (cutoff is identical).
- `dti_pct` (fragility threshold) has minimal effect on labeled counts at frag_prob=0.10-0.20
  and a small effect at frag_prob=0.30. No effect on OOS strategy metrics.

### "Too easy" and "too hard" configurations

**Too easy — crash_shock=0.050:**
- 4 training positive labels; pseudo_R²=1.000 (perfect separation in training)
- Logit is degenerate; coefficients unreliable
- OOS looks good (sharpe=0.19) but only because the underlying crashes are mild —
  this is not a regime where a DTI filter is tested under stress

**Too hard — crash_shock=0.100:**
- 9 training positive labels; better label balance (9/16 = 56%)
- But crashes are so deep that p05=−0.211 and ov_maxdd=−0.653; overlay negative Sharpe
- Difficult to distinguish skill from survival bias; filter cannot keep up with extreme events

**Sweet spot — crash_shock=0.075 (current baseline):**
- 7 positive labels at 44% prevalence
- Pseudo R²=0.488, meaningful but logit still non-convergent
- OOS maxdd reduction is real (4.6pp), Sharpe cost is visible
- Near-convergent regime — interesting transition zone

---

## Summary Recommendations

### 1. Which model specification is most stable?

**`main_effects_logit`** is the most stable:
- Converges (unlike interaction logit)
- Near-identical pseudo R² (0.478 vs 0.488)
- No degenerate interaction term (regime coefficient −279 in interaction model is an artifact
  of near-separation, not an economically meaningful estimate)
- Three parameters on 16 observations is still underpowered, but more defensible

### 2. Which synthetic configurations are too easy / too hard?

| Config | Verdict | Reason |
|---|---|---|
| crash_shock=0.050 | Too easy | Pseudo R²=1.0, near-perfect separation, logit degenerate |
| crash_shock=0.075 | Acceptable | Baseline; near-convergent; realistic label balance |
| crash_shock=0.100 | Borderline | More positive labels but extreme crashes reduce interpretability |
| frag_prob=0.10 | Fine | Fewer test crashes but doesn't affect training |
| frag_prob=0.30 | Fine | More test crashes, higher test prevalence |
| dti_pct=70/75/80 | Minimal effect | Only matters for Layer 4 (probabilistic); current config fine |

### 3. Best next move for making the OOS result more believable

The current bottleneck is **16 labeled training rows with 4 logit parameters**. Near-separation
is mathematically almost guaranteed at this ratio. Three targeted improvements, in priority order:

**Priority 1 — Extend the training window backward.**
Changing `start="1992-01-01"` would give ~64 training rows instead of 36, and ~44 labeled
rows instead of 16. This is the single change most likely to produce a convergent logit and
stable coefficients. No other pipeline changes required.

**Priority 2 — Switch the training model to `main_effects_logit`.**
Reduce from 4 to 3 parameters. With 16 rows this halves the parameter-to-observation ratio.
The `dti×regime` interaction is the theoretically motivated term, but its coefficient is
currently estimated with p=0.978 — it is adding noise, not signal. Verify whether dropping
it meaningfully degrades pseudo R² before committing.

**Priority 3 — Validate the walk-forward cutoff selection.**
The wf_cutoff (93.75) is identical across all 27 sweep configurations. This is structurally
robust (DTI-based, not return-based in the cross-sectional sense) but means the strategy
is not adapting to crash severity. Consider whether a harder training window should produce
a different cutoff — if not, the walk-forward is doing very little beyond setting a fixed
DTI quantile threshold, which could be parameterized directly.
