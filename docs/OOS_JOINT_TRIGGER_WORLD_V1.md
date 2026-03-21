# OOS Joint-Trigger World V1 — Research Memo

**Date:** 2026-03-21
**Script:** `scripts/oos_joint_trigger_eval.py`
**Data:** `src/research/path_a/build_dataset.py` (rewritten)
**Artifacts:** `outputs/oos_joint_trigger/`

---

## Motivation

The regime-only overlay (V1) validated that exiting when `real_rate ≥ 0` consistently
improves downside metrics (p05 and maxdd) across all 9 tested configurations. But it left
a structural objection unanswered: the original crash mechanism never made DTI a causal
trigger. Crash placements were calendar-anchored (Layers 1a, 1b, 2, 3) or used absolute
DTI thresholds (Layer 4: `dti > p75(dti)`) that, in the rolling-percentile space, bore
no relationship to relative stress.

The consequence: even after redefining DTI as a causal rolling-percentile feature, the
walk-forward always collapsed to the boundary threshold (0.50), equivalent to a pure
regime filter. The logit coefficient inverted. The hypothesis could not be tested because
the crash mechanism did not generate the signal the hypothesis required.

This experiment redesigns the crash mechanism so that the joint condition
`regime==0 AND dti_pct_roll > threshold` is explicitly the causal crash trigger. If DTI
adds value beyond the regime filter, this is the world where it should appear.

---

## Crash Mechanism Redesign

`src/research/path_a/build_dataset.py` was rewritten with a 3-layer architecture:

| Layer | Name | Trigger | Shock | Priority |
|---|---|---|---|---|
| A | Structural GFC | Calendar: 2007-Q1 to 2009-Q1 | −0.065/q (fixed) | Secondary |
| B | Background noise | 3% per quarter, any period | −0.020 to −0.040/q | Minor |
| C | Joint trigger | `regime==0 AND dti_pct_roll > 0.65`, 45% prob | −0.070 to −0.090/q | **Primary** |

**Key parameters:**
- Rolling window: 20 quarters (inline, causal — `dti_pct_roll[t]` uses only `[t-19, t]`)
- Joint threshold: `_JOINT_THRESH = 0.65`
- Trigger probability: `_JOINT_PROB = 0.45`
- Duration: 3–4 quarters per event
- GFC suppress set: rows GFC_START−2 to GFC_END+2 (prevents double-crash overlap)
- Layer C takes priority; Layer B fires only if Layer C does not trigger

This is the first design where the joint hypothesis is *structurally embedded*: the
synthetic world was built so that the DTI dimension, conditioned on regime, genuinely
matters for crash causation.

---

## Training Results

Training window: up to 2007-12-31 (44 labeled rows after `add_correction_label`).

| Split | y=1 | y=0 |
|---|---|---|
| Train | 19 | 25 |
| Test | 30 | 18 |

**Walk-forward result: pct_cutoff = 0.5000** — the boundary of the search grid (0.50–0.95).

The walk-forward objective is p05 of 4-quarter filtered returns on the training window.
Even in the joint-trigger world, the walk-forward finds that filtering all regime-0 rows
(threshold=0.50 ≈ removing all rows with elevated pct within a typical 20q window)
maximizes p05. Any threshold above 0.50 admits regime-0 rows with lower pct, which still
have negative returns in training — making the joint gate looser and p05 worse.

The walk-forward correctly diagnoses: "in training, all regime-0 quarters are bad
regardless of DTI percentile rank." This is a direct consequence of the crash architecture:
Layer C generates crashes whenever `pct > 0.65`, covering ~60% of regime-0 rows; the
remaining ~40% (pct ≤ 0.65) have the structural GFC or background noise bleeding through.
The result is that regime-0 is uniformly negative in training — no DTI threshold separates
good from bad regime-0 quarters.

---

## OOS Results

Test period: 2008-Q1 → 2023-Q4 (64 quarters, 48 labeled).

| Strategy | mean | sharpe | p05 | maxdd | pct_invested |
|---|---|---|---|---|---|
| always_in | −0.030 | −0.292 | −0.248 | −0.860 | 100% |
| regime_only | +0.000 | **+0.006** | −0.131 | −0.617 | 50.0% |
| joint_overlay | −0.018 | −0.190 | −0.215 | −0.772 | 81.3% |
| trend_regime | **+0.006** | **+0.108** | **−0.014** | **−0.346** | 29.7% |

**Overlay lift vs always_in:**

| Strategy | Δsharpe | Δp05 | Δmaxdd |
|---|---|---|---|
| regime_only | +0.298 | +0.118 | +0.243 |
| joint_overlay | +0.102 | +0.033 | +0.088 |
| trend_regime | +0.400 | +0.235 | +0.514 |

**DTI contribution (joint_overlay vs regime_only):**

| Metric | Value |
|---|---|
| Δsharpe | **−0.196** |
| Δp05 | **−0.084** |
| Δmaxdd | **−0.155** |
| Verdict | **DTI_HURTS** |

---

## Why DTI Hurts in Test

The joint_overlay fires on `regime==0 AND dti_pct_roll > 0.50`. With a threshold of 0.50,
this covers roughly half of all regime-0 test rows. The other half — regime-0 rows with
`dti_pct_roll ≤ 0.50` — remain invested under joint_overlay but are exited under
regime_only.

In test, those low-pct regime-0 rows do not have better outcomes. The joint-trigger crash
mechanism fires primarily at `pct > 0.65` in training; but in the test period, the GFC
structural layer (2007–2009) dominates regime-0 outcomes, and background noise is
uniformly present regardless of pct level. The result is that low-pct regime-0 quarters in
test are nearly as bad as high-pct ones — staying invested in them (as joint_overlay does)
degrades every risk metric.

Regime-only, by exiting all 32 regime-0 test quarters, avoids this problem entirely.

---

## What the Three Worlds Agree On

Across all pipeline variants tested:

| Finding | Old pipeline (abs DTI) | Pct pipeline (rolling pct) | Joint-trigger world |
|---|---|---|---|
| Regime filter improves p05 | Yes | Yes | Yes |
| Regime filter reduces maxdd | Yes | Yes | Yes |
| DTI adds incremental value | No (gate vacuous) | No (Δsharpe −0.24) | No (Δsharpe −0.20) |
| Walk-forward DTI cutoff | 98.81 (all test DTI exceed) | 0.50 (boundary) | 0.50 (boundary) |
| trend+regime strongest | Yes | Yes | Yes |

The regime dimension is the only consistent signal. The DTI dimension, in every form
tested, fails to add incremental downside protection beyond the pure regime exit.

---

## Answers to the Three Decision Questions

### 1. Does regime-only still work in the joint-trigger world?

**Yes.** regime_only improves Sharpe from −0.292 to +0.006, p05 from −0.248 to −0.131,
and maxdd from −0.860 to −0.617. These are large, directionally consistent improvements.
The regime filter works regardless of whether the synthetic crash mechanism is
calendar-anchored (old design) or joint-trigger-based (new design).

### 2. Does rolling-percentile DTI now add incremental value beyond the regime filter?

**No.** With the joint-trigger crash mechanism — the most favorable possible design for
the DTI hypothesis — the joint_overlay still underperforms regime_only on every metric
(Δsharpe=−0.196, Δp05=−0.084, Δmaxdd=−0.155). The walk-forward again collapses to the
boundary threshold (0.50), equivalent to a pure regime filter in training.

Even when the synthetic world is explicitly constructed so that `regime==0 AND pct>0.65`
is the causal crash trigger, the walk-forward cannot isolate DTI as a separable condition
from the regime indicator. The regime dimension dominates because regime-0 quarters are
uniformly adverse — the DTI percentile rank cannot subdivide them into safe and unsafe
subsets in either training or test.

### 3. Is the joint hypothesis (regime + high relative DTI) supported, unsupported, or still ambiguous?

**Unsupported in this synthetic framework.**

After three independent experiments (absolute DTI, rolling-percentile DTI, joint-trigger
causal redesign), the DTI dimension has never added stable signal beyond the macro regime:

- **Absolute DTI**: non-stationary; gate vacuous in test (all test DTI > training cutoff)
- **Rolling-percentile DTI (original crash mechanism)**: coefficient inverted; walk-forward
  boundary; Δsharpe=−0.241 vs regime_only
- **Rolling-percentile DTI (joint-trigger crash mechanism)**: walk-forward boundary again;
  Δsharpe=−0.196 vs regime_only; DTI_HURTS verdict

The hypothesis is not ambiguous — it has been tested under the most favorable possible
conditions (a crash mechanism explicitly designed to make DTI the causal trigger) and
still fails. The DTI dimension cannot be recovered by further threshold tuning or crash
redesign within the current quarterly synthetic panel structure.

**The defensible characterization of Leviathan remains:** a macro-regime overlay that
exits during positive real rate environments. DTI, in any form tested, does not survive
rigorous OOS scrutiny in this framework.

---

## Caveats

**1. Synthetic data only.** The joint-trigger world makes DTI causal by construction in
the data-generating process. Even so, the walk-forward cannot learn a separable DTI
threshold. This is partly a consequence of quarterly resolution: 20 quarters of DTI
history is a narrow window; within-regime DTI percentile variation is limited.

**2. Regime uniformity problem.** In both training and test, regime-0 quarters are
uniformly adverse regardless of DTI level. This is a structural property of the crash
mechanism (crashes are concentrated in regime-0 periods by design), not a solvable
calibration problem. No walk-forward objective can separate regime-0/low-pct quarters
from regime-0/high-pct quarters when both have negative expected returns.

**3. The joint hypothesis may hold at higher frequency or finer geography.** Quarterly
national (synthetic) data may wash out within-regime DTI variation that would be visible
in monthly MSA-level or loan-level data. The negative result here is specific to this
data structure, not a universal refutation.

**4. trend_regime remains the recommended overlay.** It achieves Sharpe +0.108, p05
−0.014, maxdd −0.346 — substantially better than regime_only alone (Sharpe +0.006,
p05 −0.131, maxdd −0.617). The momentum + regime combination identifies complementary
exit conditions without requiring DTI data.
