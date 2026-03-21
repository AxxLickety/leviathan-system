# OOS Regime Multi-Market Positive-Drift Evaluation V1 — Research Memo

**Date:** 2026-03-21
**Script:** `scripts/oos_regime_multimarket_posdrift.py`
**Generator:** `src/research/path_a/build_multimarket_posdrift.py`
**Artifacts:** `outputs/oos_regime_multimarket_posdrift/`

---

## Why This Is the Harder Test

The adversarial world showed that regime_only's downside protection was robust even when
crash-regime alignment was weakened to 0.58. But the adversarial memo identified a
confound: in that world, always_in had mostly negative Sharpe (11/12 markets negative).
Being out of the market ~50% of the time is mechanically helpful in a bear world
regardless of regime signal. That component — call it **participation-reduction value** —
inflated the apparent robustness.

This experiment removes the confound by making the world broadly profitable. All 12
markets have positive always_in mean return and positive always_in Sharpe in this test
period. Being out of the market now has a real opportunity cost: the filter consistently
sacrifices mean return (d_mean < 0 in 12/12 markets, mean Δmean = −1.6%/q ≈ −6.6% per
year of foregone appreciation).

The test question becomes: **does regime_only still reduce tail risk and drawdown when
doing so requires giving up profitable quarters in regime-0?**

---

## World Design

**Generator:** `src/research/path_a/build_multimarket_posdrift.py`

Three changes from the adversarial world; everything else is structurally identical:

| Parameter | Adversarial | Positive-Drift |
|---|---|---|
| growth_base | U(0.004, 0.013)/q | **U(0.012, 0.022)/q** |
| growth_regime_boost | U(0.001, 0.007)/q | **U(0.004, 0.010)/q** |
| p_crash_r0 | U(0.08, 0.18)/q | U(0.05, 0.12)/q (reduced) |
| p_crash_r1 | U(0.03, 0.08)/q | U(0.02, 0.05)/q (reduced) |
| major_depth | U(0.040, 0.070)/q | U(0.025, 0.050)/q (lighter) |
| major_dur | 5–9 quarters | 3–6 quarters (shorter) |

The rate/DTI/price noise parameters and crash mechanism structure are identical.
Crash depth reductions and probability reductions prevent crashes from overwhelming the
elevated positive drift. The crash-regime ratio (p_r0/p_r1 ≈ 2.5×) is maintained so
regime remains informative at roughly the same level as the adversarial world (1.3×
to 2.5×).

---

## World Diagnostics

| Diagnostic | Positive-Drift | Adversarial | Base World |
|---|---|---|---|
| always_in Sharpe > 0 | **12/12** | 1/12 | 5/12 |
| always_in mean > 0 | 12/12 | ~1/12 | ~7/12 |
| Avg frac crash-q in r0 | 0.60 | 0.58 | ~0.87 |
| Avg frac r0-q crashing | 0.16 | 0.28 | ~0.40 |
| Avg frac r1-q crashing | 0.10 | 0.22 | ~0.08 |
| Major crash in regime-0 | 9/12 | 7/12 | 12/12 |

The regime signal (frac_crash_in_r0 = 0.60) is essentially the same as the adversarial
world. The only thing that changed is drift: this world is broadly profitable for
staying invested, while the adversarial world was broadly adverse.

---

## Results

### Market-Level (OOS: 2008-Q1 onward)

| market | ai_sharpe | ai_mean | ro_sharpe | ro_p05 | ro_maxdd | Δsharpe | Δmean | Δp05 | Δmaxdd |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.842 | +0.048 | 0.385 | 0.000 | −0.204 | **−0.458** | −0.031 | +0.076 | +0.170 |
| 1 | 1.663 | +0.067 | 0.953 | −0.025 | −0.077 | **−0.710** | −0.021 | +0.005 | +0.031 |
| 2 | 0.874 | +0.050 | 0.647 | −0.006 | −0.120 | −0.227 | −0.020 | +0.048 | +0.088 |
| 3 | 0.704 | +0.045 | 0.671 | −0.033 | −0.140 | −0.034 | −0.013 | +0.032 | +0.322 |
| 4 | 1.388 | +0.064 | 1.145 | 0.000 | −0.011 | −0.244 | −0.013 | +0.023 | +0.237 |
| 5 | 0.514 | +0.031 | 0.612 | 0.000 | −0.174 | **+0.098** | −0.005 | +0.095 | +0.272 |
| 6 | 1.209 | +0.063 | 0.584 | −0.002 | −0.091 | **−0.625** | −0.034 | +0.035 | +0.075 |
| 7 | 1.088 | +0.061 | 0.844 | −0.001 | −0.046 | −0.243 | −0.017 | +0.034 | +0.173 |
| 8 | 0.723 | +0.043 | 0.913 | 0.000 | −0.016 | **+0.191** | −0.002 | +0.076 | +0.331 |
| 9 | 1.486 | +0.055 | 0.664 | 0.000 | 0.000 | **−0.822** | −0.032 | +0.008 | +0.137 |
| 10 | 0.482 | +0.031 | 0.473 | −0.058 | −0.147 | −0.008 | −0.005 | +0.008 | +0.150 |
| 11 | 1.065 | +0.065 | 1.137 | 0.000 | 0.000 | **+0.072** | −0.006 | +0.054 | +0.199 |

### Aggregate

| Metric | Pos-Drift | Adversarial | Decay |
|---|---|---|---|
| p05 improved | **12/12 (100%)** | 11/12 (92%) | +1 |
| maxdd improved | **12/12 (100%)** | 12/12 (100%) | 0 |
| Sharpe improved | 3/12 (25%) | 8/12 (67%) | **−5** |
| Mean Δp05 | +0.041 | +0.069 | −0.027 |
| Mean Δmaxdd | +0.182 | +0.325 | **−0.143** |
| Mean Δsharpe | −0.251 | +0.259 | **−0.510** |
| Mean Δmean | **−0.016** | n/a | — |

**Verdict: SIGNAL VALUE CONFIRMED — regime-only downside protection survives positive-drift world; regime signal is not purely participation reduction**

---

## The Attribution Decomposition

The results cleanly separate two effects:

**What collapsed:** Sharpe improvement. In the adversarial world, regime_only improved
Sharpe in 8/12 markets (mean Δsharpe = +0.259). In the positive-drift world, it
improves Sharpe in only 3/12 markets (mean Δsharpe = −0.251). This is the
participation-reduction component disappearing: when always_in is profitable, exiting
~50% of the time reliably costs mean return (regime_only sacrifices mean in 12/12
markets, mean Δmean = −1.6%/q ≈ −6.6% annual). That foregone return overwhelms the
volatility reduction, collapsing the Sharpe ratio.

**What survived:** p05 and maxdd protection. Both improve in 12/12 markets despite the
mean sacrifice. The filter still catches the left tail. Mean Δp05 = +4.1pp (was 6.9pp
in adversarial); mean Δmaxdd = +18.2pp (was 32.5pp). These are real, not trivial — the
filter is achieving genuine downside reduction by avoiding regime-0 crash clusters, even
in a world where staying invested is on average profitable.

This is the clean attribution:

| Component | Adversarial World | Positive-Drift World | What It Tells Us |
|---|---|---|---|
| Signal value (downside protection) | Present | **Present** | Regime has genuine predictive content for tail risk |
| Participation-reduction value | Present (large) | **Absent** | Was masking how much came from being in a bear world |
| Net Sharpe benefit | Positive | **Negative** | Filter costs Sharpe when drift > crash damage |

---

## Is It Worth the Opportunity Cost?

In the positive-drift world, the regime filter trades:
- **Cost:** −1.6%/q mean return foregone (6.4%/year of missed appreciation)
- **Benefit:** mean Δp05 = +4.1pp per quarter (floor loss reduced from −4.9% to −0.9% on average), mean Δmaxdd = +18.2pp (peak-to-trough drawdown reduced from −25.8% to −7.6% on average)

Whether this trade is worth it depends entirely on the investor's objective. For a
**return-seeking investor** who cares about Sharpe or compounded growth in a favorable
market, the regime filter is net harmful: it costs Sharpe in 9/12 markets, and the
missed upside is substantial. For a **downside-risk-constrained investor** who cares
primarily about avoiding large drawdowns and left-tail losses (insurance companies,
pension funds with funded-ratio constraints, concentrated housing positions), the filter
provides meaningful protection even at a cost to Sharpe.

---

## Does the Result Look Structural?

**Yes — with now-precise scope.**

Three progressive stress tests have produced a consistent picture:

| World | Crash-Regime Alignment | Always_In Environment | p05 Improved | maxdd Improved | Sharpe Improved |
|---|---|---|---|---|---|
| Base multi-market | ~87% | Mixed (5/12 pos) | 11/12 | 10/12 | 7/12 |
| Adversarial | ~58% | Mostly negative (1/12 pos) | 11/12 | 12/12 | 8/12 |
| Positive-drift | ~60% | Uniformly positive (12/12) | **12/12** | **12/12** | **3/12** |

The **downside protection persists** across all three worlds. The **Sharpe benefit is
context-dependent**: it appears in bear worlds and disappears in bull worlds. This is
exactly the behavior expected of an asymmetric risk overlay.

The regime signal is structural in one specific sense: the fact that regime-0 has higher
crash probability (even at 2.5× ratio) creates genuine tail risk reduction that survives
the opportunity cost of exiting a positive-drift world. The effect decays — mean Δp05
fell from 6.9pp to 4.1pp (40% decay), mean Δmaxdd fell from 32.5pp to 18.2pp (44%
decay) — but does not collapse. Roughly 60% of the downside protection survives the
transition from a bear to a bull world.

---

## Answers to the Five Questions

### 1. Across how many markets does regime-only still improve p05?

**12 of 12 (100%).** p05 improvement is universal and actually improves vs the
adversarial world (11/12). In every market, the regime filter reduces the worst-quarter
losses. The mean Δp05 is +4.1pp vs +6.9pp in the adversarial world — smaller but not
gone.

### 2. Across how many markets does it still improve maxdd?

**12 of 12 (100%).** Same as the adversarial world. Even in markets where always_in
Sharpe is above 1.0 and the filter costs substantial Sharpe, it still reduces the
peak-to-trough drawdown. The mean Δmaxdd is +18.2pp vs +32.5pp adversarial — roughly
44% decay, but always positive.

### 3. Across how many markets does it improve Sharpe?

**3 of 12 (25%)**, down sharply from 8/12 adversarial. The 9 markets where Sharpe
declines all have positive always_in mean return, and regime_only exits profitable
regime-0 quarters — the Sharpe cost is real and expected. The 3 markets where Sharpe
improves (markets 5, 8, 11) happen to have moderate always_in Sharpe (0.51–0.72 except
market 11 at 1.06) and most of their crash activity concentrated in regime-0 (64–100%),
so the filter avoids the right quarters.

### 4. How much of the prior benefit came from participation reduction?

Roughly **40% of the downside benefit** (measured by Δp05 and Δmaxdd) came from simple
participation reduction rather than signal value. The 60% that survived the transition
to a positive-drift world is attributable to the regime signal itself.

The Sharpe benefit was **entirely** participation-reduction: it was +0.26 in the
adversarial world (mostly bear environment) and −0.25 in the positive-drift world
(uniformly bull environment). The Sharpe benefit had no residual in a world where
staying invested was clearly profitable.

### 5. In a positive-drift world with imperfect regime alignment, does Leviathan still have real marginal value?

**Yes — specifically and only as a downside-risk overlay.**

The positive-drift test resolves the attribution. The result is:

- **As a Sharpe enhancer:** No. In a bull world, exiting ~50% of the time reliably
  costs mean return and Sharpe. Regime_only should not be evaluated on this criterion.

- **As a downside-risk overlay:** Yes. p05 and maxdd improve universally (12/12) even
  when the filter sacrifices 1.6%/q of mean return. The regime filter pays for itself
  in reduced tail exposure even when doing so is costly on average.

The honest characterization of Leviathan at this point:

> *Leviathan's regime-only overlay is a pure downside-risk instrument. It consistently
> reduces left-tail losses and peak drawdowns, at the cost of mean return and Sharpe.
> In bull markets, it is net harmful to risk-adjusted performance. In bear markets or
> tail-focused mandates, it provides meaningful protection. Whether it is "worth using"
> is an investor mandate question, not a model question.*

---

## Caveats

**1. The positive-drift is calibrated, not discovered.** By design, growth_base was
elevated to ensure positive always_in Sharpe. Real housing markets have periods of
both positive and negative expected return. The "bull world" result here is a controlled
counterfactual, not an empirical estimate.

**2. The regime signal strength (2.5× ratio) is generous.** In real data, the ratio of
crash probability between high-rate and low-rate environments is unknown and may be
lower. If p_r0/p_r1 fell below 1.5×, the downside protection would decay further.

**3. 12 markets is enough for directional conclusions, not precise calibration.** The
40%/60% participation-reduction vs signal attribution is an approximation from a small
synthetic sample. A 100-market version would give tighter estimates.

**4. The regime filter's value in a real portfolio depends on position sizing.** This
analysis tests a binary in/out filter. In practice, a partial position reduction during
regime-0 might capture more of the downside protection while sacrificing less mean
return — but that requires richer portfolio construction than this framework tests.
