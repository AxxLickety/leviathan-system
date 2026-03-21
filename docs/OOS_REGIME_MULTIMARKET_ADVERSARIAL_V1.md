# OOS Regime Multi-Market Adversarial Evaluation V1 — Research Memo

**Date:** 2026-03-21
**Script:** `scripts/oos_regime_multimarket_adversarial.py`
**Generator:** `src/research/path_a/build_multimarket_adversarial.py`
**Artifacts:** `outputs/oos_regime_multimarket_adversarial/`

---

## Why the Previous World May Have Been Too Favorable

The base multi-market framework had two structural advantages for the regime filter:

**1. Synchronized crash.** All 12 markets shared a single GFC crash anchored at
2007-Q1 to 2009-Q1. This event always falls in the test period, always coincides with a
regime-0 episode (positive real rates), and dominates test-period drawdowns. The regime
filter exits precisely during this event in every market, guaranteeing large maxdd
improvements regardless of what the probabilistic crashes do.

**2. Regime-dominated crash probability.** The joint-trigger mechanism required
`regime==0 AND dti_pct_roll > 0.65` for crash eligibility. Combined with the
synchronized GFC, roughly 85–90% of crash-active quarters in the base world fell in
regime-0. This is a nearly perfect world for a regime filter.

Neither property is likely to hold in real housing markets. Crashes can and do occur in
accommodative rate environments (e.g., local supply shocks, foreign capital flows,
sentiment reversals). The "correct" fraction of crashes that fall in adverse rate regimes
is unknown and probably varies substantially across markets and time periods.

---

## How the Adversarial World Differs

**`src/research/path_a/build_multimarket_adversarial.py`**

Three specific changes from the base world:

| Feature | Base World | Adversarial World |
|---|---|---|
| Synchronized GFC | Yes — all markets, 2007-Q1 to 2009-Q1 | **No** — market-specific major crash (uniform onset over 20–75% of panel) |
| Major crash regime | Always regime-0 by GFC calendar | Random — falls in regime-0 in 7/12 markets, regime-1 in 5/12 |
| Crash probability structure | regime==0 AND pct>0.65: 45% → pure regime-0 gate | regime-0: 8–18%/q; regime-1: 3–8%/q — imperfect, both > 0 |
| Background noise | 2–5%/q, regime-agnostic | 3–7%/q, slightly elevated, regime-agnostic |

**Key world properties achieved (Phase 4 diagnostics):**

| Diagnostic | Adversarial | Base World |
|---|---|---|
| Avg frac crash-quarters in regime-0 | **0.58** | ~0.85–0.90 |
| Avg frac regime-0 quarters crashing | 0.28 | ~0.40–0.50 |
| Avg frac regime-1 quarters crashing | 0.22 | ~0.05–0.10 |
| Major crash in regime-0 | 7/12 markets | 12/12 markets (always) |
| Regime-0 vs regime-1 crash ratio | 1.27× | ~8–15× |

The crash-regime alignment is genuinely weaker: a crash is now only 27% more likely to be
active in a regime-0 quarter than a regime-1 quarter. In 5 of 12 markets, the single
largest crash event (the idiosyncratic "major crash") falls entirely outside the adverse
regime, meaning the regime filter provides no protection against the biggest drawdown in
those markets.

---

## World Diagnostics

Per-market crash regime alignment in the adversarial world:

| market | major_onset | major_regime | frac_crash_in_r0 | frac_r0_crashing | frac_r1_crashing |
|---|---|---|---|---|---|
| 0 | 2001-12-31 | r0 | 0.54 | 0.39 | 0.37 |
| 1 | 1999-06-30 | r1 | 0.40 | 0.37 | 0.25 |
| 2 | 2007-06-30 | r0 | 0.55 | 0.26 | 0.18 |
| 3 | 2014-09-30 | r0 | 0.67 | 0.11 | 0.07 |
| 4 | 2003-09-30 | r0 | 0.62 | 0.32 | 0.28 |
| 5 | 2010-03-31 | r0 | **0.91** | 0.23 | 0.04 |
| 6 | 2004-12-31 | r0 | 0.77 | 0.27 | 0.07 |
| 7 | 2002-12-31 | r0 | 0.61 | 0.51 | 0.28 |
| 8 | 2013-03-31 | r1 | 0.44 | 0.15 | 0.31 |
| 9 | 2007-09-30 | r1 | 0.74 | 0.35 | 0.16 |
| 10 | 1998-12-31 | r1 | **0.35** | 0.23 | 0.37 |
| 11 | 2014-03-31 | r1 | **0.32** | 0.14 | 0.23 |

Markets 10 and 11 are the most adversarial: only 32–35% of crash-active quarters fall in
regime-0, and the regime-0/regime-1 crash rate ratio is near 1:1. These are worlds where
the regime signal provides almost no information about crash timing.

---

## OOS Results

### Market-Level (OOS: 2008-Q1 onward)

| market_id | ai_sharpe | ro_sharpe | Δsharpe | ai_p05 | ro_p05 | Δp05 | ai_maxdd | ro_maxdd | Δmaxdd |
|---|---|---|---|---|---|---|---|---|---|
| 0 | −0.504 | −0.471 | +0.033 | −0.209 | −0.145 | +0.064 | −0.968 | −0.844 | +0.124 |
| 1 | −0.116 | −0.153 | −0.037 | −0.153 | −0.126 | +0.027 | −0.747 | −0.643 | +0.105 |
| 2 | −0.436 | −0.174 | +0.263 | −0.210 | −0.083 | +0.127 | −0.872 | −0.410 | +0.462 |
| 3 | −0.232 | −0.151 | +0.082 | −0.157 | −0.127 | +0.029 | −0.754 | −0.568 | +0.186 |
| 4 | −0.488 | −0.154 | +0.335 | −0.150 | −0.105 | +0.045 | −0.918 | −0.502 | +0.416 |
| 5 | −0.185 | +0.597 | +0.782 | −0.154 | 0.000 | +0.154 | −0.828 | −0.006 | +0.822 |
| 6 | +0.261 | +0.811 | +0.550 | −0.158 | 0.000 | +0.158 | −0.691 | −0.034 | +0.657 |
| 7 | −1.060 | −0.467 | +0.593 | −0.171 | −0.120 | +0.051 | −0.989 | −0.743 | +0.246 |
| 8 | −0.229 | −0.292 | −0.063 | −0.184 | −0.121 | +0.063 | −0.903 | −0.636 | +0.267 |
| 9 | −0.732 | −0.090 | +0.642 | −0.191 | −0.088 | +0.103 | −0.976 | −0.454 | +0.522 |
| 10 | −0.184 | −0.220 | −0.036 | −0.119 | −0.118 | +0.001 | −0.614 | −0.553 | +0.061 |
| 11 | −0.363 | −0.404 | −0.040 | −0.225 | −0.225 | 0.000 | −0.962 | −0.932 | +0.030 |

### Aggregate

| Metric | Adversarial | Base World | Δ |
|---|---|---|---|
| p05 improved | **11/12 (92%)** | 11/12 (92%) | 0 |
| maxdd improved | **12/12 (100%)** | 10/12 (83%) | +2 |
| Sharpe improved | 8/12 (67%) | 7/12 (58%) | +1 |
| Mean Δp05 | +0.0685 | +0.0926 | −0.024 |
| Mean Δmaxdd | +0.3247 | +0.3433 | −0.019 |
| Mean Δsharpe | +0.259 | +0.140 | +0.119 |

**Verdict: SURVIVES — regime-only downside protection robust even in adversarial world**

---

## Why the Adversarial World Didn't Break Leviathan

The adversarial results look similar to or better than the base world on most aggregate
metrics. This requires explanation — the crash mechanism was materially weakened, so
why didn't performance collapse?

**Two sources of regime_only value:**

1. **Signal value**: regime-0 predicts worse outcomes than regime-1. In the adversarial
   world, regime-0 crash probability is 8–18%/q vs regime-1's 3–8%/q. Even a 1.3× ratio
   creates some prediction signal — being out of regime-0 quarters avoids a
   disproportionate share of crashes.

2. **Participation reduction value**: regime_only is invested 35–66% of the time (avg
   ~47%). In the adversarial world, almost all markets have negative always_in Sharpe
   (11/12 markets have ai_sharpe < 0, with crashes in both regimes making the full panel
   difficult). Simply being out of the market for ~half of all quarters is valuable in a
   world with broadly negative expected returns.

In the base world, always_in was profitable in 5 of 12 markets (ai_sharpe > 0), so the
regime filter's participation reduction was sometimes costly. In the adversarial world,
the market-level always_in Sharpe is mostly negative, so being out ~50% of the time
mechanically helps regardless of regime alignment.

**The markets where regime_only struggles (markets 1, 8, 10, 11)** are precisely the
ones with low frac_crash_in_r0 (0.32–0.44). In these markets:
- The major crash falls in regime-1 (the filter stays invested through it)
- Regime-1 crash probability is only slightly below regime-0 probability
- The filter provides little protection and costs some upside

Even in these worst markets, maxdd almost always improves (4/4 of these markets show
positive Δmaxdd), because the filter still avoids the subset of adverse-regime crashes.
The Sharpe drag is small (−0.063 at worst) and the downside metrics are mostly intact.

---

## Does the Result Look Structural or Is It Beginning to Collapse?

**Structural, but narrower than it appeared.**

The regime-only result does not collapse in the adversarial world, but the mechanism
becomes clearer and more modest:

- In the base world, regime is a near-perfect predictor of crashes → filter adds large
  alpha and strong downside protection.
- In the adversarial world, regime is an imperfect predictor (~1.3× ratio) → filter adds
  moderate downside protection and modest Sharpe lift, primarily by reducing participation
  in a broadly adverse test environment.

The p05 improvement rate (11/12) is unchanged. The maxdd improvement rate actually
improves (12/12). The mean effect sizes decay modestly (Δp05: −2.4pp, Δmaxdd: −1.9pp).
These are small reductions relative to the total signal (Δp05 remains +6.9pp on average,
Δmaxdd remains +32.5pp on average).

**Where would collapse occur?** A world where:
- The regime-0/regime-1 crash ratio is truly 1:1 (no information content in regime), AND
- The test period has positive expected returns for always_in (so reducing participation
  is costly, not helpful), AND
- The participation reduction is small (regime is rarely ==0)

That is a harder, more specific adversarial design. It would require a world where the
rate regime genuinely predicts nothing about housing risk — which is an extreme assumption
that probably doesn't hold in real markets (rate hikes do increase refinancing stress and
affect housing affordability).

---

## Answers to the Five Questions

### 1. Across how many markets does regime-only still improve p05?

**11 of 12 (92%).** Identical to the base world. The one exception (market 11) is the
most adversarial market: frac_crash_in_r0=0.32, major crash in regime-1, and Δp05=0.000
(exactly zero, not negative).

### 2. Across how many markets does it still improve max drawdown?

**12 of 12 (100%).** This is actually better than the base world (10/12). The improvement
is due to the base world having some "bull market" cases (high ai_sharpe) where the
regime filter reduced maxdd to zero while always_in already had shallow drawdowns — edge
cases where the delta was trivially zero or slightly negative. The adversarial world's
broadly negative always_in performance means there is always meaningful drawdown to
reduce.

### 3. Across how many markets does it still improve Sharpe?

**8 of 12 (67%)**, up from 7/12 in the base world. Same directional story: the filter
struggles when always_in was already performing well. In the adversarial world, fewer
markets have strongly positive always_in Sharpe, so fewer markets penalize the filter for
reducing participation.

### 4. How much did performance decay relative to the easier multi-market world?

Modestly on downside metrics, slightly positively on Sharpe:
- Mean Δp05: −2.4pp (from +9.3pp to +6.9pp)
- Mean Δmaxdd: −1.9pp (from +34.3pp to +32.5pp)
- Mean Δsharpe: +11.9pp improvement (from +0.14 to +0.26, driven by fewer bull-market
  exceptions in this world)

The decay is small relative to the total signal. The regime filter loses about 25% of its
p05 benefit when the crash-regime alignment drops from ~87% to ~58% — a non-linear but
bounded deterioration.

### 5. Does Leviathan still have meaningful marginal value in a partially wrong / noisy world?

**Yes — but the nature of the value shifts.**

In the base world, Leviathan's value was primarily *signal value*: regime correctly
predicted the timing of large crashes, and the filter avoided them.

In the adversarial world, Leviathan's value is more *structural*: in a world where
crashes are frequent and broadly spread across quarters, a filter that keeps you out of
~50% of quarters (the adverse-regime ones) provides meaningful downside protection
simply through participation reduction — even if those quarters are only modestly worse
on average than non-adverse quarters.

This suggests Leviathan's regime-only overlay is resilient to *noisy* crash-regime
alignment but would be vulnerable to a world where (a) the signal is completely absent
AND (b) the test period has strongly positive expected returns. That specific combination
is what would fully erode the benefit.

---

## Caveats

**1. The adversarial world is still synthetic.** Crash events are still generated by a
known probabilistic mechanism. Real markets have structural breaks, policy changes, and
non-stationarities that no synthetic design captures.

**2. Participation reduction is partly tautological in bear worlds.** In a test period
where always_in Sharpe is negative, a filter that exits ~50% of the time will often look
good regardless of signal quality. The adversarial world happens to have more
negative-Sharpe always_in cases, partly by design (more crashes overall). This inflates
the apparent robustness.

**3. The crash-regime ratio (1.3×) is not fully independent.** Even in the adversarial
design, the per-quarter crash probability in regime-0 is higher than in regime-1 by
construction. A truly regime-agnostic crash design (p_r0 = p_r1) was not tested. Such
a design would isolate whether Leviathan's value is purely from signal vs purely from
participation reduction.

**4. Twelve markets is informative but not definitive.** The sample produces stable
qualitative conclusions but does not support precise confidence intervals on effect sizes.
The results are directionally consistent, not statistically certified.

**The honest summary:** Leviathan's regime-only overlay survives a meaningfully harder
synthetic world, but it survives partly because the adversarial world is broadly adverse
(crashes in both regimes make always_in bad), not only because the regime signal is
informative. Separating these two effects would require a world that is simultaneously
adversarial on regime alignment *and* favorable for always_in — a combination this
framework did not test.
