# OOS Overlay Multi-Baseline Experiment (Percentile-Normalized DTI) V1 — Research Memo

**Date:** 2026-03-21
**Depends on:** `docs/OOS_OVERLAY_MULTIBASELINE_V1.md`
**Script:** `scripts/oos_overlay_multibaseline_percentile.py`
**Artifacts:** `outputs/oos/overlay_multibaseline_percentile/`

---

## Motivation

The previous experiment (`OOS_OVERLAY_MULTIBASELINE_V1.md`) found that `valuation_tilt` was
structurally degenerate: 0% test-period participation because absolute DTI in the test window
(min=108.4) exceeded the training p60 threshold (101.1) for all 64 quarters.

This experiment replaces absolute DTI with **rolling 20-quarter causal percentile rank**, then
re-runs the multi-baseline overlay comparison.

---

## Why Expanding Percentile Also Fails

The first attempt at normalization used expanding percentile rank:

> *For each time t, the expanding rank is the fraction of all values [0:t] that are ≤ value[t].*

This appears causal but fails for a trending series. Because DTI rises ~50 units over 132 quarters
(~0.38 units/quarter), each new observation tends to be near the historical maximum. As a result,
the expanding percentile rank clusters near 1.0 in the test period:

| Window | Train expanding pct | Test expanding pct |
|---|---|---|
| describe | mean=0.84, p50=0.92 | mean=0.92, p50=0.95 |
| rows ≤ 0.60 | 9/64 (14%) | 0/64 (0%) |

The expanding percentile rank is degenerate for the same structural reason as the raw p60: DTI
has never been this high before, so the percentile rank of every test observation is high.

---

## Rolling 20-Quarter Percentile Rank

The rolling 20q rank measures DTI relative to the **recent 20 quarters only**, not all history.
Over 20 quarters, the DTI trend contributes ~7.5 units, which is comparable to the cross-sectional
noise (~8 DTI std). This means the within-window variation from cycle and noise is large enough
to give the rank a near-uniform distribution.

| Window | Train rolling-20q pct | Test rolling-20q pct |
|---|---|---|
| describe | usable range | 0.05–1.00, mean=0.69 |
| rows ≤ 0.60 | 15/64 (23%) | 23/64 (36%) |

**Rolling-20q percentile is a functional normalization: valuation_tilt now participates 35.9%
of test quarters, versus 0% under either absolute or expanding methods.**

---

## Parameters

| Parameter | Value | Source |
|---|---|---|
| Frozen absolute dti_cutoff | 98.8142 | oos_train.py (production) |
| Walk-forward rolling-pct cutoff | 0.5000 | Training walk-forward only |
| Valuation threshold | dti_pct_roll20 ≤ 0.60 | Fixed rule |
| Rolling window | 20 quarters | Design choice |

Note: `abs_cutoff=98.8142 < min(test DTI)=108.4`, so the absolute Leviathan gate
(`lev_abs`) reduces to `regime != 0` for all test periods. The DTI condition is vacuous.

---

## Results

Test period: 2008-Q1 → 2023-Q4 (64 quarters).

### All strategies

| Strategy | mean | sharpe | p05 | maxdd | pct_invested | turnover |
|---|---|---|---|---|---|---|
| always_in_raw | −0.0179 | −0.208 | −0.181 | −0.894 | 100% | 0 |
| always_in_lev_abs | +0.0010 | +0.017 | −0.108 | −0.590 | 50.0% | 11 |
| always_in_lev_pct | −0.0163 | −0.203 | −0.181 | −0.853 | 81.3% | 18 |
| valuation_roll20_raw | −0.0154 | −0.270 | −0.147 | −0.650 | 35.9% | 18 |
| valuation_roll20_lev_abs | +0.0009 | +0.125 | 0.000 | 0.000 | **1.6%** | 1 |
| valuation_roll20_lev_pct | −0.0165 | −0.293 | −0.147 | −0.650 | 32.8% | 18 |
| trend_raw | +0.0040 | +0.067 | −0.131 | −0.624 | 57.8% | 7 |
| trend_lev_abs | +0.0054 | +0.109 | −0.031 | −0.435 | 35.9% | 5 |
| trend_lev_pct | +0.0010 | +0.016 | −0.131 | −0.624 | 48.4% | 11 |

### Overlay lift: lev_abs (absolute DTI gate, ≡ regime filter in test)

| Baseline | Δmean | Δsharpe | Δp05 | Δmaxdd | Δpct_invest |
|---|---|---|---|---|---|
| always_in | +0.019 | +0.225 | +0.073 | +0.305 | −50.0% |
| valuation_roll20 | +0.016 | +0.395 | +0.147 | +0.650 | −34.4% |
| trend | +0.001 | +0.042 | +0.100 | +0.189 | −21.9% |

**→ Verdict: ALPHA-ENABLING — p05: 3/3, maxdd: 3/3, Sharpe: 3/3**

### Overlay lift: lev_pct (rolling-percentile DTI gate, cutoff=0.50)

| Baseline | Δmean | Δsharpe | Δp05 | Δmaxdd | Δpct_invest |
|---|---|---|---|---|---|
| always_in | +0.002 | +0.005 | 0.000 | +0.041 | −18.8% |
| valuation_roll20 | −0.001 | −0.022 | 0.000 | 0.000 | −3.1% |
| trend | −0.003 | −0.051 | 0.000 | 0.000 | −9.4% |

**→ Verdict: STANDALONE FLAG — p05: 0/3, maxdd: 1/3, Sharpe: 1/3**

---

## Key Findings

### 1. Rolling-20q percentile fixes the degeneracy

| Normalization | Train invested | Test invested | Status |
|---|---|---|---|
| Raw DTI (abs p60=101.1) | 100% (by def.) | 0% | Degenerate |
| Expanding pct ≤ 0.60 | 14.1% | 0% | Degenerate |
| Rolling-20q pct ≤ 0.60 | 23.4% | 35.9% | **Functional** |

The rolling-20q valuation_tilt is now a usable baseline with reasonable test-period participation.

### 2. lev_abs improves all three baselines — but via regime, not DTI

With the absolute gate, all 64 test-period DTI values exceed the frozen cutoff (98.81). The
gate therefore fires on **every** `regime==0` observation, making it purely a regime filter.
This is the same as gating on `regime != 0` with no reference to DTI at all.

The lev_abs results confirm the regime filter is effective: p05 improves across all three
baselines, and maxdd improves by 18.9–65.0pp. However, this does **not** validate the DTI
dimension of Leviathan — it only confirms that filtering out positive-real-rate periods
removes the worst test-period quarters. This was known from the regime assignment itself.

### 3. lev_pct (rolling percentile gate) is weak

The walk-forward selected `pct_cutoff=0.50` — the minimum of the search grid. This is the
walk-forward's way of saying: no DTI percentile threshold in the training window meaningfully
improved p05 of filtered returns; the optimizer converged on the lowest (most permissive)
threshold available. In test, `lev_pct` at 0.50 passes 81% of periods and improves metrics
only marginally for always_in (+4.1pp maxdd). For valuation and trend baselines it slightly
hurts both mean and Sharpe by removing some positive-return periods.

### 4. Valuation_roll20 + lev_abs = near-empty intersection

`valuation_roll20_lev_abs` invests only **1.6% of test quarters (1 of 64)**. This is because
the two signals select nearly opposite periods:

- `valuation_roll20` invests when DTI is **low relative to recent history** (bottom 40% of
  rolling 20q window). This tends to happen during falling or recovering DTI episodes.
- `lev_abs` in test is equivalent to `regime == 1` (accommodative policy, negative real rates).
  Accommodative periods tend to coincide with rising DTI, not falling DTI.

The near-zero intersection reveals a structural anti-correlation between the two signals in
this synthetic panel. Combining them eliminates almost all exposure. The large Δmaxdd (+0.650)
for valuation_roll20_lev_abs is an artifact of 98.4% cash allocation, not meaningful protection.

---

## Comparison to Raw-DTI Experiment (V1)

| Aspect | V1 (raw DTI) | This experiment |
|---|---|---|
| valuation_tilt test participation | 0% (degenerate) | 35.9% (functional) |
| lev_abs verdict | ALPHA-ENABLING (2/3) | ALPHA-ENABLING (3/3) |
| DTI component of lev_abs active | No (all DTI > cutoff) | No (same cutoff) |
| lev_pct tested | No | Yes — STANDALONE FLAG |
| Regime filter vs DTI filter isolated | No | Yes |

The key new result is the isolation: **the lev_abs "Leviathan" signal is effectively a regime
filter with an inert DTI condition**. The DTI level chosen by the walk-forward (98.81) is
always exceeded in test, so the DTI dimension adds nothing. Introducing a rolling-percentile
DTI gate (`lev_pct`) does not recover the result — it is weak (STANDALONE FLAG).

---

## Answers to Experiment Questions

### 1. Did percentile normalization fix the valuation_tilt degeneracy?

**Yes** — rolling 20q percentile makes valuation_tilt functional (35.9% test participation vs 0%).
Expanding percentile did not fix it because it suffers from the same problem: a trending series
has expanding ranks that cluster near 1.0 over time.

### 2. Does Leviathan still improve multiple baselines after removing raw-DTI drift effects?

**Only if lev_abs (regime-only gate) is counted as "Leviathan."** The absolute DTI gate
improves all 3 baselines (3/3), but the DTI component is vacuous — it is purely a regime
filter. The rolling-percentile gate that actually uses DTI variation (`lev_pct`) does not
improve multiple baselines (0/3 on p05).

### 3. Is the case for calling Leviathan alpha-enabling now stronger or weaker?

**Weaker than V1, but the honest interpretation is clearer.**

- In V1, the DTI component of Leviathan appeared to be active (the cutoff 98.81 sat within
  the training DTI distribution). The regime×DTI interaction seemed to matter.
- Here, we find the DTI condition is vacuous in test: all 64 test periods have DTI > cutoff.
  The regime filter alone accounts for the full effect. The DTI walk-forward adds nothing.
- When we introduce a genuinely active percentile DTI gate (`lev_pct`), it does not work.

**The regime filter is real and effective. The DTI filter has not been validated.**

---

## Implications

The core Leviathan hypothesis is: risk is elevated when `regime == 0` **AND** `DTI is high`.
The joint condition should identify periods when both channels of stress are active simultaneously.

What we actually found:
- In the test period, `dti > 98.81` is always true (DTI trend exceeded the training cutoff).
- The gate reduces to `regime == 0` alone — one channel, not two.
- A rolling-percentile DTI gate that does preserve cross-sectional variation fails to improve
  outcomes — suggesting the DTI relative-rank dimension does not add signal above regime alone
  (at least in this synthetic design).

**Next research step:** Redesign the Leviathan gate to use rolling-percentile DTI directly in
the walk-forward training and test evaluation (not as an add-on overlay). Test whether
`regime == 0 AND dti_pct_roll > threshold` (jointly optimized) outperforms regime-only
filtering. This would test the interaction hypothesis more directly than the current approach.
