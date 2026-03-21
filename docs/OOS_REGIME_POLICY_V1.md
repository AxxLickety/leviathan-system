# OOS Regime Policy Experiment V1 — Research Memo

**Date:** 2026-03-21
**Script:** `scripts/oos_regime_policy_experiment.py`
**World:** Positive-drift adversarial multi-market (12 markets, always_in Sharpe > 0 in all)
**Signal:** Unchanged — regime = (real_rate ≥ 0) → adverse
**Artifacts:** `outputs/oos_regime_policy/`

---

## Signal Quality and Policy Design Are Different Questions

The positive-drift test established that Leviathan has genuine signal value: the
regime filter reliably reduces p05 and maxdd even when staying invested is profitable.
But it also established a real cost: full_exit sacrifices 1.6%/q of mean return and
Sharpe in 9 of 12 markets.

Signal quality is a property of the regime indicator. **Policy design is a separate
question**: how should the signal be translated into exposure, and for whom?

A regime filter that is always fully on (full_exit) is a blunt instrument. The signal
says "adverse conditions are elevated." It does not say "exit completely." The
interpretation between signal and action is a mandate decision:
- Capital preservation mandates may want the full exit
- Balanced mandates may want partial derisking
- Growth mandates may want to wait for confirmed persistence before reacting

This experiment holds the signal fixed and varies only the activation policy.

---

## What Each Policy Does

| Policy | Exposure in adverse regime | Exposure in accommodative |
|---|---|---|
| always_in | 100% | 100% |
| full_exit | 0% | 100% |
| partial_derisk_50 | 50% | 100% |
| persistent_2q | 100% for first adverse q; 0% after 2 consecutive | 100% |
| persistent_3q | 100% for first 2 adverse q; 0% after 3 consecutive | 100% |
| partial_persistent | 50% after 1 adverse q; 0% after 2+ consecutive | 100% |

**persistent policies**: The persistence filter is a false-positive suppressor. A 1-quarter
adverse regime signal may be transient (noise in the rate estimate, a brief cycle
overshoot). Requiring 2 or 3 consecutive quarters of adverse signal before exiting
filters those transient signals, avoiding the opportunity cost of exiting a profitable
quarter and immediately re-entering.

**partial_persistent**: A graduated policy that stages the exit — partial derisking on
the first confirmed adverse quarter, full exit only after the regime has persisted for a
second consecutive quarter. This combines both mechanisms.

---

## Results

### Policy Comparison Table (averaged across 12 markets)

| Policy | pct_inv | turnover | mean Δsharpe | mean Δp05 | mean Δmaxdd | p05 retained | cost retained |
|---|---|---|---|---|---|---|---|
| always_in | 100% | 0 | ref | ref | ref | ref | ref |
| full_exit | 52.6% | 12.8 | −0.251 | +0.041 | +0.182 | 100% | 100% |
| partial_derisk_50 | 76.3% | 12.8 | **−0.032** | +0.019 | +0.091 | 45% | **50%** |
| persistent_2q | 63.8% | 7.9 | −0.152 | +0.030 | +0.144 | 72.5% | 67.7% |
| **persistent_3q** | 70.4% | **4.8** | −0.096 | +0.027 | +0.118 | **66.2%** | **49.4%** |
| partial_persistent | 58.2% | 17.0 | −0.176 | +0.035 | +0.162 | 85.6% | 84.1% |

*p05 retained = policy mean Δp05 / full_exit mean Δp05. Cost retained = policy |Δmean| / full_exit |Δmean|.*

### Win Rates vs always_in

| Policy | p05 improved | maxdd improved | Sharpe improved |
|---|---|---|---|
| full_exit | 12/12 | 12/12 | 3/12 |
| partial_derisk_50 | 12/12 | 12/12 | **6/12** |
| persistent_2q | 11/12 | 10/12 | 3/12 |
| persistent_3q | 10/12 | 8/12 | 5/12 |
| partial_persistent | 12/12 | 12/12 | 4/12 |

---

## What the Numbers Reveal

### 1. Partial derisking is nearly proportional

`partial_derisk_50` retains 45% of p05 protection at 50% of the mean cost — essentially
a linear scale-down of full_exit. There is no efficiency magic from halving exposure;
you get approximately half the protection and pay approximately half the cost. The
practical benefit is the Sharpe outcome: mean Δsharpe improves from −0.251 to −0.032,
and Sharpe improves vs always_in in 6/12 markets (vs 3/12 for full_exit). For a
growth-oriented mandate that wants to stay meaningfully invested, partial_derisk_50 is
nearly Sharpe-neutral — the protection comes "for almost free" on a risk-adjusted basis,
because the reduced crash exposure roughly offsets the participation reduction.

### 2. Persistent filters show genuine super-linear efficiency

`persistent_3q` retains **66%** of p05 protection at only **49%** of the cost — a
1.34× efficiency ratio vs the 1:1 of partial_derisk_50. The efficiency gain comes from
filtering short-duration adverse signals. In a positive-drift world, a 1–2 quarter
adverse regime episode may have positive or near-zero expected return (rate briefly
positive, but no crash has occurred). Waiting 3 consecutive quarters to exit avoids
exiting those episodes while still protecting against genuinely persistent adverse
conditions (3+ consecutive quarters of elevated real rates) where crash risk is
meaningfully concentrated.

**Turnover is the bonus**: persistent_3q averages only 4.8 direction changes per market
vs 12.8 for full_exit. This is a 63% reduction in portfolio turnover — a practical
benefit in real implementation (transaction costs, tracking error, operational overhead).

### 3. partial_persistent is the "best-of-both" for conservative balanced mandates

It combines staging (partial exit after 1q, full after 2q) with a light persistence
filter (requires the regime to persist for the full exit). It retains 85.6% of p05
protection and 89.2% of maxdd protection at 84.1% of the cost. Its win rate on downside
metrics matches full_exit (12/12) while Δsharpe improves to −0.176 vs −0.251.
The turnover penalty (17.0, highest of all policies) is the practical cost — the staged
exit creates more exposure changes than a binary in/out.

### 4. The full_exit dominance question

Full_exit is not dominated on a single metric (it has the best p05 and maxdd numbers),
but it **is** dominated on the efficiency frontier by persistent_3q and partial_derisk_50:

- persistent_3q: same cost as partial_derisk_50, nearly 50% more protection
- partial_derisk_50: nearly Sharpe-neutral, universal win-rates on downside metrics

For any mandate that is not purely capital-preservation focused, full_exit is the
wrong activation policy — it maximizes protection but at unnecessary Sharpe cost.

---

## Which Policy Fits Which Mandate

### (a) Capital preservation mandate
**→ full_exit or partial_persistent**

A mandate focused on avoiding drawdowns above a threshold (e.g., a pension fund with a
funded-ratio floor, a housing fund with a 20% drawdown limit) should use full_exit.
It is the only policy that universally achieves zero p05 loss in many markets
(p05=0.000 in 7 of 12 markets). partial_persistent is a close second — nearly all the
protection at slightly lower turnover disruption.

### (b) Balanced mandate
**→ persistent_3q**

A mandate that cares about both drawdown control and long-run return should use
persistent_3q. It provides two-thirds of full_exit's p05 protection at half the mean
cost, requires 63% fewer portfolio changes, and improves Sharpe vs always_in in 5/12
markets (vs 3/12 for full_exit). The 3-quarter persistence requirement acts as a
natural "conviction filter" — the regime signal must be sustained before the portfolio
acts on it, reducing the cost of short-duration false positives.

### (c) Growth / return-seeking mandate
**→ partial_derisk_50**

A mandate that prioritizes compounded return but wants some downside protection (a
growth fund with a soft drawdown guideline, a leveraged investor who wants to reduce
cycle exposure) should use partial_derisk_50. It is nearly Sharpe-neutral in this
positive-drift world (mean Δsharpe = −0.032), improves Sharpe in 6/12 markets, and
retains 45% of the p05 protection universally. It requires no persistence threshold to
calibrate — simply half the exposure when adverse. Transparent and implementable.

---

## Policy Efficiency Summary

| Policy | p05 retained | Cost paid | Efficiency ratio |
|---|---|---|---|
| full_exit | 100% | 100% | 1.00 |
| partial_derisk_50 | 45% | 50% | 0.90 |
| persistent_2q | 72.5% | 67.7% | 1.07 |
| **persistent_3q** | **66.2%** | **49.4%** | **1.34** |
| partial_persistent | 85.6% | 84.1% | 1.02 |

*Efficiency = p05_retained / cost_paid. Higher means more protection per unit of mean sacrifice.*

**persistent_3q is the most efficient policy**: it extracts 34% more p05 protection per
unit of opportunity cost than full_exit, by selectively avoiding short false-positive
adverse episodes while protecting against sustained adverse regimes.

---

## Should Leviathan Be Thought of as an Always-On Overlay or a Conditional Risk Tool?

**A mandate-specific module.**

The signal (regime) is always-on and always computed. But whether to act on it, and how
strongly, depends on the mandate:

```
┌─────────────────────────────────────────────────────────────┐
│  Mandate         │  Policy            │  Outcome            │
├─────────────────────────────────────────────────────────────┤
│  Capital pres.   │  full_exit         │  Max protection      │
│  Balanced        │  persistent_3q     │  Best efficiency     │
│  Growth          │  partial_derisk_50 │  Near-neutral Sharpe │
└─────────────────────────────────────────────────────────────┘
```

Using a single policy for all mandates is a category error. Full_exit is appropriate
for capital preservation mandates. Applied to a growth mandate, it sacrifices Sharpe
unnecessarily. Applied to a balanced mandate, it overexits relative to what the signal's
information content warrants given the opportunity cost.

---

## Answers to the Five Questions

### 1. Which policy preserves the most downside protection?

**full_exit** — 100% of the p05 and maxdd improvement baseline, 12/12 win-rates.
No other policy matches it on raw downside metrics.

### 2. Which policy preserves the most mean return / Sharpe?

**partial_derisk_50** — mean Δsharpe = −0.032 (vs −0.251 for full_exit), Sharpe
improved in 6/12 markets. It is the most Sharpe-preserving policy with non-trivial
downside benefit.

### 3. Which policy offers the best protection/cost tradeoff?

**persistent_3q** — efficiency ratio 1.34×, meaning 66% of full_exit's p05 protection
at only 49% of its mean cost. It also has the lowest turnover (4.8 changes per market,
vs 12.8 for full_exit) — practically important for real portfolios.

### 4. Is full_exit too blunt relative to partial/persistent variants?

**Yes, for most mandates.** Full_exit is only optimal when (a) the mandate is purely
capital-preservation focused, and (b) the opportunity cost of missing adverse-regime
upswings is explicitly acceptable. For balanced and growth mandates, persistent_3q
and partial_derisk_50 respectively offer better tradeoffs. Using full_exit for all
mandates leaves protection efficiency on the table.

### 5. Does Leviathan become more practically usable once the policy layer is tuned?

**Yes, materially.** Three specific improvements from policy tuning:

1. **Sharpe neutrality is achievable**: partial_derisk_50 brings the Sharpe penalty from
   −0.251 to −0.032. For growth mandates, the filter becomes nearly cost-free on a
   risk-adjusted basis while retaining universal downside benefit.

2. **Turnover is dramatically reducible**: persistent_3q cuts turnover by 63% relative
   to full_exit. In a real portfolio, 4.8 direction changes per 16-year period is
   operationally low-maintenance. 12.8 is more disruptive.

3. **Win-rates on Sharpe improve**: full_exit improves Sharpe in 3/12 markets.
   partial_derisk_50 improves it in 6/12. The filter stops being a drag in half the
   markets for growth mandates.

The core Leviathan insight — regime predicts tail risk — survives all five policies.
The policy layer determines whether that insight is translated into a blunt protection
tool or a calibrated mandate-aware overlay. Leviathan is better understood as the
latter.

---

## Caveats

**1. All results are in-sample to the positive-drift world.** The policy comparison
uses the same 12 markets throughout. A holdout comparison across independent worlds
would provide tighter evidence that persistence thresholds are not fit to noise.

**2. Persistence thresholds (2q, 3q) were not search-optimized.** They were selected
on intuitive grounds. Running the same experiment with persistence = 1, 4, 5 quarters
would establish whether 3q is genuinely efficient or a local optimum in this 12-market
sample.

**3. The partial_persistent policy has more exposure states (0, 0.5, 1.0) but no
turnover advantage.** Its higher turnover (17.0 vs 12.8) offsets the benefit of
graduated exposure for real portfolios that face transaction costs.

**4. Policy efficiency assumes the crash-regime alignment in this world.** With the
alignment tested here (frac_crash_in_r0 ≈ 0.60), persistent_3q's efficiency of 1.34×
is credible. In a world with lower alignment (0.40), the persistence filter might wait
too long and miss crashes that don't require 3 quarters to develop. Policy selection
should account for the expected regime signal persistence in the target market.
