# Leviathan (Housing Affordability & Risk) — Start Here

## What this is
A research prototype testing whether **housing affordability stress** (DTI-based proxy) functions as a **regime-dependent risk filter** for housing/credit outcomes.

## Why it matters (1 sentence)
Affordability is often treated as a valuation/timing signal; this project tests whether it instead conditions **downside risk severity**, especially under certain rate regimes.

## Core results (read this first)
- **Phase 1:** Affordability shows **strong regime dependence**; it behaves more like a **downside risk filter** than a return-seeking factor (especially when real rates are non-negative).
- **Phase 2A:** ΔDTI is **not** a stable turning-point/timing predictor for house-price declines/slowdowns.
- **Phase 2B:** Conditional on downturn regimes (and excluding COVID), **high prior ΔDTI** is associated with **materially worse mortgage delinquency** outcomes (US quarterly delinquency data).

## What to open (suggested reading order)
1) Phase 1 summary (HTML): `reports/phase1_summary.html`  
2) Phase 2 key evidence figure: `reports/phase2B_us_delinquency.png`  
3) Phase 2 notebook (reproducibility): `notebooks/phase2B_us_delinquency.ipynb`  

## One question I’d like your guidance on
Given Phase 2 validates “risk severity” rather than “timing,” should the next step be:
(a) formalizing the note + tightening statistical framing, or  
(b) extending to additional datasets/regions (e.g., Canada), or  
(c) building a small backtest / risk dashboard around the filter?

