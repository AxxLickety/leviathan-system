# Phase 2 — Mechanism validation

## Phase 2A (boundary test): Timing vs. severity
**Result:** ΔDTI does **not** show a stable lead–lag relationship with subsequent house-price declines/slowdowns.  
**Implication:** affordability is unlikely to be a turning-point timing signal.

## Phase 2B (credit-risk validation): Delinquency severity
**Setup:** Merge project dataset with US mortgage delinquency (quarterly), focus on downturn regime, exclude COVID window (2020Q4–2022Q2).  
**Result:** High prior ΔDTI periods exhibit **materially worse delinquency** outcomes during stress episodes.  
**Implication:** affordability stress functions as a **conditional risk amplifier**.

## Key evidence
- Figure: `reports/phase2B_us_delinquency.png`
- Notebook (repro): `notebooks/phase2B_us_delinquency.ipynb`
