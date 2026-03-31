# PHASE_OOS_DESIGN.md
# Out-of-Sample Evaluation: Pre-Specified Research Design

**Status**: Pre-registered design. Must not be modified after OOS evaluation begins.

---

## Train Window
- Start: 1990Q1 (earliest available data)
- End: 2007Q4

## Test Window
- Start: 2008Q1
- End: [latest available data]

## Thresholds
- FIXED (prior): real_rate < 0 (zero real rate has macroeconomic justification, not data-derived)
- LEARNED from train only: DTI fragility cutoff, Path A logit coefficients
- After train window fit, all learned values must be hardcoded as constants in OOS notebook

## Mandatory Outputs
- Table 1: Cell counts (regime × supply) for train period and test period separately
- Table 2: Conditional crash frequency with Wilson 95% CI for each cell
- Table 3: Strategy comparison — always-in, regime overlay, always-out — showing annualized return, vol, sharpe, max drawdown, time-in-market %
- Figure 1: Three equity curves on one plot with regime transition dates marked
- Figure 2: Threshold sensitivity — vary DTI cutoff ±20% and show impact on key metrics

## Pre-Specified Success Criteria
- In test window: fragile regime conditional crash frequency > stable regime conditional crash frequency, with non-overlapping Wilson CIs
- Regime overlay max drawdown < always-in max drawdown in test window

## Pre-Specified Failure Criteria (must be reported honestly if observed)
- No crash events in test window fragile regime cells
- Always-out outperforms overlay on risk-adjusted basis
- Conclusions only hold at exact trained threshold, not robust to ±20% perturbation

## Data Revision Note
All data pulled from FRED as of [date of analysis]. Real-time vintage data not used. Revision risk acknowledged as acceptable given typical magnitude of housing data revisions.
