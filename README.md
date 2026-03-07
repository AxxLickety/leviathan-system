## Phase V–VII extension

Recent notebook-based extensions expand Leviathan beyond the original Phase I–IV public snapshot into a regime-aware housing risk prototype.

### Phase V — Affordability regime definition
This phase tests whether affordability behaves like a **regime switch** rather than a continuous predictor.  
Using DTI threshold behavior, the notebook identifies a fragile affordability regime in which forward housing downside risk rises sharply.

Notebook:
- `notebooks/phase5/phase5_regime_definition.ipynb`

### Phase VI — Affordability × supply interaction
This phase evaluates whether supply expansion amplifies housing crash probability once the market has entered a fragile affordability regime.

Key finding:
- crash probability is negligible in stable affordability regimes
- crash probability rises materially in fragile regimes
- crash probability becomes most severe when fragile affordability coincides with supply expansion

Notebook:
- `notebooks/phase6/phase6_interaction_test.ipynb`

### Phase VII — Risk overlay backtest
This phase tests a simple regime-aware exposure rule:

- remain invested in normal conditions
- exit housing exposure when fragile affordability and supply expansion occur simultaneously

The overlay reduces left-tail exposure and improves maximum drawdown relative to the baseline.

Notebook:
- `notebooks/phase7/phase7_affordability_overlay.ipynb`
