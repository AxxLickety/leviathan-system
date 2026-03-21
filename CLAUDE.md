# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Leviathan** is a housing market research project testing whether **housing affordability stress** (DTI-based proxy) functions as a **regime-dependent risk filter** rather than a traditional return-seeking signal. The core hypothesis is that affordability conditions downside risk severity, especially under certain interest rate regimes.

## Development Commands

```bash
# Activate virtual environment (required for all commands)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Path A pipeline (logistic regression, thresholds, predictions)
python run_path_a.py

# Start Jupyter for notebook analysis
jupyter lab

# Run specific analysis scripts
python scripts/strategy_dti_filter.py
python scripts/calc_ic.py
python scripts/rolling_ic.py
```

## Environment Setup

The project uses a Python virtual environment (`.venv/`) and requires `PYTHONPATH` to be set for imports to work correctly. Use the `.envrc` file with direnv or source it manually:

```bash
source .envrc
```

This sets:
- `PYTHONPATH` to include the project root
- Activates `.venv/bin/activate`
- Preserves user CLI tools in PATH

## Architecture

### Directory Structure

- **`notebooks/`** - Primary research artifacts organized by phase:
  - `00_readme/` - Project overview notebooks
  - `01_data_eda/` - Exploratory data analysis
  - `02_signal_research/` - Signal construction and regime analysis (Phases 1-7)
  - `03_decision_layer/` - Decision surface learning (Phases 9-10, core contribution)
  - Key files:
    - `Leviathan_Phase1_Summary.ipynb` - Main Phase 1 findings
    - `phase2B_us_delinquency.ipynb` - Delinquency validation
    - `phase9_oos_decision.ipynb` - Out-of-sample decision robustness
    - `phase10_decision_surface.ipynb` - Learned decision surface (main contribution)

- **`src/`** - Reusable Python modules:
  - `core/` - Pipeline infrastructure, backtest engine, factor builder
  - `loaders/` - Data loading utilities
  - `signals/` - Signal implementations (DTI, affordability, migration, composite)
  - `features/` - Feature engineering utilities
  - `evaluation/` - Performance metrics and evaluation
  - `backtests/` - Backtesting strategies and timing models
  - `models/` - Model implementations
  - `utils/` - General utilities
  - `research/path_a/` - Path A specific logic (logit fitting, threshold computation, labeling)

- **`scripts/`** - Standalone analysis scripts:
  - `strategy_dti_filter.py` - DTI filtering strategy
  - `strategy_dti_rate_filter.py` - DTI + rate regime filtering
  - `calc_ic.py` / `rolling_ic.py` - Information coefficient analysis
  - `fred_smoke_test.py` - FRED API connectivity test

- **`config/`** - Configuration files:
  - `universe.py` - Geographic universe (Austin, Toronto currently)
  - YAML files for data/model config (empty templates)

- **`docs/`** - Project documentation:
  - `DATA_SOURCES.md` - Data sources (DTI proxy, real rates, delinquency from FRED)
  - `PHASE_1_SUMMARY.md` - Phase 1 findings (regime dependence)
  - `PHASE_2_SUMMARY.md` - Phase 2 findings (mechanism validation)

- **`data/`**, **`outputs/`**, **`reports/`** - Gitignored directories for generated artifacts

### Research Phases

**Phase 1**: Regime dependence testing
- Finding: Affordability behaves as a **risk/overheating filter** rather than return-seeking factor
- Artifact: `reports/phase1_summary.html`

**Phase 2A**: Timing vs. severity boundary test
- Finding: ΔDTI does **not** show stable lead-lag with house price declines (not a timing signal)

**Phase 2B**: Delinquency severity validation
- Finding: High prior ΔDTI associated with **materially worse mortgage delinquency** during stress
- Evidence: `reports/phase2B_us_delinquency.png`, `notebooks/phase2B_us_delinquency.ipynb`

**Phases 9-10**: Decision layer (core contribution)
- Phase 9: Out-of-sample decision robustness under regime transitions
- Phase 10: Learned decision surface showing volatility compression and improved risk-adjusted behavior
- Location: `notebooks/03_decision_layer/`

### Key Implementation Patterns

**Path A Execution**:
The `run_path_a.py` script orchestrates the full pipeline:
1. `build_master_df()` - Constructs master dataset
2. `add_correction_label()` - Labels correction events
3. `fit_interaction_logit()` - Fits logistic regression with interactions
4. `compute_dti_thresholds()` - Derives risk thresholds from coefficients
5. Output to `outputs/path_a/` directory

**Module Imports**:
All imports use absolute paths from `src.` (e.g., `from src.loaders.fred import load_fred_series`). The `PYTHONPATH` must include the project root for this to work.

**Data Loading**:
- Internal series (DTI, real rates, house price index) constructed in pipeline
- External validation (US mortgage delinquency) pulled from FRED
- Universe currently limited to Austin and Toronto (see `config/universe.py`)

## Tech Stack

- **Language**: Python 3.x
- **Core Libraries**:
  - `pandas`, `numpy` - Data manipulation
  - `statsmodels` - Statistical modeling (logit, time series)
  - `scikit-learn` - ML utilities
  - `matplotlib`, `seaborn` - Visualization
  - `yfinance`, `pandas-datareader` - Financial data
  - `QuantStats` - Performance metrics
- **Notebook Environment**: Jupyter Lab
- **Database**: `psycopg2-binary` (PostgreSQL support, if used)
- **Data Validation**: `pydantic`

## Important Notes

- **Data & Outputs**: The `data/`, `outputs/`, and `reports/` directories are gitignored. Generated artifacts will not be committed.
- **Virtual Environment**: Always activate `.venv` before running any code. Use `.envrc` for automatic activation with direnv.
- **PYTHONPATH**: Must be set to project root for imports to work. The `.envrc` file handles this.
- **Research Focus**: This is a research prototype, not production code. Emphasis is on reproducibility and clarity of findings rather than optimization.
- **Geographic Scope**: Currently limited to Austin and Toronto (Phase 1 & 2 evaluation universe)
- **Primary Contribution**: The decision layer work (Phases 9-10) represents the main research contribution, showing that macro-aware gating improves decision stability versus traditional return-seeking approaches.

## Core System Conventions

These definitions are canonical. All new code must follow them. Existing notebooks
predate some of these conventions and may use older patterns (see notes below).

### Regime

- **Operational regime**: `assign_fragility_regime()` in `src/evaluation/regime.py`
  - Input: `real_rate` column (real interest rate level)
  - Formula: `(real_rate < threshold).astype(int)`, default `threshold=0.0`
  - Encoding: **integer** — `1` when `real_rate < 0` (negative real rates); `0` when `real_rate >= 0`
  - `REGIME_FOR_RISK = 0` throughout scripts; `affordability_active = (regime == 0)` in pipeline
  - Ex-ante: observable at decision time, no look-ahead bias
  - `assign_fragility_regime()` is the canonical definition; `src/core/pipeline.py` calls it directly
  - Path A (`src/research/path_a/build_dataset.py`) keeps an equivalent inline implementation
    (`(real_rate < 0).astype(int)`) to avoid dependency coupling — intentional, not an inconsistency
- **Diagnostic regime**: `assign_directional_regime()` in `src/evaluation/regime.py`
  - Input: realized forward return sign
  - Encoding: string `"up"` / `"down"`
  - Ex-post: contains look-ahead bias — for historical attribution and IC decomposition only
  - **Must not be used in decision logic or backtest filters**
- **Deprecated alias**: `assign_regime()` — calls `assign_directional_regime`, emits
  `DeprecationWarning`. Exists only for backward compatibility with existing notebooks.

### Forward Return

- **Definition**: 4-quarter log return — `log(p[t+4]) - log(p[t])`
- **Horizon**: 4 quarters = 1 year on quarterly data
- **Formula**: log difference (not arithmetic percent change)
- **Column name**: `"fwd_return"` in pipeline context (`src/core/pipeline.py`, notebooks);
  `"fwd_ret_4q"` in scripts (`scripts/`)
- **Authoritative implementation**: `compute_forward_return()` in `src/evaluation/backtest.py`
- Note: `"ret_1q_fwd"` in `scripts/phase2_backtest.py` is a separate 1-quarter return
  used for strategy simulation — intentionally different, not an error

### Performance Metrics

- **Canonical module**: `src/backtests/evaluation.py`
- **`sharpe(x)`**: `mean / std` using `ddof=1` (sample standard deviation)
- **`max_drawdown(log_r)`**: equity-curve drawdown — converts log returns to price levels
  via `exp(cumsum)`, then computes peak-to-trough ratio
- **`summarize(x, name="")`**: standard summary interface returning:
  `name, n, mean, vol, sharpe, p05, p50, p95, min, max, maxdd`
- All scripts in `scripts/` import from this module. Do not define inline metric
  functions in new scripts.
