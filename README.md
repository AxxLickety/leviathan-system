# Leviathan — Regime-Aware Risk Gate (Phase I–IV)

Status: Phase IV complete (Feb 2026)
Thesis: Regime-conditional gating to reduce left-tail exposure under fragile affordability and credit conditions.
Scope: IC diagnostics → regime breakdown → gating rule → tail validation → OOS checks. (Methodological research only.)

## 5-minute tour (start here)
1. Overview notebook: notebooks/00_readme/00_project_overview.ipynb
2. Latest runner: src/phase4/run_phase4.py
3. Logit/floor calibration: run_path_a.py + src/research/path_a/

## Phase memos (public HTML)
- Phase II: https://axxlickety.github.io/leviathan-phase2-report/
- Phase III: https://axxlickety.github.io/leviathan-phase3-deliverable/
- Phase I: https://axxlickety.github.io/AxL_Leviathan_Mispricing_Model_DTI/

## Repo structure
- src/: core framework (features, signals, evaluation, backtests, phase runners)
- notebooks/: research notebooks (EDA + phase experiments)
- scripts/: utilities (IC, rolling IC, backtests)
- docs/: methodology and data notes

## Quick run
pip install -r requirements.txt
python src/phase4/run_phase4.py
