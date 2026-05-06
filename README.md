# California Housing Price Prediction

An end-to-end machine learning project that predicts California housing prices from tabular features using a training pipeline and a Streamlit interface.

## Project Goals
- Clean and preprocess raw data from `housing.csv`.
- Engineer meaningful predictive features.
- Train and evaluate a regression model.
- Serve predictions through a simple interactive UI.

## Repository Files
- `clean_data.py`: cleaning, missing value handling, feature engineering, one-hot encoding.
- `feature_selection.py`: exports model-ready feature list.
- `train_model.py`: trains `HistGradientBoostingRegressor` and saves model.
- `evaluate_model.py`: computes RMSE, MAE, and R2.
- `tune_model.py`: optional randomized hyperparameter tuning.
- `run_pipeline.py`: executes the full pipeline in order.
- `app.py`: Streamlit app for live predictions and metrics display.
- `artifacts/`: generated files after pipeline execution.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```

### 3. Run the UI
```bash
streamlit run app.py
```

## Pipeline Overview
Data -> Cleaning -> Feature Engineering -> Encoding -> Feature Selection -> Training -> Evaluation -> UI Prediction

## Outputs
After running `python run_pipeline.py`, expected artifacts:
- `artifacts/housing_cleaned.csv`
- `artifacts/selected_features.json`
- `artifacts/housing_model.pkl`
- `artifacts/location_cluster_model.pkl`

Optional tuning output:
- `artifacts/housing_model_tuned.pkl`

## Model Metrics
Current reference metrics (may vary slightly by environment):
- RMSE: about 44,596
- MAE: about 29,355
- R2: about 0.8482

## Full Report With All Code
For internship/report submission with full script code included:
- `PROJECT_REPORT_WITH_FULL_CODE.md`

## GitHub Documentation and Governance
This repository now includes full GitHub guideline files:
- `GITHUB_GUIDELINES.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/pull_request_template.md`
- `.github/workflows/ci.yml`

## Recommended GitHub Workflow
1. Create branch: `feature/<topic>`, `fix/<topic>`, or `docs/<topic>`.
2. Make changes and test locally.
3. Commit with clear messages (e.g. `feat: add model tuning notes`).
4. Open PR using the provided template.
5. Merge after review and CI pass.

## Notes
- If you prefer not to track generated artifacts in Git, uncomment artifact lines in `.gitignore`.
- Keep `PROJECT_REPORT_WITH_FULL_CODE.md` updated whenever source files change.

