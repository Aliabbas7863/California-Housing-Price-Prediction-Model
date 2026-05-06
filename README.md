# California Housing Price Prediction

End-to-end machine learning project for predicting California house values from tabular census-style features. The repository includes a full training pipeline, model evaluation, optional hyperparameter tuning, and a Streamlit web interface for interactive predictions.

## Overview

This project is designed to be simple to run and easy to extend:

- Data is loaded from `housing.csv`.
- Preprocessing handles duplicates, missing values, and feature engineering.
- Spatial clustering and categorical one-hot encoding are applied.
- A `HistGradientBoostingRegressor` is trained and evaluated.
- Artifacts are saved under `artifacts/` for reuse in the Streamlit app.

## Key Features

- Reproducible multi-step pipeline (`run_pipeline.py`)
- Structured preprocessing with engineered ratio features
- Geographic clustering using `KMeans` on latitude/longitude
- Model training with gradient boosting for strong tabular performance
- Standard regression metrics: RMSE, MAE, and R2
- Optional randomized hyperparameter tuning
- User-friendly Streamlit inference app

## Tech Stack

- Python 3.9+
- pandas
- scikit-learn
- streamlit

## Project Structure

```text
.
|-- app.py
|-- clean_data.py
|-- evaluate_model.py
|-- feature_selection.py
|-- run_pipeline.py
|-- train_model.py
|-- tune_model.py
|-- housing.csv
|-- requirements.txt
|-- artifacts/
|   |-- housing_cleaned.csv
|   |-- selected_features.json
|   |-- housing_model.pkl
|   |-- location_cluster_model.pkl
|   `-- housing_model_tuned.pkl (optional)
|-- PROJECT_REPORT_WITH_FULL_CODE.md
|-- CONTRIBUTING.md
|-- CODE_OF_CONDUCT.md
`-- GITHUB_GUIDELINES.md
```

## Data Pipeline

### 1. Data Cleaning and Feature Engineering (`clean_data.py`)

Operations performed:

- Removes duplicate records
- Fills missing values:
	- Numeric columns: median
	- Categorical columns: mode
- Creates engineered features:
	- `rooms_per_household`
	- `bedrooms_per_room`
	- `population_per_household`
- Builds a location cluster feature using `KMeans(n_clusters=10)` from latitude/longitude
- Applies one-hot encoding to `ocean_proximity` and `loc_cluster`
- Saves:
	- `artifacts/housing_cleaned.csv`
	- `artifacts/location_cluster_model.pkl`

### 2. Feature Selection (`feature_selection.py`)

- Selects all model-ready columns except target `median_house_value`
- Stores selected feature names in:
	- `artifacts/selected_features.json`

### 3. Model Training (`train_model.py`)

- Loads cleaned data and selected features
- Splits data into train/test (80/20, `random_state=42`)
- Trains `HistGradientBoostingRegressor` with configured hyperparameters
- Saves trained model:
	- `artifacts/housing_model.pkl`
- Prints model performance on the test split

### 4. Evaluation (`evaluate_model.py`)

- Loads saved model and test split
- Reports:
	- RMSE
	- MAE
	- R2

### 5. Optional Hyperparameter Tuning (`tune_model.py`)

- Runs `RandomizedSearchCV` on `HistGradientBoostingRegressor`
- Uses RMSE-based scoring (`neg_root_mean_squared_error`)
- Saves best tuned estimator to:
	- `artifacts/housing_model_tuned.pkl`

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd mathtask
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full training pipeline

```bash
python run_pipeline.py
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

## Script Usage

Run each stage independently if needed:

```bash
python clean_data.py
python feature_selection.py
python train_model.py
python evaluate_model.py
python tune_model.py
```

## Streamlit Application

The app (`app.py`) provides:

- Sidebar inputs for all required housing attributes
- Real-time single-record prediction
- Automatic recomputation and display of RMSE, MAE, and R2 on a held-out test split

If artifacts are missing, the app prompts you to run:

```bash
python run_pipeline.py
```

## Generated Artifacts

After running the pipeline, expected outputs are:

- `artifacts/housing_cleaned.csv`
- `artifacts/selected_features.json`
- `artifacts/housing_model.pkl`
- `artifacts/location_cluster_model.pkl`

Optional output from tuning:

- `artifacts/housing_model_tuned.pkl`

## Reproducibility Notes

- The project uses fixed random seeds in key steps (`random_state=42`) to improve reproducibility.
- Exact metrics may still vary slightly across environments and package versions.

## Documentation and Governance

- Contribution guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Additional project guidelines: `GITHUB_GUIDELINES.md`
- Full report with complete code: `PROJECT_REPORT_WITH_FULL_CODE.md`

## Common Issues

- `FileNotFoundError` for artifacts:
	- Run `python run_pipeline.py` before evaluation or launching the app.
- Missing Python packages:
	- Re-run `pip install -r requirements.txt` in the active environment.
- Streamlit command not found:
	- Ensure `streamlit` is installed in the same environment used to run commands.

## Future Improvements

- Add model versioning and experiment tracking
- Add cross-validation reporting to the main training script
- Add unit tests for preprocessing and feature alignment
- Add containerized deployment (Docker)

