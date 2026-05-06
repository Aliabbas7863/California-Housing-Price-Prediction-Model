import json
import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "housing_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "selected_features.json")
CLEANED_DATA_PATH = os.path.join(ARTIFACTS_DIR, "housing_cleaned.csv")
CLUSTER_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "location_cluster_model.pkl")

OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None

    if not os.path.exists(FEATURES_PATH):
        return None, None, None, None

    if not os.path.exists(CLEANED_DATA_PATH):
        return None, None, None, None

    cluster_model = None
    if os.path.exists(CLUSTER_MODEL_PATH):
        with open(CLUSTER_MODEL_PATH, "rb") as f:
            cluster_model = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "r") as f:
        selected_features = json.load(f)

    df = pd.read_csv(CLEANED_DATA_PATH)
    return model, selected_features, df, cluster_model


def build_input_row(user_input, selected_features, cluster_model=None):
    row = {
        "longitude": user_input["longitude"],
        "latitude": user_input["latitude"],
        "housing_median_age": user_input["housing_median_age"],
        "total_rooms": user_input["total_rooms"],
        "total_bedrooms": user_input["total_bedrooms"],
        "population": user_input["population"],
        "households": user_input["households"],
        "median_income": user_input["median_income"],
    }

    if row["households"] <= 0:
        row["households"] = 1
    if row["total_rooms"] <= 0:
        row["total_rooms"] = 1

    row["rooms_per_household"] = row["total_rooms"] / row["households"]
    row["bedrooms_per_room"] = row["total_bedrooms"] / row["total_rooms"]
    row["population_per_household"] = row["population"] / row["households"]

    if cluster_model is not None:
        try:
            cluster_id = int(
                cluster_model.predict(
                    pd.DataFrame([[row["latitude"], row["longitude"]]], columns=["latitude", "longitude"])
                )[0]
            )
        except Exception:
            cluster_id = 0
        for i in range(10):
            row[f"loc_cluster_{i}"] = 1 if i == cluster_id else 0
    else:
        for i in range(10):
            row[f"loc_cluster_{i}"] = 0

    for option in OCEAN_OPTIONS:
        col = f"ocean_proximity_{option}"
        row[col] = 1 if user_input["ocean_proximity"] == option else 0

    row_df = pd.DataFrame([row])

    for feature in selected_features:
        if feature not in row_df.columns:
            row_df[feature] = 0

    return row_df[selected_features]


def compute_metrics(model, selected_features, df):
    x = df[selected_features]
    y = df["median_house_value"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    result = pd.DataFrame(
        {
            "Actual Price": y_test.values,
            "Predicted Price": predictions,
        }
    ).reset_index(drop=True)

    return rmse, mae, r2, result


def main():
    st.set_page_config(page_title="California Housing Price Prediction", layout="centered")
    st.title("California Housing Price Prediction")
    st.write("Predict house value using the trained model.")

    model, selected_features, df, cluster_model = load_artifacts()

    if model is None:
        st.error("Required files are missing in artifacts/.")
        st.info("Run: python run_pipeline.py")
        return

    # Move input controls to the sidebar for cleaner UI
    st.sidebar.header("Input features")
    longitude = st.sidebar.number_input("Longitude", value=-122.23, format="%.4f")
    latitude = st.sidebar.number_input("Latitude", value=37.88, format="%.4f")
    housing_median_age = st.sidebar.number_input("Housing Median Age", min_value=1.0, value=25.0)
    total_rooms = st.sidebar.number_input("Total Rooms", min_value=1.0, value=2000.0)
    total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1.0, value=400.0)
    population = st.sidebar.number_input("Population", min_value=1.0, value=1200.0)
    households = st.sidebar.number_input("Households", min_value=1.0, value=350.0)
    median_income = st.sidebar.number_input("Median Income", min_value=0.1, value=4.5)
    ocean_proximity = st.sidebar.selectbox("Ocean Proximity", OCEAN_OPTIONS)

    predict_button = st.sidebar.button("Predict")

    st.subheader("Prediction")
    input_col = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    if predict_button:
        input_df = build_input_row(input_col, selected_features, cluster_model)
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted House Price: ${prediction:,.2f}")

    st.markdown("---")
    st.subheader("Model Accuracy")
    rmse, mae, r2, _ = compute_metrics(model, selected_features, df)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:,.2f}")
    c2.metric("MAE", f"{mae:,.2f}")
    c3.metric("R2", f"{r2:.4f}", f"{r2*100:.2f}%")

    st.caption("Metrics computed on a held-out test split (20%).")


if __name__ == "__main__":
    main()
