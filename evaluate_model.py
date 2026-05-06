import json
import os
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    input_path = os.path.join("artifacts", "housing_cleaned.csv")
    features_path = os.path.join("artifacts", "selected_features.json")
    model_path = os.path.join("artifacts", "housing_model.pkl")

    df = pd.read_csv(input_path)

    with open(features_path, "r") as f:
        selected_features = json.load(f)

    x = df[selected_features]
    y = df["median_house_value"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(x_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model evaluation")
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))
    print("R2:", round(r2, 4))


if __name__ == "__main__":
    main()
