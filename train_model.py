import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    input_path = os.path.join("artifacts", "housing_cleaned.csv")
    features_path = os.path.join("artifacts", "selected_features.json")
    model_path = os.path.join("artifacts", "housing_model.pkl")

    df = pd.read_csv(input_path)

    # load selected features; if missing, use all features except target
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            selected_features = json.load(f)
    else:
        selected_features = [c for c in df.columns if c != "median_house_value"]

    x = df[selected_features]
    y = df["median_house_value"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # stronger gradient boosting model tuned for higher accuracy on all features
    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=12,
        learning_rate=0.04,
        max_iter=1000,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(x_train, y_train)

    # save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # metrics
    preds = model.predict(x_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Model training completed")
    print("Saved model to:", model_path)
    print("Training rows:", x_train.shape[0])
    print("Test rows:", x_test.shape[0])
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    main()
