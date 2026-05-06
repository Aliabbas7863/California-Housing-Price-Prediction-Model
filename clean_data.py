import os
import pickle

import pandas as pd
from sklearn.cluster import KMeans


def main():
    data_path = "housing.csv"
    output_dir = "artifacts"
    output_path = os.path.join(output_dir, "housing_cleaned.csv")
    cluster_model_path = os.path.join(output_dir, "location_cluster_model.pkl")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)

    # Fill missing values
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    # Feature engineering
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    # Location clustering (simple spatial feature)
    try:
        coords = df[["latitude", "longitude"]].copy()
        kmeans = KMeans(n_clusters=10, random_state=42)
        df["loc_cluster"] = kmeans.fit_predict(coords).astype(str)
        with open(cluster_model_path, "wb") as f:
            pickle.dump(kmeans, f)
    except Exception:
        df["loc_cluster"] = "0"

    # One-hot encode the categorical columns
    df = pd.get_dummies(df, columns=["ocean_proximity", "loc_cluster"], drop_first=False)

    df.to_csv(output_path, index=False)

    print("Cleaning completed")
    print("Saved file:", output_path)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])


if __name__ == "__main__":
    main()
