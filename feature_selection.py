import json
import os

import pandas as pd


def main():
    input_path = os.path.join("artifacts", "housing_cleaned.csv")
    output_path = os.path.join("artifacts", "selected_features.json")

    df = pd.read_csv(input_path)

    # Use all model-ready features for best performance.
    final_features = [col for col in df.columns if col != "median_house_value"]

    # save
    with open(output_path, "w") as f:
        json.dump(final_features, f, indent=2)

    print("Saved selected features to:", output_path)
    print("Total features used:", len(final_features))


if __name__ == "__main__":
    main()
