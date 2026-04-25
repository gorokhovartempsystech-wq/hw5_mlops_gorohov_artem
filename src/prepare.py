"""
Готовит данные: читает сырой csv, делит на train и test,
сохраняет в data/processed.
"""
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["prepare"]

    raw_path = "data/raw/iris.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = df.dropna()

    train_df, test_df = train_test_split(
        df,
        test_size=params["split_ratio"],
        random_state=params["random_state"],
        stratify=df["variety"],
    )

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print(f"train: {len(train_df)} строк, test: {len(test_df)} строк")


if __name__ == "__main__":
    main()
