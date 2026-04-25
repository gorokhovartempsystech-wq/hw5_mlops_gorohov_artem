"""
Обучает RandomForest на подготовленных данных,
логирует параметры, метрики и артефакты в MLflow,
сохраняет модель в model.pkl.
"""
import os
import json
import pickle
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    feature_cols = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
    target_col = "variety"

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hw5_iris")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("model", "RandomForestClassifier")

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # confusion matrix как график-артефакт
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title("Confusion matrix")
        fig.tight_layout()
        fig.savefig("confusion_matrix.png", dpi=120)
        plt.close(fig)
        mlflow.log_artifact("confusion_matrix.png")

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        metrics = {"accuracy": acc, "f1_macro": f1}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact("model.pkl")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        print(f"accuracy={acc:.4f}, f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
