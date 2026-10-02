"""Inference evaluation module for registered champion model on test data."""

import os
import sys
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, log_loss

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.data import load_data  # noqa: E402

MODEL_REGISTRY_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"


def get_champion_model():
    """Load the champion model from MLflow Model Registry."""
    db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    model_uri = f"models:/{MODEL_REGISTRY_NAME}@{CHAMPION_ALIAS}"
    try:
        model = mlflow.sklearn.load_model(model_uri=model_uri)
    except Exception:
        # Fallback to fetching latest version from client if alias resolution requires version ID
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, CHAMPION_ALIAS)
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{model_version.version}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
    return model


def evaluate_champion_on_test() -> Dict[str, float]:
    """Load champion model and compute final metrics on unseen test split."""
    _, X_test, _, y_test = load_data()
    model = get_champion_model()

    predictions = model.predict(X_test)
    pred_probas = model.predict_proba(X_test)
    classes = np.unique(y_test)

    test_f1 = float(f1_score(y_test, predictions, average="macro"))
    test_acc = float(accuracy_score(y_test, predictions))
    test_loss = float(log_loss(y_test, pred_probas, labels=classes))

    results = {
        "test_macro_f1": test_f1,
        "test_accuracy": test_acc,
        "test_log_loss": test_loss,
    }

    print("\n=== Champion Model Evaluation on Test Split ===")
    print(f"Test Samples: {len(y_test)}")
    print(f"Test Macro F1-Score: {test_f1:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"Test Log Loss:       {test_loss:.4f}")
    print("===============================================\n")

    return results


if __name__ == "__main__":
    evaluate_champion_on_test()
