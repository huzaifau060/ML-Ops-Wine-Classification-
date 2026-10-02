"""Model training, hyperparameter tuning, and MLflow registry promotion module."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.data import load_data  # noqa: E402

# Experiment configurations
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_REGISTRY_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
RANDOM_STATE = 42

# Define candidate hyperparameter search grids (3 configurations per model family = 6 total)
SEARCH_GRID: List[Dict[str, Any]] = [
    # Model Family A: RandomForestClassifier
    {
        "family": "RandomForest",
        "model_class": RandomForestClassifier,
        "config_name": "rf_config_1",
        "params": {
            "n_estimators": 50,
            "max_depth": 3,
            "min_samples_split": 2,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "family": "RandomForest",
        "model_class": RandomForestClassifier,
        "config_name": "rf_config_2",
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "family": "RandomForest",
        "model_class": RandomForestClassifier,
        "config_name": "rf_config_3",
        "params": {
            "n_estimators": 150,
            "max_depth": None,
            "min_samples_split": 4,
            "random_state": RANDOM_STATE,
        },
    },
    # Model Family B: GradientBoostingClassifier
    {
        "family": "GradientBoosting",
        "model_class": GradientBoostingClassifier,
        "config_name": "gbm_config_1",
        "params": {
            "n_estimators": 50,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "family": "GradientBoosting",
        "model_class": GradientBoostingClassifier,
        "config_name": "gbm_config_2",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "family": "GradientBoosting",
        "model_class": GradientBoostingClassifier,
        "config_name": "gbm_config_3",
        "params": {
            "n_estimators": 150,
            "learning_rate": 0.2,
            "max_depth": 4,
            "random_state": RANDOM_STATE,
        },
    },
]


def evaluate_with_cv(
    model_class, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, float]:
    """Perform 5-fold stratified cross-validation and compute train/val metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    classes = np.unique(y_train)

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = model_class(**params)
        fold_model.fit(X_tr, y_tr)

        # Predictions
        y_tr_pred = fold_model.predict(X_tr)
        y_val_pred = fold_model.predict(X_val)
        y_tr_proba = fold_model.predict_proba(X_tr)
        y_val_proba = fold_model.predict_proba(X_val)

        # Metrics
        train_f1s.append(f1_score(y_tr, y_tr_pred, average="macro"))
        val_f1s.append(f1_score(y_val, y_val_pred, average="macro"))
        train_accs.append(accuracy_score(y_tr, y_tr_pred))
        val_accs.append(accuracy_score(y_val, y_val_pred))
        train_losses.append(log_loss(y_tr, y_tr_proba, labels=classes))
        val_losses.append(log_loss(y_val, y_val_proba, labels=classes))

    return {
        "train_macro_f1": float(np.mean(train_f1s)),
        "val_macro_f1": float(np.mean(val_f1s)),
        "train_accuracy": float(np.mean(train_accs)),
        "val_accuracy": float(np.mean(val_accs)),
        "train_log_loss": float(np.mean(train_losses)),
        "val_log_loss": float(np.mean(val_losses)),
    }


def run_experiments() -> str:
    """Run all candidate experiments, log to MLflow, and promote champion model."""
    # Ensure tracking URI is set (SQLite backend supports Model Registry)
    db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    X_train, X_test, y_train, y_test = load_data(random_state=RANDOM_STATE)
    input_example = X_train.iloc[:5]

    run_records = []

    print(f"Starting experiments for {len(SEARCH_GRID)} configurations...")

    for config in SEARCH_GRID:
        family = config["family"]
        name = config["config_name"]
        params = config["params"]
        model_cls = config["model_class"]

        print(f"Evaluating {family} - {name}...")

        # 5-fold cross-validation
        cv_metrics = evaluate_with_cv(model_cls, params, X_train, y_train)

        # Fit model on entire training split for artifact serialization
        full_model = model_cls(**params)
        full_model.fit(X_train, y_train)

        signature = infer_signature(X_train, full_model.predict(X_train))

        with mlflow.start_run(run_name=f"{family}_{name}") as run:
            # Log params
            mlflow.log_params(params)
            mlflow.log_param("model_family", family)
            mlflow.log_param("cv_folds", 5)

            # Log metrics
            mlflow.log_metrics(cv_metrics)

            # Log tags
            mlflow.set_tag("model_family", family)
            mlflow.set_tag("candidate_name", name)
            mlflow.set_tag("dataset", "load_wine")

            # Log model artifact with signature and input example
            mlflow.sklearn.log_model(
                sk_model=full_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )

            run_records.append({
                "run_id": run.info.run_id,
                "config_name": name,
                "family": family,
                "val_macro_f1": cv_metrics["val_macro_f1"],
                "val_accuracy": cv_metrics["val_accuracy"],
                "val_log_loss": cv_metrics["val_log_loss"],
                "params": params,
            })

    # Champion Selection
    best_run = max(run_records, key=lambda r: r["val_macro_f1"])
    print("\n--- Hyperparameter Tuning Results ---")
    for r in run_records:
        print(
            f"Config: {r['config_name']:<12} | Family: {r['family']:<16} | "
            f"Val F1: {r['val_macro_f1']:.4f} | Val Acc: {r['val_accuracy']:.4f} | "
            f"Val Loss: {r['val_log_loss']:.4f}"
        )

    print(f"\nWinning Champion Model: {best_run['config_name']} ({best_run['family']}) "
          f"with Validation Macro F1 = {best_run['val_macro_f1']:.4f}")

    # Register Model in MLflow Model Registry
    model_uri = f"runs:/{best_run['run_id']}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

    # Assign alias 'champion'
    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias=CHAMPION_ALIAS,
        version=model_version.version,
    )
    print(f"Registered version {model_version.version} of '{MODEL_REGISTRY_NAME}' "
          f"with alias '{CHAMPION_ALIAS}'.")

    return best_run["run_id"]


if __name__ == "__main__":
    run_experiments()
