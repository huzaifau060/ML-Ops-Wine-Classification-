"""Automated MLOps Quality Gate test suite enforcing model criteria."""

import os
import time
import pytest
import numpy as np

from src.data import load_data
from src.train import run_experiments, evaluate_with_cv
from src.evaluate import get_champion_model


@pytest.fixture(scope="session")
def ensure_champion_model():
    """Ensure that the MLflow champion model exists and return it."""
    db_path = os.path.abspath("mlflow.db")
    if not os.path.exists(db_path):
        print("\n[Quality Gate Fixture] No existing MLflow DB found. Training models...")
        run_experiments()
    try:
        model = get_champion_model()
    except Exception:
        print("\n[Quality Gate Fixture] Champion model not registered. Training models...")
        run_experiments()
        model = get_champion_model()
    return model


def test_metric_threshold_gate(ensure_champion_model):
    """
    Quality Gate 1: Metric Threshold Gate.

    Validation Macro F1-score must be >= 0.88.
    """
    X_train, _, y_train, _ = load_data()
    model = ensure_champion_model
    model_class = model.__class__
    params = model.get_params()

    cv_results = evaluate_with_cv(model_class, params, X_train, y_train)
    val_macro_f1 = cv_results["val_macro_f1"]

    print(f"\n[Quality Gate 1] Validation Macro F1: {val_macro_f1:.4f} (Threshold >= 0.88)")
    assert val_macro_f1 >= 0.88, (
        f"Quality Gate FAILED: Validation Macro F1 is {val_macro_f1:.4f}, "
        f"which is below the mandatory threshold of 0.88."
    )


def test_inference_latency_gate(ensure_champion_model):
    """
    Quality Gate 2: Inference Latency Gate.

    Batch inference time for the test batch (36 samples) must be <= 30 ms.
    """
    _, X_test, _, _ = load_data()
    model = ensure_champion_model

    # Warm-up run to eliminate first-call overhead
    _ = model.predict(X_test)

    # Measure batch latency over multiple iterations for stability
    latencies_ms = []
    for _ in range(20):
        start_time = time.perf_counter()
        _ = model.predict(X_test)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(elapsed_ms)

    avg_latency_ms = float(np.mean(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))

    print(f"\n[Quality Gate 2] Batch Inference Latency: Avg={avg_latency_ms:.2f}ms, "
          f"P95={p95_latency_ms:.2f}ms (Threshold <= 30 ms)")

    assert avg_latency_ms <= 30.0, (
        f"Quality Gate FAILED: Average batch latency {avg_latency_ms:.2f}ms exceeds 30ms limit."
    )


def test_output_schema_integrity_gate(ensure_champion_model):
    """
    Quality Gate 3: Output Schema Integrity Gate.

    Model predictions must strictly contain valid class indices only (0, 1, or 2).
    """
    _, X_test, _, _ = load_data()
    model = ensure_champion_model

    predictions = model.predict(X_test)
    unique_classes = set(np.unique(predictions))
    valid_classes = {0, 1, 2}

    print(f"\n[Quality Gate 3] Predicted classes in test batch: {unique_classes}")

    assert unique_classes.issubset(valid_classes), (
        f"Quality Gate FAILED: Model predicted invalid classes: {unique_classes - valid_classes}"
    )
    assert len(predictions) == len(X_test), "Prediction batch size mismatch."
