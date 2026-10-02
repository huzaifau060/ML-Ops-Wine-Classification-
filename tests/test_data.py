"""Unit tests for data pipeline and data validation."""

import pytest
import pandas as pd
import numpy as np
from src.data import load_data, validate_data


def test_load_data_shapes():
    """Verify that load_data returns an 80/20 train-test split."""
    X_train, X_test, y_train, y_test = load_data(test_size=0.2, random_state=42)
    total_samples = len(X_train) + len(X_test)

    assert total_samples == 178, f"Expected 178 total samples, got {total_samples}"
    assert len(X_train) == 142, f"Expected 142 train samples, got {len(X_train)}"
    assert len(X_test) == 36, f"Expected 36 test samples, got {len(X_test)}"
    assert len(y_train) == 142
    assert len(y_test) == 36


def test_feature_count():
    """Verify that feature count is exactly 13."""
    X_train, X_test, _, _ = load_data()
    assert X_train.shape[1] == 13, f"Expected 13 features in train, got {X_train.shape[1]}"
    assert X_test.shape[1] == 13, f"Expected 13 features in test, got {X_test.shape[1]}"


def test_no_null_values():
    """Verify that there are no null or NaN values in train or test splits."""
    X_train, X_test, y_train, y_test = load_data()
    assert not X_train.isnull().any().any(), "X_train contains null values."
    assert not X_test.isnull().any().any(), "X_test contains null values."
    assert not y_train.isnull().any(), "y_train contains null values."
    assert not y_test.isnull().any(), "y_test contains null values."


def test_target_classes():
    """Verify that target contains only classes 0, 1, and 2."""
    _, _, y_train, y_test = load_data()
    expected_classes = {0, 1, 2}
    assert set(np.unique(y_train)) == expected_classes
    assert set(np.unique(y_test)) == expected_classes


def test_validation_rejects_nulls():
    """Verify that validate_data raises ValueError if features or target contain nulls."""
    df_with_null = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    target = pd.Series([0, 1])

    with pytest.raises(ValueError, match="null values"):
        validate_data(df_with_null, target, expected_features=2)

    df_valid = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    target_with_null = pd.Series([0, np.nan])

    with pytest.raises(ValueError, match="null values"):
        validate_data(df_valid, target_with_null, expected_features=2)


def test_validation_rejects_wrong_feature_count():
    """Verify that validate_data raises ValueError when feature count is not 13."""
    df_wrong = pd.DataFrame(np.random.randn(10, 10))
    target = pd.Series([0] * 10)

    with pytest.raises(ValueError, match="Expected 13 features"):
        validate_data(df_wrong, target, expected_features=13)
