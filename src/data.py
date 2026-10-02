"""Data pipeline and validation module for the Wine Cultivar Classification system."""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def validate_data(features: pd.DataFrame, target: pd.Series, expected_features: int = 13) -> None:
    """
    Validate data integrity.

    Checks that:
    1. Features DataFrame has no missing/null values.
    2. Target Series has no missing/null values.
    3. Features count exactly matches expected_features (13).
    """
    if features.isnull().any().any():
        raise ValueError("Data validation failed: Features contain null values.")
    if target.isnull().any():
        raise ValueError("Data validation failed: Target contains null values.")
    if features.shape[1] != expected_features:
        raise ValueError(
            f"Data validation failed: Expected {expected_features} features, "
            f"got {features.shape[1]}."
        )


def load_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load Wine dataset from scikit-learn, validate integrity, and perform stratified 80/20 split.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training targets.
        y_test (pd.Series): Testing targets.
    """
    wine = load_wine(as_frame=True)
    X: pd.DataFrame = wine.data
    y: pd.Series = wine.target

    validate_data(X, y, expected_features=13)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("Data successfully loaded and validated.")
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Class distribution in train:\n{y_train.value_counts()}")
