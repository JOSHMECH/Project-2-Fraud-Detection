"""
preprocessing.py
------------------
Data loading, inspection, and leak-free train/test splitting utilities
for the fraud detection pipeline.

Critically: this module NEVER applies SMOTE or scaling to the full
dataset before splitting -- both are only ever fit on the training fold,
inside the imblearn Pipeline, to avoid data leakage.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load the raw credit-card transactions dataset."""
    return pd.read_csv(path)


def inspect_target(df: pd.DataFrame, target_col: str = "Class") -> dict:
    """
    Summarize the class balance of the target variable.
    Returns counts and proportions for each class.
    """
    counts = df[target_col].value_counts().to_dict()
    proportions = df[target_col].value_counts(normalize=True).to_dict()
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "proportions_pct": {int(k): round(float(v) * 100, 4) for k, v in proportions.items()},
        "total_rows": int(len(df)),
    }


def check_missing_and_duplicates(df: pd.DataFrame) -> dict:
    """Basic data-quality check before modeling."""
    return {
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def stratified_split(df: pd.DataFrame, target_col: str = "Class",
                      test_size: float = 0.2, random_state: int = 42):
    """
    Perform a stratified train/test split BEFORE any resampling or
    scaling, so the test set reflects the real-world class imbalance.

    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
