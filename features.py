"""
features.py — Preprocessing pipeline: encode, scale, split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import config


def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, scale numericals, return train/test arrays.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = df.drop(columns=["customer_id"]).copy()

    # Encode categorical columns
    le = LabelEncoder()
    for col in config.CATEGORICAL_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"Features: {X.shape[1]}")
    return X_train, X_test, y_train, y_test, list(X.columns)
