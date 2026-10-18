"""Leak-free preprocessing: fitted on training data only, inside Pipelines."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import BINARY, CATEGORICAL, FEATURES, NUMERIC, TARGET_CLS, TARGET_REG, TEST_SIZE
from src.validation import assert_no_leakage


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics, one-hot encode season, pass binary through."""
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


def split_data(df: pd.DataFrame, seed: int):
    """Single fixed split (random_state=LLLL), stratified on the risk label."""
    assert_no_leakage(FEATURES)
    X = df[FEATURES]
    return train_test_split(X, df[TARGET_REG], df[TARGET_CLS], test_size=TEST_SIZE,
                            random_state=seed, stratify=df[TARGET_CLS])
