"""Identity, seed, schema, range, duplicate and leakage checks."""
import re
import pandas as pd
from src.config import FORBIDDEN_FEATURES, RANGES, REQUIRED_COLUMNS, TARGET_CLS, TARGET_REG


class DataValidationError(ValueError):
    """Raised when identity or data checks fail."""


def parse_identity(student_id: str, seed: int) -> dict:
    """Derive REGDIGITS, LLLL and the verification code; reject a wrong seed."""
    username = student_id.split("@")[0]
    digits = "".join(re.findall(r"\d", username))
    if len(digits) < 4:
        raise DataValidationError(f"Student id '{student_id}' needs at least 4 digits.")
    last4 = digits[-4:]
    if int(seed) != int(last4):
        raise DataValidationError(
            f"Seed {seed} is invalid: it must equal the last four registration digits ({int(last4)}).")
    return {"student_id": student_id, "regdigits": digits, "seed": int(last4),
            "verification_code": f"AI-A2-{last4}"}


def validate_raw(df: pd.DataFrame) -> dict:
    """Check schema/ranges and return a data-quality report (missing + duplicates)."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    problems = []
    for col, (lo, hi) in RANGES.items():
        s = df[col].dropna()
        bad = int(((s < lo) | (s > hi)).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    if not set(df["season"].dropna().unique()) <= {"A", "B"}:
        problems.append("season must be A or B")
    if not set(df["cooperative_support"].dropna().unique()) <= {0, 1}:
        problems.append("cooperative_support must be 0 or 1")
    if df[[TARGET_REG, TARGET_CLS]].isna().any().any():
        problems.append("targets contain missing values")
    if (df[TARGET_REG] < 0.5).any():
        problems.append("yield_tons below the 0.50 clip")
    if problems:
        raise DataValidationError("; ".join(problems))
    return {"rows_raw": int(len(df)),
            "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n},
            "duplicate_rows": int(df.duplicated().sum())}


def assert_no_leakage(feature_columns) -> None:
    """Fail if any forbidden target-derived column is used as a feature."""
    leaked = FORBIDDEN_FEATURES & set(feature_columns)
    if leaked:
        raise DataValidationError(f"Target leakage: {sorted(leaked)} used as features.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates (keeps first). Missing values are imputed later, inside the Pipeline."""
    return df.drop_duplicates(keep="first")
