import pandas as pd
from typing import Optional, Dict

from .constants import MISSINGNESS_THRESHOLD


def analyze_target(
    df: pd.DataFrame,
    target_column: Optional[str]
) -> Optional[Dict]:
    """
    Analyze the target column, if provided.

    Returns a dictionary describing the target,
    or None if no target is specified.
    """
    if target_column is None:
        return None

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    series = df[target_column]
    total_rows = len(series)
    non_missing = series.dropna()
    non_missing_count = len(non_missing)

    missing_percentage = (
        0.0 if total_rows == 0
        else (1 - non_missing_count / total_rows) * 100
    )

    unique_values = non_missing.nunique()

    target_info = {
        "target_column": target_column,
        "missing_percentage": round(missing_percentage, 2),
        "unique_values": int(unique_values),
        "problem_type": None,
        "class_distribution": None,
        "is_imbalanced": None,
        "imbalance_ratio": None
    }

    # -----------------------------
    # Degenerate target checks
    # -----------------------------
    if non_missing_count == 0:
        target_info["problem_type"] = "invalid"
        return target_info

    if unique_values == 1:
        target_info["problem_type"] = "constant"
        return target_info

    # -----------------------------
    # Problem type detection
    # -----------------------------
    if pd.api.types.is_numeric_dtype(non_missing) and unique_values > 10:
        # Heuristic: many unique numeric values → regression
        target_info["problem_type"] = "regression"
        return target_info

    # Otherwise treat as classification
    target_info["problem_type"] = "classification"

    # -----------------------------
    # Classification analysis
    # -----------------------------
    value_counts = non_missing.value_counts(normalize=True) * 100
    class_distribution = {
        str(k): round(v, 2) for k, v in value_counts.items()
    }

    majority_pct = value_counts.iloc[0]
    minority_pct = value_counts.iloc[-1]

    imbalance_ratio = (
        round(majority_pct / minority_pct, 2)
        if minority_pct > 0 else None
    )

    target_info["class_distribution"] = class_distribution
    target_info["imbalance_ratio"] = (
        float(imbalance_ratio) if imbalance_ratio is not None else None
    )
    target_info["is_imbalanced"] = bool(
        imbalance_ratio is not None and imbalance_ratio >= 5
    )

    return target_info
