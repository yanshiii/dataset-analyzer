import pandas as pd
import numpy as np

from .constants import (
    NEAR_CONSTANT_THRESHOLD,
    ID_UNIQUENESS_RATIO,
    HIGH_CARDINALITY_LIMIT
)


def infer_column_properties(df: pd.DataFrame) -> dict:
    """
    Infer column properties using deterministic, rule-based logic.
    """
    results = {}

    for col in df.columns:
        series = df[col]
        total_rows = len(series)
        non_missing = series.dropna()
        non_missing_count = len(non_missing)

        # Missingness
        missing_percentage = (
            0.0 if total_rows == 0
            else (1 - non_missing_count / total_rows) * 100
        )

        unique_values = int(non_missing.nunique())

        # Base structure
        info = {
            "original_dtype": str(series.dtype),
            "missing_percentage": round(missing_percentage, 2),
            "unique_values": unique_values,
            "is_constant": False,
            "is_near_constant": False,
            "is_id_like": False,
            "inferred_type": None,
            "cardinality_level": None
        }

        # -----------------------------
        # Rule 0: All-missing
        # -----------------------------
        if non_missing_count == 0:
            info["inferred_type"] = "unknown"
            results[col] = info
            continue

        # -----------------------------
        # Rule 1: Constant
        # -----------------------------
        if unique_values == 1:
            info["is_constant"] = True
            info["inferred_type"] = "constant"
            results[col] = info
            continue

        # -----------------------------
        # Rule 2: Near-constant
        # -----------------------------
        dominant_pct = (
            non_missing.value_counts(normalize=True).iloc[0] * 100
        )

        if dominant_pct >= NEAR_CONSTANT_THRESHOLD:
            info["is_near_constant"] = True

        # -----------------------------
        # Rule 3: ID-like
        # -----------------------------
        uniqueness_ratio = unique_values / non_missing_count
        is_integer_like = pd.api.types.is_integer_dtype(series)

        is_monotonic = series.is_monotonic_increasing or series.is_monotonic_decreasing

        if (
            uniqueness_ratio >= ID_UNIQUENESS_RATIO
            and unique_values > 10
            and is_integer_like
            and is_monotonic
        ):
            info["is_id_like"] = True
            info["inferred_type"] = "id_like"
            results[col] = info
            continue

        # -----------------------------
        # Rule 4: Datetime
        # -----------------------------
        if pd.api.types.is_datetime64_any_dtype(series):
            info["inferred_type"] = "datetime"
            results[col] = info
            continue

        # Only attempt parsing for object/string columns
        if series.dtype == object:
            try:
                parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
                parsed_ratio = parsed.notna().sum() / non_missing_count

                if parsed_ratio >= 0.8:
                    info["inferred_type"] = "datetime"
                    results[col] = info
                    continue
            except Exception:
                pass

        # -----------------------------
        # Rule 5: Numerical
        # -----------------------------
        if pd.api.types.is_numeric_dtype(series):
            info["inferred_type"] = "numerical"

            # Discrete vs continuous (annotation only)
            is_integer_like = pd.api.types.is_integer_dtype(series)
            unique_values = info["unique_values"]

            if is_integer_like and unique_values <= 20:
                info["numerical_kind"] = "discrete"
            else:
                info["numerical_kind"] = "continuous"

            results[col] = info
            continue

        # -----------------------------
        # Rule 6: Categorical (fallback)
        # -----------------------------
        info["inferred_type"] = "categorical"

        if unique_values <= HIGH_CARDINALITY_LIMIT:
            info["cardinality_level"] = "low"
        else:
            info["cardinality_level"] = "high"

        results[col] = info

    return results
