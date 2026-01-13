from typing import List, Dict

from .constants import (
    MISSINGNESS_THRESHOLD,
    HIGH_CARDINALITY_LIMIT,
    SMALL_DATASET_ROWS
)


def detect_data_quality_issues(
    column_analysis: Dict[str, dict],
    dataset_summary: Dict[str, int]
) -> List[dict]:
    """
    Detect explicit data quality issues based on column properties
    and dataset-level statistics.

    Returns a list of issue dictionaries.
    """
    issues = []

    # -----------------------------
    # Dataset-level issue
    # -----------------------------
    if dataset_summary["num_rows"] < SMALL_DATASET_ROWS:
        issues.append({
            "issue_type": "small_dataset",
            "column": None,
            "evidence": {
                "num_rows": dataset_summary["num_rows"],
                "threshold": SMALL_DATASET_ROWS
            }
        })

    # -----------------------------
    # Column-level issues
    # -----------------------------
    for column, info in column_analysis.items():

        # All-missing
        if info["missing_percentage"] == 100.0:
            issues.append({
                "issue_type": "all_missing",
                "column": column,
                "evidence": {
                    "missing_percentage": 100.0
                }
            })
            continue  # no further checks make sense

        # Constant
        if info["is_constant"]:
            issues.append({
                "issue_type": "constant_column",
                "column": column,
                "evidence": {
                    "unique_values": info["unique_values"]
                }
            })

        # Near-constant
        if info["is_near_constant"]:
            issues.append({
                "issue_type": "near_constant_column",
                "column": column,
                "evidence": {
                    "missing_percentage": info["missing_percentage"]
                }
            })

        # High missingness
        if info["missing_percentage"] >= MISSINGNESS_THRESHOLD:
            issues.append({
                "issue_type": "high_missingness",
                "column": column,
                "evidence": {
                    "missing_percentage": info["missing_percentage"],
                    "threshold": MISSINGNESS_THRESHOLD
                }
            })

        # ID-like
        if info["is_id_like"]:
            issues.append({
                "issue_type": "id_like_column",
                "column": column,
                "evidence": {
                    "unique_values": info["unique_values"]
                }
            })

        # High-cardinality categorical
        if (
            info["inferred_type"] == "categorical"
            and info["cardinality_level"] == "high"
        ):
            issues.append({
                "issue_type": "high_cardinality_categorical",
                "column": column,
                "evidence": {
                    "unique_values": info["unique_values"],
                    "threshold": HIGH_CARDINALITY_LIMIT
                }
            })

    return issues
