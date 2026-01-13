from typing import List, Dict


def generate_recommendations(
    issues: List[Dict]
) -> List[Dict]:
    """
    Generate conservative, evidence-backed recommendations
    based strictly on detected data quality issues.
    """
    recommendations = []

    for issue in issues:
        issue_type = issue["issue_type"]
        column = issue["column"]

        # -----------------------------
        # Small dataset
        # -----------------------------
        if issue_type == "small_dataset":
            recommendations.append({
                "related_issue": issue_type,
                "column": None,
                "statement": (
                    "The dataset contains a relatively small number of rows, "
                    "which may limit the reliability of patterns learned during modeling."
                ),
                "severity": "medium"
            })

        # -----------------------------
        # All-missing column
        # -----------------------------
        elif issue_type == "all_missing":
            recommendations.append({
                "related_issue": issue_type,
                "column": column,
                "statement": (
                    "This column contains no observed values, which means it cannot "
                    "contribute information to a model in its current form."
                ),
                "severity": "high"
            })

        # -----------------------------
        # Constant / near-constant
        # -----------------------------
        elif issue_type in {"constant_column", "near_constant_column"}:
            recommendations.append({
                "related_issue": issue_type,
                "column": column,
                "statement": (
                    "This column shows little to no variation, which may limit its "
                    "usefulness for learning meaningful patterns."
                ),
                "severity": "low"
            })

        # -----------------------------
        # High missingness
        # -----------------------------
        elif issue_type == "high_missingness":
            recommendations.append({
                "related_issue": issue_type,
                "column": column,
                "statement": (
                    "A large proportion of values are missing in this column, which "
                    "may affect how reliably it can be used during modeling."
                ),
                "severity": "medium"
            })

        # -----------------------------
        # ID-like column
        # -----------------------------
        elif issue_type == "id_like_column":
            recommendations.append({
                "related_issue": issue_type,
                "column": column,
                "statement": (
                    "This column behaves like an identifier and may pose a risk of "
                    "information leakage if used as a feature."
                ),
                "severity": "high"
            })

        # -----------------------------
        # High-cardinality categorical
        # -----------------------------
        elif issue_type == "high_cardinality_categorical":
            recommendations.append({
                "related_issue": issue_type,
                "column": column,
                "statement": (
                    "This categorical column has a large number of unique values, "
                    "which may require careful handling during modeling."
                ),
                "severity": "medium"
            })

    return recommendations
