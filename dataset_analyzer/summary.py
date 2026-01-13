import pandas as pd


def compute_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Compute basic, high-level dataset facts.
    No interpretation, no assumptions.
    """
    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum())
    }
