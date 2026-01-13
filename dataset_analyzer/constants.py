# Thresholds used across the analyzer.
# These are deliberately conservative and explainable.

MISSINGNESS_THRESHOLD = 30.0        # percentage
NEAR_CONSTANT_THRESHOLD = 99.0      # dominant value percentage
HIGH_CARDINALITY_LIMIT = 50         # unique values
ID_UNIQUENESS_RATIO = 0.95          # unique / non-missing ratio
SMALL_DATASET_ROWS = 100            # informational flag only
