# Dataset Analyzer — Early ML Readiness Check

## What this tool solves

This project provides an **early-stage, read-only diagnostic** for tabular datasets to answer one focused question:

> **“Is this dataset safe and reasonable to start machine learning with?”**

Instead of preprocessing or modeling, the analyzer surfaces **structural risks** in a dataset that commonly cause downstream ML failures, such as data leakage, extreme missingness, high-cardinality features, or unreliable targets.

The goal is **clarity before complexity**.

---

## What this tool explicitly does NOT do

To preserve trust and interpretability, this tool deliberately avoids:

* Data preprocessing or cleaning
* Feature engineering
* Model training or evaluation
* Automatic fixes or transformations
* Domain-specific assumptions

All outputs are **diagnostic only**.
The tool highlights risks; it does not prescribe actions.

---

## Heuristics used (transparent by design)

All checks are **rule-based, conservative, and explainable**.
Thresholds are defaults meant to prompt review, **not strict requirements for ML**.

### Dataset-level checks

* **Small dataset**: fewer than 100 rows (informational)
* **High duplicate rows**: ≥ 20% exact duplicates

### Column-level checks

* **Missingness**: ≥ 30% missing values
* **Constant / near-constant features**
* **ID-like columns**: highly unique, monotonic integer patterns
* **High-cardinality categorical features**
* **Discrete vs continuous numerical annotation**

  * Discrete: integer-valued with low unique counts (e.g., counts, encoded categories)
  * Continuous: real-valued measurements

### Target checks (if a target is provided)

* Classification vs regression detection
* Constant or invalid targets
* Class distribution and imbalance ratio (reported, not judged)

All heuristics are **read-only** and **evidence-backed**.

---

## Datasets tested (real-world validation)

This analyzer has been validated on multiple Kaggle datasets with very different characteristics:

* **Titanic**
  Mixed numerical/categorical data, missing values, ID leakage risk, binary classification.

* **Heart Disease**
  Encoded medical features, discrete vs continuous numerical distinction, extreme duplicate rows.

* **Netflix Movies & TV Shows**
  Text-heavy, high-cardinality metadata, no supervised target (unsupervised / exploratory use case).

Testing on diverse datasets ensures the tool behaves conservatively across scenarios.

---

## Example output (Titanic dataset)

```text
DATASET SUMMARY
{'num_rows': 891, 'num_columns': 12, 'duplicate_rows': 0}

DATA QUALITY ISSUES
- ID-like column detected: PassengerId
- High missingness: Cabin (77%)
- High-cardinality categoricals: Name, Ticket

TARGET ANALYSIS
- Problem type: classification
- Class distribution: 61.6% / 38.4%
- No strong class imbalance detected

RECOMMENDATIONS
- Identifier columns may cause information leakage if used as features.
- High-cardinality categorical features may require careful handling during modeling.
- High missingness may affect feature reliability.
```

The tool intentionally avoids telling the user *what to do*.
It focuses on **what to be aware of** before modeling.

---

## Design philosophy

* Conservative over clever
* Explainable over automated
* Evidence before recommendations
* Clear enough for interns, precise enough for senior analysts

If a sentence cannot be justified in a technical review, the tool does not say it.

---

## Status

* **Current version**: v1.1
* **Focus**: Dataset readiness diagnostics
* **Planned direction**: Deeper data-quality signals and improved reporting (no modeling)

---

## Usage (basic)

```bash
python main.py examples/datasets/Titanic.csv
```

Optionally specify a target column inside `main.py` for supervised analysis.

---

## License

MIT
