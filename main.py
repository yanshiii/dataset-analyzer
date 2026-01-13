import pandas as pd

from dataset_analyzer.summary import compute_dataset_summary
from dataset_analyzer.inference import infer_column_properties


def main():
    df = pd.read_csv("examples/sample.csv")

    summary = compute_dataset_summary(df)
    columns = infer_column_properties(df)

    print("DATASET SUMMARY")
    print(summary)

    print("\nCOLUMN ANALYSIS")
    for col, info in columns.items():
        print(f"{col}: {info}")


if __name__ == "__main__":
    main()
