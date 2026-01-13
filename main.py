import pandas as pd

from dataset_analyzer.summary import compute_dataset_summary
from dataset_analyzer.inference import infer_column_properties
from dataset_analyzer.issues import detect_data_quality_issues
from dataset_analyzer.target import analyze_target
from dataset_analyzer.recommendations import generate_recommendations

def main():
    df = pd.read_csv("examples/sample.csv")

    summary = compute_dataset_summary(df)
    columns = infer_column_properties(df)
    issues = detect_data_quality_issues(columns, summary)

    # CHANGE HERE: specify target column (or None)
    target_analysis = analyze_target(df, target_column="outcome")

    print("DATASET SUMMARY")
    print(summary)

    print("\nCOLUMN ANALYSIS")
    for col, info in columns.items():
        print(f"{col}: {info}")

    print("\nDATA QUALITY ISSUES")
    for issue in issues:
        print(issue)

    print("\nTARGET ANALYSIS")
    print(target_analysis)

    recommendations = generate_recommendations(issues)
    print("\nRECOMMENDATIONS")
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    main()
