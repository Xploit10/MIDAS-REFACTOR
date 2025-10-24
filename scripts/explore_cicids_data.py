"""
Explore CICIDS2017 dataset to understand structure, class distribution, and data quality.

This script analyzes all CSV files in the MachineLearningCVE directory and provides
comprehensive statistics about the dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict


def explore_csv_file(file_path: Path) -> dict:
    """
    Explore a single CSV file and return statistics.

    Args:
        file_path: Path to CSV file

    Returns:
        Dictionary with file statistics
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path.name}")
    print(f"{'='*80}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Basic info
    stats = {
        "file_name": file_path.name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
    }

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Label column (last column)
    label_col = df.columns[-1]
    print(f"\nLabel column: '{label_col}'")

    # Class distribution
    label_counts = df[label_col].value_counts()
    print(f"\nClass distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label:40s}: {count:8,} ({percentage:5.2f}%)")

    stats["label_column"] = label_col
    stats["class_distribution"] = label_counts.to_dict()
    stats["num_classes"] = len(label_counts)

    # Feature statistics
    feature_cols = df.columns[:-1]  # All except label
    print(f"\nFeature statistics:")
    print(f"  Number of features: {len(feature_cols)}")

    # Check for missing values
    missing = df[feature_cols].isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n  Columns with missing values:")
        for col, count in missing_cols.items():
            print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print(f"  No missing values found")

    stats["missing_values"] = missing_cols.to_dict() if len(missing_cols) > 0 else {}

    # Check for infinite values
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = int(inf_count)

    if inf_counts:
        print(f"\n  Columns with infinite values:")
        for col, count in inf_counts.items():
            print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print(f"  No infinite values found")

    stats["infinite_values"] = inf_counts

    # Data types
    non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"\n  Non-numeric feature columns:")
        for col in non_numeric_cols:
            print(f"    {col}: {df[col].dtype}")

    stats["non_numeric_features"] = list(non_numeric_cols)

    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nMemory usage: {memory_mb:.2f} MB")
    stats["memory_mb"] = memory_mb

    return stats


def main():
    """Main exploration function."""
    # Path to data directory
    data_dir = Path("/Users/kieranrendall/git/MIDAS/MachineLearningCVE")

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    # Find all CSV files
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Analyze each file
    all_stats = []
    total_samples = 0
    all_classes = defaultdict(int)

    for csv_file in csv_files:
        try:
            stats = explore_csv_file(csv_file)
            all_stats.append(stats)
            total_samples += stats["total_rows"]

            # Aggregate class counts
            for label, count in stats["class_distribution"].items():
                all_classes[label] += count

        except Exception as e:
            print(f"\nError processing {csv_file.name}: {e}")
            continue

    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {len(all_stats)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Total unique classes: {len(all_classes)}")

    print(f"\nCombined class distribution:")
    sorted_classes = sorted(all_classes.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_classes:
        percentage = (count / total_samples) * 100
        print(f"  {label:40s}: {count:8,} ({percentage:5.2f}%)")

    # Calculate class imbalance ratio
    class_counts = [count for _, count in sorted_classes]
    imbalance_ratio = max(class_counts) / min(class_counts)
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1 (max:min)")

    # Recommendations for subset size
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}")

    # Suggest subset sizes
    for percentage in [10, 15, 20]:
        subset_size = int(total_samples * (percentage / 100))
        print(f"\n{percentage}% subset: {subset_size:,} samples")
        print(f"  Smallest class would have: ~{int(min(class_counts) * percentage / 100):,} samples")

    # Check for potential issues
    print(f"\nPotential issues to address:")
    issues = []

    # Check for missing values
    total_missing = sum(len(s.get("missing_values", {})) for s in all_stats)
    if total_missing > 0:
        issues.append(f"- {total_missing} columns have missing values")

    # Check for infinite values
    total_inf = sum(len(s.get("infinite_values", {})) for s in all_stats)
    if total_inf > 0:
        issues.append(f"- {total_inf} columns have infinite values")

    # Check for non-numeric features
    non_numeric = set()
    for s in all_stats:
        non_numeric.update(s.get("non_numeric_features", []))
    if non_numeric:
        issues.append(f"- {len(non_numeric)} non-numeric feature columns")

    # Check for class imbalance
    if imbalance_ratio > 10:
        issues.append(f"- Severe class imbalance (ratio: {imbalance_ratio:.0f}:1)")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("- No major issues detected!")

    # Save summary to JSON
    output_file = data_dir.parent / "data_exploration_summary.json"
    summary = {
        "total_files": len(all_stats),
        "total_samples": total_samples,
        "file_stats": all_stats,
        "combined_class_distribution": dict(sorted_classes),
        "class_imbalance_ratio": imbalance_ratio,
        "recommendations": {
            "subset_10pct": int(total_samples * 0.10),
            "subset_15pct": int(total_samples * 0.15),
            "subset_20pct": int(total_samples * 0.20),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
