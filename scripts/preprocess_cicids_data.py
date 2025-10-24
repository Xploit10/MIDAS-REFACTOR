"""
Preprocess CICIDS2017 dataset for MIDAS training.

This script:
1. Loads all CSV files
2. Cleans data (handles missing/infinite values)
3. Optionally creates binary classification (BENIGN vs ATTACK)
4. Creates stratified subset for faster training
5. Splits into train/val/test
6. Saves processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse


def clean_label(label: str) -> str:
    """Clean label string (fix encoding issues)."""
    # Remove leading/trailing whitespace
    label = label.strip()

    # Fix encoding issues (� character)
    label = label.replace('�', '-')

    return label


def load_and_combine_data(data_dir: Path, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load and combine all CSV files.

    Args:
        data_dir: Directory containing CSV files
        sample_frac: Fraction of each file to load (for memory efficiency)

    Returns:
        Combined DataFrame
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    dfs = []
    for csv_file in csv_files:
        print(f"\nLoading: {csv_file.name}")

        # Load with sampling if requested
        if sample_frac < 1.0:
            # Read first few rows to get total
            df_temp = pd.read_csv(csv_file, nrows=1000)
            total_rows = sum(1 for _ in open(csv_file)) - 1
            skip_rows = sorted(np.random.choice(
                range(1, total_rows + 1),
                size=int(total_rows * (1 - sample_frac)),
                replace=False
            ))
            df = pd.read_csv(csv_file, skiprows=skip_rows)
        else:
            df = pd.read_csv(csv_file)

        print(f"  Loaded: {len(df):,} rows")
        dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df):,} rows")

    return combined_df


def clean_data(df: pd.DataFrame, min_class_samples: int = 100) -> pd.DataFrame:
    """
    Clean the dataset.

    Args:
        df: Input DataFrame
        min_class_samples: Minimum samples per class (remove rare classes)

    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("CLEANING DATA")
    print("="*80)

    # Get label column (last column, usually has leading space)
    label_col = df.columns[-1]
    print(f"Label column: '{label_col}'")

    # Clean labels
    print("\nCleaning labels...")
    df[label_col] = df[label_col].apply(clean_label)

    # Check class distribution before filtering
    print("\nClass distribution before filtering:")
    class_counts = df[label_col].value_counts()
    for label, count in class_counts.items():
        print(f"  {label:40s}: {count:8,}")

    # Filter out rare classes
    print(f"\nFiltering classes with < {min_class_samples} samples...")
    rare_classes = class_counts[class_counts < min_class_samples].index.tolist()
    if rare_classes:
        print(f"Removing {len(rare_classes)} rare classes:")
        for cls in rare_classes:
            count = class_counts[cls]
            print(f"  - {cls} ({count} samples)")
        df = df[~df[label_col].isin(rare_classes)].copy()
        print(f"Remaining samples: {len(df):,}")

    # Get feature columns (all except label)
    feature_cols = df.columns[:-1]

    # Handle missing values
    print("\nHandling missing values...")
    missing_before = df[feature_cols].isnull().sum().sum()
    if missing_before > 0:
        print(f"  Total missing values: {missing_before:,}")
        # Fill with column median
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        print(f"  Filled with column medians")
    else:
        print(f"  No missing values found")

    # Handle infinite values
    print("\nHandling infinite values...")
    inf_count = 0
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32]:
            inf_mask = np.isinf(df[col])
            if inf_mask.any():
                inf_count += inf_mask.sum()
                # Replace inf with column max (excluding inf)
                finite_values = df.loc[~inf_mask, col]
                if len(finite_values) > 0:
                    max_val = finite_values.max()
                    df.loc[inf_mask, col] = max_val

    if inf_count > 0:
        print(f"  Replaced {inf_count:,} infinite values with column max")
    else:
        print(f"  No infinite values found")

    # Remove non-numeric features (if any, except label)
    non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"\nRemoving {len(non_numeric_cols)} non-numeric feature columns:")
        for col in non_numeric_cols:
            print(f"  - {col}")
        df = df.drop(columns=non_numeric_cols)

    # Final class distribution
    print("\nFinal class distribution:")
    class_counts = df[label_col].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label:40s}: {count:8,} ({percentage:5.2f}%)")

    return df


def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert to binary classification: BENIGN (0) vs ATTACK (1).

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with binary labels
    """
    label_col = df.columns[-1]

    print("\n" + "="*80)
    print("CREATING BINARY LABELS")
    print("="*80)

    # Create binary label: 0 = BENIGN, 1 = ATTACK
    df['label_binary'] = (df[label_col] != 'BENIGN').astype(int)

    # Show distribution
    print("\nBinary distribution:")
    binary_counts = df['label_binary'].value_counts().sort_index()
    for label, count in binary_counts.items():
        label_name = "BENIGN" if label == 0 else "ATTACK"
        percentage = (count / len(df)) * 100
        print(f"  {label} ({label_name}): {count:,} ({percentage:.2f}%)")

    return df


def create_multiclass_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Create integer labels for multi-class classification.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (DataFrame with encoded labels, label mapping dict)
    """
    label_col = df.columns[-1]

    print("\n" + "="*80)
    print("CREATING MULTI-CLASS LABELS")
    print("="*80)

    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df[label_col])

    # Create label mapping
    label_mapping = {
        int(i): label
        for i, label in enumerate(le.classes_)
    }

    print(f"\nLabel mapping:")
    for idx, label in label_mapping.items():
        count = (df['label_encoded'] == idx).sum()
        print(f"  {idx}: {label:40s} ({count:,} samples)")

    return df, label_mapping


def create_subset(df: pd.DataFrame, subset_fraction: float, random_state: int = 42) -> pd.DataFrame:
    """
    Create stratified subset of data.

    Args:
        df: Input DataFrame
        subset_fraction: Fraction of data to keep (e.g., 0.15 for 15%)
        random_state: Random seed

    Returns:
        Subset DataFrame
    """
    label_col = [col for col in df.columns if 'label' in col.lower()][-1]

    print("\n" + "="*80)
    print(f"CREATING {subset_fraction*100:.0f}% STRATIFIED SUBSET")
    print("="*80)

    print(f"Original size: {len(df):,} samples")

    # Stratified sampling
    _, df_subset = train_test_split(
        df,
        test_size=subset_fraction,
        stratify=df[label_col],
        random_state=random_state
    )

    print(f"Subset size: {len(df_subset):,} samples")

    # Verify class distribution is preserved
    print("\nSubset class distribution:")
    class_counts = df_subset[label_col].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(df_subset)) * 100
        # Handle both int and str labels
        if isinstance(label, int):
            label_str = f"{label}"
        else:
            label_str = str(label)
        print(f"  {label_str:40s}: {count:8,} ({percentage:5.2f}%)")

    return df_subset


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets.

    Args:
        df: Input DataFrame
        test_size: Fraction for test set
        val_size: Fraction for validation set (of remaining data after test)
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    label_col = [col for col in df.columns if 'label' in col.lower()][-1]

    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df[label_col],
        random_state=random_state
    )

    print(f"Train set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val set:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    label_mapping: dict = None,
    classification_type: str = "binary"
):
    """
    Save processed datasets and metadata.

    Args:
        train_df, val_df, test_df: DataFrames to save
        output_dir: Output directory
        label_mapping: Label mapping for multi-class
        classification_type: "binary" or "multiclass"
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)

    # Determine which label column to use
    if classification_type == "binary":
        label_col = "label_binary"
    else:
        label_col = "label_encoded"

    # Get feature columns (numeric only, exclude original label columns)
    all_label_cols = [col for col in train_df.columns if 'label' in col.lower() or col == ' Label']
    feature_cols = [col for col in train_df.columns if col not in all_label_cols]

    # Save with only features + chosen label
    save_cols = feature_cols + [label_col]

    # Rename label column to 'label' for consistency
    train_save = train_df[save_cols].rename(columns={label_col: 'label'})
    val_save = val_df[save_cols].rename(columns={label_col: 'label'})
    test_save = test_df[save_cols].rename(columns={label_col: 'label'})

    # Save CSVs
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    print(f"\nSaving train set to: {train_path}")
    train_save.to_csv(train_path, index=False)

    print(f"Saving val set to: {val_path}")
    val_save.to_csv(val_path, index=False)

    print(f"Saving test set to: {test_path}")
    test_save.to_csv(test_path, index=False)

    # Save metadata
    metadata = {
        "classification_type": classification_type,
        "num_features": len(feature_cols),
        "num_classes": len(train_save['label'].unique()),
        "train_samples": len(train_save),
        "val_samples": len(val_save),
        "test_samples": len(test_save),
        "feature_columns": feature_cols,
    }

    if label_mapping:
        metadata["label_mapping"] = label_mapping

    metadata_path = output_dir / "metadata.json"
    print(f"\nSaving metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nProcessed data saved to: {output_dir}")
    print(f"  - train.csv: {len(train_save):,} samples")
    print(f"  - val.csv: {len(val_save):,} samples")
    print(f"  - test.csv: {len(test_save):,} samples")
    print(f"  - metadata.json")
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Classes: {len(train_save['label'].unique())}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CICIDS2017 dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="MachineLearningCVE",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--classification",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
        help="Binary (BENIGN vs ATTACK) or multi-class classification"
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=0.15,
        help="Fraction of data to use (e.g., 0.15 for 15%%)"
    )
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=100,
        help="Minimum samples per class (remove rarer classes)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction for test set"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction for validation set"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.random_seed)

    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print("="*80)
    print("CICIDS2017 DATA PREPROCESSING")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Classification type: {args.classification}")
    print(f"Subset fraction: {args.subset_fraction*100:.0f}%")
    print(f"Min samples per class: {args.min_class_samples}")
    print(f"Random seed: {args.random_seed}")

    # Step 1: Load and combine data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    df = load_and_combine_data(data_dir, sample_frac=1.0)

    # Step 2: Clean data
    df_clean = clean_data(df, min_class_samples=args.min_class_samples)

    # Step 3: Create labels
    label_mapping = None
    if args.classification == "binary":
        df_clean = create_binary_labels(df_clean)
    else:
        df_clean, label_mapping = create_multiclass_labels(df_clean)

    # Step 4: Create subset
    if args.subset_fraction < 1.0:
        df_subset = create_subset(df_clean, args.subset_fraction, args.random_seed)
    else:
        df_subset = df_clean

    # Step 5: Split data
    train_df, val_df, test_df = split_data(
        df_subset,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_seed
    )

    # Step 6: Save processed data
    save_processed_data(
        train_df,
        val_df,
        test_df,
        output_dir,
        label_mapping=label_mapping,
        classification_type=args.classification
    )


if __name__ == "__main__":
    main()
