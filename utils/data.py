"""
Data loading utilities for netflow datasets.

Supports both the legacy prediction CSV format and new netflow feature CSVs.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


class NetflowDataset(Dataset):
    """
    PyTorch Dataset for netflow classification.

    Args:
        features: Feature matrix (num_samples, num_features)
        labels: Class labels (num_samples,)
        transform: Optional feature transformation
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        transform=None,
    ):
        if isinstance(features, np.ndarray):
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels.long()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, label


def load_netflow_data(
    data_path: str,
    target_column: str = "label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    normalize: bool = True,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler | None]:
    """
    Load netflow data from CSV and create train/val/test dataloaders.

    Args:
        data_path: Path to CSV file with netflow features
        target_column: Name of the target label column
        test_size: Fraction of data for test set
        val_size: Fraction of training data for validation
        normalize: Whether to standardize features
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Load data
    df = pd.read_csv(data_path)

    # Separate features and labels
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Train/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )

    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Create datasets
    train_dataset = NetflowDataset(X_train, y_train)
    val_dataset = NetflowDataset(X_val, y_val)
    test_dataset = NetflowDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def load_split_netflow_data(
    train_path: str,
    val_path: str,
    test_path: str,
    target_column: str = "label",
    normalize: bool = True,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler | None]:
    """
    Load pre-split netflow data from separate CSV files.

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        target_column: Name of the target label column
        normalize: Whether to standardize features
        batch_size: Batch size for dataloaders

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Separate features and labels
    X_train = train_df.drop(columns=[target_column]).values
    y_train = train_df[target_column].values

    X_val = val_df.drop(columns=[target_column]).values
    y_val = val_df[target_column].values

    X_test = test_df.drop(columns=[target_column]).values
    y_test = test_df[target_column].values

    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Create datasets and loaders
    train_loader = DataLoader(
        NetflowDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(NetflowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NetflowDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


# Legacy function for backwards compatibility
def load_datasets(l1_path, l2_path):
    """
    LEGACY: Load L1/L2 prediction datasets (deprecated).

    This function is kept for backwards compatibility with old experiments.
    New code should use load_netflow_data() instead.
    """
    l1 = pd.read_csv(l1_path)
    l2 = pd.read_csv(l2_path)
    l1['binary'] = l1['prediction'].apply(lambda prob: int(prob > 0.5))
    l2['binary'] = l2['prediction'].apply(lambda prob: int(prob > 0.5))
    if len(l1) != len(l2):
        raise ValueError("Dataset size mismatch between L1 and L2")
    return l1, l2
