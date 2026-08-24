"""
data.py

Loads the UCI Breast Cancer Wisconsin Diagnostic (WDBC) dataset and produces
a fixed, leakage-free 70/15/15 stratified train/val/test split.

The split and the standardization statistics are computed ONCE and reused
for every optimizer / seed combination, so that all training runs are
compared on identical data.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataBundle:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    feature_names: list
    scaler: StandardScaler


def load_wdbc(
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DataBundle:
    """
    Load WDBC and produce a fixed stratified 70/15/15 split.

    IMPORTANT (data leakage):
    StandardScaler is fit ONLY on the training split. The fitted transform
    is then applied to validation and test data. This mirrors what the plan
    calls out explicitly: standardization must not see val/test statistics.

    random_state is fixed (default 42) so this split is identical across
    every optimizer and every seed in {0,...,4}. Training randomness (seeds)
    is applied only to model init / minibatch order downstream, never to
    this split.
    """
    bunch = load_breast_cancer()
    X = bunch.data  # (569, 30) - identifier column is not present in sklearn's copy
    y = bunch.target  # sklearn encodes malignant=0, benign=1 -- see note below

    # sklearn's load_breast_cancer encodes target as 0=malignant, 1=benign.
    # The plan specifies benign -> 0, malignant -> 1. Flip to match the plan.
    y = 1 - y

    feature_names = list(bunch.feature_names)

    # First split off test (15%), then split remaining into train/val.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # val_size relative to the remaining (1 - test_size) fraction
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=relative_val_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit ONLY on train
    X_val = scaler.transform(X_val)           # transform only
    X_test = scaler.transform(X_test)         # transform only

    to_t = lambda a, dt: torch.tensor(a, dtype=dt)

    return DataBundle(
        X_train=to_t(X_train, torch.float32),
        y_train=to_t(y_train, torch.float32).unsqueeze(1),
        X_val=to_t(X_val, torch.float32),
        y_val=to_t(y_val, torch.float32).unsqueeze(1),
        X_test=to_t(X_test, torch.float32),
        y_test=to_t(y_test, torch.float32).unsqueeze(1),
        feature_names=feature_names,
        scaler=scaler,
    )


if __name__ == "__main__":
    data = load_wdbc()
    print("Train:", data.X_train.shape, "pos rate:", data.y_train.mean().item())
    print("Val:  ", data.X_val.shape, "pos rate:", data.y_val.mean().item())
    print("Test: ", data.X_test.shape, "pos rate:", data.y_test.mean().item())