"""Split the train_set in to a train and validation set, save their indices"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from omegaconf import DictConfig # loading config yaml
import os

def split_train_val(
        config:DictConfig = None,
        labels_path: str = "data/train_labels.csv",
        val_fraction: float = 0.1, # TODO reference yaml
        seed: int = 42
        ) -> tuple[list, list]:
    """Split the dataset into training and validation sets (lists of indices) based on the provided fraction."""

    if config:
        labels_path = config.data.labels_path
        val_fraction = config.data.val_fraction
        seed = config.seed

    # Load the labels
    labels_df = pd.read_csv(labels_path)

    # Get the total number of samples
    num_samples = len(labels_df)

    # shuffle the indices
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    # Determine number of validation samples
    num_val_samples = int(val_fraction * num_samples)

    # Split the indices into training and validation sets
    val_indices = indices[:num_val_samples].tolist()
    train_indices = indices[num_val_samples:].tolist()

    # return the indices as lists
    return train_indices, val_indices

def split_kfold(
        labels_path: str = "data/train_labels.csv",
        k: int = 5,
        config: DictConfig = None,
        shuffle: bool = True,
        seed: int = 42,
        ) -> list[tuple[list, list]]:
    """Return a list of k (train_indices, val_indices) tuples for k-fold cross-validation.
    if k=1, this will just return one split based on the val_fraction, as in split_train_val."""

    if config:
        labels_path = config.data.labels_path
        k           = config.data.k
        shuffle     = config.data.shuffle
    
    if k == 1:
        # Just do a single split based on val_fraction
        return [split_train_val(config=config)]
 
    labels_df   = pd.read_csv(labels_path)
    num_samples = len(labels_df)
    all_indices = np.arange(num_samples)
 
    kfold  = KFold(n_splits=k, shuffle=shuffle, random_state=seed if shuffle else None)
    splits = [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in kfold.split(all_indices)
    ]
 
    return splits