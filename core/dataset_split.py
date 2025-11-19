"""
Dataset splitting utilities for train/val split
"""

import random
import numpy as np
from pathlib import Path


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Split dataset into train and validation sets
    
    Args:
        dataset: StereoDataset instance
        train_ratio: ratio of training samples
        seed: random seed
    
    Returns:
        train_dataset, val_dataset
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_samples = len(dataset.image_list)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    train_size = int(total_samples * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train dataset
    train_dataset = type(dataset).__new__(type(dataset))
    train_dataset.__dict__.update(dataset.__dict__)
    train_dataset.image_list = [dataset.image_list[i] for i in train_indices]
    train_dataset.disparity_list = [dataset.disparity_list[i] for i in train_indices]
    
    # Create val dataset
    val_dataset = type(dataset).__new__(type(dataset))
    val_dataset.__dict__.update(dataset.__dict__)
    val_dataset.image_list = [dataset.image_list[i] for i in val_indices]
    val_dataset.disparity_list = [dataset.disparity_list[i] for i in val_indices]
    
    return train_dataset, val_dataset


def sample_vkitti2(dataset, train_size=1000, val_size=200, seed=42):
    """
    Sample specific number of images from vKITTI2
    
    Args:
        dataset: vKITTI2 dataset
        train_size: number of training samples
        val_size: number of validation samples
        seed: random seed
    
    Returns:
        train_dataset, val_dataset
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_samples = len(dataset.image_list)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    # Create train dataset
    train_dataset = type(dataset).__new__(type(dataset))
    train_dataset.__dict__.update(dataset.__dict__)
    train_dataset.image_list = [dataset.image_list[i] for i in train_indices]
    train_dataset.disparity_list = [dataset.disparity_list[i] for i in train_indices]
    
    # Create val dataset
    val_dataset = type(dataset).__new__(type(dataset))
    val_dataset.__dict__.update(dataset.__dict__)
    val_dataset.image_list = [dataset.image_list[i] for i in val_indices]
    val_dataset.disparity_list = [dataset.disparity_list[i] for i in val_indices]
    
    return train_dataset, val_dataset
