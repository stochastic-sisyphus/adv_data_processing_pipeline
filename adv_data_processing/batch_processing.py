"""Batch processing utilities for efficient data handling."""

from typing import Iterator, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import psutil
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    prefetch_factor: int = 2

class TabularDataset(Dataset):
    """Dataset wrapper for tabular data."""
    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray] = None) -> None:
        """
        Initialize the dataset with features and optional targets.

        Args:
            features (np.ndarray): Feature data.
            targets (Optional[np.ndarray], optional): Target data. Defaults to None.
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a sample and its target from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Feature and target tensors.
        """
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx], None

def get_optimal_batch_size(data_size: int, sample_size_bytes: int) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        data_size (int): Total number of samples.
        sample_size_bytes (int): Size of a single sample in bytes.

    Returns:
        int: Optimal batch size.
    """
    available_memory = psutil.virtual_memory().available
    target_memory_usage = available_memory * 0.7  # Use 70% of available memory
    
    batch_size = int(target_memory_usage / (sample_size_bytes * 2))  # Factor of 2 for safety
    logger.info(f"Calculated optimal batch size: {batch_size}")
    return min(batch_size, data_size)

def get_memory_efficiency_score(batch_size: int, features_shape: Tuple[int, ...]) -> float:
    """
    Calculate memory efficiency score for given batch size.

    Args:
        batch_size (int): Batch size.
        features_shape (Tuple[int, ...]): Shape of the feature data.

    Returns:
        float: Memory efficiency score.
    """
    sample_size = np.prod(features_shape) * 4  # 4 bytes per float32
    batch_memory = sample_size * batch_size
    available_memory = psutil.virtual_memory().available
    score = 1 - (batch_memory / available_memory)
    logger.info(f"Memory efficiency score for batch size {batch_size}: {score}")
    return score

def optimize_batch_configuration(
    data_shape: Tuple[int, ...],
    target_memory_usage: float = 0.7,
    max_workers: int = 8
) -> BatchConfig:
    """
    Optimize batch processing configuration based on system resources.

    Args:
        data_shape (Tuple[int, ...]): Shape of the data.
        target_memory_usage (float, optional): Target memory usage as a fraction. Defaults to 0.7.
        max_workers (int, optional): Maximum number of workers. Defaults to 8.

    Returns:
        BatchConfig: Optimized batch configuration.
    """
    cpu_count = psutil.cpu_count(logical=False)
    num_workers = min(cpu_count - 1, max_workers)
    
    optimal_batch_size = get_optimal_batch_size(
        data_shape[0],
        np.prod(data_shape[1:]) * 4
    )
    
    config = BatchConfig(
        batch_size=optimal_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2
    )
    logger.info(f"Optimized batch configuration: {config}")
    return config

def create_data_loader(
    features: np.ndarray,
    targets: Optional[np.ndarray] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader with optimal batch size.

    Args:
        features (np.ndarray): Feature data.
        targets (Optional[np.ndarray], optional): Target data. Defaults to None.
        batch_size (Optional[int], optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    if batch_size is None:
        sample_size = features[0].nbytes
        batch_size = get_optimal_batch_size(len(features), sample_size)
    
    dataset = TabularDataset(features, targets)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )
    logger.info(f"Created DataLoader with batch size {batch_size} and {num_workers} workers")
    return data_loader
