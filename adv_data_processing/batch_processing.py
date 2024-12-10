"""Batch processing utilities for efficient data handling."""

from typing import Iterator, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import math
import psutil
from dataclasses import dataclass

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    prefetch_factor: int = 2

class TabularDataset(Dataset):
    """Dataset wrapper for tabular data."""
    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx], None

def get_optimal_batch_size(data_size: int, sample_size_bytes: int) -> int:
    """Calculate optimal batch size based on available memory."""
    available_memory = psutil.virtual_memory().available
    target_memory_usage = available_memory * 0.7  # Use 70% of available memory
    
    batch_size = int(target_memory_usage / (sample_size_bytes * 2))  # Factor of 2 for safety
    return min(batch_size, data_size)

def get_memory_efficiency_score(batch_size: int, features_shape: Tuple[int, ...]) -> float:
    """Calculate memory efficiency score for given batch size."""
    sample_size = np.prod(features_shape) * 4  # 4 bytes per float32
    batch_memory = sample_size * batch_size
    available_memory = psutil.virtual_memory().available
    return 1 - (batch_memory / available_memory)

def optimize_batch_configuration(
    data_shape: Tuple[int, ...],
    target_memory_usage: float = 0.7,
    max_workers: int = 8
) -> BatchConfig:
    """Optimize batch processing configuration based on system resources."""
    cpu_count = psutil.cpu_count(logical=False)
    num_workers = min(cpu_count - 1, max_workers)
    
    optimal_batch_size = get_optimal_batch_size(
        data_shape[0],
        np.prod(data_shape[1:]) * 4
    )
    
    return BatchConfig(
        batch_size=optimal_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2
    )

def create_data_loader(
    features: np.ndarray,
    targets: Optional[np.ndarray] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader with optimal batch size."""
    if batch_size is None:
        sample_size = features[0].nbytes
        batch_size = get_optimal_batch_size(len(features), sample_size)
    
    dataset = TabularDataset(features, targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )