
"""Tests for batch processing functionality."""

import pytest
import numpy as np
from adv_data_processing.batch_processing import (
    get_optimal_batch_size,
    create_data_loader
)

@pytest.fixture
def sample_data():
    return np.random.randn(1000, 32)

def test_optimal_batch_size(sample_data):
    batch_size = get_optimal_batch_size(
        len(sample_data),
        sample_data[0].nbytes
    )
    assert isinstance(batch_size, int)
    assert batch_size > 0
    assert batch_size <= len(sample_data)

def test_data_loader_creation(sample_data):
    loader = create_data_loader(sample_data)
    assert len(loader) > 0
    
    for batch in loader:
        assert isinstance(batch, tuple)
        assert batch[0].shape[0] <= get_optimal_batch_size(
            len(sample_data),
            sample_data[0].nbytes
        )
        break