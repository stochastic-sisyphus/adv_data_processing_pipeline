"""Tests for batch processing functionality."""

import pytest
import numpy as np
from adv_data_processing.batch_processing import (
    get_optimal_batch_size,
    create_data_loader,
    get_memory_efficiency_score,
    optimize_batch_configuration
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

def test_memory_efficiency_score(sample_data):
    batch_size = get_optimal_batch_size(len(sample_data), sample_data[0].nbytes)
    score = get_memory_efficiency_score(batch_size, sample_data.shape)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_optimize_batch_configuration(sample_data):
    config = optimize_batch_configuration(sample_data.shape)
    assert config.batch_size > 0
    assert config.num_workers > 0
    assert isinstance(config.pin_memory, bool)
    assert config.prefetch_factor == 2

def test_edge_case_empty_data():
    empty_data = np.array([])
    with pytest.raises(ValueError):
        create_data_loader(empty_data)

def test_edge_case_large_batch_size(sample_data):
    large_batch_size = len(sample_data) * 2
    loader = create_data_loader(sample_data, batch_size=large_batch_size)
    assert len(loader) == 1
    for batch in loader:
        assert batch[0].shape[0] == len(sample_data)
