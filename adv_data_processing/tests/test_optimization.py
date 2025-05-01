"""Tests for optimization functionality."""

import pytest
import pandas as pd
from adv_data_processing.optimization import (
    optimize_dataframe_dtypes,
    get_memory_usage
)

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [1.1, 2.2, 3.3, 4.4, 5.5],
        'C': ['a', 'b', 'c', 'd', 'e']
    }
    return pd.DataFrame(data)

def test_optimize_dataframe_dtypes(sample_data):
    optimized_df = optimize_dataframe_dtypes(sample_data)
    assert optimized_df['A'].dtype == 'int8'
    assert optimized_df['B'].dtype == 'float32'
    assert optimized_df['C'].dtype.name == 'category'

def test_get_memory_usage(sample_data):
    memory_stats = get_memory_usage(sample_data)
    assert 'total_memory_mb' in memory_stats
    assert 'memory_per_column' in memory_stats
    assert memory_stats['total_memory_mb'] > 0
    assert all(mem > 0 for mem in memory_stats['memory_per_column'].values())

def test_edge_case_empty_dataframe():
    empty_df = pd.DataFrame()
    optimized_df = optimize_dataframe_dtypes(empty_df)
    assert optimized_df.empty

def test_edge_case_single_column():
    single_col_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    optimized_df = optimize_dataframe_dtypes(single_col_df)
    assert optimized_df['A'].dtype == 'int8'

def test_integration_optimization_and_memory_usage(sample_data):
    optimized_df = optimize_dataframe_dtypes(sample_data)
    memory_stats = get_memory_usage(optimized_df)
    assert 'total_memory_mb' in memory_stats
    assert 'memory_per_column' in memory_stats
    assert memory_stats['total_memory_mb'] > 0
    assert all(mem > 0 for mem in memory_stats['memory_per_column'].values())
