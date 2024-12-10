"""Tests for imbalanced data handling functionality."""

import pytest
import dask.dataframe as dd
import pandas as pd
from adv_data_processing.imbalanced_data import handle_imbalanced_data

@pytest.fixture
def sample_data():
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    return dd.from_pandas(pd.DataFrame(data), npartitions=1)

def test_handle_imbalanced_data(sample_data):
    balanced_df = handle_imbalanced_data(sample_data, 'target')
    assert balanced_df is not None
    assert balanced_df.shape[0].compute() > sample_data.shape[0].compute()

def test_edge_case_empty_dataframe():
    empty_df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(ValueError):
        handle_imbalanced_data(empty_df, 'target')

def test_edge_case_invalid_method(sample_data):
    with pytest.raises(ValueError):
        handle_imbalanced_data(sample_data, 'target', method='invalid')
