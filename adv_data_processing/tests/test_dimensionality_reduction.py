"""Tests for dimensionality reduction functionality."""

import pytest
import dask.dataframe as dd
import pandas as pd
from adv_data_processing.dimensionality_reduction import reduce_dimensions

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10],
        'C': [11, 12, 13, 14, 15]
    }
    return dd.from_pandas(pd.DataFrame(data), npartitions=1)

def test_reduce_dimensions_pca(sample_data):
    df = reduce_dimensions(sample_data, method='pca', n_components=2)
    assert df.shape[1] == 2

def test_reduce_dimensions_tsne(sample_data):
    df = reduce_dimensions(sample_data, method='tsne', n_components=2)
    assert df.shape[1] == 2

def test_reduce_dimensions_invalid_method(sample_data):
    with pytest.raises(ValueError):
        reduce_dimensions(sample_data, method='invalid', n_components=2)

def test_edge_case_empty_dataframe():
    empty_df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(ValueError):
        reduce_dimensions(empty_df, method='pca', n_components=2)

def test_edge_case_single_column():
    single_col_df = dd.from_pandas(pd.DataFrame({'A': [1, 2, 3, 4, 5]}), npartitions=1)
    with pytest.raises(ValueError):
        reduce_dimensions(single_col_df, method='pca', n_components=2)
