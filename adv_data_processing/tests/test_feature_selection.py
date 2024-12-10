"""Tests for feature selection functionality."""

import pytest
import dask.dataframe as dd
import pandas as pd
from adv_data_processing.feature_selection import select_features

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10],
        'C': [11, 12, 13, 14, 15],
        'target': [0, 1, 0, 1, 0]
    }
    return dd.from_pandas(pd.DataFrame(data), npartitions=1)

def test_select_features_mutual_info(sample_data):
    df = select_features(sample_data, target_col='target', n_features=2, method='mutual_info')
    assert df.shape[1] == 2

def test_select_features_f_classif(sample_data):
    df = select_features(sample_data, target_col='target', n_features=2, method='f_classif')
    assert df.shape[1] == 2

def test_select_features_invalid_method(sample_data):
    with pytest.raises(ValueError):
        select_features(sample_data, target_col='target', n_features=2, method='invalid')

def test_edge_case_empty_dataframe():
    empty_df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(ValueError):
        select_features(empty_df, target_col='target', n_features=2, method='mutual_info')

def test_edge_case_single_column():
    single_col_df = dd.from_pandas(pd.DataFrame({'target': [0, 1, 0, 1, 0]}), npartitions=1)
    with pytest.raises(ValueError):
        select_features(single_col_df, target_col='target', n_features=2, method='mutual_info')
