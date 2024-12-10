"""Tests for feature engineering functionality."""

import pytest
import dask.dataframe as dd
import pandas as pd
from adv_data_processing.feature_engineering import (
    auto_feature_engineering,
    create_polynomial_features,
    create_interaction_features
)

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    }
    return dd.from_pandas(pd.DataFrame(data), npartitions=1)

def test_auto_feature_engineering(sample_data):
    config = {
        'create_polynomial_features': True,
        'polynomial_degree': 2,
        'create_interaction_features': True
    }
    df = auto_feature_engineering(sample_data, 'target', config)
    assert 'A' in df.columns
    assert 'B' in df.columns
    assert 'A^2' in df.columns
    assert 'interaction_A_B' in df.columns

def test_create_polynomial_features(sample_data):
    df = create_polynomial_features(sample_data, degree=2)
    assert 'A' in df.columns
    assert 'B' in df.columns
    assert 'A^2' in df.columns

def test_create_interaction_features(sample_data):
    df = create_interaction_features(sample_data)
    assert 'A' in df.columns
    assert 'B' in df.columns
    assert 'interaction_A_B' in df.columns

def test_edge_case_empty_dataframe():
    empty_df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(ValueError):
        auto_feature_engineering(empty_df, 'target', {})

def test_edge_case_invalid_target_column(sample_data):
    with pytest.raises(ValueError):
        auto_feature_engineering(sample_data, 'invalid_target', {})
