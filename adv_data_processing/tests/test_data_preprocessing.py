"""Tests for data preprocessing functionality."""

import pytest
import pandas as pd
from adv_data_processing.data_preprocessing import (
    handle_missing_values,
    encode_categorical_variables,
    scale_numerical_features,
    preprocess_dataset
)

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, None, 4, 5],
        'B': [None, 2, 3, 4, 5],
        'C': [1, 2, 3, 4, 5],
        'D': ['a', 'b', 'c', 'd', 'e']
    }
    return pd.DataFrame(data)

def test_handle_missing_values(sample_data):
    df = handle_missing_values(sample_data)
    assert not df.isnull().any().any()

def test_encode_categorical_variables(sample_data):
    df = encode_categorical_variables(sample_data)
    assert 'D_a' in df.columns
    assert 'D_b' in df.columns

def test_scale_numerical_features(sample_data):
    df = scale_numerical_features(sample_data)
    assert df['A'].mean() < 1e-6
    assert df['A'].std() - 1 < 1e-6

def test_preprocess_dataset(sample_data):
    df = preprocess_dataset(sample_data)
    assert not df.isnull().any().any()
    assert 'D_a' in df.columns
    assert df['A'].mean() < 1e-6
    assert df['A'].std() - 1 < 1e-6

def test_edge_case_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        preprocess_dataset(empty_df)

def test_edge_case_invalid_encoding_type(sample_data):
    with pytest.raises(ValueError):
        encode_categorical_variables(sample_data, encoding_type='invalid')

def test_edge_case_invalid_scaling_type(sample_data):
    with pytest.raises(ValueError):
        scale_numerical_features(sample_data, scaling_type='invalid')
