"""Tests for data transformation functionality."""

import pytest
import dask.dataframe as dd
import pandas as pd
from adv_data_processing.data_transformation import (
    transform_data,
    transform_numerical_features,
    transform_categorical_features,
    transform_text_features
)

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': ['text1', 'text2', 'text3', 'text4', 'text5']
    }
    return dd.from_pandas(pd.DataFrame(data), npartitions=1)

def test_transform_numerical_features(sample_data):
    df = transform_numerical_features(sample_data, ['A'])
    assert 'A' in df.columns
    assert df['A'].mean().compute() < 1e-6
    assert df['A'].std().compute() - 1 < 1e-6

def test_transform_categorical_features(sample_data):
    df = transform_categorical_features(sample_data, ['B'])
    assert 'B_a' in df.columns
    assert 'B_b' in df.columns

def test_transform_text_features(sample_data):
    df = transform_text_features(sample_data, ['C'])
    assert 'C_text1' in df.columns
    assert 'C_text2' in df.columns

def test_transform_data(sample_data):
    config = {
        'numerical_features': ['A'],
        'categorical_features': ['B'],
        'text_features': ['C'],
        'scaling_method': 'standard',
        'encoding_method': 'onehot',
        'text_vectorization_method': 'tfidf'
    }
    df = transform_data(sample_data, config)
    assert 'A' in df.columns
    assert 'B_a' in df.columns
    assert 'C_text1' in df.columns

def test_edge_case_empty_dataframe():
    empty_df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(ValueError):
        transform_data(empty_df, {})

def test_edge_case_invalid_scaling_method(sample_data):
    with pytest.raises(ValueError):
        transform_numerical_features(sample_data, ['A'], method='invalid')

def test_edge_case_invalid_encoding_method(sample_data):
    with pytest.raises(ValueError):
        transform_categorical_features(sample_data, ['B'], method='invalid')

def test_edge_case_invalid_text_vectorization_method(sample_data):
    with pytest.raises(ValueError):
        transform_text_features(sample_data, ['C'], method='invalid')
