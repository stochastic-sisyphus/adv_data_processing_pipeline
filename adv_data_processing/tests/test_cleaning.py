"""Tests for data cleaning functionality."""

import pytest
import dask.dataframe as dd
import pandas as pd
from adv_data_processing.cleaning import (
    validate_data,
    clean_data,
    handle_missing_values,
    handle_outliers,
    remove_duplicates,
    convert_datatypes,
    impute_missing_values
)

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, None, 4, 5],
        'B': [None, 2, 3, 4, 5],
        'C': [1, 2, 3, 4, 5],
        'D': ['a', 'b', 'c', 'd', 'e']
    }
    return dd.from_pandas(pd.DataFrame(data), npartitions=1)

@pytest.fixture
def sample_schema():
    return {
        'A': {'type': 'float64', 'range': (0, 10)},
        'B': {'type': 'float64', 'range': (0, 10)},
        'C': {'type': 'int64', 'range': (0, 10)},
        'D': {'type': 'object'}
    }

def test_validate_data(sample_data, sample_schema):
    assert validate_data(sample_data, sample_schema)

def test_clean_data(sample_data):
    config = {
        'schema': {
            'A': {'type': 'float64', 'range': (0, 10)},
            'B': {'type': 'float64', 'range': (0, 10)},
            'C': {'type': 'int64', 'range': (0, 10)},
            'D': {'type': 'object'}
        },
        'missing_values': {
            'A': {'missing': 'mean'},
            'B': {'missing': 'median'}
        },
        'outlier_columns': ['A', 'B'],
        'dtype_conversions': {'A': 'float64', 'B': 'float64', 'C': 'int64', 'D': 'object'}
    }
    cleaned_df = clean_data(sample_data, config)
    assert not cleaned_df.isnull().any().compute()

def test_handle_missing_values(sample_data):
    strategy = {'missing': 'mean'}
    df = handle_missing_values(sample_data, 'A', strategy)
    assert not df['A'].isnull().any().compute()

def test_handle_outliers(sample_data):
    df = handle_outliers(sample_data, ['A', 'B'])
    assert df['A'].max().compute() <= 5
    assert df['A'].min().compute() >= 1

def test_remove_duplicates(sample_data):
    df = remove_duplicates(sample_data)
    assert df.shape[0].compute() == 5

def test_convert_datatypes(sample_data):
    dtype_dict = {'A': 'float64', 'B': 'float64', 'C': 'int64', 'D': 'object'}
    df = convert_datatypes(sample_data, dtype_dict)
    assert df['A'].dtype == 'float64'
    assert df['B'].dtype == 'float64'
    assert df['C'].dtype == 'int64'
    assert df['D'].dtype == 'object'

def test_impute_missing_values(sample_data):
    strategies = {'A': 'mean', 'B': 'median'}
    df = impute_missing_values(sample_data, strategies)
    assert not df['A'].isnull().any().compute()
    assert not df['B'].isnull().any().compute()

def test_edge_case_empty_dataframe():
    empty_df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(ValueError):
        clean_data(empty_df, {})

def test_edge_case_invalid_schema(sample_data):
    invalid_schema = {'A': {'type': 'str'}}
    with pytest.raises(ValueError):
        validate_data(sample_data, invalid_schema)
