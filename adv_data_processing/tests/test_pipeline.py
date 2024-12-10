"""Tests for pipeline functionality."""

import pytest
import dask.dataframe as dd
import pandas as pd
from adv_data_processing.pipeline import process_data

@pytest.fixture
def sample_data():
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    }
    return dd.from_pandas(pd.DataFrame(data), npartitions=1)

@pytest.fixture
def sample_config():
    return {
        'source': 'sample.csv',
        'steps': ['load', 'clean', 'transform'],
        'cleaning_strategies': {
            'feature1': {'missing': 'mean'},
            'feature2': {'missing': 'median'}
        },
        'numeric_features': ['feature1', 'feature2'],
        'categorical_features': [],
        'scale_strategy': 'standard',
        'encode_strategy': 'onehot'
    }

def test_pipeline_execution(sample_data, sample_config):
    processed_data = process_data(
        source=sample_config['source'],
        steps=sample_config['steps'],
        cleaning_strategies=sample_config['cleaning_strategies'],
        numeric_features=sample_config['numeric_features'],
        categorical_features=sample_config['categorical_features'],
        scale_strategy=sample_config['scale_strategy'],
        encode_strategy=sample_config['encode_strategy']
    )
    assert processed_data is not None
    assert isinstance(processed_data, dd.DataFrame)

def test_pipeline_edge_case_empty_data():
    empty_data = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(ValueError):
        process_data(
            source='empty.csv',
            steps=['load', 'clean', 'transform'],
            cleaning_strategies={},
            numeric_features=[],
            categorical_features=[],
            scale_strategy='standard',
            encode_strategy='onehot'
        )

def test_pipeline_integration(sample_data, sample_config):
    processed_data = process_data(
        source=sample_config['source'],
        steps=sample_config['steps'],
        cleaning_strategies=sample_config['cleaning_strategies'],
        numeric_features=sample_config['numeric_features'],
        categorical_features=sample_config['categorical_features'],
        scale_strategy=sample_config['scale_strategy'],
        encode_strategy=sample_config['encode_strategy']
    )
    assert processed_data is not None
    assert isinstance(processed_data, dd.DataFrame)
    assert 'feature1' in processed_data.columns
    assert 'feature2' in processed_data.columns
    assert 'target' in processed_data.columns
