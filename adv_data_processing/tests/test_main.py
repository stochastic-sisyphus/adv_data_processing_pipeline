"""Tests for main pipeline functionality."""

import pytest
import dask.dataframe as dd
import pandas as pd
from adv_data_processing.main import (
    setup_dask_client,
    initialize_pipeline_state,
    handle_text_analytics,
    handle_entity_extraction,
    handle_model_training,
    perform_auto_tuning,
    process_pipeline_step,
    main
)
from dask.distributed import Client

@pytest.fixture
def sample_data():
    data = {
        'text': ['This is a test.', 'Another test sentence.'],
        'target': [1, 0]
    }
    return dd.from_pandas(pd.DataFrame(data), npartitions=1)

@pytest.fixture
def sample_config():
    return {
        'text_column': 'text',
        'target_column': 'target',
        'model_type': 'random_forest',
        'evaluation_metrics': ['accuracy'],
        'param_grid': {'n_estimators': [10, 50]}
    }

def test_setup_dask_client():
    client = setup_dask_client(argparse.Namespace(n_workers=2, scheduler_address=None))
    assert isinstance(client, Client)
    client.close()

def test_initialize_pipeline_state(sample_data):
    state = initialize_pipeline_state(argparse.Namespace(resume=None))
    assert state == (None, [])

def test_handle_text_analytics(sample_data, sample_config):
    results = handle_text_analytics(sample_data, sample_config, argparse.Namespace(analyze_text=True))
    assert 'sentiment' in results
    assert 'summary' in results

def test_handle_entity_extraction(sample_data, sample_config):
    results = handle_entity_extraction(sample_data, sample_config, argparse.Namespace(extract_entities=True))
    assert 'entities' in results

def test_handle_model_training(sample_data, sample_config):
    model = handle_model_training(sample_data, sample_config, argparse.Namespace(auto_tune=False))
    assert model is not None

def test_perform_auto_tuning(sample_data, sample_config):
    model = perform_auto_tuning(None, sample_data.drop(columns=['target']), sample_data['target'], sample_config)
    assert model is not None

def test_process_pipeline_step(sample_data, sample_config):
    processed_data = process_pipeline_step('load', sample_data, sample_config, argparse.Namespace(), {})
    assert processed_data is not None

def test_main_function():
    args = argparse.Namespace(config='config.yaml', output=None, visualize=False, analyze_text=False,
                              extract_entities=False, model_topics=False, resume=None, select_features=False,
                              reduce_dimensions=False, validate_schema=False, summary_stats=False,
                              save_intermediate=False, intermediate_path='./intermediate/', n_workers=4,
                              scheduler_address=None, plugins=None, generate_report=False, use_cache=False,
                              auto_feature_engineering=False, handle_imbalanced=False, auto_tune=False,
                              memory_limit=None)
    main(args)
