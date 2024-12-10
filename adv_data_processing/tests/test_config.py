import pytest
from pathlib import Path
import yaml
from adv_data_processing.config import PipelineConfig, validate_config

@pytest.fixture
def sample_config():
    return {
        'data_source': {'type': 'csv', 'path': 'data/input.csv'},
        'preprocessing': {'handle_missing': True, 'scale': True},
        'feature_engineering': {'polynomial_features': True},
        'model': {'type': 'random_forest', 'params': {'n_estimators': 100}},
        'evaluation': {'metrics': ['accuracy', 'f1']},
        'output': {'path': 'results/'}
    }

def test_config_loading(tmp_path, sample_config):
    config_path = tmp_path / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    config = PipelineConfig.from_yaml(str(config_path))
    assert config.data_source == sample_config['data_source']
    assert config.preprocessing == sample_config['preprocessing']

def test_config_validation(sample_config):
    config = PipelineConfig(**sample_config)
    assert validate_config(config)

def test_invalid_config():
    invalid_config = {'data_source': {}}  # Missing required sections
    with pytest.raises(ValueError):
        PipelineConfig(**invalid_config)

def test_edge_case_empty_config():
    empty_config = {}
    with pytest.raises(TypeError):
        PipelineConfig(**empty_config)

def test_edge_case_missing_sections():
    missing_sections_config = {
        'data_source': {'type': 'csv', 'path': 'data/input.csv'}
    }
    with pytest.raises(ValueError):
        validate_config(PipelineConfig(**missing_sections_config))

def test_integration_config_loading_and_validation(tmp_path, sample_config):
    config_path = tmp_path / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    config = PipelineConfig.from_yaml(str(config_path))
    assert validate_config(config)
