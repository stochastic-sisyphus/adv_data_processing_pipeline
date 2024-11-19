import yaml
import json
import joblib
import logging
from typing import Dict, Any, List
import dask.dataframe as dd
from pathlib import Path
from schema import Schema, Optional, And, Or, Use
import pandas as pd

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    try:
        if path.suffix in {'.yaml', '.yml'}:
            with open(path) as f:
                return yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML or JSON file."""
    path = Path(output_path)
    
    try:
        if path.suffix in {'.yaml', '.yml'}:
            with open(path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    except Exception as e:
        logger.error(f"Error saving config file: {str(e)}")
        raise

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration against schema."""
    config_schema = Schema({
        'source': str,
        'output_file': str,
        'steps': [str],
        Optional('cleaning_strategies'): {
            str: {
                'missing': Or('drop', 'mean', 'median', 'mode', 'constant'),
                Optional('constant_value'): Or(int, float, str)
            }
        },
        Optional('numeric_features'): [str],
        Optional('categorical_features'): [str],
        Optional('scale_strategy'): Or('standard', 'minmax', 'robust', None),
        Optional('encode_strategy'): Or('onehot', 'label', 'target', None),
        Optional('n_workers'): And(Use(int), lambda n: n > 0),
        Optional('chunk_size'): And(Use(int), lambda n: n > 0),
        Optional('memory_limit'): And(Use(int), lambda n: n > 0)
    })
    
    try:
        config_schema.validate(config)
        return True
    except Exception as e:
        logger.error(f"Config validation failed: {str(e)}")
        return False

def get_pipeline_steps(config: Dict[str, Any]) -> List[str]:
    """Get ordered list of pipeline steps from config."""
    return config.get('steps', ['load', 'clean', 'transform'])

def log_step(step: str) -> None:
    """Log the current pipeline step."""
    logger.info(f"Executing step: {step}")

def get_model(model_type: str) -> Any:
    """Get model instance based on type."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    models = {
        'random_forest_classifier': RandomForestClassifier,
        'random_forest_regressor': RandomForestRegressor,
        'logistic_regression': LogisticRegression,
        'linear_regression': LinearRegression
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return models[model_type]()

def save_pipeline_state(state: Dict[str, Any], filepath: str) -> None:
    """Save pipeline state to disk."""
    joblib.dump(state, filepath)
    logger.info(f"Pipeline state saved to {filepath}")

def load_pipeline_state(filepath: str) -> Dict[str, Any]:
    """Load pipeline state from disk."""
    state = joblib.load(filepath)
    logger.info(f"Pipeline state loaded from {filepath}")
    return state

def load_cached_result(step: str) -> dd.DataFrame:
    """Load cached result for a pipeline step."""
    cache_file = f"cache/{step}_result.parquet"
    if Path(cache_file).exists():
        try:
            return dd.read_parquet(cache_file)
        except Exception as e:
            logger.error(f"Error loading cached result: {str(e)}")
            return None
    return None

def cache_result(step: str, data: dd.DataFrame) -> None:
    """Cache result of a pipeline step."""
    cache_file = f"cache/{step}_result.parquet"
    try:
        Path("cache").mkdir(exist_ok=True)
        if isinstance(data, (pd.DataFrame, dd.DataFrame)):
            data.to_parquet(cache_file)
            logger.info(f"Cached result for step {step}")
        else:
            logger.error(f"Invalid data type for caching: {type(data)}")
            raise ValueError("Data must be a pandas or dask DataFrame")
    except Exception as e:
        logger.error(f"Error caching result: {str(e)}")
        raise

def load_custom_plugins(plugin_paths: List[str]) -> Dict[str, Any]:
    """Load custom plugin functions."""
    plugins = {}
    if plugin_paths:
        import importlib.util
        for path in plugin_paths:
            try:
                spec = importlib.util.spec_from_file_location("plugin", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                plugins = plugins | module.get_plugins()
            except Exception as e:
                logger.error(f"Error loading plugin {path}: {str(e)}")
    return plugins

def validate_feature_engineering_config(config: Dict[str, Any]) -> bool:
    """Validate feature engineering configuration."""
    feature_engineering_schema = Schema({
        Optional('create_polynomial_features'): bool,
        Optional('polynomial_degree'): And(int, lambda n: 1 < n < 4),
        Optional('create_interaction_features'): bool,
        Optional('create_time_features'): bool,
        Optional('time_column'): str,
        Optional('create_text_features'): bool,
        Optional('text_columns'): [str],
        Optional('select_top_features'): bool,
        Optional('n_top_features'): And(int, lambda n: n > 0),
        Optional('extract_html_features'): bool,
        Optional('html_column'): str
    })
    
    try:
        feature_engineering_schema.validate(config)
        return True
    except Exception as e:
        logger.error(f"Feature engineering config validation failed: {e}")
        return False

