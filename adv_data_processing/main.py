import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
import yaml
from pipeline import process_data
import dask.dataframe as dd
import argparse
from utils import *
from visualization import *
from text_analytics import *
from entity_recognition import *
from topic_modeling import *
from feature_selection import *
from dimensionality_reduction import *
from model_evaluation import *
from data_validation import *
from error_handling import *
from dask.distributed import Client, progress
from tqdm import tqdm
import joblib
import os
from feature_engineering import auto_feature_engineering
from imbalanced_data import handle_imbalanced_data
from dask_ml.model_selection import GridSearchCV
from transformation import (
    transform_data, handle_transform_step, get_scaler, 
    get_encoder, get_encoded_feature_names
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dask_client(args: argparse.Namespace) -> Client:
    """Set up and return a Dask client based on arguments."""
    if args.scheduler_address:
        client = Client(args.scheduler_address)
    else:
        client = Client(n_workers=args.n_workers)
    logger.info(f"Dask client set up with {client.ncores} cores")
    return client

def initialize_pipeline_state(args: argparse.Namespace) -> Tuple[dd.DataFrame, List[str]]:
    """Initialize or resume pipeline state."""
    if args.resume:
        pipeline_state = load_pipeline_state(args.resume)
        logger.info(f"Resumed from state: {args.resume}")
        return pipeline_state['data'], pipeline_state['completed_steps']
    return None, []

def handle_text_analytics(data: dd.DataFrame, config: Dict, args: argparse.Namespace) -> Dict:
    """Process text analytics steps and return results."""
    results = {}
    if args.analyze_text:
        text_column = config['text_column']
        results['sentiment'] = perform_sentiment_analysis(data[text_column])
        results['summary'] = summarize_text(data[text_column])
        logger.info(f"Sentiment analysis results: {results['sentiment']}")
        logger.info(f"Text summary: {results['summary']}")
    return results

def handle_entity_extraction(data: dd.DataFrame, config: Dict, args: argparse.Namespace) -> Dict:
    """Process entity extraction and return results."""
    results = {}
    if args.extract_entities:
        text_column = config['text_column']
        results['entities'] = extract_entities(data[text_column])
        logger.info(f"Extracted entities: {results['entities']}")
        plot_entity_distribution(results['entities'])
    return results

def handle_model_training(data: dd.DataFrame, config: Dict, args: argparse.Namespace) -> Any:
    """Handle model training and evaluation."""
    if 'model_type' not in config:
        return None

    model = get_model(config['model_type'])
    X = data.drop(columns=[config['target_column']])
    y = data[config['target_column']]

    if args.auto_tune:
        model = perform_auto_tuning(model, X, y, config)
    else:
        model.fit(X, y)

    evaluation_results = evaluate_model(model, X, y, config['evaluation_metrics'])
    logger.info(f"Model evaluation results: {evaluation_results}")
    return model

def perform_auto_tuning(model: Any, X: dd.DataFrame, y: dd.Series, config: Dict) -> Any:
    """Perform hyperparameter tuning."""
    try:
        param_grid = config.get('param_grid', {})
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(X, y)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error during auto-tuning: {str(e)}")
        logger.info("Falling back to default model fitting")
        model.fit(X, y)
        return model

def process_pipeline_step(step: str, data: dd.DataFrame, config: Dict, args: argparse.Namespace, 
                         custom_plugins: Dict) -> dd.DataFrame:
    """Process a single pipeline step and return updated data."""
    if args.use_cache:
        cached_result = load_cached_result(step)
        if cached_result is not None:
            logger.info(f"Loaded cached result for step: {step}")
            return cached_result

    log_step(step)

    # Map of step names to their handling functions
    step_handlers = {
        'load': lambda: process_data(config['source'], steps=['load'], **config.get('load_options', {})),
        'clean': lambda: process_data(data, steps=['clean'], cleaning_strategies=config.get('cleaning_strategies')),
        'transform': lambda: handle_transform_step(data, config),
        'validate_schema': lambda: handle_validation_step(data, config, args),
        'select_features': lambda: handle_feature_selection(data, config, args),
        'reduce_dimensions': lambda: handle_dimension_reduction(data, config, args),
        'auto_feature_engineering': lambda: handle_feature_engineering(data, config, args),
        'handle_imbalanced': lambda: handle_imbalanced_data(data, config['target_column'], 
                                                          config.get('imbalance_method', 'smote'))
    }

    # Get the appropriate handler or use custom plugin
    if handler := step_handlers.get(step):
        processed_data = handler()
    elif step in custom_plugins:
        processed_data = custom_plugins[step](data, config)
    else:
        processed_data = data

    if args.use_cache:
        cache_result(step, processed_data)

    return processed_data

def handle_transform_step(data: dd.DataFrame, config: Dict) -> dd.DataFrame:
    """Handle the transform pipeline step."""
    return process_data(data, steps=['transform'],
                       numeric_features=config.get('numeric_features'),
                       categorical_features=config.get('categorical_features'),
                       scale_strategy=config.get('scale_strategy', 'standard'),
                       encode_strategy=config.get('encode_strategy', 'onehot'))

def handle_validation_step(data: dd.DataFrame, config: Dict, args: argparse.Namespace) -> dd.DataFrame:
    """Handle the validation pipeline step."""
    if args.validate_schema:
        schema_valid = validate_data_schema(data, config['data_schema'])
        logger.info(f"Data schema validation result: {schema_valid}")
    return data

def handle_feature_selection(data: dd.DataFrame, config: Dict, args: argparse.Namespace) -> dd.DataFrame:
    """Handle feature selection step."""
    if args.select_features:
        selected_features = select_features(data, config['target_column'], 
                                         config.get('feature_selection_method', 'mutual_info'))
        return data[selected_features + [config['target_column']]]
    return data

def handle_dimension_reduction(data: dd.DataFrame, config: Dict, args: argparse.Namespace) -> dd.DataFrame:
    """Handle dimensionality reduction step."""
    if args.reduce_dimensions:
        reduced_data = reduce_dimensions(data, config.get('n_components', 2), 
                                      config.get('reduction_method', 'pca'))
        return dd.concat([data, reduced_data], axis=1)
    return data

def handle_feature_engineering(data: dd.DataFrame, config: Dict, args: argparse.Namespace) -> dd.DataFrame:
    """Handle feature engineering step."""
    if args.auto_feature_engineering:
        return auto_feature_engineering(data, config['target_column'])
    return data

def main(args: argparse.Namespace) -> None:
    client = None
    try:
        config = load_and_validate_config(args.config)
        client = setup_dask_client(args)
        custom_plugins = load_custom_plugins(args.plugins) if args.plugins else {}
        processed_data, completed_steps = initialize_pipeline_state(args)

        steps = get_pipeline_steps(config)
        process_pipeline(steps, args, config, processed_data, completed_steps, custom_plugins)

        save_processed_data(processed_data, args, config)
        if args.generate_report:
            generate_report(processed_data, completed_steps, config, args.output or config['output_file'])

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except IOError as e:
        logger.error(f"I/O error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        if client:
            client.close()

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Separate function for config loading and validation."""
    config = load_config(config_path)
    if not validate_config(config):
        raise ValueError("Invalid configuration file")
    return config

def process_pipeline(steps: List[str], args: argparse.Namespace, config: Dict[str, Any], 
                     processed_data: Any, completed_steps: List[str], custom_plugins: Dict[str, Any]) -> None:
    """Dedicated function for pipeline processing."""
    with tqdm(total=len(steps), desc="Processing Pipeline") as pbar:
        for step in steps:
            if step not in completed_steps:
                processed_data = execute_step(step, args, config, processed_data, custom_plugins)
                completed_steps.append(step)
                pbar.update(1)
                save_pipeline_state({'data': processed_data, 'completed_steps': completed_steps}, 
                                  f'pipeline_state_{step}.pkl')

def save_processed_data(processed_data: Any, args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Dedicated function for saving processed data."""
    output_file = args.output or config['output_file']
    processed_data.to_csv(output_file, index=False, single_file=True)
    logger.info(f"Processed data saved to {output_file}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Data Processing Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--output', type=str, help='Output file path (overrides config file)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--analyze_text', action='store_true', help='Perform text analytics')
    parser.add_argument('--extract_entities', action='store_true', help='Perform named entity recognition')
    parser.add_argument('--model_topics', action='store_true', help='Perform topic modeling')
    parser.add_argument('--resume', type=str, help='Resume from a saved pipeline state')
    parser.add_argument('--select_features', action='store_true', help='Perform feature selection')
    parser.add_argument('--reduce_dimensions', action='store_true', help='Perform dimensionality reduction')
    parser.add_argument('--validate_schema', action='store_true', help='Validate data schema')
    parser.add_argument('--summary_stats', action='store_true', help='Generate summary statistics')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate results')
    parser.add_argument('--intermediate_path', type=str, default='./intermediate/', 
                       help='Path to save intermediate results')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for parallel processing')
    parser.add_argument('--scheduler_address', type=str, help='Address of the Dask scheduler')
    parser.add_argument('--plugins', type=str, nargs='+', help='Custom plugins to load')
    parser.add_argument('--generate_report', action='store_true', help='Generate a comprehensive report')
    parser.add_argument('--use_cache', action='store_true', help='Use cached results if available')
    parser.add_argument('--auto_feature_engineering', action='store_true', 
                       help='Perform automatic feature engineering')
    parser.add_argument('--handle_imbalanced', action='store_true', help='Handle imbalanced datasets')
    parser.add_argument('--auto_tune', action='store_true', help='Perform automatic hyperparameter tuning')
    parser.add_argument('--memory_limit', type=int, help='Set memory limit for Dask workers')
    return parser.parse_args()

def generate_report(data: dd.DataFrame, steps: List[str], config: Dict[str, Any], output_file: str) -> None:
    """Generate a comprehensive report of the pipeline execution."""
    report = [
        "Data Processing Pipeline Report",
        "=" * 30,
        "",
        f"Configuration: {config}",
        "",
        f"Steps Completed: {steps}",
        "",
        f"Data Shape: {data.shape}",
        "",
        f"Data Types:\n{data.dtypes}",
        "",
        f"Summary Statistics:\n{data.describe()}",
        "",
        f"Output File: {output_file}",
        ""
    ]

    report_file = 'pipeline_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(str(item) for item in report))
    logger.info(f"Comprehensive report generated: {report_file}")

def execute_step(step: str, args: argparse.Namespace, config: Dict[str, Any], 
                processed_data: dd.DataFrame, custom_plugins: Dict[str, Any]) -> dd.DataFrame:
    """Execute a single pipeline step."""
    if args.use_cache:
        cached_result = load_cached_result(step)
        if cached_result is not None:
            logger.info(f"Loaded cached result for step: {step}")
            return cached_result

    log_step(step)

    # Process the step using process_pipeline_step
    processed_data = process_pipeline_step(step, processed_data, config, args, custom_plugins)

    if args.use_cache:
        cache_result(step, processed_data)

    return processed_data

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
