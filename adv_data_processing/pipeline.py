from typing import Dict, List, Optional, Any, Callable
from .loading import load_data
from .cleaning import clean_data
from .transformation import transform_data
import logging
import time
from tqdm import tqdm
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from dask import delayed
import dask.bag as db
import contextlib

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from .text_analytics import perform_sentiment_analysis, summarize_text
    TEXT_ANALYTICS_AVAILABLE = True
except ImportError:
    TEXT_ANALYTICS_AVAILABLE = False
    logger.warning("Text analytics features not available.")

try:
    from .entity_recognition import extract_entities
    ENTITY_RECOGNITION_AVAILABLE = True
except ImportError:
    ENTITY_RECOGNITION_AVAILABLE = False
    logger.warning("Entity recognition features not available.")

try:
    from .topic_modeling import perform_topic_modeling
    TOPIC_MODELING_AVAILABLE = True
except ImportError:
    TOPIC_MODELING_AVAILABLE = False
    logger.warning("Topic modeling features not available.")

try:
    from .feature_selection import select_features
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False
    logger.warning("Feature selection features not available.")

try:
    from .dimensionality_reduction import reduce_dimensions
    DIMENSIONALITY_REDUCTION_AVAILABLE = True
except ImportError:
    DIMENSIONALITY_REDUCTION_AVAILABLE = False
    logger.warning("Dimensionality reduction features not available.")

from .data_validation import validate_data_schema
from .feature_engineering import auto_feature_engineering
from .imbalanced_data import handle_imbalanced_data

def apply_custom_transformations(df: dd.DataFrame, custom_funcs: List[Callable], pbar: tqdm) -> dd.DataFrame:
    """
    Apply custom transformation functions to the dataframe.

    Args:
        df (dd.DataFrame): The dataframe to transform.
        custom_funcs (List[Callable]): List of custom transformation functions.
        pbar (tqdm): Progress bar.

    Returns:
        dd.DataFrame: Transformed dataframe.
    """
    for func in custom_funcs:
        df = func(df)
        pbar.update(1 / len(custom_funcs))
    return df

def _execute_step(step: str, df: dd.DataFrame, params: dict, pbar: tqdm) -> dd.DataFrame:
    """
    Execute a single pipeline step.

    Args:
        step (str): The pipeline step to execute.
        df (dd.DataFrame): The dataframe to process.
        params (dict): Parameters for the step.
        pbar (tqdm): Progress bar.

    Returns:
        dd.DataFrame: Processed dataframe.
    """
    step_handlers = {
        'load': lambda: load_data(params['source'], chunk_size=params['chunk_size'], **params['kwargs']),
        'clean': lambda: clean_data(df, params['cleaning_strategies']),
        'transform': lambda: transform_data(df, params['numeric_features'], params['categorical_features'],
                                          params['scale_strategy'], params['encode_strategy']),
        'custom': lambda: apply_custom_transformations(df, params['custom_transformations'], pbar)
    }
    
    pbar.set_description(f"Executing {step}")
    return step_handlers.get(step, lambda: df)()

def _handle_large_dataset(df: dd.DataFrame, memory_limit: Optional[int], n_workers: int) -> dd.DataFrame:
    """
    Handle large datasets if they exceed memory limit.

    Args:
        df (dd.DataFrame): The dataframe to handle.
        memory_limit (Optional[int]): Memory limit in bytes.
        n_workers (int): Number of workers.

    Returns:
        dd.DataFrame: Handled dataframe.
    """
    if memory_limit and df.memory_usage().sum().compute() > memory_limit:
        return df.to_bag().repartition(npartitions=n_workers)
    return df

def process_data(
    source: str,
    steps: Optional[List[str]] = None,
    cleaning_strategies: Optional[Dict[str, Dict[str, str]]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_strategy: str = 'standard',
    encode_strategy: str = 'onehot',
    custom_transformations: Optional[List[Callable]] = None,
    chunk_size: Optional[int] = None,
    n_workers: int = 4,
    save_intermediate: bool = False,
    intermediate_path: str = './intermediate/',
    memory_limit: Optional[int] = None,
    **kwargs: Any
) -> dd.DataFrame:
    """
    Main function to load, clean, and transform data, optimized for large files.

    Args:
        source (str): Path to file or URL or SQL connection string.
        steps (Optional[List[str]]): List of steps to perform in the pipeline.
        cleaning_strategies (Optional[Dict[str, Dict[str, str]]]): Cleaning strategies for each column.
        numeric_features (Optional[List[str]]): List of numeric column names.
        categorical_features (Optional[List[str]]): List of categorical column names.
        scale_strategy (str): Strategy for scaling numeric features.
        encode_strategy (str): Strategy for encoding categorical features.
        custom_transformations (Optional[List[Callable]]): List of custom transformation functions.
        chunk_size (Optional[int]): Size of chunks for processing large files.
        n_workers (int): Number of workers for parallel processing.
        save_intermediate (bool): Whether to save intermediate results.
        intermediate_path (str): Path to save intermediate results.
        memory_limit (Optional[int]): Memory limit for handling large datasets.
        kwargs (Any): Additional arguments for data loading.

    Returns:
        dd.DataFrame: Processed dask DataFrame.
    """
    if steps is None:
        steps = ['load', 'clean', 'transform']

    params = {
        'source': source, 'chunk_size': chunk_size, 'kwargs': kwargs,
        'cleaning_strategies': cleaning_strategies, 'numeric_features': numeric_features,
        'categorical_features': categorical_features, 'scale_strategy': scale_strategy,
        'encode_strategy': encode_strategy, 'custom_transformations': custom_transformations or []
    }

    client = None
    try:
        start_time = time.time()
        client = Client(n_workers=n_workers)
        df = None

        with tqdm(total=len(steps), desc="Processing Data") as pbar:
            for step_group in [['load'], ['clean', 'transform'], ['custom']]:
                if step_results := [
                    delayed(_execute_step)(step, df, params, pbar)
                    for step in step_group if step in steps
                ]:
                    step_results = client.compute(step_results)
                    df = step_results[-1]
                    df = _handle_large_dataset(df, memory_limit, n_workers)

                    if save_intermediate:
                        for step in step_group:
                            if step in steps:
                                df.to_parquet(f"{intermediate_path}{step}_result.parquet")

        return df.compute()

    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise
    finally:
        if client:
            client.close()
