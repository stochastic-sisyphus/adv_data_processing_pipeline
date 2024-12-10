"""Memory and performance optimization utilities."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by selecting appropriate dtypes.

    Args:
        df (pd.DataFrame): The DataFrame to optimize.

    Returns:
        pd.DataFrame: Optimized DataFrame with reduced memory usage.
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type == 'object':
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                df_optimized[col] = pd.Categorical(df_optimized[col])
        elif col_type == 'int64':
            if df_optimized[col].min() >= -128 and df_optimized[col].max() <= 127:
                df_optimized[col] = df_optimized[col].astype('int8')
            elif df_optimized[col].min() >= 0 and df_optimized[col].max() <= 255:
                df_optimized[col] = df_optimized[col].astype('uint8')
            elif df_optimized[col].min() >= -32768 and df_optimized[col].max() <= 32767:
                df_optimized[col] = df_optimized[col].astype('int16')
        elif col_type == 'float64':
            df_optimized[col] = df_optimized[col].astype('float32')
    
    logger.info("Optimized DataFrame dtypes")
    return df_optimized

def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get memory usage statistics for DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        Dict[str, float]: Memory usage statistics including total memory and memory per column.
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024**2  # Convert to MB
    
    memory_stats = {
        'total_memory_mb': total_memory,
        'memory_per_column': {col: mem/1024**2 for col, mem in memory_usage.items()}
    }
    
    logger.info(f"Memory usage: {memory_stats}")
    return memory_stats
