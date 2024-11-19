# feature_engineering.py
import logging
from typing import Dict, Any, Optional
import pandas as pd
import dask.dataframe as dd
import numpy as np

logger = logging.getLogger(__name__)

try:
    from dask_ml.preprocessing import PolynomialFeatures
    DASK_ML_AVAILABLE = True
except ImportError:
    DASK_ML_AVAILABLE = False
    logger.warning("dask_ml not available. Feature engineering will be limited.")

def auto_feature_engineering(
    df: dd.DataFrame,
    target_col: str,
    config: Dict[str, Any]
) -> dd.DataFrame:
    """Automatically engineer features based on configuration."""
    if not isinstance(df, dd.DataFrame):
        raise TypeError("Input must be a dask DataFrame")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    result = df.copy()
    
    if config.get('create_polynomial_features'):
        result = create_polynomial_features(
            result, 
            degree=config.get('polynomial_degree', 2)
        )
    
    if config.get('create_interaction_features'):
        result = create_interaction_features(result)
        
    return result

def create_polynomial_features(df: dd.DataFrame, degree: int = 2) -> dd.DataFrame:
    """Create polynomial features."""
    if not DASK_ML_AVAILABLE:
        logger.warning("dask_ml not available. Returning original dataframe.")
        return df
        
    try:
        return (PolynomialFeatures(degree=degree, include_bias=False).fit_transform(df.select_dtypes(include=['int64', 'float64']).compute().values)
                if df.select_dtypes(include=['int64', 'float64']).columns.size 
                else df)
        
    except Exception as e:
        logger.error(f"Error creating polynomial features: {str(e)}")
        return df

def create_interaction_features(df: dd.DataFrame) -> dd.DataFrame:
    """Create interaction features between numeric columns."""
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        result = df.copy()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                result[f'interaction_{col1}_{col2}'] = df[col1] * df[col2]
                
        return result
        
    except Exception as e:
        logger.error(f"Error creating interaction features: {str(e)}")
        return df
