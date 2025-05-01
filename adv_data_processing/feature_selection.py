"""Feature selection functions for the data processing pipeline."""

import logging
from typing import List, Optional
import dask.dataframe as dd

logger = logging.getLogger(__name__)

try:
    from dask_ml.feature_selection import SelectKBest, mutual_info_classif, f_classif
    DASK_ML_AVAILABLE = True
except ImportError:
    DASK_ML_AVAILABLE = False
    logger.warning("dask_ml not available. Feature selection will be limited.")

def select_features(
    df: dd.DataFrame,
    target_col: str,
    n_features: int = 10,
    method: str = 'mutual_info'
) -> dd.DataFrame:
    """
    Select top features based on statistical tests.

    Args:
        df (dd.DataFrame): The dataframe to select features from.
        target_col (str): The target column name.
        n_features (int, optional): The number of top features to select. Defaults to 10.
        method (str, optional): The method to use for feature selection. Defaults to 'mutual_info'.

    Returns:
        dd.DataFrame: The dataframe with selected features.
    """
    if not DASK_ML_AVAILABLE:
        logger.warning("dask_ml not available. Returning original dataframe.")
        return df
        
    try:
        selector = SelectKBest(
            {'mutual_info': mutual_info_classif, 'f_classif': f_classif}[method],
            k=n_features
        )
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        selected_features = selector.fit_transform(X, y)
        logger.info(f"Selected top {n_features} features using {method} method")
        return selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        return df
