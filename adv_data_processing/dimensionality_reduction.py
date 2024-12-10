"""Dimensionality reduction functions for the data processing pipeline."""

import logging
from typing import Optional
import dask.dataframe as dd

logger = logging.getLogger(__name__)

try:
    from dask_ml.manifold import TSNE
    from dask_ml.decomposition import PCA
    DASK_ML_AVAILABLE = True
except ImportError:
    DASK_ML_AVAILABLE = False
    logger.warning("dask_ml not available. Dimensionality reduction will be limited.")

def reduce_dimensions(
    df: dd.DataFrame,
    method: str = 'pca',
    n_components: int = 2
) -> dd.DataFrame:
    """
    Reduce dimensionality of the dataset.

    Args:
        df (dd.DataFrame): The dataframe to reduce dimensions of.
        method (str, optional): The method to use for dimensionality reduction. Defaults to 'pca'.
        n_components (int, optional): The number of components to reduce to. Defaults to 2.

    Returns:
        dd.DataFrame: The dataframe with reduced dimensions.
    """
    if not DASK_ML_AVAILABLE:
        logger.warning("dask_ml not available. Returning original dataframe.")
        return df
        
    try:
        reducer = {
            'pca': lambda: PCA(n_components=n_components),
            'tsne': lambda: TSNE(n_components=n_components)
        }.get(method, lambda: ValueError(f"Unsupported dimensionality reduction method: {method}"))()
        
        reduced_df = reducer.fit_transform(df)
        logger.info(f"Reduced dimensions using {method} to {n_components} components")
        return reduced_df
        
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {str(e)}")
        return df
