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
    """Reduce dimensionality of the dataset."""
    if not DASK_ML_AVAILABLE:
        logger.warning("dask_ml not available. Returning original dataframe.")
        return df
        
    try:
        reducer = {
            'pca': lambda: PCA(n_components=n_components),
            'tsne': lambda: TSNE(n_components=n_components)
        }.get(method, lambda: ValueError(f"Unsupported dimensionality reduction method: {method}"))()
        
        return reducer.fit_transform(df)
        
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {str(e)}")
        return df

