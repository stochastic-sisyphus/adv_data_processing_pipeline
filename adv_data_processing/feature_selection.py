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
    """Select top features based on statistical tests."""
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
        
        return selector.fit_transform(X, y)
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        return df

