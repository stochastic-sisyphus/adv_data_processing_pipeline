
"""Centralized metrics management for model evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, roc_auc_score,
    confusion_matrix
)
from typing import Dict, Any, List, Union, Callable
import logging

logger = logging.getLogger(__name__)

class MetricsManager:
    """Manages evaluation metrics for the pipeline."""
    
    CLASSIFICATION_METRICS = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'auc_roc': roc_auc_score
    }
    
    REGRESSION_METRICS = {
        'mse': mean_squared_error,
        'rmse': lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score
    }
    
    @classmethod
    def get_metrics(cls, task_type: str) -> Dict[str, Callable]:
        """Get metrics for specified task type."""
        if task_type == 'classification':
            return cls.CLASSIFICATION_METRICS
        elif task_type == 'regression':
            return cls.REGRESSION_METRICS
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str],
        task_type: str
    ) -> Dict[str, float]:
        """Calculate specified metrics."""
        results = {}
        available_metrics = MetricsManager.get_metrics(task_type)
        
        for metric in metrics:
            if metric not in available_metrics:
                logger.warning(f"Unknown metric: {metric}")
                continue
            try:
                results[metric] = available_metrics[metric](y_true, y_pred)
            except Exception as e:
                logger.error(f"Error calculating {metric}: {str(e)}")
                results[metric] = None
                
        return results