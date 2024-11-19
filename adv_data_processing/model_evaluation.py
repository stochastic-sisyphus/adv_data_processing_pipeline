"""Module for evaluating machine learning models."""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_classification_model(model, X, y, metrics=None):
    """Evaluate a classification model using various metrics.
    
    Args:
        model: Fitted classifier with predict method
        X: Features
        y: True labels
        metrics: List of metrics to compute
        
    Returns:
        dict: Metric names and values
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty data provided")
        
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
    available_metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }
    
    invalid_metrics = set(metrics) - set(available_metrics.keys())
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")
    
    y_pred = model.predict(X)
    results = {}
    
    for metric in metrics:
        results[metric] = available_metrics[metric](y, y_pred)
        
    return results

def evaluate_regression_model(model, X, y, metrics=None):
    """Evaluate a regression model using various metrics.
    
    Args:
        model: Fitted regressor with predict method
        X: Features
        y: True values
        metrics: List of metrics to compute
        
    Returns:
        dict: Metric names and values
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty data provided")
        
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'r2']
        
    y_pred = model.predict(X)
    results = {}
    
    available_metrics = {
        'mse': lambda y, yp: mean_squared_error(y, yp),
        'rmse': lambda y, yp: np.sqrt(mean_squared_error(y, yp)),
        'mae': lambda y, yp: mean_absolute_error(y, yp),
        'r2': lambda y, yp: r2_score(y, yp)
    }
    
    invalid_metrics = set(metrics) - set(available_metrics.keys())
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")
    
    for metric in metrics:
        results[metric] = available_metrics[metric](y, y_pred)
        
    return results

def cross_validate_model(model, X, y, cv=5, metrics=None):
    """Perform cross-validation on a model.
    
    Args:
        model: Unfitted model
        X: Features
        y: Target
        cv: Number of folds
        metrics: List of scoring metrics
        
    Returns:
        dict: Metric names and lists of scores
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty data provided")
        
    if metrics is None:
        metrics = ['accuracy'] if hasattr(model, 'predict_proba') else ['neg_mean_squared_error']
    
    # Map our metric names to sklearn's scoring parameter names
    metric_mapping = {
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    results = {}
    for metric in metrics:
        try:
            sklearn_metric = metric_mapping.get(metric, metric)
            scores = cross_val_score(
                model, X, y, 
                cv=cv, 
                scoring=sklearn_metric
            )
            
            # Convert negative metrics back to positive
            if sklearn_metric.startswith('neg_'):
                scores = -scores
                
            results[metric] = scores
        except ValueError as e:
            raise ValueError(f"Invalid metric: {metric}. Available metrics: {list(metric_mapping.keys())}") from e
            
    return results
