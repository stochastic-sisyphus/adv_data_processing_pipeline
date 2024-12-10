
import pytest
import numpy as np
from adv_data_processing.metrics import MetricsManager

@pytest.fixture
def classification_data():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    return y_true, y_pred

@pytest.fixture
def regression_data():
    y_true = np.array([1.5, 2.1, 3.3, 4.7, 5.8])
    y_pred = np.array([1.7, 2.0, 3.1, 4.9, 5.7])
    return y_true, y_pred

def test_classification_metrics(classification_data):
    y_true, y_pred = classification_data
    metrics = ['accuracy', 'precision', 'recall']
    results = MetricsManager.calculate_metrics(y_true, y_pred, metrics, 'classification')
    assert all(metric in results for metric in metrics)
    assert 0 <= results['accuracy'] <= 1

def test_regression_metrics(regression_data):
    y_true, y_pred = regression_data
    metrics = ['mse', 'rmse', 'r2']
    results = MetricsManager.calculate_metrics(y_true, y_pred, metrics, 'regression')
    assert all(metric in results for metric in metrics)
    assert results['mse'] >= 0