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

def test_advanced_classification_metrics(classification_data):
    y_true, y_pred = classification_data
    results = MetricsManager.calculate_advanced_metrics(y_true, y_pred)
    assert 'confusion_matrix' in results
    assert 'per_class_precision' in results
    assert 'per_class_recall' in results
    assert 'weighted_metrics' in results

def test_generate_metrics_report(tmp_path, classification_data):
    y_true, y_pred = classification_data
    metrics = ['accuracy', 'precision', 'recall']
    results = MetricsManager.calculate_metrics(y_true, y_pred, metrics, 'classification')
    report_file = tmp_path / "metrics_report.txt"
    MetricsManager.generate_metrics_report(results, str(report_file))
    assert report_file.exists()
    with open(report_file, 'r') as f:
        content = f.read()
        assert "Model Evaluation Report" in content
        assert "accuracy" in content
        assert "precision" in content
        assert "recall" in content

def test_edge_case_empty_classification_data():
    y_true = np.array([])
    y_pred = np.array([])
    metrics = ['accuracy', 'precision', 'recall']
    with pytest.raises(ValueError):
        MetricsManager.calculate_metrics(y_true, y_pred, metrics, 'classification')

def test_edge_case_empty_regression_data():
    y_true = np.array([])
    y_pred = np.array([])
    metrics = ['mse', 'rmse', 'r2']
    with pytest.raises(ValueError):
        MetricsManager.calculate_metrics(y_true, y_pred, metrics, 'regression')
