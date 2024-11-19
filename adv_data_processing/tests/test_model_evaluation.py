import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from adv_data_processing.model_evaluation import (
    evaluate_classification_model,
    evaluate_regression_model,
    cross_validate_model
)

@pytest.fixture
def classification_data():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y

@pytest.fixture
def regression_data():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y = pd.Series([2.1, 4.2, 6.3, 8.4, 10.5, 12.6, 14.7, 16.8, 18.9, 21.0])
    return X, y

def test_evaluate_classification_model(classification_data):
    X, y = classification_data
    model = DummyClassifier(strategy='stratified', random_state=42)
    model.fit(X, y)
    
    metrics = evaluate_classification_model(model, X, y)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0 <= v <= 1 for v in metrics.values())

def test_evaluate_regression_model(regression_data):
    X, y = regression_data
    model = DummyRegressor(strategy='mean')
    model.fit(X, y)
    
    metrics = evaluate_regression_model(model, X, y)
    
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['r2'] <= 1

def test_cross_validate_model_classification(classification_data):
    X, y = classification_data
    model = DummyClassifier(strategy='stratified', random_state=42)
    
    cv_results = cross_validate_model(
        model, X, y, 
        cv=3, 
        metrics=['accuracy', 'precision', 'recall']
    )
    
    assert isinstance(cv_results, dict)
    assert 'accuracy' in cv_results
    assert 'precision' in cv_results
    assert 'recall' in cv_results
    assert all(len(scores) == 3 for scores in cv_results.values())
    assert all(all(0 <= score <= 1 for score in scores) 
              for scores in cv_results.values())

def test_cross_validate_model_regression(regression_data):
    X, y = regression_data
    model = DummyRegressor(strategy='mean')
    
    cv_results = cross_validate_model(
        model, X, y, 
        cv=3, 
        metrics=['mse', 'r2']
    )
    
    assert isinstance(cv_results, dict)
    assert 'mse' in cv_results
    assert 'r2' in cv_results
    assert all(len(scores) == 3 for scores in cv_results.values())
    assert all(score >= 0 for score in cv_results['mse'])
    assert all(score <= 1 for score in cv_results['r2'])

def test_invalid_metric():
    X, y = np.random.rand(10, 2), np.random.randint(0, 2, 10)
    model = DummyClassifier()
    model.fit(X, y)
    
    with pytest.raises(ValueError):
        evaluate_classification_model(model, X, y, metrics=['invalid_metric'])
        
    with pytest.raises(ValueError):
        evaluate_regression_model(model, X, y, metrics=['invalid_metric'])
        
    with pytest.raises(ValueError):
        cross_validate_model(model, X, y, metrics=['invalid_metric'])

def test_empty_data():
    X, y = np.array([]).reshape(0, 2), np.array([])
    model = DummyClassifier()
    
    with pytest.raises(ValueError):
        evaluate_classification_model(model, X, y)
        
    with pytest.raises(ValueError):
        evaluate_regression_model(model, X, y)
        
    with pytest.raises(ValueError):
        cross_validate_model(model, X, y)