import numpy as np
from adv_data_processing.model_evaluation import evaluate_model

def smoke_test():
    # Create dummy data
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    
    # Create a simple model mock
    class DummyModel:
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
    
    model = DummyModel()
    
    # Test all metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'mse', 'rmse', 'r2']
    results = evaluate_model(model, X, y, metrics)
    
    # Check if all metrics were calculated
    assert all(metric in results for metric in metrics), "Not all metrics were calculated"
    
    print("Smoke test passed successfully!")

if __name__ == "__main__":
    smoke_test()
