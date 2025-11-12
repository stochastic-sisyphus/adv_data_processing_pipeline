# GitHub Copilot Instructions for Advanced Data Processing Pipeline

## Project Overview

This is an advanced data processing pipeline for machine learning workflows, designed to handle large-scale data processing tasks efficiently. The project is published on PyPI as `advanced-data-processing` and provides a comprehensive suite of tools for data loading, cleaning, transformation, feature engineering, text analytics, and machine learning model integration.

## Repository Structure

```
adv_data_processing_pipeline/
├── adv_data_processing/          # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── batch_processing.py       # Batch processing with memory optimization
│   ├── cleaning.py               # Data cleaning utilities
│   ├── config.py                 # Configuration management
│   ├── data_preprocessing.py     # Preprocessing pipeline
│   ├── data_transformation.py    # Data transformation utilities
│   ├── data_validation.py        # Data validation logic
│   ├── dimensionality_reduction.py  # PCA, t-SNE, UMAP implementations
│   ├── entity_recognition.py     # Named Entity Recognition
│   ├── error_handling.py         # Error handling utilities
│   ├── feature_engineering.py    # Feature engineering functions
│   ├── feature_selection.py      # Feature selection methods
│   ├── imbalanced_data.py        # Handling imbalanced datasets
│   ├── loading.py                # Data loading from various sources
│   ├── main.py                   # Main entry point
│   ├── metrics.py                # Performance metrics
│   ├── model_evaluation.py       # Model evaluation utilities
│   ├── monitoring.py             # Monitoring and logging
│   ├── optimization.py           # Hyperparameter optimization
│   ├── pipeline.py               # Pipeline orchestration
│   ├── text_analytics.py         # Text analysis (sentiment, summarization)
│   ├── topic_modeling.py         # Topic modeling (LDA, NMF)
│   ├── transformation.py         # Data transformation
│   ├── utils.py                  # Utility functions
│   ├── validation.py             # Validation logic
│   ├── visualization.py          # Data visualization
│   └── tests/                    # Test suite
│       ├── test_*.py             # Unit tests for each module
│       └── smoke_test.py         # Smoke tests
├── config/                       # Configuration files directory
├── .github/
│   └── workflows/                # CI/CD workflows
│       ├── ci.yml                # Main CI/CD pipeline
│       └── python-publish.yml    # PyPI publishing workflow
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # Package setup configuration
├── setup.cfg                     # Additional package configuration
├── mypy.ini                      # MyPy type checking configuration
├── .pre-commit-config.yaml       # Pre-commit hooks configuration
├── .coveragerc                   # Coverage configuration
└── README.md                     # Project documentation
```

## Development Workflow

### Installation

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -r requirements-dev.txt
```

### Testing

- **Test Framework**: pytest
- **Coverage Tool**: pytest-cov
- **Test Location**: `adv_data_processing/tests/`
- **Run Tests**: `pytest --cov=adv_data_processing --cov-report=term-missing`
- **Test Files**: Follow pattern `test_*.py`
- **Minimum Coverage**: Aim for high coverage on new code

### Linting and Code Quality

The project uses multiple linting and formatting tools:

- **Black**: Code formatter (line length: 88 characters)
  - Run: `black adv_data_processing/`
- **isort**: Import sorting (profile: black)
  - Run: `isort adv_data_processing/`
- **Flake8**: Style guide enforcement (max line length: 100)
  - Ignored rules: D100, D104, E203, W503
  - Run: `flake8 adv_data_processing/`
- **MyPy**: Static type checking (strict mode)
  - Run: `mypy adv_data_processing/`
- **Pre-commit hooks**: Automatically run before commits
  - Install: `pre-commit install`
  - Run manually: `pre-commit run --all-files`

### Building and Publishing

```bash
# Build package
python -m build

# Publish to PyPI (requires credentials)
twine upload dist/*
```

## Coding Standards

### Python Style

- **Python Version**: 3.8+ (support up to 3.12)
- **Code Style**: Follow PEP 8 with Black formatter
- **Import Order**: Use isort with black profile
- **Line Length**: 88 characters (Black default), 100 for Flake8
- **Type Hints**: Required for all function signatures (enforced by mypy)
- **Docstrings**: Required for public functions and classes

### Type Annotations

- All functions must have type hints for parameters and return values
- Use `from typing import` for complex types
- MyPy strict mode is enabled - ensure type safety

### Error Handling

- Use custom exceptions defined in `error_handling.py`
- Always catch specific exceptions, avoid bare `except:`
- Log errors appropriately using the logging setup
- Provide meaningful error messages

### Logging

- Use the logging utilities from `logging_setup.py`
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Include context in log messages

## Key Dependencies

### Core Data Processing
- **pandas** (>=1.5.0): Primary data manipulation
- **numpy** (>=1.23.0): Numerical operations
- **dask** (>=2023.1.0): Distributed computing for large datasets
- **scikit-learn** (>=1.0.0): Machine learning algorithms

### Deep Learning
- **torch** (>=2.1.0): PyTorch for GPU acceleration

### ML Experiment Tracking
- **mlflow** (>=2.8.0): Experiment tracking and model registry
- **optuna** (>=3.3.0): Hyperparameter optimization
- **wandb** (>=0.15.0): Experiment tracking (optional)

### NLP Libraries
- **nltk**: Natural language processing
- **spacy**: NLP and entity recognition
- **gensim**: Topic modeling
- **textblob**: Sentiment analysis

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive visualizations

### Other
- **psutil**: System monitoring
- **pyyaml**: Configuration file parsing
- **boto3**: AWS S3 integration
- **sqlalchemy**: Database connectivity

## Common Tasks

### Adding a New Processing Step

1. Create a new module in `adv_data_processing/` (e.g., `new_feature.py`)
2. Implement the processing function with proper type hints
3. Add docstrings following existing patterns
4. Create corresponding tests in `adv_data_processing/tests/test_new_feature.py`
5. Update `pipeline.py` to integrate the new step if needed
6. Update documentation in README.md

### Adding a New Data Source

1. Add loader function to `loading.py`
2. Handle errors appropriately
3. Support both single-file and batch processing
4. Add tests for the new loader
5. Update documentation

### Modifying Configuration

- Configuration is managed through YAML files in `config/` directory
- Configuration schema is defined in `config.py`
- Use `pyyaml` for parsing
- Validate configuration using the validation utilities

## Testing Guidelines

### Test Structure

- Place tests in `adv_data_processing/tests/`
- Mirror the structure of the main package
- Use descriptive test names: `test_<function_name>_<scenario>`

### Test Categories

- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test complete pipeline workflows
- **Smoke Tests**: Quick validation tests

### Mocking and Fixtures

- Use `pytest-mock` for mocking external dependencies
- Create reusable fixtures in `conftest.py`
- Mock file I/O, network requests, and expensive computations

### Test Data

- Use small, representative datasets for testing
- Generate test data programmatically when possible
- Don't commit large test files to the repository

## CI/CD Pipeline

### Automated Checks (on PR and Push)

1. Pre-commit hooks (trailing whitespace, YAML validation, etc.)
2. Black formatting check
3. isort import ordering check
4. Flake8 linting
5. Pytest with coverage
6. Coverage report to Codecov

### Python Versions Tested

- Python 3.8
- Python 3.9
- Python 3.10

### Deployment

- Automatic deployment to PyPI on push to `main` branch
- Requires PYPI_USERNAME and PYPI_PASSWORD secrets

## Performance Considerations

- Use Dask for large datasets that don't fit in memory
- Implement batch processing for iterative operations
- Use GPU acceleration via PyTorch when available
- Cache intermediate results when appropriate
- Monitor memory usage with psutil

## Documentation

- Keep README.md up to date with new features
- Document configuration options
- Provide usage examples for new functionality
- Update CHANGELOG for version increments

## Contributing

When making changes:

1. Create a feature branch
2. Make minimal, focused changes
3. Write tests for new functionality
4. Ensure all tests pass
5. Run pre-commit hooks
6. Update documentation
7. Submit a pull request with clear description

## Security Considerations

- Never commit credentials or secrets
- Use environment variables for sensitive configuration
- Validate all external input
- Keep dependencies updated
- Use the security scanning in CI/CD pipeline

## Troubleshooting

### Common Issues

- **Import Errors**: Ensure package is installed with `pip install -e .`
- **Type Checking Errors**: Check mypy.ini for ignored imports
- **Test Failures**: Check test data and mock configurations
- **Memory Issues**: Use Dask for large datasets or reduce batch size
- **GPU Issues**: Code should gracefully fall back to CPU if GPU unavailable

## Additional Resources

- PyPI Package: https://pypi.org/project/advanced-data-processing/
- Repository: https://github.com/stochastic-sisyphus/adv_data_processing_pipeline
- Issue Tracker: Use GitHub Issues for bug reports and feature requests
