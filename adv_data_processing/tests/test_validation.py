
import pytest
import pandas as pd
import numpy as np
from adv_data_processing.validation import DataValidator

@pytest.fixture
def sample_schema():
    return {
        'numeric_col': {'type': 'float', 'min': 0, 'max': 100},
        'categorical_col': {'type': 'string', 'allowed': ['A', 'B', 'C']},
        'id_col': {'type': 'integer', 'min': 1}
    }

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'numeric_col': [25.5, 75.2, 50.0],
        'categorical_col': ['A', 'B', 'C'],
        'id_col': [1, 2, 3]
    })

def test_validate_dataframe(sample_schema, sample_dataframe):
    validator = DataValidator(sample_schema)
    assert validator.validate_dataframe(sample_dataframe)

def test_check_missing_values(sample_schema, sample_dataframe):
    validator = DataValidator(sample_schema)
    df_with_missing = sample_dataframe.copy()
    df_with_missing.loc[0, 'numeric_col'] = np.nan
    missing_report = validator.check_missing_values(df_with_missing)
    assert missing_report['numeric_col'] > 0

def test_check_data_types(sample_schema, sample_dataframe):
    validator = DataValidator(sample_schema)
    type_results = validator.check_data_types(sample_dataframe)
    assert all(type_results.values())