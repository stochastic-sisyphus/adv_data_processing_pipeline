
"""Data validation and schema management."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from cerberus import Validator
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data against defined schemas and constraints."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.validator = Validator(schema)
        
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame against schema."""
        try:
            data_dict = df.to_dict('records')[0]
            return self.validator.validate(data_dict)
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False
    
    def check_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """Check for missing values in DataFrame."""
        missing_ratios = df.isnull().mean()
        problematic_columns = missing_ratios[missing_ratios > threshold]
        
        if not problematic_columns.empty:
            logger.warning(
                f"Columns with high missing values (>{threshold*100}%):\n"
                f"{problematic_columns}"
            )
            
        return missing_ratios.to_dict()
    
    def check_data_types(
        self,
        df: pd.DataFrame
    ) -> Dict[str, bool]:
        """Validate column data types."""
        results = {}
        for column, expected_type in self.schema.items():
            if column in df.columns:
                actual_type = df[column].dtype
                results[column] = self._check_type_compatibility(
                    actual_type,
                    expected_type
                )
        return results
    
    @staticmethod
    def _check_type_compatibility(
        actual_type: np.dtype,
        expected_type: str
    ) -> bool:
        """Check if actual type matches expected type."""
        type_mapping = {
            'integer': ['int16', 'int32', 'int64'],
            'float': ['float16', 'float32', 'float64'],
            'string': ['object', 'string'],
            'boolean': ['bool']
        }
        
        return str(actual_type) in type_mapping.get(expected_type, [])