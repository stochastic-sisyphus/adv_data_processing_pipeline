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
        
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame against schema with detailed reporting."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }

        try:
            # Basic schema validation
            data_dict = df.to_dict('records')[0]
            schema_valid = self.validator.validate(data_dict)
            
            if not schema_valid:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(
                    [f"Column '{k}': {v}" for k, v in self.validator.errors.items()]
                )

            # Check for missing required columns
            missing_cols = set(self.schema.keys()) - set(df.columns)
            if missing_cols:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"Missing required columns: {', '.join(missing_cols)}"
                )

            # Check data types
            type_validation = self.check_data_types(df)
            invalid_types = {k: v for k, v in type_validation.items() if not v}
            if invalid_types:
                validation_results['errors'].extend(
                    [f"Invalid type for column '{k}'" for k in invalid_types]
                )

            # Check for missing values
            missing_stats = self.check_missing_values(df)
            validation_results['summary']['missing_values'] = missing_stats

            return validation_results

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            return validation_results
    
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