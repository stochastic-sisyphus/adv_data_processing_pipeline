from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import dask.dataframe as dd

def validate_data_schema(df, schema):
    """Validate dataframe against schema.
    
    Args:
        df: pandas DataFrame to validate
        schema: dict with validation rules
        
    Returns:
        dict with validation results
    """
    results = {
        'valid': True,
        'violations': [],
        'type_mismatches': [],
        'range_violations': []
    }
    
    # Convert to pandas if it's a dask dataframe
    if hasattr(df, 'compute'):
        df = df.compute()
        
    for column, rules in schema.items():
        if column not in df.columns:
            continue
            
        # Check type
        if 'type' in rules:
            actual_type = str(df[column].dtype)
            if actual_type != rules['type']:
                results['type_mismatches'].append({
                    'column': column,
                    'expected': rules['type'],
                    'actual': actual_type
                })
                results['valid'] = False
                
        # Check range
        if 'range' in rules:
            min_val, max_val = rules['range']
            range_violations = df[
                (df[column] < min_val) | (df[column] > max_val)
            ].index.tolist()
            
            if range_violations:
                results['range_violations'].extend([{
                    'column': column,
                    'index': idx,
                    'value': df.loc[idx, column]
                } for idx in range_violations])
                results['valid'] = False
                
        # Add violations to the main list
        if results['type_mismatches'] or results['range_violations']:
            results['violations'].extend([{
                'column': column,
                'type': 'type_mismatch' if results['type_mismatches'] else 'range_violation',
                'details': results['type_mismatches'] or results['range_violations']
            }])
            
    return results

