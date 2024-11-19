"""Module for data preprocessing tasks."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = 'mean',
    categorical_strategy: str = 'most_frequent',
    numeric_fill_value: float = None,
    categorical_fill_value: str = None
) -> pd.DataFrame:
    """Handle missing values in both numeric and categorical columns."""
    df = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Handle numeric columns
    if len(numeric_cols) > 0:
        if numeric_strategy == 'constant':
            df[numeric_cols] = df[numeric_cols].fillna(numeric_fill_value)
        else:
            numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    # Handle categorical columns
    if len(categorical_cols) > 0:
        if categorical_strategy == 'constant':
            df[categorical_cols] = df[categorical_cols].fillna(categorical_fill_value)
        else:
            for col in categorical_cols:
                # Convert to string type and handle missing values
                df[col] = df[col].astype(str).replace('None', np.nan)
                imputer = SimpleImputer(strategy=categorical_strategy)
                df[col] = imputer.fit_transform(df[[col]]).ravel()
    
    return df

def encode_categorical_variables(
    df: pd.DataFrame,
    encoding_type: str = 'onehot'
) -> pd.DataFrame:
    """Encode categorical variables using specified method."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        # Convert all categorical columns to string type
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            
        if encoding_type == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=df.index
            )
            # Drop original categorical columns and join encoded ones
            df = df.drop(columns=categorical_cols).join(encoded_df)
            
        elif encoding_type == 'label':
            for col in categorical_cols:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                
        else:
            raise ValueError(f"Invalid encoding type: {encoding_type}")
    
    return df

def scale_numerical_features(
    df: pd.DataFrame,
    scaling_type: str = 'standard'
) -> pd.DataFrame:
    """Scale numerical features using specified method."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 0:
        if scaling_type == 'standard':
            # Manual standardization using pandas methods
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / std
                
        elif scaling_type == 'minmax':
            # Manual min-max scaling using pandas methods
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
                
        else:
            raise ValueError(f"Invalid scaling type: {scaling_type}")
    
    return df

def preprocess_dataset(
    df: pd.DataFrame,
    numeric_missing_strategy: str = 'mean',
    categorical_missing_strategy: str = 'most_frequent',
    numeric_fill_value: float = None,
    categorical_fill_value: str = None,
    encoding_type: str = 'onehot',
    scaling_type: str = 'standard'
) -> pd.DataFrame:
    """Apply full preprocessing pipeline to the dataset."""
    df = df.copy()
    
    # Handle missing values
    df = handle_missing_values(
        df,
        numeric_strategy=numeric_missing_strategy,
        categorical_strategy=categorical_missing_strategy,
        numeric_fill_value=numeric_fill_value,
        categorical_fill_value=categorical_fill_value
    )
    
    # Encode categorical variables
    df = encode_categorical_variables(df, encoding_type=encoding_type)
    
    # Scale numerical features
    df = scale_numerical_features(df, scaling_type=scaling_type)
    
    return df

