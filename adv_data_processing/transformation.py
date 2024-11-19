import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
import dask.dataframe as dd
from dask.delayed import delayed
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

def transform_data(data: Union[pd.DataFrame, dd.DataFrame],
                  numeric_features: List[str] = None,
                  categorical_features: List[str] = None,
                  scale_strategy: str = 'standard',
                  encode_strategy: str = 'onehot') -> Union[pd.DataFrame, dd.DataFrame]:
    """Transform data using specified strategies."""
    try:
        # Convert to pandas if it's a dask dataframe
        is_dask = isinstance(data, dd.DataFrame)
        if is_dask:
            data = data.compute()

        if numeric_features:
            data = scale_numeric_features(data, numeric_features, strategy=scale_strategy)

        if categorical_features:
            data = encode_categorical_features(data, categorical_features, strategy=encode_strategy)

        # Convert back to dask if input was dask
        if is_dask:
            data = dd.from_pandas(data, npartitions=4)

        return data

    except Exception as e:
        logger.error(f"Error during data transformation: {str(e)}")
        raise

def get_scaler(strategy: str = 'standard') -> Any:
    """Get the appropriate scaler based on strategy."""
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
    }
    return scalers.get(strategy, StandardScaler())

def get_encoder(strategy: str = 'onehot') -> Any:
    """Get the appropriate encoder based on strategy."""
    return LabelEncoder() if strategy == 'label' else DictVectorizer(sparse=False)

def scale_numeric_features(data: pd.DataFrame,
                         numeric_features: List[str],
                         strategy: str = 'standard') -> pd.DataFrame:
    """Scale numeric features using specified strategy."""
    try:
        scaler = get_scaler(strategy)
        data[numeric_features] = scaler.fit_transform(data[numeric_features])
        return data
    except Exception as e:
        logger.error(f"Error scaling numeric features: {str(e)}")
        raise

def encode_categorical_features(data: pd.DataFrame,
                              categorical_features: List[str],
                              strategy: str = 'onehot') -> pd.DataFrame:
    """Encode categorical features using specified strategy."""
    try:
        if strategy == 'label':
            for feature in categorical_features:
                encoder = get_encoder(strategy)
                data[feature] = encoder.fit_transform(data[feature])
        else:
            # Convert categorical columns to dict format for DictVectorizer
            dict_data = data[categorical_features].to_dict('records')
            encoder = get_encoder(strategy)
            encoded_data = encoder.fit_transform(dict_data)
            
            # Get feature names and create new dataframe
            feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=data.index)
            
            # Drop original categorical columns and join encoded ones
            data = data.drop(columns=categorical_features)
            data = pd.concat([data, encoded_df], axis=1)

        return data
    except Exception as e:
        logger.error(f"Error encoding categorical features: {str(e)}")
        raise

def get_encoded_feature_names(encoder: Any, features: List[str]) -> List[str]:
    """Get the names of encoded features."""
    if isinstance(encoder, LabelEncoder):
        return features
    return [f"{feature}_{val}" for feature in features 
            for val in encoder.get_feature_names_out([feature])]

def handle_transform_step(data: Union[pd.DataFrame, dd.DataFrame], 
                        config: Dict[str, Any]) -> Union[pd.DataFrame, dd.DataFrame]:
    """Handle the transform pipeline step."""
    return transform_data(
        data,
        numeric_features=config.get('numeric_features'),
        categorical_features=config.get('categorical_features'),
        scale_strategy=config.get('scale_strategy', 'standard'),
        encode_strategy=config.get('encode_strategy', 'onehot')
    )

