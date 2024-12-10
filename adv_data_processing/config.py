
"""Configuration management for the data processing pipeline."""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Data class for pipeline configuration."""
    data_source: Dict[str, Any]
    preprocessing: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    model: Dict[str, Any]
    evaluation: Dict[str, Any]
    output: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return cls(**config)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

def validate_config(config: PipelineConfig) -> bool:
    """Validate pipeline configuration."""
    required_sections = [
        'data_source', 'preprocessing', 'feature_engineering',
        'model', 'evaluation', 'output'
    ]
    
    try:
        for section in required_sections:
            if not hasattr(config, section):
                raise ValueError(f"Missing required config section: {section}")
        return True
    except Exception as e:
        logger.error(f"Config validation failed: {str(e)}")
        raise