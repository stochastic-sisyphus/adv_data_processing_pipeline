"""Enhanced error handling for the pipeline."""

import sys
import traceback
import logging
from typing import Type, Callable, Any, Optional
from functools import wraps
from contextlib import contextmanager
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Base class for pipeline errors."""
    pass

class DataValidationError(PipelineError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(PipelineError):
    """Raised when configuration validation fails."""
    pass

@contextmanager
def error_context(error_type: Type[Exception], message: str):
    """Context manager for handling specific types of errors."""
    try:
        yield
    except Exception as e:
        logger.error(f"{message}: {str(e)}")
        raise error_type(f"{message}: {str(e)}") from e

def handle_errors(func: Callable) -> Callable:
    """Decorator for consistent error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise PipelineError(f"Pipeline error in {func.__name__}: {str(e)}") from e
    return wrapper

def setup_logging(log_file: str = 'pipeline.log', log_level: int = logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def handle_error(error: Exception, step: str = 'Unknown') -> None:
    """Handle and log errors with more context."""
    error_type = type(error).__name__
    error_message = str(error)
    stack_trace = traceback.format_exc()
    
    logger.error(f"Error in step: {step}")
    logger.error(f"Error Type: {error_type}")
    logger.error(f"Error Message: {error_message}")
    logger.error(f"Stack Trace:\n{stack_trace}")

    # You can add custom error handling logic here, such as sending notifications or writing to a database

def log_step(step: str, message: str):
    """Log the start and end of each pipeline step."""
    logger.info(f"Starting step: {step}")
    logger.info(message)
    logger.info(f"Completed step: {step}")
