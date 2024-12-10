
"""Performance monitoring and profiling utilities."""

import time
import logging
import psutil
from typing import Any, Callable, Dict, Optional
from functools import wraps
from contextlib import contextmanager
import tracemalloc
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    peak_memory: float

@contextmanager
def performance_monitor():
    """Context manager for monitoring performance metrics."""
    start_time = time.time()
    tracemalloc.start()
    process = psutil.Process()
    
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage=process.memory_info().rss / 1024 / 1024,  # MB
            cpu_usage=process.cpu_percent(),
            peak_memory=peak / 1024 / 1024  # MB
        )
        
        logger.info(f"Performance metrics: {metrics}")

def monitor_performance(func: Callable) -> Callable:
    """Decorator for monitoring function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with performance_monitor():
            result = func(*args, **kwargs)
        return result
    return wrapper