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

class MetricsManager:
    """Manager for logging detailed performance metrics."""

    def log_step_performance(self, step_name: str, metrics: PerformanceMetrics):
        """Log detailed step performance metrics."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Step: {step_name}")
        logger.info(f"Execution Time: {metrics.execution_time:.2f}s")
        logger.info(f"Memory Usage: {metrics.memory_usage:.2f}MB")
        logger.info(f"CPU Usage: {metrics.cpu_usage:.1f}%")
        logger.info(f"Peak Memory: {metrics.peak_memory:.2f}MB")
        logger.info(f"{'='*50}\n")

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

@contextmanager
def step_performance_monitor(step_name: str):
    """Enhanced context manager for monitoring step performance."""
    start_time = time.time()
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        final_memory = process.memory_info().rss
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage=(final_memory - initial_memory) / 1024 / 1024,
            cpu_usage=process.cpu_percent(),
            peak_memory=peak / 1024 / 1024
        )
        
        MetricsManager().log_step_performance(step_name, metrics)

def monitor_performance(func: Callable) -> Callable:
    """Decorator for monitoring function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with performance_monitor():
            result = func(*args, **kwargs)
        return result
    return wrapper