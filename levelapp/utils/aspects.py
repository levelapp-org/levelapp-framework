"""levelapp/utils.aspects.py"""
import logging
import time

from functools import wraps, lru_cache
from threading import Lock

from typing import Callable, Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS: Dict[str, Callable[[str, str], Any]] = {}
_metrics_lock = Lock()  # Thread-safe registration

def register_metric(
        name: str,
        cached: bool = False,
        maxsize: int | None = 128,
        typed: bool = False,
):
    """
    Decorator to register a metric function under a given name.

    Args:
        name: Unique identifier for the metric
        cached: Whether to enable LRU caching (default: False)
        maxsize: Maximum cache size (None for unlimited, default: 128)
        typed: Cache different argument types separately (default: False)
    """
    def decorator(func: Callable[[str, str], Any]):
        if func.__code__.co_argcount != 2:
            raise ValueError(f"Metric '{name}' must take exactly 2 arguments (generated, reference)")

        if cached:
            func = lru_cache(maxsize=maxsize, typed=typed)(func)

        @wraps(func)
        def wrapped(generated: str, reference: str) -> Any:
            start_time = time.perf_counter()
            result = func(generated, reference)
            logger.info(f"Executed metric '{name}' in {time.perf_counter() - start_time:.4f} seconds")
            return result

        with _metrics_lock:
            if name in METRICS:
                raise KeyError(f"Metric '{name}' already registered!")

            METRICS[name] = wrapped
            logger.info(f"Registered metric: {name}")

        return wrapped

    return decorator