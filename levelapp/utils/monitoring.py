"""levelapp/utils.monitoring.py"""
import logging
import inspect
import time

from typing import Dict, Callable, Any, ParamSpec, TypeVar

from threading import Lock
from functools import lru_cache, wraps, update_wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class FunctionMonitor:
    """Thread-safe function monitoring and registry."""
    _monitored_functions: Dict[str, Callable[..., Any]] = {}
    _lock = Lock()

    @classmethod
    def list_monitored_functions(cls) -> Dict[str, Callable[..., Any]]:
        return cls._monitored_functions

    @classmethod
    def _apply_caching(cls, func: Callable[P, T], maxsize: int | None) -> Callable[P, T]:
        """
        Apply LRU caching to a function and ensure cache methods are exposed.
        """
        cached_func = lru_cache(maxsize=maxsize)(func)

        # Create wrapper that preserves cache methods
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return cached_func(*args, **kwargs)

        # Copy cache methods to wrapper
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear

        return wrapper

    @classmethod
    def _wrap_execution(
            cls,
            func: Callable[P, T],
            name: str,
            enable_timing: bool
    ) -> Callable[P, T]:
        """
        Wrap function execution with timing and error handling.

        Args:
            func: Function to be wrapped
            name: Unique identifier for the function
            enable_timing: Enable execution time logging

        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                if enable_timing:
                    duration = time.perf_counter() - start_time
                    logger.info(f"Executed '{name}' in {duration:.4f}s")
                return result

            except Exception as e:
                logger.error(f"Error in '{name}': {str(e)}", exc_info=True)
                raise

        return wrapped

    @classmethod
    def monitor(
            cls,
            name: str,
            cached: bool = False,
            maxsize: int | None = 128,
            enable_timing: bool = True
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Decorator factory for monitoring functions.

        Args:
            name: Unique identifier for the function
            cached: Enable LRU caching
            maxsize: Maximum cache size
            enable_timing: Record execution time
        """
        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            wrapped_func = cls._apply_caching(func=func, maxsize=maxsize) if cached else func
            monitored_func = cls._wrap_execution(func=wrapped_func, name=name, enable_timing=enable_timing)

            with cls._lock:
                if name in cls._monitored_functions:
                    raise ValueError(f"Function '{name}' is already registered.")

                cls._monitored_functions[name] = monitored_func

            return monitored_func
        return decorator

    @classmethod
    def get_stats(cls, name: str) -> Dict[str, Any] | None:
        """
        Retrieve statistics for a registered function.

        Args:
            name: Name of the registered function

        Returns:
            Dict[str, Any]: Statistics including:
                - name: str
                - cache_info: Optional[CacheInfo]
                - is_cached: bool
        """
        if name not in cls._monitored_functions:
            print(f"Function '{name}' is not registered.")
            return None

        func = cls._monitored_functions[name]

        # Safely get the original function
        original_func = inspect.unwrap(func)

        stats = {
            'name': name,
            'is_cached': hasattr(func, 'cache_info'),
            'cache_info': func.cache_info() if hasattr(func, 'cache_info') else None
        }

        # Only try to get arg count if we can safely access it
        if hasattr(original_func, '__code__'):
            stats['execution_count'] = original_func.__code__.co_argcount

        return stats
