"""levelapp/utils.monitoring.py"""
import logging
import time

from typing import Dict, Callable, Any, ParamSpec, TypeVar

from threading import Lock
from functools import lru_cache, wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class FunctionMonitor:
    """Thread-safe function monitoring and registry."""
    _monitored_functions: Dict[str, Callable[..., Any]] = {}
    _Lock = Lock()

    @classmethod
    def register(
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
            if cached:
                cached_func = lru_cache(maxsize=maxsize)(func)
                wrapped_func = cached_func
            else:
                wrapped_func = func

            @wraps(func)
            def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
                start_time = time.perf_counter()
                try:
                    result = wrapped_func(*args, **kwargs)
                    if enable_timing:
                        if enable_timing:
                            duration = time.perf_counter() - start_time
                            logger.info(f"Executed '{name}' in {duration:.4f}s")

                    return result

                except Exception as e:
                    logger.error(f"Error in '{name}': {str(e)}", exc_info=True)
                    raise

            if cached:
                wrapped.cache_clear = cached_func.cache_clear
                wrapped.cache_info = cached_func.cache_info

            with cls._Lock:
                if name in cls._monitored_functions:
                    raise ValueError(f"Function '{name}' is already registered.")

                cls._monitored_functions[name] = wrapped
                logger.debug(f"Registered function '{name}'")

            return wrapped
        return decorator

    @classmethod
    def get_stats(cls, name: str) -> Dict[str, Any] | None:
        """
        Retrieve statistics for a registered function.

        Args:
            name: Name of the registered function

        Returns:
            Dict[str, Any]: Statistics including execution count and cache info
        """
        if name not in cls._monitored_functions:
            return None

        func = cls._monitored_functions[name]
        stats = {
            'name': name,
            'execution_count': func.__wrapped__.__code__.co_argcount,
            'cache_info': func.cache_info() if hasattr(func, 'cache_info') else None
        }
        return stats