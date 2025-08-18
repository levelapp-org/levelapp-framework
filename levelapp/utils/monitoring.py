"""levelapp/utils.monitoring.py"""
import logging
import threading
import time
import tracemalloc
from collections import defaultdict

from enum import Enum
from dataclasses import dataclass, fields
from typing import List, Dict, Callable, Any, Union, ParamSpec, TypeVar, runtime_checkable, Protocol, Type

from datetime import datetime
from threading import RLock
from functools import lru_cache, wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class MetricType(Enum):
    """Types of metrics that can be collected."""
    API_CALL = "api_call"
    SCORING = "scoring"
    CUSTOM = "custom"


@dataclass
class ExecutionMetrics:
    """Comprehensive metrics for a function execution."""
    procedure: str
    category: MetricType = MetricType.CUSTOM
    start_time: float | None = None
    end_time: float | None = None
    duration: float | None = None
    memory_before: int | None = None
    memory_after: int | None = None
    memory_peak: int | None = None
    cache_hit: bool = False
    error: str | None = None

    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time

    def update_duration(self, value: float) -> None:
        self.duration = value

    def to_dict(self) -> dict:
        """Returns the content of the ExecutionMetrics as a structured dictionary."""
        metrics_dict = {}
        for field in fields(self):
            value = getattr(self, field.name)

            # Special handling for enum types to convert them to their value
            if isinstance(value, Enum):
                metrics_dict[field.name] = value.name
            else:
                metrics_dict[field.name] = value

        return metrics_dict


@dataclass
class AggregatedStats:
    """Aggregated metrics for monitored functions."""
    total_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    memory_peak: int = 0
    last_called: datetime | None = None

    def update(self, metrics: ExecutionMetrics) -> None:
        """Update aggregated metrics with new execution metrics."""
        self.total_calls += 1
        self.last_called = datetime.now()

        if metrics.duration is not None:
            self.total_duration += metrics.duration
            self.min_duration = min(self.min_duration, metrics.duration)
            self.max_duration = max(self.max_duration, metrics.duration)

        if metrics.error:
            self.error_count += 1

        if metrics.cache_hit:
            self.cache_hits += 1

        if metrics.memory_peak:
            self.memory_peak = max(self.memory_peak, metrics.memory_peak)

    @property
    def average_duration(self) -> float:
        """Average execution duration."""
        return (self.total_duration / self.total_calls) if self.total_calls > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        return (self.cache_hits / self.total_calls * 100) if self.total_calls > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as a percentage."""
        return (self.error_count / self.total_calls * 100) if self.total_calls > 0 else 0.0


@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for custom metrics collectors."""
    def collect_before(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics before function execution."""
        ...

    def collect_after(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics after function execution."""
        ...


class MemoryTracker(MetricsCollector):
    """Memory usage metrics collector."""
    def __init__(self):
        self._tracking = False

    def collect_before(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self._tracking:
            tracemalloc.start()
            self._tracking = True

        current, peak = tracemalloc.get_traced_memory()
        return {"memory_before": current / 10**6, "memory_peak": peak / 10**6}

    def collect_after(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if self._tracking:
            current, peak = tracemalloc.get_traced_memory()
            return {
                "memory_after": current / 10**6,
                "memory_peak": peak / 10**6,
            }
        return {}

    def __del__(self):
        if self._tracking:
            tracemalloc.stop()


class APICallTracker(MetricsCollector):
    """API call metrics collector for LLM clients."""

    def __init__(self):
        self._api_calls = defaultdict(int)
        self._lock = threading.Lock()

    def collect_before(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {"api_calls_history": dict(self._api_calls)}

    def collect_after(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            api_calls = 0
            if metadata['category'] == MetricType.API_CALL:
                api_calls += 1

            if api_calls > 0:
                func_name = metadata.get('procedure', 'unknown')
                self._api_calls[func_name] += api_calls

        return {"api_calls_detected": api_calls, "total_api_calls": dict(self._api_calls)}


class FunctionMonitor:
    """Core function monitoring system."""

    def __init__(self, max_history: int = 1000):
        self._lock = RLock()
        self._monitored_procedures: Dict[str, Callable[..., Any]] = {}
        self._execution_history: Dict[str, List[ExecutionMetrics]] = defaultdict(list)
        self._aggregated_stats: Dict[str, AggregatedStats] = defaultdict(AggregatedStats)
        self._collectors: List[MetricsCollector] = []
        self.add_collector(MemoryTracker())
        self.add_collector(APICallTracker())

    def update_procedure_duration(self, name: str, value: float) -> None:
        """
        Update the duration of a monitored procedure by name.

        Args:
            name: The name of the procedure to retrieve.
            value: The value to retrieve for the procedure.
        """
        with self._lock:
            history = self._execution_history[name]
            if not history:
                return

            for entry in history:
                entry.update_duration(value=value)
                self._aggregated_stats[name].update(metrics=entry)

    def add_collector(self, collector: MetricsCollector) -> None:
        """
        Add a custom metrics collector to the monitor.

        Args:
            collector: An instance of a class implementing MetricsCollector protocol.
        """
        with self._lock:
            self._collectors.append(collector)

    def remove_collector(self, collector: MetricsCollector) -> None:
        """
        Remove a custom metrics collector from the monitor.

        Args:
            collector: An instance of a class implementing MetricsCollector protocol.
        """
        with self._lock:
            if collector in self._collectors:
                self._collectors.remove(collector)

    def _collect_metrics_before(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect metrics before function execution using registered collectors.
        """
        metrics = {}
        for collector in self._collectors:
            try:
                metrics.update(collector.collect_before(metadata=metadata))
            except Exception as e:
                logger.warning(f"[FunctionMonitor] Metrics collector failed: {e}")

        return metrics

    def _collect_metrics_after(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect metrics after function execution using registered collectors.
        """
        metrics = {}
        for collector in self._collectors:
            try:
                metrics.update(collector.collect_after(metadata=metadata))
            except Exception as e:
                logger.warning(f"[FunctionMonitor] Metrics collector failed: {e}")

        return metrics

    @staticmethod
    def _apply_caching(func: Callable[P, T], maxsize: int | None) -> Callable[P, T]:
        if maxsize is None:
            return func

        def make_args_hashable(args, kwargs):
            hashable_args = tuple(_make_hashable(a) for a in args)
            hashable_kwargs = tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items()))
            return hashable_args, hashable_kwargs

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Build a hashable cache key without altering the original args
            cache_key = make_args_hashable(args, kwargs)

            if not hasattr(wrapper, "_cache"):
                wrapper._cache = {}
                wrapper._cache_info = {"hits": 0, "misses": 0}

            if cache_key in wrapper._cache:
                wrapper._cache_info["hits"] += 1
                return wrapper._cache[cache_key]

            wrapper._cache_info["misses"] += 1
            result = func(*args, **kwargs)  # pass ORIGINAL args
            wrapper._cache[cache_key] = result
            return result

        wrapper.cache_info = wrapper._cache_info
        wrapper.cache_clear = wrapper._cache.clear()
        return wrapper

    def _wrap_execution(
            self,
            func: Callable[P, T],
            name: str,
            category: MetricType,
            enable_timing: bool,
            track_memory: bool,
    ) -> Callable[P, T]:
        """
        Wrap function execution with timing and error handling.

        Args:
            func: Function to be wrapped
            name: Unique identifier for the function
            enable_timing: Enable execution time logging
            track_memory: Enable memory tracking

        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            # Initialize execution metadata
            exec_metadata = {'procedure': name, 'category': category}

            # Initialize execution metrics
            metrics = ExecutionMetrics(
                procedure=name,
                category=category,
            )

            # Collect pre-execution metrics
            if track_memory and self._collectors:
                # TODO-0: I don't like this, but it works for now.
                pre_metrics = self._collect_metrics_before(metadata=exec_metadata)
                metrics.memory_before = pre_metrics.get('memory_before')

            if enable_timing:
                metrics.start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)

                # Check for cache hit
                cache_hit_info = getattr(func, 'cache_hit_info', None)
                if hasattr(func, 'cache_info') and cache_hit_info is not None:
                    metrics.cache_hit = getattr(cache_hit_info, 'is_hit', False)

                # Collect post-execution metrics
                if track_memory and self._collectors:
                    post_metrics = self._collect_metrics_after(metadata=exec_metadata)
                    metrics.memory_after = post_metrics.get('memory_after')
                    metrics.memory_peak = post_metrics.get('memory_peak')

                return result

            except Exception as e:
                metrics.error = str(e)
                logger.error(f"Error in '{name}': {str(e)}", exc_info=True)
                raise

            finally:
                metrics.end_time = time.perf_counter()
                metrics.finalize()

                # store metrics
                with self._lock:
                    self._execution_history[name].append(metrics)
                    self._aggregated_stats[name].update(metrics)

                if enable_timing and metrics.duration:
                    log_message = f"[FunctionMonitor] Executed '{name}' in {metrics.duration:.4f}s"
                    if metrics.cache_hit:
                        log_message += " (cache hit)"
                    if metrics.memory_peak:
                        log_message += f" (memory peak: {metrics.memory_peak / 1024 / 1024:.2f} MB)"
                    logger.info(log_message)

        return wrapped

    def monitor(
            self,
            name: str,
            category: MetricType,
            cached: bool = False,
            maxsize: int | None = 128,
            enable_timing: bool = True,
            track_memory: bool = True,
            collectors: List[Type[MetricsCollector]] | None = None
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Decorator factory for monitoring functions.

        Args:
            name: Unique identifier for the function
            category: Category of the metric (e.g., API_CALL, SCORING)
            cached: Enable LRU caching
            maxsize: Maximum cache size
            enable_timing: Record execution time
            track_memory: Track memory usage
            collectors: Optional list of custom metrics collectors

        Returns:
            Callable[[Callable[P, T]], Callable[P, T]]: Decorator function
        """
        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            if collectors:
                for collector in collectors:
                    self.add_collector(collector)

            if cached:
                func = self._apply_caching(func=func, maxsize=maxsize)

            monitored_func = self._wrap_execution(
                func=func,
                name=name,
                category=category,
                enable_timing=enable_timing,
                track_memory=track_memory,
            )

            with self._lock:
                if name in self._monitored_procedures:
                    raise ValueError(f"Function '{name}' is already registered.")

                self._monitored_procedures[name] = monitored_func

            return monitored_func

        return decorator

    def list_monitored_functions(self) -> Dict[str, Callable[..., Any]]:
        """
        List all registered monitored functions.

        Returns:
            List[str]: Names of all registered functions
        """
        with self._lock:
            return dict(self._monitored_procedures)

    def get_stats(self, name: str) -> Dict[str, Any] | None:
        """
        Get comprehensive statistics for a monitored function.

        Args:
            name (str): Name of the monitored function.

        Returns:
            Dict[str, Any] | None: Dictionary containing function statistics or None if not found.
        """
        with self._lock:
            if name not in self._monitored_procedures:
                return None

            func = self._monitored_procedures[name]
            stats = self._aggregated_stats[name]
            history = self._execution_history[name]

            return {
                'name': name,
                'total_calls': stats.total_calls,
                'avg_duration': stats.average_duration,
                'min_duration': stats.min_duration if stats.min_duration != float('inf') else 0,
                'max_duration': stats.max_duration,
                'error_rate': stats.error_rate,
                'cache_hit_rate': stats.cache_hit_rate if hasattr(func, 'cache_info') else None,
                'memory_peak_mb': stats.memory_peak / 1024 / 1024 if stats.memory_peak else 0,
                'last_called': stats.last_called.isoformat() if stats.last_called else None,
                'recent_executions': len(history),
                'is_cached': hasattr(func, 'cache_info'),
                'cache_info': func.cache_info() if hasattr(func, 'cache_info') else None
            }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all monitored functions."""
        with self._lock:
            return {
                name: self.get_stats(name)
                for name in self._monitored_procedures.keys()
            }

    def get_execution_history(
            self,
            name: str | None = None,
            category: MetricType | None = None,
            limit: int | None = None
    ) -> list[ExecutionMetrics]:
        """Get execution history filtered by procedure name or category."""
        with self._lock:
            if name:
                history = self._execution_history.get(name, [])
            else:
                history = [m for h in self._execution_history.values() for m in h]

            if category:
                history = [m for m in history if m.category == category]

            history.sort(key=lambda m: m.start_time or 0)
            return history[-limit:] if limit else history

    def clear_history(self, procedure: str | None = None) -> None:
        """Clear execution history."""
        with self._lock:
            if procedure:
                if procedure in self._execution_history:
                    self._execution_history[procedure].clear()
                if procedure in self._aggregated_stats:
                    self._aggregated_stats[procedure] = AggregatedStats()
            else:
                self._execution_history.clear()
                self._aggregated_stats.clear()

    def export_metrics(self, output_format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        Export all metrics in various formats.

        Args:
            output_format (str): Format for exporting metrics ('dict' or 'json').

        Returns:
            Union[Dict[str, Any], str]: Exported metrics in the specified format.
        """
        with self._lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'functions': self.get_all_stats(),
                'total_executions': sum(
                    len(history) for history in self._execution_history.values()
                ),
                'collectors': [type(c).__name__ for c in self._collectors]
            }

        if output_format == 'dict':
            return data
        elif output_format == 'json':
            import json
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {output_format}")


MonitoringAspect = FunctionMonitor()


# Global monitoring functions for backward compatibility.
def monitor(name: str, **kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to monitor function execution with global FunctionMonitor.

    Args:
        name: Unique identifier for the function
        **kwargs: Additional parameters for FunctionMonitor

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: Decorator function
    """
    return MonitoringAspect.monitor(name=name, **kwargs)


def get_stats(name: str) -> Dict[str, Any] | None:
    """
    Get statistics for a monitored function.

    Args:
        name (str): Name of the monitored function.

    Returns:
        Dict[str, Any] | None: Function statistics or None if not found.
    """
    return MonitoringAspect.get_stats(name=name)


def list_monitored_functions() -> Dict[str, Callable[..., Any]]:
    """
    List all monitored functions.

    Returns:
        Dict[str, Callable[..., Any]]: Dictionary of monitored function names and their callable objects.
    """
    return MonitoringAspect.list_monitored_functions()


def clear_history(procedure: str | None = None) -> None:
    """
    Clear execution history.

    Args:
        procedure (str | None): Name of the function to clear history for.
    """
    return MonitoringAspect.clear_history(procedure=procedure)


def export_metrics(output_format: str = 'dict') -> Union[Dict[str, Any], str]:
    """
    Export all metrics.

    Args:
        output_format (str): Format for exporting metrics ('dict' or 'json').

    Returns:
        Union[Dict[str, Any], str]: Exported metrics in the specified format.
    """
    return MonitoringAspect.export_metrics(output_format)


# Convenience decorators
def monitor_with_cache(name: str, maxsize: int = 128, **kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Monitor with caching enabled.

    Args:
        name (str): Unique identifier for the function.
        maxsize (int): Maximum size of the cache.
        **kwargs: Additional parameters for FunctionMonitor.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: Decorator function with caching enabled.
    """
    return monitor(name, cached=True, maxsize=maxsize, **kwargs)


def monitor_memory(name: str, **kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Monitor with memory tracking.

    Args:
        name (str): Unique identifier for the function.
        **kwargs: Additional parameters for FunctionMonitor.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: Decorator function with memory
    """
    return monitor(name, track_memory=True, **kwargs)


def monitor_api_calls(name: str, **kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Monitor API calls (includes API call tracker by default).

    Args:
        name (str): Unique identifier for the function.
        **kwargs: Additional parameters for FunctionMonitor.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: Decorator function with API call tracking.
    """
    return monitor(name, track_memory=True, enable_timing=True, **kwargs)


def _make_hashable(obj):
    """Convert potentially unhashable objects to a hashable representation."""
    if isinstance(obj, defaultdict):
        return 'defaultdict', _make_hashable(dict(obj))

    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))

    elif isinstance(obj, (list, set, tuple)):
        return tuple(_make_hashable(v) for v in obj)

    return obj
