"""levelapp/aspects.monitor.py"""
import time
import logging
import threading
import tracemalloc
import weakref
from contextlib import contextmanager

from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import List, Dict, Callable, Any, Union, ParamSpec, TypeVar, runtime_checkable, Protocol, Type

from threading import RLock
from functools import wraps, lru_cache
from datetime import datetime, timedelta
from humanize import precisedelta, naturalsize


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class MetricType(Enum):
    """Types of metrics that can be collected."""
    SETUP = "setup"
    DATA_LOADING = "data_loading"
    EXECUTION = "execution"
    RESULTS_COLLECTION = "results_collection"

    API_CALL = "api_call"
    SCORING = "scoring"
    CUSTOM = "custom"


@dataclass
class ExecutionMetrics:
    """Comprehensive metrics for a function execution."""
    procedure: str
    category: MetricType = MetricType.CUSTOM
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: float | None = None
    total_api_calls: int = 0
    memory_before: int | None = None
    memory_after: int | None = None
    memory_peak: int | None = None
    cache_hit: bool = False
    error: str | None = None

    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def update_duration(self, value: float) -> None:
        """Update duration with explicit value."""
        if value < 0:
            raise ValueError("Duration value cannot be negative.")
        self.duration = value

    def to_dict(self) -> dict:
        """Returns the content of the ExecutionMetrics as a structured dictionary."""
        metrics_dict = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)

            # Special handling for enum types to convert them to their value
            if isinstance(value, Enum):
                metrics_dict[field_info.name] = value.name
            elif isinstance(value, datetime):
                metrics_dict[field_info.name] = value.isoformat()
            else:
                metrics_dict[field_info.name] = value

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
    recent_call: datetime | None = None

    def update(self, metrics: ExecutionMetrics) -> None:
        """Update aggregated metrics with new execution metrics."""
        self.recent_call = datetime.now()
        self.total_calls += 1

        if metrics.duration is not None:
            self.total_duration += metrics.duration

            if self.min_duration == float('inf'):
                self.min_duration = metrics.duration
            else:
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

    def collect_before(self, collected_metrics: ExecutionMetrics) -> ExecutionMetrics:
        """Collect metrics before function execution."""
        ...

    def collect_after(self, collected_metrics: ExecutionMetrics) -> ExecutionMetrics:
        """Collect metrics after function execution."""
        ...


class MemoryTracker(MetricsCollector):
    """Memory usage metrics collector."""
    def __init__(self):
        self._tracking = False
        self._lock = threading.Lock()

    @contextmanager
    def _ensure_tracking(self):
        """Context manager to ensure tracemalloc is properly managed."""
        with self._lock:
            if not self._tracking:
                tracemalloc.start()
                self._tracking = True
        try:
            yield
        finally:
            pass

    def collect_before(self, collected_metrics: ExecutionMetrics) -> ExecutionMetrics:
        with self._ensure_tracking():
            try:
                current, _ = tracemalloc.get_traced_memory()
                collected_metrics.memory_before = current

            except Exception as e:
                logger.warning(f"[MemoryTracker] Memory tracking failed: {e}")

            return collected_metrics

    def collect_after(self, collected_metrics: ExecutionMetrics) -> ExecutionMetrics:
        if self._tracking:
            try:
                current, peak = tracemalloc.get_traced_memory()
                collected_metrics.memory_after = current
                collected_metrics.memory_peak = peak
                return collected_metrics

            except Exception as e:
                logger.warning(f"Memory tracking failed: {e}")
        return collected_metrics

    def cleanup(self):
        """Explicit cleanup method."""
        with self._lock:
            if self._tracking:
                try:
                    tracemalloc.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop tracemalloc: {e}")
                finally:
                    self._tracking = False


class APICallTracker(MetricsCollector):
    """API call metrics collector for LLM clients."""

    def __init__(self):
        self._api_calls = defaultdict(int)
        self._lock = threading.Lock()

    def collect_before(self, collected_metrics: ExecutionMetrics) -> ExecutionMetrics:
        return collected_metrics

    def collect_after(self, collected_metrics: ExecutionMetrics) -> ExecutionMetrics:
        with self._lock:
            if collected_metrics.category == MetricType.API_CALL:
                self._api_calls[collected_metrics.procedure] += 1
                collected_metrics.total_api_calls = self._api_calls[collected_metrics.procedure]

        return collected_metrics


class FunctionMonitor:
    """Core function monitoring system."""

    def __init__(self, max_history: int = 1000):
        self._lock = RLock()
        self._max_history = max_history
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
            history = self._execution_history.get(name, [])
            if not history:
                logger.warning(f"[FunctionMonitor] ne execution history found for proc: {name}")
                return

            latest_entry = history[-1]
            latest_entry.update_duration(value=value)
            self._aggregated_stats[name] = AggregatedStats()

            for entry in history:
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

    def _collect_metrics_before(self, execution_metrics: ExecutionMetrics) -> ExecutionMetrics:
        """Collect metrics before function execution using registered collectors."""
        result_metrics = execution_metrics
        for collector in self._collectors:
            try:
                result_metrics = collector.collect_before(result_metrics)
            except Exception as e:
                logger.warning(f"Metrics collector {type(collector).__name__} failed: {e}")

        return result_metrics

    def _collect_metrics_after(self, execution_metrics: ExecutionMetrics) -> ExecutionMetrics:
        """Collect metrics after function execution using registered collectors."""
        result_metrics = execution_metrics
        for collector in self._collectors:
            try:
                result_metrics = collector.collect_after(result_metrics)
            except Exception as e:
                logger.warning(f"Metrics collector {type(collector).__name__} failed: {e}")
        return result_metrics

    @staticmethod
    def _apply_caching(func: Callable[P, T], maxsize: int | None) -> Callable[P, T]:
        if maxsize is None:
            return func

        def make_args_hashable(args, kwargs):
            hashable_args = tuple(_make_hashable(a) for a in args)
            hashable_kwargs = tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items()))
            return hashable_args, hashable_kwargs

        cache = {}
        cache_info = {"hits": 0, "misses": 0}

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal cache_info
            cache_key = make_args_hashable(args, kwargs)

            if cache_key in cache:
                cache_info["hits"] += 1
                return cache[cache_key]

            cache_info["misses"] += 1
            result = func(*args, **kwargs)
            cache[cache_key] = result

            # Enforce maxsize
            if len(cache) > maxsize:
                cache.pop(next(iter(cache)))
            return result

        # Add cache management methods
        def get_cache_info():
            return dict(cache_info)

        def clear_cache():
            nonlocal cache, cache_info
            cache.clear()
            cache_info = {"hits": 0, "misses": 0}

        wrapper.cache_info = get_cache_info
        wrapper.cache_clear = clear_cache
        wrapper._cache = cache  # For debugging/inspection

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
            metrics = ExecutionMetrics(
                procedure=name,
                category=category,
            )

            if enable_timing:
                metrics.start_time = datetime.now()

            # Collect pre-execution metrics
            if track_memory:
                self._collect_metrics_before(execution_metrics=metrics)

            try:
                result = func(*args, **kwargs)

                cache_hit_info = getattr(func, 'cache_hit_info', None)
                if hasattr(func, 'cache_info') and cache_hit_info is not None:
                    metrics.cache_hit = getattr(cache_hit_info, 'is_hit', False)

                return result

            except Exception as e:
                metrics.error = str(e)
                logger.error(f"Error in '{name}': {str(e)}", exc_info=True)
                raise

            finally:
                if enable_timing:
                    metrics.end_time = datetime.now()
                    metrics.finalize()

                if track_memory:
                    metrics = self._collect_metrics_after(execution_metrics=metrics)

                # store metrics
                with self._lock:
                    history = self._execution_history[name]
                    history.append(metrics)

                    if len(history) > self._max_history:
                        history.pop(0)

                    self._aggregated_stats[name].update(metrics=metrics)

                if enable_timing and metrics.duration is not None:
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
            category: MetricType = MetricType.CUSTOM,
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

            min_duration = stats.min_duration if stats.min_duration != float('inf') else 0.0

            return {
                'name': name,
                'total_calls': stats.total_calls,
                'avg_duration': precisedelta(
                    timedelta(seconds=stats.average_duration),
                    suppress=['minutes'],
                    format='%.4f'
                ) if stats.average_duration >0 else "0.000s",
                'min_duration': precisedelta(
                    timedelta(seconds=min_duration),
                    suppress=['minutes'],
                    format='%.4f'
                ),
                'max_duration': precisedelta(
                    timedelta(seconds=stats.max_duration),
                    suppress=['minutes'],
                    format='%.4f'
                ),
                'error_rate': f"{stats.error_rate:.2f}%",
                'cache_hit_rate': f"{stats.cache_hit_rate}%",
                'memory_peak_mb': naturalsize(stats.memory_peak) if stats.memory_peak > 0 else "0 B",
                'last_called': stats.recent_call.isoformat() if stats.recent_call else None,
                'recent_execution': stats.recent_call.isoformat() if stats.recent_call else None,
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

    def cleanup(self):
        """Cleanup resources."""
        with self._lock:
            for collector in self._collectors:
                if hasattr(collector, 'cleanup'):
                    try:
                        collector.cleanup()
                    except Exception as e:
                        logger.warning(f"Collector cleanup failed: {e}")


MonitoringAspect = FunctionMonitor()


def _make_hashable(obj):
    """Convert potentially unhashable objects to a hashable representation."""
    if isinstance(obj, defaultdict):
        return 'defaultdict', _make_hashable(dict(obj))

    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))

    elif isinstance(obj, (list, set, tuple)):
        return tuple(_make_hashable(v) for v in obj)

    return obj
