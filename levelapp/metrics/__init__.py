"""levelapp/metrics/__init__.py"""
from typing import List, Dict, Type, Any

from levelapp.aspects import logger
from levelapp.core.base import MetricProtocol
from levelapp.metrics.exact import EXACT_METRICS
from levelapp.metrics.fuzzy import FUZZY_METRICS
from levelapp.metrics.rag.retrieval import RAG_RETRIEVAL_METRICS


_metrics: Dict[str, Type[MetricProtocol]] = {}


class MetricRegistry:
    """Registry for metric classes."""

    @classmethod
    def register(cls, name: str, metric_class: Type[MetricProtocol]) -> None:
        """
        Register a metric class under a given name.

        Args:
            name (str): Unique identifier for the metric.
            metric_class (Type[MetricProtocol]): The metric class to register.
        """
        if name in _metrics:
            raise KeyError(f"Metric '{name}' is already registered")

        _metrics[name] = metric_class

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> MetricProtocol:
        """
        Retrieve an instance of a registered metric by its name.

        Args:
            name (str): The name of the metric to retrieve.

        Returns:
            Type[BaseMetric]: The metric class associated with the given name.

        Raises:
            KeyError: If the metric is not found.
        """
        if name not in _metrics:
            raise KeyError(f"Metric '{name}' is not registered")

        metric_class = _metrics[name]

        return metric_class[name](**kwargs)

    @classmethod
    def list_metrics(cls) -> List[str]:
        return list(_metrics.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        _metrics.pop(name, None)


METRICS = FUZZY_METRICS | EXACT_METRICS | RAG_RETRIEVAL_METRICS

for name_, metric_class_ in METRICS.items():
    try:
        MetricRegistry.register(name_, metric_class_)

    except Exception as e:
        logger.info(f"Failed to register metric {name_}: {e}")
