"""levelapp/metrics/__init__.py"""
import logging

from typing import List, Dict, Type

from levelapp.core.base import BaseMetric
from levelapp.metrics.exact import EXACT_METRICS
from levelapp.metrics.fuzzy import FUZZY_METRICS

logger = logging.getLogger(__name__)


class MetricRegistry:
    _metrics: Dict[str, Type[BaseMetric]] = {}

    @classmethod
    def register(cls, name: str, metric_class: Type[BaseMetric]) -> None:
        """
        Register a metric class under a given name.

        Args:
            name (str): Unique identifier for the metric.
            metric_class (Type[BaseMetric]): The metric class to register.
        """
        if not issubclass(metric_class, BaseMetric):
            raise TypeError(f"Metric '{name}' must be a subclass of BaseMetric")

        if name in cls._metrics:
            raise KeyError(f"Metric '{name}' is already registered")

        cls._metrics[name] = metric_class
        logger.info(f"Metric '{name}' registered successfully.")

    @classmethod
    def get(cls, name: str, **kwargs) -> Type[BaseMetric]:
        """
        Retrieve a registered metric class by its name.

        Args:
            name (str): The name of the metric to retrieve.

        Returns:
            Type[BaseMetric]: The metric class associated with the given name.

        Raises:
            KeyError: If the metric is not found.
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' is not registered")

        return cls._metrics[name](**kwargs)

    @classmethod
    def list_metrics(cls) -> List[str]:
        return list(cls._metrics.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._metrics.pop(name, None)


METRICS = FUZZY_METRICS | EXACT_METRICS

for name, metric_class in METRICS.items():
    try:
        MetricRegistry.register(name, metric_class)

    except Exception as e:
        logger.info(f"Failed to register metric {name}: {e}")
