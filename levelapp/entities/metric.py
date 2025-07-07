"""levelapp/entities/metric.py"""
from typing import Callable, Any

from pydantic import BaseModel


class Metric(BaseModel):
    """Represents a metric for evaluation"""
    name: str
    compute: Callable[[Any, Any], float]
    description: str = ""
