"""levelapp/core/base.py"""
from abc import ABC, abstractmethod


class Simulator(ABC):
    """Abstract base for simulator components."""
    @abstractmethod
    def simulate(self):
        """Run a stress test simulation based on the provided configuration."""


class Comparator(ABC):
    """Abstract base class for comparator components."""
    @abstractmethod
    def evaluate(self):
        """Evaluate system output against expected output."""
        pass
      