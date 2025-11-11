"""levelapp/assessor/strategy.pu"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any

from levelapp.assessor.schemas import Document, MetricSpec


# TODO-0: Maybe we can place this in "core/base.py"?
class BaseStrategy(ABC):
    """Base class for all RAG strategies."""
    name: str
    level: str
    config: Dict[str, Any]

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Execute the strategy logic."""
        pass


class ChunkingStrategy(BaseStrategy):
    """Chunking strategy."""
    level = "chunking"

    async def run(self, document: Document) -> List[List[float]]:
        """Generate embeddings for chunks."""
        pass


class EmbeddingStrategy(BaseStrategy):
    """Embedding strategy."""
    level = "embedding"

    async def run(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        pass


class RetrievalStrategy(BaseStrategy):
    """Retrieval strategy."""
    level = "retrieval"

    @abstractmethod
    async def run(self, query: str, embedded_chunks: List[Dict]) -> List[Document]:
        """Retrieve most relevant chunks given query embeddings."""
        pass


class GenerationStrategy(BaseStrategy):
    """Generation strategy."""
    level = "generation"

    @abstractmethod
    async def run(self, query: str, retrieved_docs: List[Document]) -> str:
        """Generate augmented answer from retrieved context."""
        pass


class StrategyRegistry:
    """Maintains available strategy classes for each type."""
    def __init__(self) -> None:
        self.registry: Dict[str, Dict[str, Type[BaseStrategy]]] = {
            "chunking": {},
            "embedding": {},
            "retrieval": {},
            "generation": {},
        }

        self.metric_map: Dict[str, Dict[str, List[MetricSpec]]] = {
            "chunking": {},
            "embedding": {},
            "retrieval": {},
            "generation": {},
        }

    def register_strategy(
            self,
            level: str,
            name: str,
            strategy: Type[BaseStrategy],
            metrics: List[MetricSpec] | None = None,
    ) -> None:
        """Register a new strategy implementation and its metrics."""
        if level not in self.registry:
            raise ValueError(f"[StrategyRegistry] Invalid strategy level '{level}'")

        self.registry[level][name] = strategy

        if metrics:
            self.metric_map[level][name] = metrics

    def get_strategy(self, level: str, name: str, **kwargs) -> BaseStrategy:
        """Instantiate and return a strategy by its level and name."""
        try:
            strategy_cls = self.registry[level][name]
            return strategy_cls(name=name, config=kwargs.get("config"))

        except KeyError:
            raise KeyError(f"[StrategyRegistry] Strategy '{name}' not found under level '{level}'")

    def list_strategies(self, level: str | None) -> Dict[str, List[str]]:
        """Return the metrics associated with a given strategy."""
        if level:
            return {level: list(self.registry[level].keys())}

        return {lvl: list(names.keys()) for lvl, names in self.registry.items()}

    def resolve_metrics(self, level: str, name: str) -> List[MetricSpec]:
        """Return the metrics associated with a given strategy."""
        try:
            return self.metric_map[level][name]

        except KeyError:
            return []

    def clear(self) -> None:
        """Clear all registered strategies and metrics (useful for testing)."""
        for level in self.registry:
            self.registry[level].clear()
            self.metric_map[level].clear()