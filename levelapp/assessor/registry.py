"""levelapp/assessor/strategy.pu"""
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any

from levelapp.assessor.schemas import Document


# TODO-0: Maybe we can place this in "core/base.py"?
class BaseStrategy(ABC):
    """Base class for all RAG strategies."""
    name: str
    config: Dict[str, Any]

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Execute the strategy logic."""
        pass


class ChunkingStrategy(BaseStrategy):
    """Chunking strategy."""
    async def run(self, document: Document) -> List[List[float]]:
        """Generate embeddings for chunks."""
        pass


class RetrievalStrategy(BaseStrategy):
    """Retrieval strategy."""
    @abstractmethod
    async def run(self, query: str, embedded_chunks: List[Dict]) -> List[Document]:
        """Retrieve most relevant chunks given query embeddings."""
        pass


class GenerationStrategy(BaseStrategy):
    """Generation strategy."""
    @abstractmethod
    async def run(self, query: str, retrieved_docs: List[Document]) -> str:
        """Generate augmented answer from retrieved context."""
        pass


class StrategyRegistry:
    """Maintains available strategy classes for each type."""
    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Type[BaseStrategy]]] = {
            "chunking": {},
            "embedding": {},
            "retrieval": {},
            "generation": {},
        }

    def register(self, strategy_type: str, name: str, cls: Type[BaseStrategy]) -> None:
        self._registry[strategy_type][name] = cls

    def get(self, strategy_type: str, name: str) -> Type[BaseStrategy]:
        return self._registry[strategy_type][name]

    def list_strategies(self) -> Dict[str, List[str]]:
        return {stype: list(names.keys()) for stype, names in self._registry.items()}
