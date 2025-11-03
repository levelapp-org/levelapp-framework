"""levelapp/assessor/strategy.pu"""
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any

from levelapp.assessor.schemas import Document


# TODO-0: Maybe we can place this in "core/base.py"?
class BaseStrategy(ABC):
    """Base class for all RAG strategies."""
    name: str
    config: Dict[str, Any]

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Execute the strategy logic."""
        pass


class ChunkingStrategy(BaseStrategy):
    """Chunking strategy."""
    async def run(self, chunks: List[str]) -> List[List[float]]:
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
        self._registry: Dict[str, Dict[str, Type[ChunkingStrategy]]] = {
            "chunking": {},
            "embedding": {},
            "retrieval": {},
            "generation": {},
        }