"""levelapp/assessor/strategy.pu"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, ValidationError
from typing import List, Dict, Type, Any, TypeVar

from levelapp.assessor.schemas import Document, MetricSpec
from levelapp.aspects.logger import logger


TConfig = TypeVar("TConfig", bound=BaseModel)


@dataclass(frozen=True)
class StrategyInfo:
    """Metadata about a registered strategy."""
    name: str
    level: str
    strategy_class: Type[BaseStrategy]
    description: str = ""
    config_schema: Type[BaseModel] | None = None
    metrics: List[MetricSpec] | None = None


# TODO-0: Maybe we can place this in "core/base.py"?
class BaseStrategy(ABC):
    """Base class for all RAG strategies with proper lifecycle management."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize strategy resources (models, connections, etc.)"""
        if not self._initialized:
            await self._initialize()
            self._initialized = True

    async def _initialize(self) -> None:
        """Strategy-specific initialization. Override in subclasses."""
        pass

    async def cleanup(self) -> None:
        """clean up strategy resources."""
        if self._initialized:
            await self._cleanup()
            self._initialized = False

    async def _cleanup(self) -> None:
        """Strategy-specific cleanup. Override in subclasses."""
        pass

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Execute the strategy logic."""
        pass

    @property
    @abstractmethod
    def level(self) -> str:
        """Return strategy level."""
        pass

    @property
    def name(self) -> str:
        """Return the strategy name derived from class name."""
        return self.__class__.__name__


class ChunkingStrategy(BaseStrategy):
    """Chunking strategy contract."""
    level = "chunking"

    @abstractmethod
    async def run(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents (List[Document]): List of input documents to chunk.

        Returns:
            List of chunked documents with metadata.
        """
        pass


class EmbeddingStrategy(BaseStrategy):
    """Embedding strategy contract."""
    level = "embedding"

    @abstractmethod
    async def run(self, chunks: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks (List[Document]): List of document chunks to embed.

        Returns:
            List of embedding vectors for each chunk.
        """
        pass


class GenerationStrategy(BaseStrategy):
    """Generation strategy contract."""
    level = "generation"

    @abstractmethod
    async def run(self, query: str, context: List[Document]) -> str:
        """
        Generate answer using query and context.

        Args:
            query (str): Input query string.
            context (List[Document]): List of relevant documents for context.

        Returns:
            Generated answer string.
        """
        pass


class RetrievalStrategy(BaseStrategy):
    """Retrieval strategy contract."""
    level = "retrieval"

    @abstractmethod
    async def run(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve most relevant documents for a query.

        Args:
            query (str): Input query string.
            top_k (int): Number of top results to retrieve.

        Returns:
            List of retrieved documents sorted by relevance.
        """
        pass


class StrategyRegistry:
    """Maintains available strategy classes for each type."""
    def __init__(self) -> None:
        self._strategies: Dict[str, Dict[str, StrategyInfo]] = {
            "chunking": {},
            "embedding": {},
            "retrieval": {},
            "generation": {},
        }

    def register_strategy(
            self,
            strategy_class: Type[BaseStrategy],
            name: str,
            metrics: List[MetricSpec] | None = None,
            description: str = "",
            config_schema: Type[BaseModel] | None = None,
    ) -> None:
        """
        Register a new strategy implementation with metadata.

        Args:
            strategy_class (Type[BaseStrategy]): The strategy class to register.
            name (str): Unique name for this strategy.
            metrics (List[MetricSpec]): Optional list of metrics for evaluation.
            description (str): Optional description for this strategy.
            config_schema (Type[BaseModel] | None): Pydantic model for configuration validation.

        Raises:
            ValueError: If strategy level is invalid or name is already registered.
        """
        level = getattr(strategy_class, "level", None)
        if level not in self._strategies:
            raise ValueError(f"[StrategyRegistry] Invalid strategy level '{level}. Must be one of {self._strategies.keys()}'")

        if name in self._strategies[level]:
            raise ValueError(f"[StrategyRegistry] Strategy '{name}' is already registered for level '{level}'.")

        # Validate strategy class?
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"[StrategyRegistry] Strategy class must inherit from 'BaseStrategy'.")

        strategy_info = StrategyInfo(
            name=name,
            level=level,
            strategy_class=strategy_class,
            description=description,
            config_schema=config_schema,
            metrics=metrics or [],
        )

        self._strategies[level][name] = strategy_info
        logger.info(f"[StrategyRegistry] Registered strategy: {level}.{name}")

    def get_strategy(
            self,
            level: str,
            name: str,
            config: Dict[str, Any] | None = None,
    ) -> BaseStrategy:
        """
        Instantiate and return a strategy with configuration.

        Args:
            level (str): The strategy level (chunking, embedding, etc.).
            name (str): The strategy name.
            config (Dict[str, Any] | None): Optional configuration dictionary.

        Returns:
            Strategy instance.

        Raises:
            KeyError: If strategy not found.
            ValidationError: If configuration validation fails.
        """
        strategy_info = self._get_strategy_info(level=level, name=name)

        # Validate config against schema if provided
        if config and strategy_info.config_schema:
            try:
                validated_config = strategy_info.config_schema.model_validate(config)

            except ValidationError as e:
                logger.info(f"[StrategyRegistry] Invalid configuration for level '{level}.{name}'.\n{e}")
                validated_config = {}

        else:
            validated_config = config or {}

        # Instantiate strategy
        strategy = strategy_info.strategy_class(config=validated_config)

        return strategy

    def _get_strategy_info(self, level: str, name: str) -> StrategyInfo:
        """
        Get StrategyInfo instance for given level and name.

        Args:
            level (str): The strategy level (chunking, embedding, etc.).
            name (str): The strategy name.

        Returns:
            StrategyInfo instance.

        Raises:
            KeyError: If strategy not found.
        """
        try:
            return self._strategies[level][name]

        except KeyError:
            available = list(self._strategies[level].keys())
            raise KeyError(
                f"[StrategyRegistry] Strategy '{name}' not found for level '{level}'."
                f"Available strategies: {available}"
            )

    def get_strategy_info(self, level: str, name: str) -> StrategyInfo:
        """Get strategy metadata without instantiation."""
        return self._get_strategy_info(level=level, name=name)

    def list_strategies(self, level: str | None = None) -> Dict[str, List[str]]:
        """
        List all available strategies, optionally filtered by level.

        Args:
            level (str): The strategy level (chunking, embedding, etc.).

        Returns:
            Dict[str, List[str]] List of available strategies.

        Raises:
            ValueError: If strategy level is invalid or name is already registered.
        """
        if level:
            if level not in self._strategies:
                raise ValueError(f"[StrategyRegistry] Invalid leve: {level}.")

            return {level: list(self._strategies[level].keys())}

        return {
            level: list(strategies.keys())
            for level, strategies in self._strategies.items()
            if strategies  # Only include levels with strategies otherwise it won't make sense.
        }

    def resolve_metrics(self, level: str, name: str) -> List[MetricSpec]:
        """
        Get metrics associated with a strategy.

        Args:
            level (str): The strategy level (chunking, embedding, etc.).
            name (str): The strategy name.

        Returns:
            List of MetricSpec object, empty list if none found.

        Raises:
            KeyError if strategy level is invalid or name is already registered.
        """
        try:
            strategy_info = self._get_strategy_info(level=level, name=name)
            return strategy_info.metrics

        except KeyError:
            return []

    def strategy_exists(self, level: str, name: str) -> bool:
        """Check if a strategy is registered."""
        try:
            self._get_strategy_info(level, name)
            return True
        except KeyError:
            return False

    def unregister_strategy(self, level: str, name: str) -> None:
        """Unregister a strategy."""
        try:
            del self._strategies[level][name]
        except KeyError:
            raise KeyError(f"Strategy '{name}' not found for level '{level}'")

    def clear(self) -> None:
        """Clear all registered strategies (primarily for testing)."""
        for level in self._strategies:
            self._strategies[level].clear()


if __name__ == '__main__':
    # Example strategy implementations for demonstration
    class SemanticSplitterStrategy(ChunkingStrategy):
        """Example chunking strategy using semantic splitting."""

        async def _initialize(self) -> None:
            # Load models, initialize components
            print("Initializing semantic splitter...")

        async def run(self, documents: List[Document]) -> List[Document]:
            # Implementation would go here
            print(f"Semantic splitting {len(documents)} documents")
            return documents  # Simplified


    class OpenAIEmbeddingStrategy(EmbeddingStrategy):
        """Example embedding strategy using OpenAI."""

        async def run(self, chunks: List[Document]) -> List[List[float]]:
            print(f"Generating embeddings for {len(chunks)} chunks")
            return [[0.1, 0.2] for _ in chunks]  # Simplified


    # Demonstration
    registry = StrategyRegistry()

    # Register strategies with metadata
    registry.register_strategy(
        strategy_class=SemanticSplitterStrategy,
        name="semantic_splitter",
        description="Splits documents based on semantic boundaries",
        metrics=[MetricSpec(name="chunk_quality", weight=1.0)]
    )

    registry.register_strategy(
        strategy_class=OpenAIEmbeddingStrategy,
        name="openai_embedding",
        description="Uses OpenAI's text-embedding-ada-002 model",
        metrics=[
            MetricSpec(name="embedding_similarity", weight=1.0),
            MetricSpec(name="latency", weight=0.5)
        ]
    )

    # List available strategies
    print("Available strategies:", registry.list_strategies())

    # Get and use a strategy
    chunker = registry.get_strategy("chunking", "semantic_splitter")
    print(f"Got strategy: {chunker.name} for level: {chunker.level}")