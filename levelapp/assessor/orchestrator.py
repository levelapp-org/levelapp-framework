"""levelapp/assessor/orchestrator.py"""
from __future__ import annotations

import asyncio

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, AsyncGenerator
from pydantic import BaseModel, ConfigDict

from levelapp.assessor.builder import ProfileBuilder, ProfileCard
from levelapp.assessor.registry import StrategyRegistry, BaseStrategy
from levelapp.assessor.schemas import Document, MetricSpec
from levelapp.aspects.logger import logger


class EvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class EvaluationRequest:
    """Immutable request for evaluation."""
    profile_name: str
    user_system_endpoint: str
    test_queries: List[str]
    evaluation_metrics: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyOutput:
    """Standardized output from strategy execution."""
    level: str
    strategy_name: str
    output: Any
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """Complete result from running a RAG pipeline."""
    query: str
    strategy_outputs: Dict[str, StrategyOutput]  # level -> output
    final_answer: str
    source_documents: List[Document]
    total_execution_time: float
    success: bool = True
    error: str | None = None


@dataclass
class EvaluationResult:
    """Result for a single query evaluation."""
    query: str
    profile_result: PipelineResult
    user_system_result: PipelineResult
    metric_scores: Dict[str, float]  # metric_name -> score
    comparison_results: Dict[str, Any]


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report."""
    model_config = ConfigDict(frozen=True)

    evaluation_id: str
    status: EvaluationStatus
    profile_card: ProfileCard
    evaluation_request: EvaluationRequest
    results: List[EvaluationResult]
    summary_metrics: Dict[str, float]
    execution_time: float


class EvaluationGauger(ABC):
    """Abstract base for evaluation metrics calculators."""

    @abstractmethod
    async def calculate(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        """Calculate metrics for comparison"""
        pass

    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """List of metrics this gauger supports."""
        pass


class PipelineExecutor:
    """Executes a complete RAG pipeline using configured strategies."""
    def __init__(self, profile_card: ProfileCard, registry: StrategyRegistry):
        self.profile_card = profile_card
        self.registry = registry
        self._initialized_strategies: Dict[str, BaseStrategy] = {}

    async def initialize(self) -> None:
        """Initialize all strategies in the pipeline."""
        logger.info(f"[PipelineExecutor] Initializing pipeline for profile: {self.profile_card.name}")

        for level_name, level_config in self.profile_card.levels.items():
            strategy = self.registry.get_strategy(
                level=level_name,
                name=level_config.strategy_name,
                config=level_config.strategy_config
            )
            await strategy.initialize()
            self._initialized_strategies[level_name] = strategy

        logger.info(f"[PipelineExecutor] Pipeline initialized with {len(self._initialized_strategies)} strategies.")

    async def cleanup(self) -> None:
        """Clean up all strategy resources."""
        logger.info(f"[PipelineExecutor] Cleaning up pipeline strategies.")

        for strategy in self._initialized_strategies.values():
            try:
                await strategy.cleanup()

            except Exception as e:
                logger.warning(f"[PipelineExecutor] Error cleaning up strategy '{strategy.name}':\n{e}")

        self._initialized_strategies.clear()

    async def execute(self, query: str) -> PipelineResult:
        """Execute the complete RAG pipeline for a query."""
        import time
        start_time = time.perf_counter()
        strategy_outputs: Dict[str, StrategyOutput] = {}

        try:
            # Execute chunking strategy
            chunking_strategy = self._initialized_strategies["chunking"]
            chunks = await self._execute_strategy("chunking", chunking_strategy, [Document(content=query)])  # hmmm? what if this is a URL or text?

            strategy_outputs["chunks"] = chunks

            # Execute embedding strategy
            embedding_strategy = self._initialized_strategies["embedding"]
            embeddings = await self._execute_strategy("embedding", embedding_strategy, chunks.output)

            strategy_outputs["embeddings"] = embeddings

            # Execute retrieval strategy
            retrieval_strategy = self._initialized_strategies["retrieval"]
            retrieved_docs = await self._execute_strategy("retrieval", retrieval_strategy, query)

            strategy_outputs["retrieved_docs"] = retrieved_docs

            # Execute generation strategy
            generation_strategy = self._initialized_strategies["generation"]
            augmented_answer = await self._execute_strategy("generation", generation_strategy, retrieved_docs.output)

            strategy_outputs["augmented_answer"] = augmented_answer

            execution_time = time.perf_counter() - start_time

            return PipelineResult(
                query=query,
                strategy_outputs=strategy_outputs,
                final_answer=augmented_answer.output,
                source_documents=retrieved_docs.output if hasattr(retrieved_docs.output, '__iter__') else [],
                total_execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"[PipelineExecutor] Pipeline execution failed for query: '{query}'\nError:\n{e}")
            return PipelineResult(
                query=query,
                strategy_outputs=strategy_outputs,
                final_answer="",
                source_documents=[],
                total_execution_time=execution_time,
                success=False,
                error=str(e),
            )

    @staticmethod
    async def _execute_strategy(level: str, strategy: BaseStrategy, *args, **kwargs) -> StrategyOutput:
        """
        Execute a single strategy with timing and error handling.

        Args:
            level (str): The level name.
            strategy (BaseStrategy): The strategy to execute.
            * args: extra arguments to pass to the strategy.
            ** kwargs: extra keyword arguments to pass to the strategy.

        Returns:
            StrategyOutput: The output of the strategy.
        """
        import time
        start_time = time.perf_counter()

        try:
            output = await strategy.run(*args, **kwargs)
            execution_time = time.perf_counter() - start_time

            return StrategyOutput(
                level=level,
                strategy_name=strategy.name,
                output=output,
                execution_time=execution_time,
                metadata={"success": True},
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"[PipelineExecutor] Strategy '{level}.{strategy.name}' failed:\n{e}")
            return StrategyOutput(
                level=level,
                strategy_name=strategy.name,
                output=None,
                execution_time=execution_time,
                metadata={"success": False, "error": str(e)},
            )