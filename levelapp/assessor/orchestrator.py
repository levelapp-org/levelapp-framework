"""levelapp/assessor/orchestrator.py"""
from __future__ import annotations

import asyncio

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict

from levelapp.assessor.builder import ProfileBuilder, ProfileCard
from levelapp.assessor.gauger import EvaluationGauger
from levelapp.assessor.registry import StrategyRegistry, BaseStrategy
from levelapp.assessor.schemas import Document
from levelapp.endpoint.manager import EndpointConfigManager
from levelapp.endpoint.client import EndpointConfig
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
    user_system_endpoint: EndpointConfig
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
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)





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
        logger.info("[PipelineExecutor] Cleaning up pipeline strategies.")

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


class UserSystemClient:
    """Client for communicating with user's RAG system endpoint."""
    def __init__(self, endpoint_config: EndpointConfig, timeout: int = 60):
        self.endpoint_config = endpoint_config
        self.timeout = timeout

    async def query(self, query: str) -> PipelineResult:
        """Send query to user system and parse response."""
        import time
        start_time = time.perf_counter()

        try:
            ecm = EndpointConfigManager()
            context = {"query": query}
            response = await ecm.send_request(
                endpoint_config=self.endpoint_config,
                context=context,
            )

            if not response or response.status_code != 200:
                execution_time = time.perf_counter() - start_time
                logger.error(f"[UserSystemClient] User system request error [{response.status_code}]:\n{response.text}")
                return PipelineResult(
                    query=query,
                    strategy_outputs={},
                    final_answer="",
                    source_documents=[],
                    total_execution_time=execution_time,
                    success=False,
                    error=response.text,
                )

            mappings = self.endpoint_config.response_mapping

            # The response mapping in the YAML file should already contain the required data for the PipelineResult.
            results = ecm.extract_response_data(
                response=response,
                mappings=mappings,
            )

            execution_time = time.perf_counter() - start_time

            return PipelineResult(
                query=query,
                strategy_outputs=results.get("strategy_outputs", {}),
                final_answer=results.get("augmented_answer", ""),
                source_documents=results.get("source_documents", []),
                total_execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"[UserSystemClient] User system query failed for query: '{query}'\nError:\n{e}")
            return PipelineResult(
                query=query,
                strategy_outputs={},
                final_answer="",
                source_documents=[],
                total_execution_time=execution_time,
                success=False,
                error=str(e),
            )


class AssessmentOrchestrator:
    """
    Main orchestrator that manages and coordinates the entire evaluation process.
    """
    def __init__(
            self,
            registry: StrategyRegistry,
            profile_builder: ProfileBuilder | None = None,
            max_concurrent_evaluations: int = 5
    ):
        self.registry = registry
        self.profile_builder = profile_builder or ProfileBuilder(registry=registry)
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self._active_evaluations: Dict[str, asyncio.Task] = {}
        self._gaugers: List[EvaluationGauger] = []

        # Initialize default gaugers
        self._initialize_default_gaugers()

    def _initialize_default_gaugers(self) -> None:
        """
        Initialize default evaluation gaugers.
        """
        # TODO: Implement actual gaugers
        # For now, add placeholder
        from levelapp.assessor.gauger import DefaultEvaluationGauger
        self._gaugers.append(DefaultEvaluationGauger())

    def register_gauger(self, gauger: EvaluationGauger) -> None:
        """Register a custom evaluation gauger."""
        self._gaugers.append(gauger)
        logger.info(f"[AssessmentOrchestrator] Registered gauger: {gauger.__class__.__name__}")

    async def evaluate(
            self,
            evaluation_request: EvaluationRequest,
    ) -> EvaluationReport:
        """
        Execute a complete evaluation comparing profile vs user system.

        Args:
            evaluation_request (EvaluationRequest): Configuration for the evaluation.

        Returns:
            Comprehensive evaluation report.
        """
        import time
        import uuid
        start_time = time.perf_counter()
        evaluation_id = f"rag_eval_{uuid.uuid4().hex[:8]}"

        logger.info(f"[AssessmentOrchestrator] Starting evaluation <{evaluation_id}> for profile: {evaluation_request.profile_name}")

        # Build profile
        profile_card = self.profile_builder.build(profile_name=evaluation_request.profile_name)
        logger.info(f"[AssessmentOrchestrator] Built profile: {profile_card.name}")

        # Initialize pipeline executor
        pipeline_executor = PipelineExecutor(profile_card=profile_card, registry=self.registry)

        try:
            await pipeline_executor.initialize()

            # Initialize user system client
            user_system_client = UserSystemClient(endpoint_config=evaluation_request.user_system_endpoint)

            # Execute evaluations for all test queries
            results = await self._execute_evaluations(
                evaluation_request.test_queries,
                pipeline_executor,
                user_system_client,
                evaluation_request
            )

            # Generate summary report
            execution_time = time.perf_counter() - start_time
            report = self._generate_report(
                evaluation_id,
                evaluation_request,
                profile_card,
                results,
                execution_time
            )

            logger.info(f"[AssessmentOrchestrator] Evaluation <{evaluation_id}> completed in {execution_time:.2f}s.")
            return report

        except Exception as e:
            logger.error(f"[AssessmentOrchestrator] Evaluation <{evaluation_id}> failed:\n{e}")
            execution_time = time.perf_counter() - start_time
            return self._create_error_report(
                evaluation_id,
                evaluation_request,
                str(e),
                execution_time,
            )

        finally:
            # Cleanup
            if 'pipeline_executor' in locals():
                await pipeline_executor.cleanup()

    async def _execute_evaluations(
            self,
            test_queries: List[str],
            pipeline_executor: PipelineExecutor,
            user_system_client: UserSystemClient,
            evaluation_request: EvaluationRequest,
    ) -> List[EvaluationResult]:
        """
        Execute evaluations for all test queries with concurrency control.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
        tasks = []

        for query in test_queries:
            task = self._evaluate_single_query(
                query,
                pipeline_executor,
                user_system_client,
                evaluation_request,
                semaphore,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in individual evaluations
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[AssessmentOrchestrator] Query evaluation failed for '{test_queries[i]}': {result}.")
                error_result = EvaluationResult(
                    query=test_queries[i],
                    profile_result=PipelineResult(
                        query=test_queries[i],
                        strategy_outputs={},
                        final_answer="",
                        source_documents=[],
                        total_execution_time=0,
                        success=False,
                        error=str(result),
                    ),
                    user_system_result=PipelineResult(
                        query=test_queries[i],
                        strategy_outputs={},
                        final_answer="",
                        source_documents=[],
                        total_execution_time=0,
                        success=False,
                        error="Dependent on profile execution"
                    ),
                    metric_scores={},
                    comparison_results={"error": str(result)},
                )
                processed_results.append(error_result)

            else:
                processed_results.append(result)

        return processed_results

    async def _evaluate_single_query(
            self,
            query: str,
            pipeline_executor: PipelineExecutor,
            user_system_client: UserSystemClient,
            evaluation_request: EvaluationRequest,
            semaphore: asyncio.Semaphore,
    ) -> EvaluationResult:
        """
        Evaluate a single query with semaphore-based concurrency control.
        """
        async with semaphore:
            logger.debug(f"[AssessmentOrchestrator] Evaluating query: '{query}'.")

            # Execute profile pipeline
            profile_result = await pipeline_executor.execute(query=query)

            # Query user system
            user_system_result = await user_system_client.query(query=query)

            # Calculate metrics
            metrics_scores = await self._calculate_metrics(
                profile_result=profile_result,
                user_system_result=user_system_result,
                query=query
            )

            # Generate comparison analysis
            comparison_analysis = self._analyze_comparison(
                profile_result=profile_result,
                user_system_result=user_system_result,
                metric_scores=metrics_scores
            )

            return EvaluationResult(
                query=query,
                profile_result=profile_result,
                user_system_result=user_system_result,
                metric_scores=metrics_scores,
                comparison_results=comparison_analysis,
            )

    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str,
    ) -> Dict[str, float]:
        """
        Calculate all metrics using registered gaugers.
        """
        metric_scores = {}

        for gauger in self._gaugers:
            try:
                scores = await gauger.calculate(profile_result, user_system_result, query)
                metric_scores.update(scores)

            except Exception as e:
                logger.warning(f"[AssessmentOrchestrator] Gauger <{gauger.__class__.__name__}> failed:\n{e}>")

        return metric_scores

    def _analyze_comparison(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            metric_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Generate comparative analysis between profile and user system.
        """
        return {
            "execution_time_ratio": (
                user_system_result.total_execution_time / profile_result.total_execution_time
                if profile_result.total_execution_time > 0 else float('inf')
            ),
            "answer_length_ratio": (
                len(user_system_result.final_answer) / len(profile_result.final_answer)
                if profile_result.final_answer else float('inf')
            ),
            "sources_count_ratio": (
                len(user_system_result.source_documents) / len(profile_result.source_documents)
                if profile_result.source_documents else float('inf')
            ),
            "performance_summary": self._generate_performance_summary(metric_scores)
        }

    def _generate_performance_summary(
            self,
            metric_scores: Dict[str, float]
    ) -> str:
        """
        Generate human-readable performance summary.
        """
        if not metric_scores:
            return "No metrics available."

        avg_score = sum(metric_scores.values()) / len(metric_scores)

        if avg_score > 0.7:
            return "User system outperforms profile."

        elif avg_score > 0.2:
            return "User system performs similarly to profile."

        else:
            return "Profile outperforms user system."

    def _generate_report(
            self,
            evaluation_id: str,
            evaluation_request: EvaluationRequest,
            profile_card: ProfileCard,
            results: List[EvaluationResult],
            execution_time: float
    ) -> EvaluationReport:
        """
        Generate comprehensive evaluation report.
        """
        successful_results = [r for r in results if r.profile_result.success and r.user_system_result.success]
        error_count = len(results) - len(successful_results)

        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(results)

        return EvaluationReport(
            evaluation_id=evaluation_id,
            status=EvaluationStatus.COMPLETED,
            profile_card=profile_card,
            evaluation_request=evaluation_request,
            results=results,
            summary_metrics=summary_metrics,
            execution_time=execution_time,
            error_count=error_count,
            warnings=["Mock implementation - real gaugers needed"] if not self._gaugers else []
        )

    def _calculate_summary_metrics(
            self,
            results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}

        successful_results = [r for r in results if r.metric_scores]

        if not successful_results:
            return {}

        # Average all metrics across successful runs
        all_metrics = {}
        metric_counts = {}

        for result in successful_results:
            for metric, score in result.metric_scores.items():
                if metric not in all_metrics:
                    all_metrics[metric] = 0.0
                    metric_counts[metric] = 0

                all_metrics[metric] += score
                metric_counts[metric] += 1

        return {metric: all_metrics[metric] / metric_counts[metric] for metric in all_metrics}

    def _create_error_report(
            self,
            evaluation_id: str,
            evaluation_request: EvaluationRequest,
            error: str,
            execution_time: float
    ) -> EvaluationReport:
        """Create error report when evaluation fails."""
        return EvaluationReport(
            evaluation_id=evaluation_id,
            status=EvaluationStatus.FAILED,
            profile_card=ProfileCard(
                name="error",
                description="Error profile",
                levels={},
                aggregation_strategy="weighted",
                weights={}
            ),
            evaluation_request=evaluation_request,
            results=[],
            summary_metrics={},
            execution_time=execution_time,
            error_count=len(evaluation_request.test_queries),
            warnings=[f"Evaluation failed: {error}"]
        )

    async def cancel_evaluation(self, evaluation_id: str) -> bool:
        """
        Cancel a running evaluation.
        """
        if evaluation_id in self._active_evaluations:
            self._active_evaluations[evaluation_id].cancel()
            return True
        return False


async def main() -> None:
    """Demonstrate orchestrator usage"""
    from levelapp.endpoint.schemas import HttpMethod, HeaderConfig

    registry = StrategyRegistry()
    profile_builder = ProfileBuilder(registry=registry)
    orchestrator = AssessmentOrchestrator(registry=registry, profile_builder=profile_builder)

    header_config = HeaderConfig(name="Content-Type", value="application/json")

    endpoint_config = EndpointConfig(
        name="user-system-rag",
        base_url="http://localhost:8080",
        path="rag/query",
        method=HttpMethod.POST,
        headers=[header_config],
    )

    # Create evaluation request
    request = EvaluationRequest(
        profile_name="quality",
        user_system_endpoint=endpoint_config,
        test_queries=["query1", "query2"],
        evaluation_metrics=["answer_quality", "retrieval_accuracy"]
    )

    # Run evaluation
    report = await orchestrator.evaluate(evaluation_request=request)
    print(f"Evaluation completed: {report.status}")
    print(f"Summary metrics: {report.summary_metrics}")


if __name__ == '__main__':
    asyncio.run(main())
