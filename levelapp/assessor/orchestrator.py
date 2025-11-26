"""levelapp/assessor/orchestrator.py"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import List, Dict

from levelapp.assessor.builder import ProfileBuilder, ProfileCard
from levelapp.assessor.gauger import EvaluationGauger
from levelapp.assessor.registry import StrategyRegistry, BaseStrategy
from levelapp.assessor.schemas import (
    Document, PipelineResult, StrategyOutput, EvaluationRequest,
    EvaluationReport, EvaluationResult, EvaluationStatus
)

from levelapp.endpoint.manager import EndpointConfigManager
from levelapp.aspects.logger import logger


class PipelineExecutor:
    """Executes a complete RAG pipeline using configured strategies."""

    def __init__(self, profile_card: ProfileCard, registry: StrategyRegistry, source_documents: List[Document]):
        self.profile_card = profile_card
        self.registry = registry
        self.source_documents = source_documents
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

        # Pre-process source documents through chunking and embedding
        await self._start_preprocessing()

        logger.info(f"[PipelineExecutor] Pipeline initialized with {len(self._initialized_strategies)} strategies.")

    @staticmethod
    async def _execute_strategy(level: str, strategy: BaseStrategy, *args, **kwargs) -> StrategyOutput:
        """Execute a single strategy with timing and error handling."""
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
            logger.error(f"[PipelineExecutor] Strategy '{level}.{strategy.name}' failed: {e}")
            return StrategyOutput(
                level=level,
                strategy_name=strategy.name,
                output=None,
                execution_time=execution_time,
                metadata={"success": False, "error": str(e)},
            )

    async def _start_preprocessing(self) -> None:
        """Pre-process source documents for retrieval."""
        try:
            # Chunk the source documents
            chunking_strategy = self._initialized_strategies["chunking"]
            chunks = await self._execute_strategy("chunking", chunking_strategy, self.source_documents)

            # Generate embeddings for chunks
            embedding_strategy = self._initialized_strategies["embedding"]
            embeddings = await self._execute_strategy("embedding", embedding_strategy, chunks.output)

            # Add documents and embeddings to retriever
            retrieval_strategy = self._initialized_strategies["retrieval"]
            if hasattr(retrieval_strategy, 'add_documents'):
                await retrieval_strategy.add_documents(chunks.output, embeddings.output)

        except Exception as e:
            logger.error(f"[PipelineExecutor] Failed to initialize document store: {e}")
            raise

    async def execute(self, query: str) -> PipelineResult:
        """Execute the complete RAG pipeline for a query."""
        start_time = time.perf_counter()
        strategy_outputs: Dict[str, StrategyOutput] = {}

        try:
            # Execute retrieval strategy (documents already pre-processed)
            retrieval_strategy = self._initialized_strategies["retrieval"]
            retrieved_docs = await self._execute_strategy("retrieval", retrieval_strategy, query)
            strategy_outputs["retrieved_docs"] = retrieved_docs

            # Execute generation strategy - Pass both query and context
            generation_strategy = self._initialized_strategies["generation"]
            augmented_answer = await self._execute_strategy(
                "generation",
                generation_strategy,
                query,  # First argument: query
                retrieved_docs.output  # Second argument: context
            )
            strategy_outputs["augmented_answer"] = augmented_answer

            execution_time = time.perf_counter() - start_time

            return PipelineResult(
                query=query,
                strategy_outputs=strategy_outputs,
                final_answer=augmented_answer.output,
                source_documents=retrieved_docs.output,
                total_execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"[PipelineExecutor] Pipeline execution failed for query: '{query}'\nError: {e}")
            return PipelineResult(
                query=query,
                strategy_outputs=strategy_outputs,
                final_answer="",
                source_documents=[],
                total_execution_time=execution_time,
                success=False,
                error=str(e),
            )


    async def cleanup(self) -> None:
        """Clean up all strategy resources."""
        logger.info("[PipelineExecutor] Cleaning up pipeline strategies.")

        for strategy in self._initialized_strategies.values():
            try:
                await strategy.cleanup()
            except Exception as e:
                logger.warning(f"[PipelineExecutor] Error cleaning up strategy '{strategy.name}': {e}")

        self._initialized_strategies.clear()


class UserSystemClient:
    """Client for communicating with user's RAG system endpoint."""

    def __init__(self, endpoint_config_manager: EndpointConfigManager, endpoint_config_name: str, timeout: int = 60):
        self.endpoint_config_manager = endpoint_config_manager
        self.endpoint_config_name = endpoint_config_name
        self.timeout = timeout

    async def query(self, query: str) -> PipelineResult:
        """Send query to user system and parse response."""
        start_time = time.perf_counter()

        try:
            # Use the injected endpoint config manager
            if self.endpoint_config_name not in self.endpoint_config_manager.endpoints:
                raise KeyError(f"Endpoint '{self.endpoint_config_name}' not found")

            endpoint_config = self.endpoint_config_manager.endpoints[self.endpoint_config_name]

            context = {"query": query, "user_message": query}  # Common field names
            response = await self.endpoint_config_manager.send_request(
                endpoint_config=endpoint_config,
                context=context,
            )

            if response.status_code != 200:
                execution_time = time.perf_counter() - start_time
                logger.error(f"[UserSystemClient] User system request error [{response.status_code}]: {response.text}")
                return PipelineResult(
                    query=query,
                    strategy_outputs={},
                    final_answer="",
                    source_documents=[],
                    total_execution_time=execution_time,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

            # Extract response data
            results = self.endpoint_config_manager.extract_response_data(
                response=response,
                mappings=endpoint_config.response_mapping,
            )
            # We can change this later
            retrieved_docs = []
            for doc in results.get("retrieved_docs", []):
                doc = Document(content=doc)
                retrieved_docs.append(doc)

            execution_time = time.perf_counter() - start_time

            return PipelineResult(
                query=query,
                strategy_outputs={},  # User system internals not visible (yet!)
                final_answer=results.get("augmented_answer", results.get("answer", "")),
                source_documents=retrieved_docs,
                total_execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"[UserSystemClient] User system query failed for query: '{query}': {e}")
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
            endpoint_config_manager: EndpointConfigManager,
            source_documents: List[Document],
            profile_builder: ProfileBuilder | None = None,
            max_concurrent_evaluations: int = 5
    ):
        self.registry = registry
        self.endpoint_config_manager = endpoint_config_manager
        self.source_documents = source_documents
        self.profile_builder = profile_builder or ProfileBuilder(registry)
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self._active_evaluations: Dict[str, asyncio.Task] = {}
        self._gaugers: List[EvaluationGauger] = []

        self._initialize_default_gaugers()

    def _initialize_default_gaugers(self) -> None:
        """Initialize default evaluation gaugers."""
        from levelapp.assessor.gauger import AnswerQualityGauger, RetrievalQualityGauger, PerformanceGauger
        self._gaugers = [
            AnswerQualityGauger(),
            RetrievalQualityGauger(),
            PerformanceGauger(),
        ]

    def register_gauger(self, gauger: EvaluationGauger) -> None:
        """Register a custom evaluation gauger."""
        self._gaugers.append(gauger)
        logger.info(f"[AssessmentOrchestrator] Registered gauger: {gauger.__class__.__name__}")

    async def evaluate(self, evaluation_request: EvaluationRequest) -> EvaluationReport:
        """
        Execute a complete evaluation comparing profile vs user system.
        """
        start_time = time.perf_counter()
        evaluation_id = f"rag_eval_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"[AssessmentOrchestrator] Starting evaluation <{evaluation_id}> "
            f"for profile: {evaluation_request.profile_name}"
        )

        # Store the evaluation task for potential cancellation
        self._active_evaluations[evaluation_id] = asyncio.current_task()

        # Build profile
        profile_card = self.profile_builder.build(evaluation_request.profile_name)
        logger.info(f"[AssessmentOrchestrator] Built profile: {profile_card.name}")

        pipeline_executor = PipelineExecutor(profile_card, self.registry, self.source_documents)

        try:
            # Initialize pipeline executor
            await pipeline_executor.initialize()

            # Initialize user system client
            user_system_client = UserSystemClient(
                endpoint_config_manager=self.endpoint_config_manager,
                endpoint_config_name=evaluation_request.endpoint_config_name
            )

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
            logger.error(f"[AssessmentOrchestrator] Evaluation <{evaluation_id}> failed: {e}")
            execution_time = time.perf_counter() - start_time
            return self._create_error_report(
                evaluation_id,
                evaluation_request,
                str(e),
                execution_time,
            )
        finally:
            # Cleanup and remove from active evaluations
            if 'pipeline_executor' in locals():
                await pipeline_executor.cleanup()

            self._active_evaluations.pop(evaluation_id, None)

    async def _execute_evaluations(
            self,
            test_queries: List[str],
            pipeline_executor: PipelineExecutor,
            user_system_client: UserSystemClient,
            evaluation_request: EvaluationRequest,
    ) -> List[EvaluationResult]:
        """Execute evaluations for all test queries with concurrency control."""
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
                query = test_queries[i]
                logger.error(f"[AssessmentOrchestrator] Query evaluation failed for '{query}': {result}")
                error_result = EvaluationResult(
                    query=query,
                    profile_result=PipelineResult(
                        query=query,
                        strategy_outputs={},
                        final_answer="",
                        source_documents=[],
                        total_execution_time=0,
                        success=False,
                        error=str(result),
                    ),
                    user_system_result=PipelineResult(
                        query=query,
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
        """Evaluate a single query with semaphore-based concurrency control."""
        async with semaphore:
            logger.debug(f"[AssessmentOrchestrator] Evaluating query: '{query}'")

            # Execute profile pipeline
            profile_result = await pipeline_executor.execute(query=query)

            # Query user system
            user_system_result = await user_system_client.query(query=query)

            # Calculate metrics
            metric_scores = await self._calculate_metrics(
                profile_result=profile_result,
                user_system_result=user_system_result,
                query=query
            )

            # Generate comparison analysis
            comparison_analysis = self._analyze_comparison(
                profile_result=profile_result,
                user_system_result=user_system_result,
                metric_scores=metric_scores
            )

            return EvaluationResult(
                query=query,
                profile_result=profile_result,
                user_system_result=user_system_result,
                metric_scores=metric_scores,
                comparison_results=comparison_analysis,  # Fixed: consistent field name
            )

    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str,
    ) -> Dict[str, float]:
        """Calculate all metrics using registered gaugers."""
        metric_scores = {}

        for gauger in self._gaugers:
            try:
                await gauger.initialize()
                scores = await gauger.calculate(profile_result, user_system_result, query)
                metric_scores.update(scores)

            except Exception as e:
                logger.warning(f"[AssessmentOrchestrator] Gauger '{gauger.__class__.__name__}' failed: {e}")

            finally:
                await gauger.cleanup()

        return metric_scores

    def _analyze_comparison(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            metric_scores: Dict[str, float],
    ) -> Dict[str, any]:  # Fixed: any instead of float for mixed types
        """Generate comparative analysis between profile and user system."""
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

    @staticmethod
    def _generate_performance_summary(metric_scores: Dict[str, float]) -> str:
        """Generate human-readable performance summary."""
        if not metric_scores:
            return "No metrics available"

        avg_score = sum(metric_scores.values()) / len(metric_scores)

        if avg_score > 0.7:
            return "User system outperforms profile"
        elif avg_score > 0.3:
            return "User system performs similarly to profile"
        else:
            return "Profile outperforms user system"

    def _generate_report(
            self,
            evaluation_id: str,
            evaluation_request: EvaluationRequest,
            profile_card: ProfileCard,
            results: List[EvaluationResult],
            execution_time: float
    ) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
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

    @staticmethod
    def _calculate_summary_metrics(results: List[EvaluationResult]) -> Dict[str, float]:
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

    @staticmethod
    def _create_error_report(
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
        """Cancel a running evaluation."""
        if evaluation_id in self._active_evaluations:
            self._active_evaluations[evaluation_id].cancel()
            return True
        return False


async def main() -> None:
    """Demonstrate orchestrator usage"""
    from levelapp.assessor.strategies import register_basic_profile_strategies
    from levelapp.assessor.schemas import Document
    from levelapp.endpoint.client import EndpointConfig
    from levelapp.endpoint.schemas import HttpMethod, HeaderConfig, RequestSchemaConfig, ResponseMappingConfig

    # Initialize components
    registry = StrategyRegistry()
    register_basic_profile_strategies(registry=registry)

    # Create endpoint configuration for your local app
    endpoint_config = EndpointConfig(
        name="user-system-rag",
        base_url="http://127.0.0.1:8000",
        path="/rag/query",  # Note: added leading slash for consistency
        method=HttpMethod.POST,
        headers=[
            HeaderConfig(
                name="Content-Type",
                value="application/json",
                secure=False
            )
        ],
        request_schema=[
            RequestSchemaConfig(
                field_path="query",
                value="user_message",  # This will be replaced with the actual query
                value_type="dynamic",
                required=True
            ),
            RequestSchemaConfig(
                field_path="output_mode",
                value="full",
                value_type="static",
                required=False
            )
        ],
        response_mapping=[
            ResponseMappingConfig(
                field_path="answer.output",
                extract_as="augmented_answer"  # Maps to final_answer in PipelineResult
            ),
            ResponseMappingConfig(
                field_path="retrieved_docs",
                extract_as="retrieved_docs"  # Maps to retrieved_docs in PipelineResult
            )
        ],
        timeout=60,
        retry_count=3,
        retry_backoff=0.5
    )

    # Create endpoint config manager and add our configuration
    endpoint_config_manager = EndpointConfigManager()
    endpoint_config_manager.set_endpoints([endpoint_config])

    # Provide actual source documents for RAG pipeline
    source_documents = [
        Document(content="Machine learning is a subset of artificial intelligence."),
        Document(content="Neural networks are inspired by the human brain."),
        Document(content="Transformers are a type of neural network architecture."),
        Document(content="Task decomposition can be done by LLM with simple prompting like 'Steps for XYZ'."),
        Document(
            content="Chain of Thought (CoT) is a standard prompting technique for enhancing model performance on complex tasks."),
        Document(
            content="LLM+P involves relying on an external classical planner to do long-horizon planning using PDDL."),
        Document(
            content="Self-reflection allows autonomous agents to improve iteratively by refining past action decisions."),
        Document(content="ReAct integrates reasoning and acting within LLM by extending the action space."),
    ]

    profile_builder = ProfileBuilder(registry=registry)

    # Pass all required parameters
    orchestrator = AssessmentOrchestrator(
        registry=registry,
        endpoint_config_manager=endpoint_config_manager,
        source_documents=source_documents,
        profile_builder=profile_builder
    )

    # Create evaluation request - using the endpoint config name we defined
    request = EvaluationRequest(
        profile_name="basic",
        endpoint_config_name="user-system-rag",  # Matches the endpoint config name above
        test_queries=[
            "What is machine learning?",
            "Explain neural networks",
            "What is the standard method for Task Decomposition?",
            "How does Chain of Thought work?"
        ],
        evaluation_metrics=["answer_quality", "retrieval_accuracy"]
    )

    # Run evaluation
    report = await orchestrator.evaluate(evaluation_request=request)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT")
    print(f"{'=' * 60}")
    print(f"Evaluation ID: {report.evaluation_id}")
    print(f"Status: {report.status}")
    print(f"Profile: {report.profile_card.name}")
    print(f"Total Queries: {len(report.results)}")
    print(f"Errors: {report.error_count}")
    print(f"Execution Time: {report.execution_time:.2f}s")

    print(f"\nSummary Metrics:")
    for metric, score in report.summary_metrics.items():
        print(f"  {metric}: {score:.3f}")

    print(f"\nDetailed Results:")
    for i, result in enumerate(report.results, 1):
        print(f"\nQuery {i}: {result.query}")
        print(f"  Profile Answer: {result.profile_result.final_answer[:100]}...")
        print(f"  User System Answer: {result.user_system_result.final_answer[:100]}...")
        print(f"  Metrics: {result.metric_scores}")

        if result.profile_result.error:
            print(f"  Profile Error: {result.profile_result.error}")
        if result.user_system_result.error:
            print(f"  User System Error: {result.user_system_result.error}")

    if report.warnings:
        print(f"\nWarnings:")
        for warning in report.warnings:
            print(f"  ⚠ {warning}")


if __name__ == '__main__':
    asyncio.run(main())