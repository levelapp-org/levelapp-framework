class AssessmentOrchestrator:
    """
    Main orchestrator that manages and coordinates the entire evaluation process.
    """

    def __init__(
            self,
            registry: StrategyRegistry,
            endpoint_config_manager: EndpointConfigManager,
            source_documents: List[Document],
            profile_builder: Optional[ProfileBuilder] = None,
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

            # Initialize user system client - FIXED: Use correct constructor
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
