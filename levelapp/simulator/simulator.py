"""
'simulators/service.py': Service layer to manage conversation simulation and evaluation.
"""
import time
import asyncio

from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List


from levelapp.core.base import BaseProcess, BaseEvaluator
from levelapp.endpoint.client import EndpointConfig
from levelapp.endpoint.manager import EndpointConfigManager

from levelapp.core.schemas import EvaluatorType
from levelapp.evaluator.schemas import JudgeEvaluationResults
from levelapp.simulator.schemas import (
    ScriptsBatch,
    ConversationScript,
    TurnSummary,
    SingleInteractionResults,
    SingleAttemptResults,
    AllAttemptsResults,
    InteractionEvaluationResults,
    SummaryResults,
    SimulationResults,
)

from levelapp.simulator.utils import calculate_average_scores, summarize_verdicts
from levelapp.aspects import logger


class ConversationSimulator(BaseProcess):
    """Conversation simulator component."""

    def __init__(
        self,
        endpoint_config: EndpointConfig | None = None,
        evaluators: Dict[EvaluatorType, BaseEvaluator] | None = None,
        providers: List[str] | None = None,

    ):
        """
        Initialize the ConversationSimulator.

        Args:
            endpoint_config (EndpointConfig): Endpoint configuration.
            evaluators (EvaluationService): Service for evaluating interactions.
            endpoint_config (EndpointConfig): Configuration object for VLA.
        """
        self._CLASS_NAME = self.__class__.__name__

        self.endpoint_config = endpoint_config
        self.evaluators = evaluators
        self.providers = providers

        self.endpoint_cm = EndpointConfigManager()

        self.test_batch: ScriptsBatch | None = None

    def setup(
        self,
        endpoint_config: EndpointConfig,
        evaluators: Dict[EvaluatorType, BaseEvaluator],
        providers: List[str],
    ) -> None:
        """
        Initialize the ConversationSimulator.

        Args:
            endpoint_config (EndpointConfig): Configuration object for user endpoint API.
            evaluators (Dict[str, BaseEvaluator]): List of evaluator objects for evaluating interactions.
            providers (List[str]): List of LLM provider names.

        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.setup.__name__}]"
        logger.info(f"{_LOG} Setting up the Conversation Simulator..")

        if not self.endpoint_cm:
            self.endpoint_cm = EndpointConfigManager()

        self.endpoint_config = endpoint_config
        self.endpoint_cm.set_endpoints(endpoints_config=[endpoint_config])

        self.evaluators = evaluators
        self.providers = providers

        if not self.providers:
            logger.warning(f"{_LOG} No LLM providers were provided. The Judge Evaluation process will not be executed.")

    def get_evaluator(self, name: EvaluatorType) -> BaseEvaluator:
        """
        Retrieve an evaluator by name.

        Args:
            name (EvaluatorType): Name of evaluator.

        Returns:
            An evaluator object.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.get_evaluator.__name__}]"

        if name not in self.evaluators:
            raise KeyError(f"{_LOG} Evaluator {name} not registered.")

        return self.evaluators[name]

    async def run(
        self,
        test_batch: ScriptsBatch,
        attempts: int = 1,
        batch_size: int = 4
    ) -> Any:
        """
        Run a batch test for the given batch name and details.

        Args:
            test_batch (ScriptsBatch): Scenario batch object.
            attempts (int): Number of attempts to run the simulation.
            batch_size (int): Maximum number of concurrent processes to run the simulation.

        Returns:
            Dict[str, Any]: The results of the batch test.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.run.__name__}]"
        logger.info(f"{_LOG} Starting batch test [attempts:{attempts}][batch-size:{batch_size}].")

        started_at = datetime.now()

        self.test_batch = test_batch
        conversation_results = await self.simulate_conversation(attempts=attempts, max_concurrency=batch_size)

        finished_at = datetime.now()

        script_results: List[AllAttemptsResults] = conversation_results["script_results"]

        batch_verdicts: Dict[str, List[str]] = defaultdict(list)
        interaction_summaries: List[str] = []

        for script in script_results:
            for attempt in script.attempts:
                interaction_summaries.extend(attempt.interaction_summaries)
                for judge, verdicts in attempt.evaluation_verdicts.items():
                    batch_verdicts[judge].extend(verdicts)

        verdict_summaries: Dict[str, SummaryResults] = {
            judge: summarize_verdicts(
                interaction_summaries=interaction_summaries,
                verdicts=verdicts,
                judge=judge
            )
            for judge, verdicts in batch_verdicts.items()
        }

        results = SimulationResults(
            started_at=started_at,
            finished_at=finished_at,
            evaluation_summary=verdict_summaries,
            average_scores=conversation_results.get("average_scores", {}),
            script_results=script_results
        )

        return results.model_dump_json(indent=2)

    async def simulate_conversation(
        self,
        attempts: int = 1,
        max_concurrency: int = 4
    ) -> Dict[str, Any]:
        """
        Simulate conversations for all scenarios in the batch.

        Args:
            attempts (int): Number of attempts to run the simulation.
            max_concurrency (int): Maximum number of concurrent conversations.

        Returns:
            Dict[str, Any]: The results of the conversation simulation.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.simulate_conversation.__name__}]"
        logger.info(f"{_LOG} starting conversation simulation..")

        semaphore = asyncio.Semaphore(value=max_concurrency)

        async def run_script(script: ConversationScript) -> AllAttemptsResults:
            async with semaphore:
                return await self.simulate_single_script(script=script, attempts=attempts)

        scripts_tasks = [run_script(script=script) for script in self.test_batch.scripts]
        script_results: List[AllAttemptsResults] = await asyncio.gather(*scripts_tasks)

        aggregate_scores: Dict[str, List[float]] = defaultdict(list)

        for result in script_results:
            for metric, value in result.average_scores.items():
                if isinstance(value, (int, float)):
                    aggregate_scores[metric].append(value)

        overall_average_scores = calculate_average_scores(aggregate_scores)

        return {"script_results": script_results, "average_scores": overall_average_scores}

    async def simulate_single_script(
        self,
        script: ConversationScript,
        attempts: int = 1
    ) -> AllAttemptsResults:
        """
        Simulate a single scenario with the given number of attempts, concurrently.

        Args:
            script (SimulationScenario): The scenario to simulate.
            attempts (int): Number of attempts to run the simulation.

        Returns:
            AllAttemptsResults: The results of the scenario simulation attempts.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.simulate_single_script.__name__}]"
        logger.info(f"{_LOG} Starting simulation for script: {script.id}")

        async def simulate_attempt(attempt_number: int) -> SingleAttemptResults:
            from uuid import uuid4
            attempt_id: str = str(uuid4())

            logger.info(f"{_LOG} Running attempt: {attempt_number + 1}/{attempts}\n---")
            start_time = time.time()

            interaction_results = await self.simulate_interactions(
                script=script,
                attempt_id=attempt_id,
            )

            collected_summaries: List[str] = []
            collected_scores: Dict[str, List[Any]] = defaultdict(list)
            collected_verdicts: Dict[str, List[str]] = defaultdict(list)

            # TODO-2: Refactor into a separate method 'collect_evaluation_data'.
            for interaction in interaction_results:
                if not interaction.evaluation_results:
                    continue

                eval_results = interaction.evaluation_results

                # Judge scores & verdicts
                for provider, judge_result in eval_results.judge_evaluations.items():
                    collected_scores[provider].append(judge_result.score)
                    collected_verdicts[provider].append(judge_result.verdict)

                # Metadata scores
                if eval_results.metadata_evaluation:
                    for _, score in eval_results.metadata_evaluation.items():
                        collected_scores["metadata"].append(score)

                # Guardrail
                if eval_results.guardrail_flag is not None:
                    collected_scores["guardrail"].append(eval_results.guardrail_flag)

                # Turn summaries
                if interaction.turn_summary:
                    collected_summaries.append(interaction.turn_summary.compact_repr)

            elapsed_time = time.time() - start_time
            collected_scores["processing_time"].append(elapsed_time)

            average_scores = calculate_average_scores(collected_scores)

            logger.info(
                f"{_LOG} Attempt {attempt_number + 1} completed in {elapsed_time:.2f}s\n---"
            )

            return SingleAttemptResults(
                attempt_nbr=attempt_number + 1,
                attempt_id=attempt_id,
                script_id=str(script.id),
                total_duration=elapsed_time,
                interaction_results=interaction_results,
                evaluation_verdicts=collected_verdicts,
                average_scores=average_scores,
                interaction_summaries=collected_summaries
            )

        attempt_tasks = [simulate_attempt(i) for i in range(attempts)]
        all_attempts: List[SingleAttemptResults] = await asyncio.gather(*attempt_tasks, return_exceptions=False)

        scenario_scores: Dict[str, List[float]] = defaultdict(list)

        for attempt in all_attempts:
            for metric, value in attempt.average_scores.items():
                if isinstance(value, (int, float)):
                    scenario_scores[metric].append(value)

        scenario_average_scores = calculate_average_scores(scenario_scores)

        return AllAttemptsResults(
            script_id=str(script.id),
            attempts=all_attempts,
            average_scores=scenario_average_scores,
        )

    async def simulate_interactions(
        self,
        script: ConversationScript,
        attempt_id: str,
    ) -> List[SingleInteractionResults]:
        """
        Simulate inbound interactions for a scenario.

        Args:
            script (ConversationScript): The script to simulate.
            attempt_id (str): The id of the attempt.

        Returns:
            List[SingleInteractionResults]: The results of the inbound interactions simulation.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.simulate_interactions.__name__}]"

        logger.info(f"{_LOG} Starting interactions simulation [ConvId:{attempt_id}]..")
        start_time = time.time()

        results = []
        contextual_mode: bool = script.variable_request_schema
        logger.info(f"{_LOG} Contextual Mode ON: {contextual_mode}")
        interactions = script.interactions

        for idx, interaction in enumerate(interactions):
            request_payload = interaction.request_payload.copy()
            if contextual_mode:
                from levelapp.simulator.utils import set_by_path

                if script.uuid_field:
                    request_payload[script.uuid_field] = attempt_id

                user_message = interaction.user_message
                set_by_path(
                    obj=request_payload,
                    path=interaction.user_message_path,
                    value=user_message,
                )
                logger.info(f"{_LOG} Request payload (Preloaded Request Schema):\n{request_payload}\n---")

            else:
                user_message = interaction.user_message
                request_payload.update({"user_message": user_message})
                logger.info(f"{_LOG} Request payload (Configured Request Schema):\n{request_payload}\n---")

            mappings = self.endpoint_config.response_mapping

            client_response = await self.endpoint_cm.send_request(
                endpoint_config=self.endpoint_config,
                context=request_payload,
                contextual_mode=contextual_mode
            )

            reference_reply = interaction.reference_reply
            reference_metadata = interaction.reference_metadata
            reference_guardrail_flag: bool = interaction.guardrail_flag

            if not client_response.response or client_response.response.status_code != 200:
                logger.error(
                    f"{_LOG} Interaction request failed [{client_response.error}]:\n{client_response.response}\n---"
                )
                output: SingleInteractionResults = SingleInteractionResults(
                    conversation_id=attempt_id,
                    user_message=user_message,
                    reference_reply=reference_reply,
                    reference_metadata=reference_metadata,
                    errors={"error": str(client_response.error), "context": str(client_response.response)}
                )
                results.append(output)
                continue

            logger.info(
                f"{_LOG} Response [{client_response.response.status_code}]:\n{client_response.response.text}\n---"
            )

            interaction_details = self.endpoint_cm.extract_response_data(
                response=client_response.response,
                mappings=mappings,
            )

            logger.info(f"{_LOG} Interaction details <ConvID:{attempt_id}>:\n{interaction_details}\n---")

            generated_reply = interaction_details.get("agent_reply", "")
            generated_metadata = interaction_details.get("metadata", {})
            extracted_guardrail_flag = interaction_details.get("guardrail_flag", False)

            logger.info(f"{_LOG} Generated reply <ConvID:{attempt_id}>:\n{generated_reply}\n---")

            evaluation_results = await self.evaluate_interaction(
                domain_context=script.domain_context,
                user_input=user_message,
                generated_reply=generated_reply,
                reference_reply=reference_reply,
                generated_metadata=generated_metadata,
                reference_metadata=reference_metadata,
                generated_guardrail=extracted_guardrail_flag,
                reference_guardrail=reference_guardrail_flag,
            )

            if evaluation_results.judge_evaluations:
                turn_summary = self.canonicalize_turn_summary(
                    judge_results=evaluation_results.judge_evaluations,
                    turn_index=idx,
                    guardrail_triggered=extracted_guardrail_flag,
                )
            else:
                turn_summary = None

            elapsed_time = time.time() - start_time
            logger.info(f"{_LOG} Interaction simulation complete in {elapsed_time:.2f} seconds.\n---")

            output: SingleInteractionResults = SingleInteractionResults(
                conversation_id=attempt_id,
                user_message=user_message,
                generated_reply=generated_reply,
                reference_reply=reference_reply,
                generated_metadata=generated_metadata,
                reference_metadata=reference_metadata,
                guardrail_details=extracted_guardrail_flag,
                evaluation_results=evaluation_results,
                turn_summary=turn_summary,
            )

            results.append(output)

        return results

    async def evaluate_interaction(
        self,
        domain_context: str,
        user_input: str,
        generated_reply: str,
        reference_reply: str,
        generated_metadata: Dict[str, Any],
        reference_metadata: Dict[str, Any],
        generated_guardrail: bool,
        reference_guardrail: bool,
    ) -> InteractionEvaluationResults:
        """
        Evaluate an interaction using OpenAI and Ionos evaluation services.

        Args:
            domain_context (str): Domain context of the conversation.
            user_input (str): user input to evaluate.
            generated_reply (str): The generated agent reply.
            reference_reply (str): The reference agent reply.
            generated_metadata (Dict[str, Any]): The generated metadata.
            reference_metadata (Dict[str, Any]): The reference metadata.
            generated_guardrail (bool): generated handoff/guardrail flag.
            reference_guardrail (bool): reference handoff/guardrail flag.

        Returns:
            InteractionEvaluationResults: The evaluation results.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.evaluate_interaction.__name__}]"

        judge_evaluator: BaseEvaluator | None = self.evaluators.get(EvaluatorType.JUDGE, None)
        metadata_evaluator: BaseEvaluator | None = self.evaluators.get(EvaluatorType.REFERENCE, None)

        evaluation_results = InteractionEvaluationResults()

        if judge_evaluator and self.providers:
            await self._judge_evaluation(
                domain_context=domain_context,
                user_input=user_input,
                generated_reply=generated_reply,
                reference_reply=reference_reply,
                providers=self.providers,
                judge_evaluator=judge_evaluator,
                evaluation_results=evaluation_results,
            )
        else:
            logger.info(f"{_LOG} Judge evaluation skipped (no evaluator or no providers).")

        if metadata_evaluator and reference_metadata:
            self._metadata_evaluation(
                metadata_evaluator=metadata_evaluator,
                generated_metadata=generated_metadata,
                reference_metadata=reference_metadata,
                evaluation_results=evaluation_results,
            )
        else:
            logger.info(f"{_LOG} Metadata evaluation skipped (no evaluator or no reference metadata).")

        evaluation_results.guardrail_flag = 1 if generated_guardrail == reference_guardrail else 0

        return evaluation_results

    async def _judge_evaluation(
            self,
            domain_context: str,
            user_input: str,
            generated_reply: str,
            reference_reply: str,
            providers: List[str],
            judge_evaluator: BaseEvaluator,
            evaluation_results: InteractionEvaluationResults,
    ) -> None:
        """
        Run LLM-as-a-judge evaluation using multiple providers (async).

        Args:
            domain_context (str): The domain context.
            user_input (str): The user input message.
            generated_reply (str): The generated agent reply.
            reference_reply (str): The reference agent reply.
            providers (List[str]): List of judge provider names.
            judge_evaluator (BaseEvaluator): Evaluator instance.
            evaluation_results (InteractionEvaluationResults): Results container (Pydantic model).

        Returns:
            None
        """
        _LOG: str = f"[{self._CLASS_NAME}][judge_evaluation]"

        tasks = {
            provider: judge_evaluator.async_evaluate(
                domain_context=domain_context,
                generated_data=generated_reply,
                reference_data=reference_reply,
                user_input=user_input,
                provider=provider,
            )
            for provider in providers
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for provider, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"{_LOG} Provider '{provider}' failed to perform Judge Evaluation.")
                evaluation_results.errors = {"provider": provider, "content": str(result)}
            else:
                evaluation_results.judge_evaluations[provider] = result

    def _metadata_evaluation(
            self,
            metadata_evaluator: BaseEvaluator,
            generated_metadata: Dict[str, Any],
            reference_metadata: Dict[str, Any],
            evaluation_results: InteractionEvaluationResults,
    ) -> None:
        """
        Run metadata evaluation using the provided evaluator.

        Args:
            metadata_evaluator (BaseEvaluator): Evaluator for metadata comparison.
            generated_metadata (Dict[str, Any]): The generated metadata.
            reference_metadata (Dict[str, Any]): The reference metadata.
            evaluation_results (InteractionEvaluationResults): Results container.
        """
        _LOG: str = f"[{self._CLASS_NAME}][metadata_evaluation]"

        try:
            evaluation_results.metadata_evaluation = metadata_evaluator.evaluate(
                generated_data=generated_metadata,
                reference_data=reference_metadata,
            )
        except Exception as e:
            logger.error(f"{_LOG} Metadata evaluation failed:\n{e}", exc_info=e)
            evaluation_results.errors = {"errors": e}

    def canonicalize_turn_summary(
            self,
            judge_results: Dict[str, JudgeEvaluationResults],
            turn_index: int,
            guardrail_triggered: bool = False
    ) -> TurnSummary | None:
        """
        Compute a consensus TurnSummary from multiple judge evaluations.

        Strategy:
            - score: mean of provider scores (rounded to 0.5)
            - engagement: mean of engagement scores.
            - gricean_violations: majority vote per maxim -> total count
            - task_type: most frequent (fallback: first)
            - task_success: All user_message (independent of judges)
            - key_facts: union of extracted facts (de-duplicated)
        """
        _LOG: str = f"[{self._CLASS_NAME}][canonicalize_turn_summary]"

        if not judge_results:
            logger.warning(f"{_LOG} No judge results. Using defaults for TurnSummary.")
            return TurnSummary(turn_index=turn_index, role="A", score=0, engagement=0, gricean_violations=0)

        # 1. Aggregate scalar scores:
        scores = [jr.score for jr in judge_results.values()]
        engagements = [jr.engagement_score for jr in judge_results.values()]
        gricean_counts = [jr.gricean.violation_count for jr in judge_results.values()]

        consensus_score = round(sum(scores) / len(scores) * 2) / 2  # Needs to be verified
        consensus_engagement = round(sum(engagements) / len(engagements), 3)
        consensus_gricean = round(sum(gricean_counts) / len(gricean_counts))

        # 2. Task metadata:
        task_types = [jr.task_metadata.task_type for jr in judge_results.values()]

        from collections import Counter
        task_type = Counter(task_types).most_common(1)[0][0]

        # 2.1. Task success & sentiment require ALL judges to say True (unanimous) to avoid false positives
        task_success = all(jr.task_metadata.task_success for jr in judge_results.values())
        sentiments = [jr.task_metadata.user_sentiment for jr in judge_results.values()]
        sentiment = Counter(sentiments).most_common(1)[0][0]

        # TODO-0: Change 'all_facts' to 'all_verdicts'
        all_verdicts = []
        for jr in judge_results.values():
            all_verdicts.append(jr.verdict)

        return TurnSummary(
            turn_index=turn_index,
            role="A",
            task_type=task_type,
            task_success=task_success,
            score=consensus_score,
            engagement=consensus_engagement,
            gricean_violations=consensus_gricean,
            sentiment=sentiment,
            key_facts=all_verdicts,
            guardrail_triggered=guardrail_triggered,
        )
