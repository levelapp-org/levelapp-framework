"""
'simulators/service.py': Service layer to manage conversation simulation and evaluation.
"""
import time
import asyncio

from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List

from levelapp.core.base import BaseRepository, BaseProcess, BaseEvaluator
from levelapp.config.endpoint import EndpointConfig
from levelapp.simulator.schemas import (
    InteractionEvaluationResults,
    ScriptsBatch,
    ConversationScript,
    SimulationResults
)
from levelapp.simulator.utils import (
    extract_interaction_details,
    async_interaction_request,
    calculate_average_scores,
    summarize_verdicts,
)
from levelapp.aspects import logger
from levelapp.workflow.schemas import EvaluatorType


class ConversationSimulator(BaseProcess):
    """Conversation simulator component."""

    def __init__(
        self,
        repository: BaseRepository | None = None,
        evaluators: Dict[EvaluatorType, BaseEvaluator] | None = None,
        endpoint_config: EndpointConfig | None = None,
    ):
        """
        Initialize the ConversationSimulator.

        Args:
            repository (BaseRepository): Service for saving simulation results.
            evaluators (EvaluationService): Service for evaluating interactions.
            endpoint_config (EndpointConfig): Configuration object for VLA.
        """
        self._CLASS_NAME = self.__class__.__name__

        self.repository = repository
        self.evaluators = evaluators
        self.endpoint_config = endpoint_config

        self._url: str | None = None
        self._credentials: str | None = None
        self._headers: Dict[str, Any] | None = None

        self.test_batch: ScriptsBatch | None = None
        self.evaluation_verdicts: Dict[str, List[str]] = defaultdict(list)
        self.verdict_summaries: Dict[str, List[str]] = defaultdict(list)

    def setup(
            self,
            repository: BaseRepository,
            evaluators: Dict[str, BaseEvaluator],
            endpoint_config: EndpointConfig,
    ) -> None:
        """
        Initialize the ConversationSimulator.

        Args:
            repository (BaseRepository): Repository object for storing simulation results.
            evaluators (Dict[str, BaseEvaluator]): List of evaluator objects for evaluating interactions.
            endpoint_config (EndpointConfig): Configuration object for VLA.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.setup.__name__}]"
        logger.info(f"{_LOG} Setting up the Conversation Simulator..")

        self.repository = repository
        self.evaluators = evaluators
        self.endpoint_config = endpoint_config

        self._url = endpoint_config.full_url
        self._credentials = endpoint_config.api_key.get_secret_value()
        self._headers = endpoint_config.headers

    def get_evaluator(self, name: EvaluatorType) -> BaseEvaluator:
        _LOG: str = f"[{self._CLASS_NAME}][{self.get_evaluator.__name__}]"

        if name not in self.evaluators:
            raise KeyError(f"{_LOG} Evaluator {name} not registered.")
        return self.evaluators[name]

    async def run(
        self,
        test_batch: ScriptsBatch,
        attempts: int = 1,
    ) -> Dict[str, Any]:
        """
        Run a batch test for the given batch name and details.

        Args:
            test_batch (ScriptsBatch): Scenario batch object.
            attempts (int): Number of attempts to run the simulation.

        Returns:
            Dict[str, Any]: The results of the batch test.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.run.__name__}]"
        logger.info(f"{_LOG} Starting batch test (attempts: {attempts}).")

        started_at = datetime.now()

        self.test_batch = test_batch
        results = await self.simulate_conversation(attempts=attempts)

        finished_at = datetime.now()

        results = SimulationResults(
            started_at=started_at,
            finished_at=finished_at,
            evaluation_summary=self.verdict_summaries,
            average_scores=results.get("average_scores", {}),
        )

        return {"results": results, "status": "COMPLETE"}

    async def simulate_conversation(self, attempts: int = 1) -> Dict[str, Any]:
        """
        Simulate conversations for all scenarios in the batch.

        Args:
            attempts (int): Number of attempts to run the simulation.

        Returns:
            Dict[str, Any]: The results of the conversation simulation.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.simulate_conversation.__name__}]"
        logger.info(f"{_LOG} starting conversation simulation..")

        semaphore = asyncio.Semaphore(value=len(self.test_batch.scripts))

        async def run_with_semaphore(script: ConversationScript) -> Dict[str, Any]:
            async with semaphore:
                return await self.simulate_single_scenario(
                    script=script, attempts=attempts
                )

        results = await asyncio.gather(
            *(run_with_semaphore(s) for s in self.test_batch.scripts)
        )

        aggregate_scores: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            for key, value in result.get("average_scores", {}).items():
                if isinstance(value, (int, float)):
                    aggregate_scores[key].append(value)

        overall_average_scores = calculate_average_scores(aggregate_scores)

        for judge, verdicts in self.evaluation_verdicts.items():
            self.verdict_summaries[judge] = summarize_verdicts(
                verdicts=verdicts, judge=judge
            )

        return {"scripts": results, "average_scores": overall_average_scores}

    async def simulate_single_scenario(
        self, script: ConversationScript, attempts: int = 1
    ) -> Dict[str, Any]:
        """
        Simulate a single scenario with the given number of attempts, concurrently.

        Args:
            script (SimulationScenario): The scenario to simulate.
            attempts (int): Number of attempts to run the simulation.

        Returns:
            Dict[str, Any]: The results of the scenario simulation.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.simulate_single_scenario.__name__}]"

        logger.info(f"{_LOG} Starting simulation for script: {script.id}")
        all_attempts_scores: Dict[str, List[float]] = defaultdict(list)
        all_attempts_verdicts: Dict[str, List[str]] = defaultdict(list)

        async def simulate_attempt(attempt_number: int) -> Dict[str, Any]:
            logger.info(f"{_LOG} Running attempt: {attempt_number + 1}/{attempts}")
            start_time = time.time()

            collected_scores: Dict[str, List[Any]] = defaultdict(list)
            collected_verdicts: Dict[str, List[str]] = defaultdict(list)

            initial_interaction_results = await self.simulate_interactions(
                script=script,
                evaluation_verdicts=collected_verdicts,
                collected_scores=collected_scores,
            )

            logger.info(f"{_LOG} collected_scores: {collected_scores}\n---")
            single_attempt_scores = calculate_average_scores(collected_scores)

            for target, scores in single_attempt_scores.items():
                all_attempts_scores[target].append(scores)

            for judge, verdicts in collected_verdicts.items():
                all_attempts_verdicts[judge].extend(verdicts)

            elapsed_time = time.time() - start_time
            all_attempts_scores["processing_time"].append(elapsed_time)

            logger.info(
                f"{_LOG} Attempt {attempt_number + 1} completed in {elapsed_time:.2f}s\n---"
            )

            return {
                "attempt": attempt_number + 1,
                "script_id": script.id,
                "total_duration": elapsed_time,
                "interaction_results": initial_interaction_results,
                "evaluation_verdicts": collected_verdicts,
                "average_scores": single_attempt_scores,
            }

        attempt_tasks = [simulate_attempt(i) for i in range(attempts)]
        attempt_results = await asyncio.gather(*attempt_tasks, return_exceptions=False)

        average_scores = calculate_average_scores(all_attempts_scores)

        for judge_, verdicts_ in all_attempts_verdicts.items():
            self.evaluation_verdicts[judge_].extend(verdicts_)

        logger.info(
            f"{_LOG} average scores:\n{average_scores}\n---"
        )

        return {
            "script_id": script.id,
            "attempts": attempt_results,
            "average_scores": average_scores,
        }

    async def simulate_interactions(
        self,
        script: ConversationScript,
        evaluation_verdicts: Dict[str, List[str]],
        collected_scores: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """
        Simulate inbound interactions for a scenario.

        Args:
            script (ConversationScript): The script to simulate.
            evaluation_verdicts(Dict[str, List[str]]): evaluation verdict for each evaluator.
            collected_scores(Dict[str, List[Any]]): collected scores for each target.

        Returns:
            List[Dict[str, Any]]: The results of the inbound interactions simulation.
        """
        _LOG: str = f"[{self._CLASS_NAME}][{self.simulate_interactions.__name__}]"

        logger.info(f"{_LOG} Starting interactions simulation..")
        start_time = time.time()

        results = []
        interactions = script.interactions

        for interaction in interactions:
            user_message = interaction.user_message
            request_payload = interaction.request_payload
            self.endpoint_config.variables = {
                "user_message": user_message,
                "request_payload": request_payload
            }

            response = await async_interaction_request(
                url=self.endpoint_config.full_url,
                headers=self.endpoint_config.headers,
                payload=self.endpoint_config.request_payload,
            )

            reference_reply = interaction.reference_reply
            reference_metadata = interaction.reference_metadata
            reference_guardrail_flag: bool = interaction.guardrail_flag

            if not response or response.status_code != 200:
                logger.error(f"{_LOG} Interaction request failed.")
                result = {
                    "user_message": user_message,
                    "generated_reply": "Interaction Request failed",
                    "reference_reply": reference_reply,
                    "generated_metadata": {},
                    "reference_metadata": reference_metadata,
                    "guardrail_details": None,
                    "evaluation_results": {},
                }
                results.append(result)
                continue

            interaction_details = extract_interaction_details(
                response=response.text,
                template=self.endpoint_config.response_payload,
            )

            generated_reply = interaction_details.generated_reply
            generated_metadata = interaction_details.generated_metadata
            extracted_guardrail_flag: bool = interaction_details.guardrail_flag

            evaluation_results = await self.evaluate_interaction(
                user_input=user_message,
                generated_reply=generated_reply,
                reference_reply=reference_reply,
                generated_metadata=generated_metadata,
                reference_metadata=reference_metadata,
                generated_guardrail=extracted_guardrail_flag,
                reference_guardrail=reference_guardrail_flag,
            )

            logger.info(f"{_LOG} Evaluation results:\n{evaluation_results.model_dump()}\n")

            self.store_evaluation_results(
                results=evaluation_results,
                evaluation_verdicts=evaluation_verdicts,
                collected_scores=collected_scores,
            )

            elapsed_time = time.time() - start_time
            logger.info(
                f"{_LOG} Interaction simulation complete in {elapsed_time:.2f} seconds.\n---"
            )

            result = {
                "user_message": user_message,
                "generated_reply": generated_reply,
                "reference_reply": reference_reply,
                "generated_metadata": generated_metadata,
                "reference_metadata": reference_metadata,
                "guardrail_details": interaction_details.guardrail_flag,
                "evaluation_results": evaluation_results.model_dump(),
            }

            results.append(result)

        return results

    async def evaluate_interaction(
        self,
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

        judge_evaluator = self.evaluators.get(EvaluatorType.JUDGE)
        metadata_evaluator = self.evaluators.get(EvaluatorType.REFERENCE)

        if not judge_evaluator:
            raise ValueError(f"{_LOG} No Judge Evaluator found.")

        openai_eval_task = judge_evaluator.async_evaluate(
            generated_data=generated_reply,
            reference_data=reference_reply,
            user_input=user_input,
            provider="openai"
        )

        ionos_eval_task = judge_evaluator.async_evaluate(
            provider="ionos",
            user_input=user_input,
            generated_data=generated_reply,
            reference_data=reference_reply,
        )

        openai_judge_evaluation, ionos_judge_evaluation = await asyncio.gather(
            openai_eval_task, ionos_eval_task
        )

        if not metadata_evaluator:
            raise ValueError(f"{_LOG} No Metadata Evaluator found.")

        metadata_evaluation = {}
        if reference_metadata:
            metadata_evaluation = metadata_evaluator.evaluate(
                generated_data=generated_metadata,
                reference_data=reference_metadata,
            )

        guardrail_flag = 1 if generated_guardrail == reference_guardrail else 0

        return InteractionEvaluationResults(
            judge_evaluations={
                openai_judge_evaluation.provider: openai_judge_evaluation,
                ionos_judge_evaluation.provider: ionos_judge_evaluation
            },
            metadata_evaluation=metadata_evaluation,
            guardrail_flag=guardrail_flag,
        )

    @staticmethod
    def store_evaluation_results(
        results: InteractionEvaluationResults,
        evaluation_verdicts: Dict[str, List[str]],
        collected_scores: Dict[str, List[Any]],
    ) -> None:
        """
        Store the evaluation results in the evaluation summary.

        Args:
            results (InteractionEvaluationResults): The evaluation results to store.
            evaluation_verdicts (Dict[str, List[str]]): The evaluation summary.
            collected_scores (Dict[str, List[Any]]): The collected scores.
        """
        for provider in results.judge_evaluations.keys():
            evaluation_verdicts[f"{provider}_verdicts_summary"].append(
                results.judge_evaluations.get(provider, "").justification
            )

            collected_scores[provider].append(results.judge_evaluations.get(provider, "").score)

        average_metadata_score = calculate_average_scores(scores=results.metadata_evaluation)
        for field, score in average_metadata_score.items():
            collected_scores["metadata"].append(score)

        collected_scores["guardrail"].append(results.guardrail_flag)
