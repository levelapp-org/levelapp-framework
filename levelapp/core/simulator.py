"""
'simulators/service.py': Service layer to manage conversation simulation and evaluation.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from http.client import HTTPException
from typing import Dict, Any, List

from .base import BaseDatastore, BaseEvaluator
from ..simulator.schemas import (
    InteractionEvaluationResult,
    BatchDetails,
    TestResults,
    EndpointConfig, ScriptsBatch, ConversationScript,
)
from ..simulator.utils import (
    extract_interaction_details,
    async_vla_request,
    date_value_setter,
    calculate_average_scores,
    summarize_verdicts,
)


class ConversationSimulator:
    """Service to simulate conversations and evaluate interactions."""

    def __init__(
        self,
        storage_service: BaseDatastore,
        evaluation_service: BaseEvaluator,
        endpoint_configuration: EndpointConfig,
    ):
        """
        Initialize the ConversationSimulator.

        Args:
            storage_service (BaseDatastore): Service for saving simulation results.
            evaluation_service (EvaluationService): Service for evaluating interactions.
            endpoint_configuration (EndpointConfig): Configuration object for VLA.
        """
        self.evaluation_service = evaluation_service
        self.storage_service = storage_service
        # TODO-0: Keep for now until we add a config method.
        self.api_configuration = endpoint_configuration
        self.logger = logging.getLogger("?")

        self._endpoint = endpoint_configuration.full_url
        self._credentials = endpoint_configuration.api_key
        self._headers = endpoint_configuration.headers

        self.test_batch: ScriptsBatch | None = None
        self.evaluation_verdicts: Dict[str, List[str]] = defaultdict(list)
        self.verdict_summaries: Dict[str, List[str]] = defaultdict(list)

    async def run_batch_test(
        self,
        test_batch: ScriptsBatch,
        batch_details: BatchDetails,
        test_details: TestResults,
        attempts: int = 1,
    ) -> Dict[str, Any]:
        """
        Run a batch test for the given batch name and details.

        Args:
            test_batch (ScriptsBatch): Scenario batch object.
            batch_details (Dict[str, Any]): Test batch test details.
            test_details (Dict[str, Any]): Test details dictionary.
            attempts (int): Number of attempts to run the simulation.

        Returns:
            Dict[str, Any]: The results of the batch test.
        """
        self.logger.info(
            f"[run_batch_test] Starting batch test for batch: {test_details.test_name}"
        )
        started_at = datetime.now().isoformat()
        start_time = time.time()

        self.test_batch = test_batch
        results = await self.simulate_conversation(attempts=attempts)

        finished_at = datetime.now().isoformat()
        elapsed_time = time.time() - start_time

        batch_details.started_at = started_at
        batch_details.finished_at = finished_at
        batch_details.elapsed_time = elapsed_time
        batch_details.evaluation_summary = self.verdict_summaries
        batch_details.average_scores = results["averageScores"]
        batch_details.simulation_results = results["scenarios"]

        test_details.batch_details = batch_details

        try:
            self.storage_service.save_batch_test_results(
                user_id=batch_details.user_id,
                project_id=batch_details.project_id,
                batch_id=batch_details.batch_id,
                data=test_details.model_dump(by_alias=True),
            )

        # TODO-1: Create custom exceptions for 'DataStore' implementations.
        except HTTPException as e:
            self.logger.error(f"[run_batch_test] Failed to save batch result: {e}")

        return {"batchId": batch_details.batch_id, "status": "COMPLETE"}

    async def simulate_conversation(self, attempts: int = 1) -> Dict[str, Any]:
        """
        Simulate conversations for all scenarios in the batch.

        Args:
            attempts (int): Number of attempts to run the simulation.

        Returns:
            Dict[str, Any]: The results of the conversation simulation.
        """
        self.logger.info("[simulate_conversation] starting conversation simulation..")
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
            for key, value in result.get("averageScores", {}).items():
                if isinstance(value, (int, float)):
                    aggregate_scores[key].append(value)

        overall_average_scores = calculate_average_scores(aggregate_scores)

        for judge, verdicts in self.evaluation_verdicts.items():
            self.verdict_summaries[judge] = summarize_verdicts(
                verdicts=verdicts, judge=judge
            )

        return {"scenarios": results, "averageScores": overall_average_scores}

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
        _FUNC_NAME: str = self.simulate_single_scenario.__name__

        self.logger.info(f"[{_FUNC_NAME}] Starting simulation for script: {script.id}")
        all_attempts_scores: Dict[str, List[float]] = defaultdict(list)
        all_attempts_verdicts: Dict[str, List[str]] = defaultdict(list)

        async def simulate_attempt(attempt_number: int) -> Dict[str, Any]:
            self.logger.info(f"[{_FUNC_NAME}] Running attempt: {attempt_number + 1}/{attempts}")
            start_time = time.time()

            collected_scores: Dict[str, List[Any]] = defaultdict(list)
            collected_verdicts: Dict[str, List[str]] = defaultdict(list)
            # TODO-2: Remove the 'conversation_id' from 'simulate_interactions' signature.
            conversation_id = f"<CONV-ID:{str(uuid.uuid4())}>"

            initial_interaction_results = await self.simulate_interactions(
                script=script,
                conversation_id=conversation_id,
                evaluation_verdicts=collected_verdicts,
                collected_scores=collected_scores,
            )

            if initial_interaction_results["interactionType"] in (
                "handoff",
                "newBooking",
            ):
                inbound_interactions_results = None
            else:
                inbound_interactions_results = await self.simulate_interactions(
                    script=script,
                    conversation_id=conversation_id,
                    evaluation_verdicts=collected_verdicts,
                    collected_scores=collected_scores,
                )

            single_attempt_scores = calculate_average_scores(collected_scores)

            for target, scores in single_attempt_scores.items():
                all_attempts_scores[target].append(scores)

            for judge, verdicts in collected_verdicts.items():
                all_attempts_verdicts[judge].extend(verdicts)

            elapsed_time = time.time() - start_time
            all_attempts_scores["processingTime"].append(elapsed_time)

            self.logger.info(
                f"[simulate_single_scenario] Attempt {attempt_number + 1} completed in {elapsed_time:.2f}s\n---"
            )

            return {
                "attemptId": attempt_number + 1,
                "conversationId": conversation_id,
                "totalDurationSeconds": elapsed_time,
                "initialInteraction": initial_interaction_results,
                "inboundInteractions": inbound_interactions_results,
                "globalJustification": collected_verdicts,
                "averageScores": single_attempt_scores,
            }

        attempt_tasks = [simulate_attempt(i) for i in range(attempts)]
        attempt_results = await asyncio.gather(*attempt_tasks, return_exceptions=False)

        average_scores = calculate_average_scores(all_attempts_scores)
        for judge_, verdicts_ in all_attempts_verdicts.items():
            self.evaluation_verdicts[judge_].extend(verdicts_)

        self.logger.info(
            f"[simulate_single_conversation] average scores:\n{average_scores}\n---"
        )

        return {
            "scenarioId": script.scenario_title.replace(" ", "-"),
            "scenarioName": script.scenario_title,
            "attempts": attempt_results,
            "averageScores": average_scores,
        }

    async def simulate_interactions(
        self,
        script: ConversationScript,
        conversation_id: str,
        evaluation_verdicts: Dict[str, List[str]],
        collected_scores: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """
        Simulate inbound interactions for a scenario.

        Args:
            script (ConversationScript): The script to simulate.
            conversation_id (str): The conversation ID.
            evaluation_verdicts(Dict[str, List[str]]): evaluation verdict for each evaluator.
            collected_scores(Dict[str, List[Any]]): collected scores for each target.

        Returns:
            List[Dict[str, Any]]: The results of the inbound interactions simulation.
        """
        _FUNC_NAME: str = self.simulate_interactions.__name__

        self.logger.info(f"[{_FUNC_NAME}] Starting interactions simulation..")
        start_time = time.time()

        results = []
        interactions = script.interactions

        for interaction in interactions:
            interaction.conversation_id = conversation_id
            user_message = interaction.user_message

            # TODO-3: Add payload prep here.

            response = await async_vla_request(
                url=self._endpoint,
                headers=self._headers,
                # TODO-4: Adjust the payload dump that needs to be passed for the request.
                payload=interaction.model_dump(by_alias=True),
            )

            reference_reply = interaction.reference_reply
            reference_metadata = interaction.reference_metadata
            reference_guardrail_flag: bool = interaction.guardrail_flag

            if not response or response.status_code != 200:
                self.logger.error("[simulate_inbound_interaction] VLA request failed.")
                result = {
                    "user_message": user_message,
                    "generated_reply": "VLA Request failed",
                    "reference_reply": reference_reply,
                    "generated_metadata": {},
                    "reference_metadata": reference_metadata,
                    "guardrail_details": None,
                    "evaluation_results": {},
                }
                results.append(result)
                continue

            # TODO-5: Use the loader to build a pydantic model instance of the agent response.
            # TODO-6: Extract directly the response text from the model.
            interaction_details = extract_interaction_details(
                response_text=response.text
            )

            generated_reply = interaction_details.generated_reply
            generated_metadata = interaction_details.generated_metadata
            extracted_guardrail_flag: bool = (
                    interaction_details.guardrail_details is not None
            )

            evaluation_results = await self.evaluate_interaction(
                generated_reply=generated_reply,
                reference_reply=reference_reply,
                generated_metadata=generated_metadata,
                reference_metadata=reference_metadata,
                generated_guardrail=extracted_guardrail_flag,
                reference_guardrail=reference_guardrail_flag,
            )

            self.store_evaluation_results(
                results=evaluation_results,
                evaluation_verdicts=evaluation_verdicts,
                collected_scores=collected_scores,
            )

            elapsed_time = time.time() - start_time
            self.logger.info(
                f"[simulate_initial_interaction] Simulation complete in {elapsed_time:.2f} seconds.\n---"
            )

            result = {
                "user_message": user_message,
                "generated_reply": generated_reply,
                "reference_reply": reference_reply,
                "generated_metadata": generated_metadata,
                "reference_metadata": reference_metadata,
                "guardrail_details": interaction_details.guardrail_details,
                "evaluation_results": evaluation_results.model_dump(),
            }

            results.append(result)

        return results

    async def evaluate_interaction(
        self,
        generated_reply: str,
        reference_reply: str,
        generated_metadata: Dict[str, Any],
        reference_metadata: Dict[str, Any],
        generated_guardrail: bool,
        reference_guardrail: bool,
    ) -> InteractionEvaluationResult:
        """
        Evaluate an interaction using OpenAI and Ionos evaluation services.

        Args:
            generated_reply (str): The generated agent reply.
            reference_reply (str): The reference agent reply.
            generated_metadata (Dict[str, Any]): The generated metadata.
            reference_metadata (Dict[str, Any]): The reference metadata.
            generated_guardrail (bool): generated handoff/guardrail flag.
            reference_guardrail (bool): reference handoff/guardrail flag.

        Returns:
            InteractionEvaluationResult: The evaluation results.
        """
        openai_eval_task = self.evaluation_service.evaluate_response(
            provider="openai",
            output_text=generated_reply,
            reference_text=reference_reply,
        )

        ionos_eval_task = self.evaluation_service.evaluate_response(
            provider="ionos",
            output_text=generated_reply,
            reference_text=reference_reply,
        )

        openai_reply_evaluation, ionos_reply_evaluation = await asyncio.gather(
            openai_eval_task, ionos_eval_task
        )

        extracted_metadata_evaluation = evaluate_metadata(
            expected=reference_metadata,
            actual=generated_metadata,
        )

        guardrail_flag = 1 if generated_guardrail == reference_guardrail else 0

        return InteractionEvaluationResult(
            openaiReplyEvaluation=openai_reply_evaluation,
            ionosReplyEvaluation=ionos_reply_evaluation,
            extractedMetadataEvaluation=extracted_metadata_evaluation,
            guardrailFlag=guardrail_flag,
        )

    @staticmethod
    def store_evaluation_results(
        results: InteractionEvaluationResult,
        evaluation_verdicts: Dict[str, List[str]],
        collected_scores: Dict[str, List[Any]],
    ) -> None:
        """
        Store the evaluation results in the evaluation summary.

        Args:
            results (InteractionEvaluationResult): The evaluation results to store.
            evaluation_verdicts (Dict[str, List[str]]): The evaluation summary.
            collected_scores (Dict[str, List[Any]]): The collected scores.
        """
        evaluation_verdicts["openaiJustificationSummary"].append(
            results.openaiReplyEvaluation.justification
        )
        evaluation_verdicts["ionosJustificationSummary"].append(
            results.ionosReplyEvaluation.justification
        )

        collected_scores["openai"].append(results.openaiReplyEvaluation.match_level)
        collected_scores["ionos"].append(results.ionosReplyEvaluation.match_level)
        collected_scores["metadata"].append(results.extractedMetadataEvaluation)
        collected_scores["guardrail"].append(results.guardrailFlag)
