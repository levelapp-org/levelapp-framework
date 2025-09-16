"""levelapp/core/evaluator.py"""
from functools import lru_cache
from importlib.metadata import metadata
from typing import List, Dict, Any
from collections import defaultdict
from pydantic import BaseModel, Field

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying,
    RetryError,
)

from levelapp.clients import ClientRegistry
from levelapp.comparator import MetricsManager, MetadataComparator
from levelapp.config.prompts import EVAL_PROMPT_TEMPLATE
from levelapp.core.base import BaseEvaluator, BaseChatClient
from levelapp.aspects import MonitoringAspect, MetricType, logger, DataLoader


class Evidence(BaseModel):
    """Evidence details for evaluation."""
    covered_points: List[str] = Field(
        default_factory=list,
        description="Key points covered the agent reply covered (<= 3 items)"
    )
    missing_or_wrong: List[str] = Field(
        default_factory=list,
        description="Key points the agent reply missed or contradicted (<= 3 items)"
    )


class JudgeEvaluationResults(BaseModel):
    """Structured result of an interaction evaluation."""
    provider: str = Field(..., description="The provider name, e.g., 'openai', 'ionos'")
    score: int = Field(..., ge=0, le=3, description="Evaluation score between 0 and 3")
    label: str = Field(..., description="The label of the evaluation result")
    justification: str = Field(..., description="Short explanation of the evaluation result")
    evidence: Evidence = Field(default_factory=Evidence, description="Detailed evidence for the evaluation")
    raw_response: Dict[str, Any] = Field(..., description="Full unprocessed API response")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the evaluation result")

    @classmethod
    def from_parsed(cls, provider: str, parsed: Dict[str, Any], raw: Dict[str, Any]) -> "JudgeEvaluationResults":
        """
        Build a model instance from the provided data.

        Args:
            provider (str): The provider name.
            parsed (Dict[str, Any]): The parsed response data.
            raw (Dict[str, Any]): The raw response data.

        Returns:
            JudgeEvaluationResults: The constructed evaluation result instance.
        """
        content = parsed.get("output", {})
        metadata = parsed.get("metadata", {})
        return cls(
            provider=provider,
            score=content.get("score", 0),
            label=content.get("label", "N/A"),
            justification=content.get("justification", "N/A"),
            evidence=Evidence(**content.get("evidence", {})),
            raw_response=raw,
            metadata=metadata,
        )


class JudgeEvaluator(BaseEvaluator):
    def __init__(self):
        self.prompt_template = EVAL_PROMPT_TEMPLATE
        self.clients = defaultdict(BaseChatClient)

    def register_client(self, provider: str, client: BaseChatClient):
        self.clients[provider] = client

    @lru_cache(maxsize=1024)
    def _build_prompt(self, user_input: str, generated_text: str, reference_text: str) -> str:
        return self.prompt_template.format(
            user_input=user_input,
            generated_text=generated_text,
            reference_text=reference_text
        )

    @retry(
        retry=retry_if_exception_type((TimeoutError, ValueError, RuntimeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def evaluate(
            self,
            generated_data: str,
            reference_data: str,
            user_input: str,
            provider: str,
    ) -> JudgeEvaluationResults | None:
        prompt = self._build_prompt(
            user_input=user_input,
            generated_text=generated_data,
            reference_text=reference_data
        )
        client = ClientRegistry.get(provider=provider)

        try:
            response = client.call(message=prompt)
            logger.info(f"[{provider}] Evaluation: {response}\n{'---' * 10}")
            parsed = client.parse_response(response=response)
            return JudgeEvaluationResults.from_parsed(provider=provider, parsed=parsed, raw=response)

        except Exception as e:
            logger.error(f"[{provider}] Evaluation failed: {e}", exc_info=True)
            return JudgeEvaluationResults(
                provider=provider,
                score=0,
                label="N/A",
                justification="N/A",
                evidence=Evidence(covered_points=[], missing_or_wrong=[]),
                raw_response={},
                metadata={}
            )

    @MonitoringAspect.monitor(name="judge_evaluation", category=MetricType.API_CALL)
    async def async_evaluate(
            self,
            generated_data: str,
            reference_data: str,
            user_input: str,
            provider: str,
    ) -> JudgeEvaluationResults | None:
        prompt = self._build_prompt(
            user_input=user_input,
            generated_text=generated_data,
            reference_text=reference_data
        )
        client = ClientRegistry.get(provider=provider)

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((TimeoutError, ValueError, RuntimeError)),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=True,
            ):
                with attempt:
                    response = await client.acall(message=prompt)
                    logger.info(f"[{provider}] Async evaluation:\n{response}\n{'---' * 10}")
                    parsed = client.parse_response(response=response)
                    return JudgeEvaluationResults.from_parsed(provider=provider, parsed=parsed, raw=response)

        except RetryError as e:
            logger.error(f"[{provider}] Async evaluation failed after retries: {e}", exc_info=True)
            return JudgeEvaluationResults(
                provider=provider,
                score=0,
                label="N/A",
                justification="N/A",
                evidence=Evidence(covered_points=[], missing_or_wrong=[]),
                raw_response={},
                metadata={}
            )


# TODO-0: Needs more refinement (to revisit later).
class MetadataEvaluator(BaseEvaluator):
    def __init__(self):
        self.data_loader = DataLoader()
        self.comparator = MetadataComparator()
        self.metrics_manager = MetricsManager()

    def evaluate(
            self,
            generated_data: str | Dict[str, Any],
            reference_data: str | Dict[str, Any],
            metrics_mapping: Any | None = None,
    ) -> Dict[str, float]:
        gen_data = self.data_loader.load_data(data=generated_data, model_name="GeneratedMetadata")
        ref_data = self.data_loader.load_data(data=reference_data, model_name="ReferenceMetadata")

        if metrics_mapping:
            self.comparator.metrics_manager = metrics_mapping

        self.comparator.metrics_manager = self.metrics_manager
        self.comparator.generated_data = gen_data
        self.comparator.reference_data = ref_data

        output = self.comparator.run(indexed_mode=False)
        logger.info(f"Comparison results:\n{output}\n---")
        results: Dict[str, float] = {}

        for k, v in output.items():
            field = v.get("field_name", "N/A")
            score = v.get("set_scores", -1)
            results[field] = int(score[0]) if isinstance(score, list) else int(score)

        return results

    async def async_evaluate(self, generated_data: str | Dict[str, Any], reference_data: str | Dict[str, Any]):
        """Not implemented yet."""
        pass


if __name__ == '__main__':

    # class Pirate(BaseModel):
    #     name: str
    #     role: str
    #
    # class Crew(BaseModel):
    #     name: str = "Straw Hats"
    #     crew: List[Pirate] = [Pirate(name="Monkey D. Luffy", role="Captain")]
    #     details: Dict[str, Any] = {"Ship": "SunnyGo", "Reputation": "Good"}
    #
    #
    # straw_hats = Crew(
    #     name="Straw Hat Pirates",
    #     crew=[
    #         Pirate(name="Monkey D. Luffy", role="Captain"),
    #         Pirate(name="Roronoa Zoro", role="Swordsman"),
    #         Pirate(name="Nami", role="Navigator"),
    #         Pirate(name="Usopp", role="Sniper"),
    #         Pirate(name="Sanji", role="Cook")
    #     ],
    #     details={"ship": "Thousand Sunny", "reputation": "Legendary", "bounty": "3,161,000,100+ Berries"}
    # )
    #
    # fake_straw_hats = Crew(
    #     name="Straw Hat Pirates",
    #     crew=[
    #         Pirate(name="Demalo Black", role="Captain"),
    #         Pirate(name="Manjaro", role="Swordsman"),
    #         Pirate(name="Chocolat", role="Navigator"),
    #         Pirate(name="Mounblutain", role="Sniper"),
    #         Pirate(name="Drip", role="Cook")
    #     ],
    #     details={"ship": "", "reputation": "Fake", "bounty": "0 Berries"}
    # )
    #
    # metadata_evaluator = MetadataEvaluator()
    # results_ = metadata_evaluator.evaluate(
    #     generated_data=fake_straw_hats.model_dump(),
    #     reference_data=straw_hats.model_dump()
    # )
    # print(f"Metadata evaluation results:\n{results_}\n---")

    loader_ = DataLoader()
    json_data = loader_.load_configuration(path="../../src/data/conversation_example_1.json")
    print(f"json data:\n{json_data}\n---")
    model_ = loader_.load_data(data=json_data, model_name="ScriptsBatch")
    print(f"model dump:\n{model_.model_dump()}\n---")
    metadata = model_.scripts[-1].reference_metadata
    print(f"reference metadata:\n{metadata}\n---")
