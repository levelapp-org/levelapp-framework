"""levelapp/core/evaluator.py"""
from functools import lru_cache
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
from levelapp.config.prompts import EVAL_PROMPT_TEMPLATE
from levelapp.core.base import BaseEvaluator, BaseChatClient
from levelapp.aspects import MonitoringAspect, MetricType, logger


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
        return cls(
            provider=provider,
            score=parsed.get("score", 0),
            label=parsed.get("label", "N/A"),
            justification=parsed.get("justification", "N/A"),
            evidence=Evidence(**parsed.get("evidence", {})),
            raw_response=raw
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
            provider: str,
            user_input: str,
            generated_text: str,
            reference_text: str
    ) -> JudgeEvaluationResults | None:
        prompt = self._build_prompt(
            user_input=user_input,
            generated_text=generated_text,
            reference_text=reference_text
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
                raw_response={}
            )

    @MonitoringAspect.monitor(name="judge_evaluation", category=MetricType.API_CALL)
    async def async_evaluate(
            self,
            provider: str,
            user_input: str,
            generated_text: str,
            reference_text: str
    ) -> JudgeEvaluationResults | None:
        prompt = self._build_prompt(
            user_input=user_input,
            generated_text=generated_text,
            reference_text=reference_text
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
                    logger.info(f"[{provider}] Async evaluation: (response type:{type(response)})\n{response}\n{'---' * 10}")
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
                raw_response={}
            )
