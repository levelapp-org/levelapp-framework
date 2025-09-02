"""levelapp/core/evaluator.py"""
import re
import json
import logging
from collections import defaultdict
from functools import lru_cache

from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying,
    RetryError,
)

from levelapp.clients import ClientRegistry
from levelapp.core.base import BaseEvaluator, BaseChatClient
from levelapp.utils.monitoring import MonitoringAspect, MetricType

logger = logging.getLogger(__name__)


class JudgeEvaluationResults(BaseModel):
    """Structured result of an interaction evaluation."""
    provider: str = Field(..., description="The provider name, e.g., 'openai', 'ionos'")
    match_level: int = Field(..., ge=-1, le=5, description="Evaluation score between -1 and 5")
    justification: str = Field(..., description="Short explanation of the evaluation result")
    raw_response: Dict[str, Any] = Field(..., description="Full unprocessed API response")

    @classmethod
    @field_validator("match_level", mode='before')
    def validate_match_level(cls, v):
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v

    @classmethod
    def from_raw(cls, provider: str, raw: Dict[str, Any]) -> "JudgeEvaluationResults":
        """
        Factory method to extract match_level and justification
        from raw responses for known providers.
        """
        match_level = -1
        justification = "Unable to parse response."

        try:
            if provider.lower() == "openai":
                # OpenAI content is in choices[0].message.content as JSON string
                content = raw["choices"][0]["message"]["content"]
                parsed = cls._safe_json_parse(content)
                match_level = parsed.get("match_level", match_level)
                justification = parsed.get("justification", justification)

            elif provider.lower() == "ionos":
                # IONOS puts it in properties.output (sometimes wrapped in ``` or extra text)
                output = raw["properties"]["output"]
                cleaned = cls._strip_code_fences(output)
                parsed = cls._safe_json_parse(cleaned)
                match_level = parsed.get("match_level", match_level)
                justification = parsed.get("justification", justification)

        except Exception as e:
            justification = f"Parsing error: {str(e)}"

        return cls(
            provider=provider,
            match_level=match_level,
            justification=justification,
            raw_response=raw
        )

    @staticmethod
    def _safe_json_parse(text: str) -> Dict[str, Any]:
        """Parse JSON safely, even if surrounded by extra spaces/newlines."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove triple backticks and language hints from code blocks."""
        return re.sub(r"^(```[a-zA-Z]*\n?)$", "", text.strip(), flags=re.MULTILINE)


# TODO-0: Move this to a separate file.
EVAL_PROMPT_TEMPLATE = """
Your task is to evaluate how well the agent's generated text matches the expected text.
Use the following classification criteria:

3 - Excellent Match: The generated text is virtually identical to the expected text with no meaningful differences.
2 - Good Match: The generated text closely matches the expected text with only minor wording differences.
1 - Moderate Match: The generated text captures the main ideas but has noticeable differences or omissions.
0 - Poor Match: The generated text has significant differences and misses several key points.

Expected Output:
\"\"\"
{reference_text}
\"\"\"

Agent's Output:
\"\"\"
{generated_text}
\"\"\"

Return your evaluation as a valid JSON object with exactly these keys:
{{"match_level": <an integer between 1 and 5>, "justification": <a brief explanation>}}

Output only the JSON object and nothing else.
"""


class JudgeEvaluator(BaseEvaluator):
    def __init__(self):
        self.prompt_template = EVAL_PROMPT_TEMPLATE
        self.clients = defaultdict(BaseChatClient)

    def register_client(self, provider: str, client: BaseChatClient):
        self.clients[provider] = client

    @lru_cache(maxsize=1024)
    def _build_prompt(self, generated_text: str, reference_text: str) -> str:
        return self.prompt_template.format(
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
            generated_text: str,
            reference_text: str
    ) -> JudgeEvaluationResults | None:
        prompt = self._build_prompt(generated_text=generated_text, reference_text=reference_text)
        client = ClientRegistry.get(provider=provider)

        try:
            response = client.call(message=prompt)
            logger.info(f"[{provider}] Evaluation: {response}\n{'---' * 10}")
            # TODO-2: Validate response structure using Pydantic or similar.
            return JudgeEvaluationResults.from_raw(provider=provider, raw=response)

        except Exception as e:
            logger.error(f"[{provider}] Evaluation failed: {e}", exc_info=True)
            return JudgeEvaluationResults(
                provider=provider,
                match_level=-1,
                justification="Unable to parse response.",
                raw_response={},
            )

    @MonitoringAspect.monitor(name="judge_evaluation", category=MetricType.API_CALL)
    async def async_evaluate(
            self,
            provider: str,
            generated_text: str,
            reference_text: str
    ) -> JudgeEvaluationResults | None:
        prompt = self._build_prompt(generated_text=generated_text, reference_text=reference_text)
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
                    return JudgeEvaluationResults.from_raw(provider=provider, raw=response)

        except RetryError as e:
            logger.error(f"[{provider}] Async evaluation failed after retries: {e}", exc_info=True)
            return JudgeEvaluationResults(
                provider=provider,
                match_level=-1,
                justification="Unable to parse response.",
                raw_response={},
            )
