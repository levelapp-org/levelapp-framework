"""levelapp/core/evaluator.py"""
import logging

from typing import Dict, Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying,
    RetryError,
)

from levelapp.core.base import BaseEvaluator, BaseChatClient


logger = logging.getLogger(__name__)

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


class InteractionEvaluator(BaseEvaluator):
    def __init__(self):
        self.clients: Dict[str, BaseChatClient] = {}

    def register_client(self, provider: str, client: BaseChatClient):
        self.clients[provider] = client

    @staticmethod
    def _build_prompt(generated_text: str, reference_text: str) -> str:
        return EVAL_PROMPT_TEMPLATE.format(generated_text=generated_text, reference_text=reference_text)

    def _get_client(self, provider: str) -> BaseChatClient:
        try:
            return self.clients[provider]
        except KeyError:
            raise ValueError(f"[InteractionEvaluator] Client for provider '{provider}' is not registered.")

    @retry(
        retry=retry_if_exception_type((TimeoutError, ValueError, RuntimeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def evaluate(self, provider: str, generated_text: str, reference_text: str) -> Dict[str, Any]:
        prompt = self._build_prompt(generated_text, reference_text)
        client = self._get_client(provider)

        try:
            response = client.call(message=prompt)
            logger.info(f"[{provider}] Evaluation: {response}")
            return response

        except Exception as e:
            logger.error(f"[{provider}] Evaluation failed: {e}", exc_info=True)
            return {"match_level": -1, "justification": f"Exception during evaluation: {str(e)}"}

    async def async_evaluate(self, provider: str, generated_text: str, reference_text: str) -> Dict[str, Any]:
        prompt = self._build_prompt(generated_text, reference_text)
        client = self._get_client(provider)

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((TimeoutError, ValueError, RuntimeError)),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=True,
            ):
                with attempt:
                    response = await client.acall(message=prompt)
                    logger.info(f"[{provider}] Async evaluation: {response}")
                    return response

        except RetryError as e:
            logger.error(f"[{provider}] Async evaluation failed after retries: {e}", exc_info=True)
            return {
                "match_level": -1,
                "justification": f"Async evaluation failed after retries: {str(e)}"
            }
