"""levelapp/clients/mistral.py"""
import os

from typing import Dict, Any
from levelapp.core.base import BaseChatClient


class MistralClient(BaseChatClient):
    """
    Client for interacting with the Mistral API.

    This implementation adapts requests and responses to the Mistral API
    format, handling authentication, request payload structure, and
    response parsing into a normalized format.

    Attributes:
        model (str): Target model identifier (default: "mistral-large-latest").
        base_url (str): Base endpoint for Mistral API (default: https://api.mistral.ai/v1).
        api_key (str): Authentication token for Mistral API.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model") or "mistral-large-latest"
        self.base_url = kwargs.get('base_url') or "https://api.mistral.ai/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('MISTRAL_API_KEY')

        if not self.api_key:
            raise ValueError("Missing API key not set.")

    @property
    def endpoint_path(self) -> str:
        """
        API-specific endpoint path for chat completions.

        Returns:
            str: "/chat/completions"
        """
        return "/chat/completions"

    def _build_endpoint(self) -> str:
        """
        Construct the full API endpoint URL.

        Returns:
            str: Concatenation of base_url and endpoint_path.
        """
        return f"{self.base_url}/{self.endpoint_path.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for the Mistral API request.

        Required headers include:
        - `Authorization`: Bearer token for API access.
        - `Content-Type`: Always "application/json".
        - `Accept`: Expected response format ("application/json").

        Returns:
            Dict[str, str]: Headers including authentication and content type.
        """
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        """
        Construct the JSON payload for the Mistral chat completions API.

        Args:
            message (str): User input or prompt to evaluate.

        Returns:
            Dict[str, Any]: Payload containing model ID and user message.
        """
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
        }

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and normalize the Mistral API response.

        - Extracts text output from `choices[0].message.content`.
        - Attempts to JSON-parse the output if structured data is detected.
        - Collects token usage metadata from `usage`.

        Args:
            response (Dict[str, Any]): Raw JSON response from Mistral API.

        Returns:
            Dict[str, Any]: {
                "output": Parsed model output (dict or str),
                "metadata": {
                    "input_tokens": int,
                    "output_tokens": int
                }
            }
        """
        input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = response.get("usage", {}).get("completion_tokens", 0)
        output = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = self.sanitizer.safe_load_json(text=output)
        return {'output': parsed, 'metadata': {'input': input_tokens, 'output': output_tokens}}
