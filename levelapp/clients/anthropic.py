"""levelapp/clients/anthropic.py"""
import os

from typing import Dict, Any

from levelapp.core.base import BaseChatClient


class AnthropicClient(BaseChatClient):
    """
    Client for interacting with Anthropic's Claude API.

    This implementation adapts requests and responses to the Anthropic API
    format, including authentication, versioning headers, and structured
    response parsing.

    Attributes:
        model (str): Target model ID (default: "claude-sonnet-4-20250514").
        version (str): API version header required by Anthropic (default: "2023-06-01").
        base_url (str): Base endpoint for Anthropic API (default: https://api.anthropic.com/v1).
        api_key (str): Authentication token for Anthropic API.
        max_tokens (int): Maximum tokens allowed in the response.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get('model') or "claude-sonnet-4-20250514"
        self.version = kwargs.get('version') or "2023-06-01"
        self.base_url = kwargs.get("base_url") or "https://api.anthropic.com/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        self.max_tokens = kwargs.get('max_tokens') or 1024

        if not self.api_key:
            raise ValueError("Anthropic API key not set.")

    @property
    def endpoint_path(self) -> str:
        """
        API-specific endpoint path for message-based chat.

        Returns:
            str: "/messages"
        """
        return "/messages"

    def _build_endpoint(self) -> str:
        """
        Construct the full API endpoint URL.

        Returns:
            str: Concatenation of base_url and endpoint_path.
        """
        return f"{self.base_url}/{self.endpoint_path.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for the Anthropic API request.

        Required headers include:
        - `x-api-key`: Authentication token.
        - `anthropic-version`: API version string.
        - `content-type`: Always "application/json".

        Returns:
            Dict[str, str]: Headers with authentication and API version.
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json"
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        """
        Construct the JSON payload for the Anthropic Messages API.

        Args:
            message (str): User input or prompt to evaluate.

        Returns:
            Dict[str, Any]: Payload containing model ID, messages, and token limit.
        """
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": self.max_tokens
        }

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and normalize the Anthropic API response.

        - Extracts text output from `content[0].text`.
        - Attempts to JSON-parse the output if it contains structured data.
        - Collects token usage metadata from `usage`.

        Args:
            response (Dict[str, Any]): Raw JSON response from Anthropic.

        Returns:
            Dict[str, Any]: {
                "output": Parsed model output (dict or str),
                "metadata": {
                    "input_tokens": int,
                    "output_tokens": int
                }
            }
        """
        input_tokens = response.get("usage", {}).get("input_tokens", 0)
        output_tokens = response.get("usage", {}).get("output_tokens", 0)
        output = response.get("content", {})[0].get("text", "")
        parsed = self.sanitizer.safe_load_json(text=output)
        return {'output': parsed, 'metadata': {'input_tokens': input_tokens, 'output_tokens': output_tokens}}
