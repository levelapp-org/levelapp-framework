"""levelapp/clients/groq.py"""
import os
from typing import Dict, Any
from levelapp.core.base import BaseChatClient


class GroqClient(BaseChatClient):
    """
    Client for interacting with Groq's Chat Completions API.

    This implementation adapts requests and responses to the Groq API
    format, which is OpenAI-compatible but with Groq-specific models and endpoints.

    Attributes:
        model (str): Target model ID (default: "llama-3.3-70b-versatile").
        base_url (str): Base endpoint for Groq API (default: https://api.groq.com/openai/v1).
        api_key (str): Authentication token for the Groq API.
        max_tokens (int): Maximum tokens allowed in the completion.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get('model') or os.environ.get('GROK_MODEL')
        self.base_url = kwargs.get('base_url') or "https://api.groq.com/openai/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('GROQ_API_KEY')
        self.max_tokens = kwargs.get('max_tokens') or 4096

        if not self.api_key:
            raise ValueError("Groq API key not set")

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
        Build HTTP headers for the Groq API request.

        Returns:
            Dict[str, str]: Headers with authentication and content type.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        """
        Construct the JSON payload for the Groq Chat Completions API.

        Args:
            message (str): User input or prompt to evaluate.

        Returns:
            Dict[str, Any]: Payload containing model ID, messages, and token limit.
        """
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": self.max_tokens,
        }

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and normalize the Groq API response.

        - Extracts text output from `choices[0].message.content`.
        - Attempts to JSON-parse the result if it contains structured content.
        - Collects token usage metadata from `usage`.

        Args:
            response (Dict[str, Any]): Raw JSON response from Groq.

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
        return {"output": parsed, "metadata": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
