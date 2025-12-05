"""levelapp/clients/gemini.py"""
import os
from typing import Dict, Any

from levelapp.core.base import BaseChatClient


class GeminiClient(BaseChatClient):
    """
    Client for interacting with Google's Gemini API.

    This implementation adapts requests and responses to the Gemini API
    format, including content structure, headers, and token usage reporting.

    Attributes:
        model (str): Target model ID (default: "gemini-2.0-flash-exp").
        base_url (str): Base endpoint for Gemini API (default: https://generativelanguage.googleapis.com/v1beta).
        api_key (str): Authentication token for the Gemini API.
        max_tokens (int): Maximum tokens allowed in the completion.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get('model') or os.environ.get("GEMINI_MODEL_ID")
        self.base_url = kwargs.get('base_url') or "https://generativelanguage.googleapis.com/v1beta"
        self.api_key = kwargs.get('api_key') or os.environ.get('GEMINI_API_KEY')
        self.max_tokens = kwargs.get('max_tokens') or 1024

        if not self.api_key:
            raise ValueError("Gemini API key not set")

    @property
    def endpoint_path(self) -> str:
        """
        API-specific endpoint path for content generation.

        Returns:
            str: Formatted endpoint with model name.
        """
        return f"/models/{self.model}:generateContent"

    def _build_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for the Gemini API request.

        Gemini uses x-goog-api-key header instead of Authorization: Bearer.

        Returns:
            Dict[str, str]: Headers with API key and content type.
        """
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        """
        Construct the JSON payload for the Gemini generateContent API.

        Args:
            message (str): User input or prompt to evaluate.

        Returns:
            Dict[str, Any]: Payload containing model ID, contents structure, and token limit.
        """
        return {
            "contents": [
                {
                    "parts": [
                        {
                            "text": message
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
            }
        }

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and normalize the Gemini API response.

        - Extracts text output from `candidates[0].content.parts[0].text`.
        - Attempts to JSON-parse the result if it contains structured content.
        - Collects token usage metadata from `usageMetadata`.

        Args:
            response (Dict[str, Any]): Raw JSON response from Gemini.

        Returns:
            Dict[str, Any]: {
                "output": Parsed model output (dict or str),
                "metadata": {
                    "input_tokens": int,
                    "output_tokens": int,
                    "total_tokens": int,
                    "finish_reason": str
                }
            }
        """
        # Extract text from candidates
        candidates = response.get("candidates", [{}])
        candidate = candidates[0] if candidates else {}
        content = candidate.get("content", {})
        parts = content.get("parts", [{}])
        part = parts[0] if parts else {}
        output_text = part.get("text", "")

        # Extract token usage
        usage_metadata = response.get("usageMetadata", {})
        input_tokens = usage_metadata.get("promptTokenCount", 0)
        output_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", 0)

        # Extract finish reason
        finish_reason = candidate.get("finishReason", "UNKNOWN")

        # Try to parse as JSON if it looks like structured data
        parsed = self.sanitizer.safe_load_json(text=output_text)

        return {
            "output": parsed,
            "metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "finish_reason": finish_reason
            }
        }
