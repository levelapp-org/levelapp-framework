"""levelapp/clients/ionos.py"""
import os
import uuid

from typing import Dict, Any
from ..core.base import BaseChatClient


class IonosClient(BaseChatClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = kwargs.get('base_url')
        self.api_key = kwargs.get('api_key') or os.environ.get("IONOS_API_KEY")
        if not self.api_key:
            raise ValueError("IONOS API key not set.")

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        return {
            "properties": {"input": message},
            "option": {
                "top-k": 5,
                "top-p": 0.9,
                "temperature": 0.0,
                "max_tokens": 150,
                "seed": uuid.uuid4().int & ((1 << 16) - 1),
            },
        }
