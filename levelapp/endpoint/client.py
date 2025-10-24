"""levelapp/endpoint/client.py"""
import os
import httpx
import asyncio
import logging

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from levelapp.endpoint.schemas import HttpMethod, HeaderConfig, RequestSchemaConfig, ResponseMappingConfig


class EndpointConfig(BaseModel):
    """Complete endpoint configuration."""
    name: str
    base_url: str
    path: str
    method: HttpMethod
    headers: List[HeaderConfig] = Field(default_factory=list)
    request_schema: List[RequestSchemaConfig] = Field(default_factory=list)
    response_mapping: List[ResponseMappingConfig] = Field(default_factory=list)
    timeout: int = Field(default=30)
    retry_count: int = Field(default=3)
    retry_backoff: float = Field(default=1.0)

    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.startswith('/'):
            return f"/{v}"
        return v


@dataclass
class APIClient:
    """HTTP client for REST API interactions"""
    config: EndpointConfig
    client: httpx.AsyncClient = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.logger = logging.getLogger(f"AsyncAPIClient.{self.config.name}")
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            follow_redirects=True
        )

    async def __aenter__(self) -> "APIClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.client.aclose()

    def _build_headers(self) -> Dict[str, str]:
        """Build headers with secure value resolution."""
        headers = {}

        for header in self.config.headers:
            if header.secure:
                value = os.getenv(header.value)
                if value is None:
                    self.logger.warning(f"Secure header '{header.name}' env var '{header.value}' not found")
                    continue
                headers[header.name] = value
            else:
                headers[header.name] = header.value

        return headers

    async def execute(
            self,
            payload: Dict[str, Any] | None = None,
            query_params: Dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Execute asynchronous REST API request with retry logic."""
        headers = self._build_headers()

        for attempt in range(self.config.retry_count):
            try:
                response = await self.client.request(
                    method=self.config.method.value,
                    url=self.config.path,
                    json=payload,
                    params=query_params,
                    headers=headers,
                )
                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                self.logger.error(f"HTTP {e.response.status_code}: {e}")
                if attempt == self.config.retry_count - 1:
                    raise

            except httpx.RequestError as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.retry_count - 1:
                    raise
                await asyncio.sleep(delay=self.config.retry_backoff * (attempt + 1))

        raise RuntimeError("Max retries exceeded")
