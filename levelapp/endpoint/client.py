"""levelapp/endpoint/client.py"""
import os
import httpx
import asyncio
import backoff
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

    # Timeouts (seconds)
    connect_timeout: int = 10
    read_timeout: int = 60
    write_timeout: int = 10
    pool_timeout: int = 10

    # Concurrency
    max_parallel_requests: int = 50
    max_connections: int = 50
    max_keepalive_connections: int = 50

    # Retries
    retry_count: int = 5
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 60.0

    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.startswith('/'):
            return f"/{v}"
        return v


@dataclass
class ClientResult:
    success: bool
    response: httpx.Response | None = None
    error: Exception | None = None


@dataclass
class APIClient:
    """HTTP client for REST API interactions"""
    config: EndpointConfig
    client: httpx.AsyncClient = field(init=False)
    semaphore: asyncio.Semaphore = field(init=False)
    logger: logging.Logger = field(init=False)

    RETRYABLE_ERRORS = (
        httpx.ConnectTimeout,
        httpx.WriteTimeout,
        httpx.ReadTimeout,
        httpx.NetworkError
    )

    def __post_init__(self):
        self.logger = logging.getLogger(f"AsyncAPIClient.{self.config.name}")

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
                write=self.config.write_timeout,
                pool=self.config.pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
            ),
            follow_redirects=True,
        )

        self.semaphore = asyncio.Semaphore(self.config.max_parallel_requests)

    async def __aenter__(self) -> "APIClient":
        return self

    async def __aexit__(self, *args) -> None:
        try:
            if hasattr(self, 'client') and not self.client.is_closed:
                self.logger.warning("[APIClient] Client not properly closed, forcing cleanup.")
                asyncio.create_task(self.client.aclose())
        except Exception as e:
            self.logger.error(f"[APIClient] Error closing client: {e}")

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

    def _on_backoff(self, details):
        """Callback for backoff logging"""
        self.logger.warning(
            f"[APIClient] Retry {details['tries']}/{self.config.retry_count} "
            f"after {details['wait']:.2f}s (error: {details['exception'].__class__.__name__})"
        )

    def _on_giveup(self, details):
        """Callback when all retries exhausted"""
        self.logger.error(
            f"[APIClient] Gave up after {details['tries']} tries, "
            f"elapsed: {details['elapsed']:.2f}s"
        )

    async def send_request(
            self,
            payload: Dict[str, Any] | None = None,
            query_params: Dict[str, Any] | None = None,
    ) -> httpx.Response:
        headers = self._build_headers()

        async with self.semaphore:
            response = await self.client.request(
                    method=self.config.method.value,
                    url=self.config.path,
                    json=payload,
                    params=query_params,
                    headers=headers,
                )

            if response.is_error:
                response.raise_for_status()

            return response

    async def execute(
            self,
            payload: Dict[str, Any] | None = None,
            query_params: Dict[str, Any] | None = None,
    ) -> ClientResult:
        """
        Execute asynchronous REST API request with retry logic using backoff.

        Retries on transient errors with exponential backoff and jitter.
        Non-retryable errors (pool exhaustion, HTTP errors) are raised immediately.
        """
        @backoff.on_exception(
            backoff.expo,
            self.RETRYABLE_ERRORS,
            max_tries=self.config.retry_count,
            max_time=self.config.retry_backoff_max,
            jitter=backoff.full_jitter,
            on_backoff=self._on_backoff,
            on_giveup=self._on_giveup,
            raise_on_giveup=True,
        )
        async def _execute_with_retry() -> httpx.Response:
            return await self.send_request(payload=payload, query_params=query_params)

        try:
            response = await _execute_with_retry()
            response.raise_for_status()
            return ClientResult(success=True, response=response)

        except httpx.HTTPStatusError as exc:
            exc_response = exc.response if hasattr(exc, "response") else None
            return ClientResult(success=False, response=exc_response, error=exc)

        except Exception as exc:
            exc_response = exc.response if hasattr(exc, "response") else None
            return ClientResult(success=False, response=exc_response, error=exc)
