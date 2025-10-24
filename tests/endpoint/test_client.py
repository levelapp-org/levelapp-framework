import os
import httpx
import pytest

from unittest.mock import AsyncMock, patch

from levelapp.endpoint.schemas import HttpMethod, HeaderConfig
from levelapp.endpoint.client import EndpointConfig, APIClient


@pytest.mark.asyncio
async def test_build_headers_with_secure_end(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "secret123")

    cfg = EndpointConfig(
        name="secure_endpoint",
        base_url="https://api.example.com",
        path="/data",
        method=HttpMethod.GET,
        headers=[
            HeaderConfig(name="Authorization", value="API_TOKEN", secure=True),
            HeaderConfig(name="Accept", value="application/json", secure=False),
        ],
    )

    client = APIClient(cfg)
    headers = client._build_headers()

    assert headers["Authorization"] == "secret123"
    assert headers["Accept"] == "application/json"


@pytest.mark.asyncio
async def test_build_headers_missing_secure_end(caplog):
    cfg = EndpointConfig(
        name="missing_env",
        base_url="https://api.example.com",
        path="/data",
        method=HttpMethod.GET,
        headers=[
            HeaderConfig(name="Authorization", value="MISSING_ENV", secure=True),
        ]
    )

    client = APIClient(cfg)
    headers = client._build_headers()

    assert "Authorization" not in headers
    assert "env var 'MISSING_ENV' not found" in caplog.text


@pytest.mark.asyncio
async def test_execute_success(monkeypatch):
    cfg = EndpointConfig(
        name="mock_success",
        base_url="https://api.example.com",
        path="/items",
        method=HttpMethod.GET,
        retry_count=2
    )

    mock_request = httpx.Request(method="GET", url="https://api.example.com")
    mock_response = httpx.Response(
        status_code=200,
        json={"status": "ok"},
        request=mock_request,
    )

    async with APIClient(cfg) as client:
        with patch.object(client.client, "request", AsyncMock(return_value=mock_response)):
            response = await client.execute()
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_execute_retries_and_fails(monkeypatch):
    cfg = EndpointConfig(
        name="mock_fail",
        base_url="https://api.example.com",
        path="/fail",
        method=HttpMethod.GET,
        retry_count=2,
        retry_backoff=0.01,
    )

    async with APIClient(cfg) as client:
        with patch.object(client.client, "request", AsyncMock(side_effect=httpx.RequestError("Sike! Gott'em!"))):
            with pytest.raises(httpx.RequestError):
                await client.execute()


@pytest.mark.asyncio
async def test_execute_http_error(monkeypatch):
    cfg = EndpointConfig(
        name="mock_http_error",
        base_url="https://api.example.com",
        path="/error",
        method=HttpMethod.GET,
        retry_count=1,
    )

    mock_request = httpx.Request(method="GET", url="https://api.example.com")
    mock_response = httpx.Response(
        status_code=500,
        json={"error": "server"},
        request=mock_request,
    )

    async with APIClient(cfg) as client:
        with patch.object(client.client, "request", AsyncMock(return_value=mock_response)):
            with pytest.raises(httpx.HTTPStatusError):
                await client.execute()
