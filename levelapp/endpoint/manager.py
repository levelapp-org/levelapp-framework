"""levelapp/endpoint/manager.py"""
import httpx
import yaml
import logging

from pathlib import Path
from typing import Dict, List, Any
from pydantic import ValidationError

from levelapp.endpoint.schemas import ResponseMappingConfig
from levelapp.endpoint.tester import ConnectivityTester
from levelapp.endpoint.client import EndpointConfig, APIClient
from levelapp.endpoint.parsers import RequestPayloadBuilder, ResponseDataExtractor


class EndpointConfigManager:
    """Manages endpoint configurations and creates testers."""
    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path
        self.endpoints: Dict[str, EndpointConfig] = {}
        self.logger = logging.getLogger("ConfigurationManager")

        if config_path:
            self._load_config()

    def _load_config(self) -> None:
        """Load and validate YAML configuration file."""
        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)

            for endpoint_data in data.get("endpoints", []):
                config = EndpointConfig.model_validate(endpoint_data)
                self.endpoints[config.name] = config
                self.logger.info(f"Loaded endpoint config: {config.name}")

        except ValidationError as e:
            self.logger.error(f"Failed to load endpoint config: {e}")

        except Exception as e:
            self.logger.error(f"Failed to load endpoint config: {e}", exc_info=e)
            raise RuntimeError("Failed to extract endpoints data from YAML file:\n{e}")

    def set_endpoints(self, endpoints_config: List[EndpointConfig]):
        for endpoint in endpoints_config:
            try:
                config = EndpointConfig.model_validate(endpoint)
                self.endpoints[config.name] = config

            except ValidationError as e:
                self.logger.error(f"Failed to load endpoint config: {e}", exc_info=e)
                continue

    def build_response_mapping(self, content: List[Dict[str, Any]]) -> List[ResponseMappingConfig]:
        mappings = []
        for el in content:
            try:
                mappings.append(ResponseMappingConfig.model_validate(el))
            except ValidationError as e:
                self.logger.error(f"Failed to validate response mapping: {e}", exc_info=e)

        return mappings

    async def send_request(
            self,
            endpoint_config: EndpointConfig,
            context: Dict[str, Any],
            contextual_mode: bool = False
    ) -> httpx.Response:
        payload_builder = RequestPayloadBuilder()
        client = APIClient(config=endpoint_config)

        if not contextual_mode:
            context = payload_builder.build(
                schema=endpoint_config.request_schema,
                context=context,
            )

        async with client:
            response = await client.execute(payload=context)

        self.logger.info(f"Response status: {response.status_code}")

        return response

    @staticmethod
    def extract_response_data(
            response: httpx.Response,
            mappings: List[ResponseMappingConfig],
    ) -> Dict[str, Any]:
        extractor = ResponseDataExtractor()
        response_data = response.json() if response.text else {}
        extracted = extractor.extract(
            response_data=response_data,
            mappings=mappings
        )

        return extracted

    def get_tester(self, endpoint_name: str) -> ConnectivityTester:
        """Factory method: create connectivity tester for endpoint."""
        if endpoint_name not in self.endpoints:
            raise KeyError(f"Endpoint '{endpoint_name}' not found in configuration")

        return ConnectivityTester(self.endpoints[endpoint_name])

    def test_all(self, context: Dict[str, Any] | None = None) -> Dict[str, Dict[str, Any]]:
        """Test all configured endpoints."""
        results = {}
        for name in self.endpoints:
            tester = self.get_tester(name)
            results[name] = tester.test(context)

        return results
