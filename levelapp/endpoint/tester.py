"""levelapp/endpoint/tester.py"""
import logging
from typing import Dict, Any

from levelapp.config.endpoint_ import ResponseExtractor
from levelapp.endpoint.client import EndpointConfig, APIClient
from levelapp.endpoint.parsers import RequestPayloadBuilder


class ConnectivityTester:
    """Tests REST endpoint connectivity with configurable behavior."""
    def __init__(self, config: EndpointConfig):
        self.config = config
        self.client = APIClient(config=config)
        self.payload_builder = RequestPayloadBuilder()
        self.response_extractor = ResponseExtractor()
        self.logger = logging.getLogger(f"ConnectivityTester.{self.config.name}")

    async def test(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute connectivity test (template method)."""
        context = context or {}

        self.logger.info(f"Starting connectivity test for '{self.config.name}'")

        try:
            payload = None
            if self.config.request_schema:
                payload = self.payload_builder.build(schema=self.config.request_schema, context=context)
                self.logger.debug(f"Request payload: {payload}")

            response = await self.client.execute(payload=payload)
            self.logger.debug(f"Response status: {response.status_code}")

            response_data = response.json() if response.text else {}
            extracted = self.response_extractor.extract(
                response_data=response_data,
                mappings=self.config.response_mapping,
            )

            return {
                "success": True,
                "status_code": response.status_code,
                "extracted_data": extracted,
                "raw_response": response,
            }

        except Exception as e:
            self.logger.error(f"Connectivity test failed: {e}", exc_info=e)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }
