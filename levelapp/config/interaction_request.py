"""levelapp/config/interaction_request.py"""
import os
import json
import yaml
import logging

from pydantic import BaseModel, HttpUrl, SecretStr, Field, computed_field
from typing import Literal, Dict, Any
from string import Template

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class EndpointConfig(BaseModel):
    # Required
    base_url: HttpUrl = Field(default=HttpUrl(os.getenv('ENDPOINT_URL', '')))
    method: Literal["POST", "GET"] = Field(default="POST")

    # Auth
    api_key: SecretStr = Field(default=SecretStr(os.getenv('ENDPOINT_API_KEY', '')))
    bearer_token: SecretStr = Field(default=SecretStr(os.getenv('BEARER_TOKEN', '')))
    model_id: str = Field(default=os.getenv('MODEL_ID', ''))

    # Data
    payload_template: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)

    @computed_field()
    @property
    def full_url(self) -> str:
        return str(self.base_url)

    @computed_field()
    @property
    def headers(self) -> Dict[str, Any]:
        headers: Dict[str, Any] = {"Content-Type": "application/json"}
        if self.model_id:
            headers["x-model-id"] = self.model_id
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token.get_secret_value()}"
        if self.api_key:
            headers["x-api-key"] = self.api_key.get_secret_value()
        return headers

    @computed_field
    @property
    def payload(self) -> Dict[str, Any]:
        """Return fully prepared payload depending on template or full payload."""
        self.load_template()
        if not self.variables:
            # Case 1: Already complete payload
            raise ValueError("[EndpointConfig] No variables defined to populate the payload template")
        # Case 2: Template substitution
        return self._replace_placeholders(self.payload_template, self.variables)

    @staticmethod
    def _replace_placeholders(obj: Any, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively replace placeholders in payload template with variables."""
        def _replace(_obj):
            if isinstance(_obj, str):
                subst = Template(_obj).safe_substitute(variables)
                if '$' in subst:
                    logger.warning(f"[EndpointConfig] Unsubstituted placeholder in payload:\n{subst}\nn")
                return subst

            elif isinstance(_obj, dict):
                return {k: _replace(v) for k, v in _obj.items()}

            elif isinstance(_obj, list):
                return [_replace(v) for v in _obj]

            return _obj

        return _replace(obj)

    def load_template(self, path: str | None = None) -> Dict[str, Any]:
        try:
            if not path:
                path = os.getenv('PAYLOAD_PATH', '')

            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                if path.endswith((".yaml", ".yml")):
                    data = yaml.safe_load(f)

                elif path.endswith(".json"):
                    data = json.load(f)

                else:
                    raise ValueError("[EndpointConfig] Unsupported file format.")

                self.payload_template = data
                # TODO-0: Remove the return statement if not required.
                return data

        except FileNotFoundError as e:
            raise FileNotFoundError(f"[EndpointConfig] Configuration file not found: {e}")

        except yaml.YAMLError as e:
            raise ValueError(f"[EndpointConfig] Error parsing YAML file: {e}")

        except json.JSONDecodeError as e:
            raise ValueError(f"[EndpointConfig] Error parsing JSON file: {e}")

        except IOError as e:
            raise IOError(f"[EndpointConfig] Error reading file: {e}")

        except Exception as e:
            raise ValueError(f"[EndpointConfig] Unexpected error loading configuration: {e}")


if __name__ == '__main__':

    cfg = EndpointConfig()
    template_ = cfg.load_template(path="../../src/data/payload_example_1.yaml")

    cfg.variables = {"user_message": "Hello, world!"}
    payload = cfg.payload
    print(f"payload with variables:\n{payload}\n\n")

    print(f"Configuration dump:\n{cfg.model_dump()}")
