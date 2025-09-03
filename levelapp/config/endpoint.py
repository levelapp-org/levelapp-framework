"""levelapp/config/endpoint.py"""
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
    """
    Configuration class for user system's endpoint.

    Parameters:
        base_url (HttpUrl): The base url of the endpoint.
        method (Literal['POST', 'GET']): The HTTP method to use (POST or GET).
        api_key (SecretStr): The API key to use.
        bearer_token (SecretStr): The Bearer token to use.
        model_id (str): The model to use (if applicable).
        default_payload_template (Dict[str, Any]): The payload template to use.
        generated_payload_template (Dict[str, Any]): The generated payload template from a provided file.
        variables (Dict[str, Any]): The variables to populate the payload template.

    Note:
        Either you use the provided configuration YAML file, providing the following:\n
        - base_url (HttpUrl): The base url of the endpoint.
        - method (Literal['POST', 'GET']): The HTTP method to use (POST or GET).
        - api_key (SecretStr): The API key to use.
        - bearer_token (SecretStr): The Bearer token to use.
        - model_id (str): The model to use (if applicable).
        - default_payload_template (Dict[str, Any]): The payload template to use.
        - generated_payload_template (Dict[str, Any]): The generated payload template from a provided file.
        - variables (Dict[str, Any]): The variables to populate the payload template.

        Or manually configure the model instance by assigning the proper values to the model fields.
    """
    # TODO-0: Adjust the code to support both GET and POST requests.
    # Required
    method: Literal["POST", "GET"] = Field(default="POST")
    base_url: HttpUrl = Field(default=HttpUrl)
    url_path: str = Field(default='')

    # Auth
    api_key: SecretStr | None = Field(default=None)
    bearer_token: SecretStr | None = Field(default=None)
    model_id: str | None = Field(default='')

    # Data
    default_payload_template: Dict[str, Any] = Field(default_factory=dict)
    generated_payload_template: Dict[str, Any] = Field(default_factory=dict)
    default_response_template: Dict[str, Any] = Field(default_factory=dict)
    generated_response_template: Dict[str, Any] = Field(default_factory=dict)

    # Variables
    variables: Dict[str, Any] = Field(default_factory=dict)

    @computed_field()
    @property
    def full_url(self) -> str:
        return str(self.base_url) + self.url_path

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
        if not self.variables:
            return self.default_payload_template

        if not self.default_payload_template:
            self.load_template()
            return self._replace_placeholders(self.generated_payload_template, self.variables)

        return self._replace_placeholders(self.default_payload_template, self.variables)

    @staticmethod
    def _replace_placeholders(obj: Any, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively replace placeholders in payload template with variables."""
        def _replace(_obj):
            if isinstance(_obj, str):
                subst = Template(_obj).safe_substitute(variables)
                if '$' in subst:
                    logger.warning(f"[EndpointConfig] Unsubstituted placeholder in payload:\n{subst}\n\n")
                return subst

            elif isinstance(_obj, dict):
                return {k: _replace(v) for k, v in _obj.items()}

            elif isinstance(_obj, list):
                return [_replace(v) for v in _obj]

            return _obj

        return _replace(obj)

    # TODO-0: Use 'Path' for path configuration.
    def load_template(self, path: str | None = None) -> Dict[str, Any]:
        try:
            if not path:
                path = os.getenv('PAYLOAD_PATH', '')

            if not os.path.exists(path):
                raise FileNotFoundError(f"The provide payload template file path '{path}' does not exist.")

            with open(path, "r", encoding="utf-8") as f:
                if path.endswith((".yaml", ".yml")):
                    data = yaml.safe_load(f)

                elif path.endswith(".json"):
                    data = json.load(f)

                else:
                    raise ValueError("[EndpointConfig] Unsupported file format.")

                self.generated_payload_template = data
                # TODO-1: Remove the return statement if not required.
                return data

        except FileNotFoundError as e:
            raise FileNotFoundError(f"[EndpointConfig] Payload template file '{e.filename}' not found in path.")

        except yaml.YAMLError as e:
            raise ValueError(f"[EndpointConfig] Error parsing YAML file:\n{e}")

        except json.JSONDecodeError as e:
            raise ValueError(f"[EndpointConfig] Error parsing JSON file:\n{e}")

        except IOError as e:
            raise IOError(f"[EndpointConfig] Error reading file:\n{e}")

        except Exception as e:
            raise ValueError(f"[EndpointConfig] Unexpected error loading configuration:\n{e}")
