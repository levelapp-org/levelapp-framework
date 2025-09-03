"""levelapp/clients/__init__.py"""
import logging
import dotenv
from typing import Dict, Type

from levelapp.clients.anthropic import AnthropicClient
from levelapp.clients.ionos import IonosClient
from levelapp.clients.mistral import MistralClient
from levelapp.clients.openai import OpenAIClient
from levelapp.core.base import BaseChatClient
from levelapp.aspects.monitor import FunctionMonitor

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


class ClientRegistry:
    """Thread-safe client registry with monitoring"""
    _clients: Dict[str, Type[BaseChatClient]] = {}

    @classmethod
    def register(cls, provider: str, client_class: Type[BaseChatClient]) -> None:
        """
        Register a client class under a provider name.
        Args:
            provider (str): Unique identifier for the provider.
            client_class (Type[BaseChatClient]): The client class to register.

        Raises:
            TypeError: If client_class is not a subclass of BaseChatClient.
            KeyError: If a client for the provider is already registered.
        """
        if not issubclass(client_class, BaseChatClient):
            raise TypeError(f"Client '{provider}' must be a subclass of BaseChatClient")

        if provider in cls._clients:
            raise KeyError(f"Client for provider '{provider}' is already registered")

        # cls._wrap_client_methods(client_class)
        cls._clients[provider] = client_class
        logger.info(f"Registered client for provider: {provider}")

    @classmethod
    def _wrap_client_methods(cls, client_class: Type[BaseChatClient]) -> None:
        """
        Apply monitoring decorators to client methods.

        Args:
            client_class (Type[BaseChatClient]): The client class whose methods to wrap.

        Raises:
            TypeError: If the methods are not callable.
        """
        client_class.call = FunctionMonitor.monitor(
            name=f"{client_class.__name__}.call",
            cached=False,
            enable_timing=True
        )(client_class.call)

        client_class.acall = FunctionMonitor.monitor(
            name=f"{client_class.__name__}.acall",
            cached=False,
            enable_timing=True
        )(client_class.acall)

    @classmethod
    def get(cls, provider: str, **kwargs) -> BaseChatClient:
        """
        Retrieve a registered chat client by provider name.

        Args:
            provider (str): The name of the provider to retrieve.
            **kwargs: Additional keyword arguments to pass to the client constructor.

        Returns:
            BaseChatClient: An instance of the registered client class.
        """
        if provider not in cls._clients:
            raise KeyError(f"Client for provider '{provider}' is not registered")

        return cls._clients[provider](**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names"""
        return list(cls._clients.keys())

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider from registry"""
        with cls._lock:
            cls._clients.pop(provider, None)


clients = {
    "openai": OpenAIClient,
    "ionos": IonosClient,
    "mistral": MistralClient,
    "anthropic": AnthropicClient
}

for provider, client_class in clients.items():
    try:
        ClientRegistry.register(provider=provider, client_class=client_class)

    except (TypeError, KeyError) as e:
        logger.error(f"Failed to register client for {provider}: {e}")
