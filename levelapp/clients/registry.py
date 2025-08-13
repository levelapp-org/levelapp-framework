# """levelapp/clients/registry.py"""
# import logging
#
# from typing import Dict
# from threading import Lock
#
# from levelapp.core.base import BaseChatClient
#
# logger = logging.getLogger(__name__)
#
#
# class ClientRegistry:
#     """Thread-safe client registry with validation"""
#     _clients: Dict[str, BaseChatClient] = {}
#     _lock = Lock()
#
#     @classmethod
#     def register(cls, provider: str, client: BaseChatClient):
#         """Register a new chat client"""
#         if not isinstance(client, BaseChatClient):
#             raise TypeError(f"Client must be an instance of BaseChatClient, got {type(client)}")
#
#         with cls._lock:
#             if provider in cls._clients:
#                 raise KeyError(f"Client for provider '{provider}' is already registered")
#             cls._clients[provider] = client
#             logger.info(f"Registered client for provider: {provider}")
#
#     @classmethod
#     def get(cls, provider: str) -> BaseChatClient:
#         """Retrieve a registered chat client"""
#         with cls._lock:
#             if provider not in cls._clients:
#                 raise KeyError(f"Client for provider '{provider}' is not registered")
#             return cls._clients[provider]