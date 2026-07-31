"""Shared HTTP clients with connection pooling.

Creating a new httpx client for every LLM call wastes sockets and adds latency.
This module keeps a small set of long-lived sync/async clients keyed by the
parts of settings that matter for connection reuse.
"""

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from app.core.config import Settings


# Reasonable defaults for an LLM backend: keep connections alive and allow a
# modest number of concurrent in-flight requests to the same host.
_DEFAULT_LIMITS = httpx.Limits(
    max_connections=64,
    max_keepalive_connections=16,
    keepalive_expiry=60.0,
)

_sync_clients: dict[str, httpx.Client] = {}
_async_clients: dict[str, httpx.AsyncClient] = {}


def _client_key(settings: Settings) -> str:
    """Return a key that identifies a unique HTTP client configuration."""
    # Only the provider endpoint and auth token affect the underlying connection.
    return (
        f"{settings.ai_provider}::"
        f"{settings.ollama_base_url}::"
        f"{settings.openai_base_url}::"
        f"{settings.openai_api_key}"
    )


def get_sync_client(settings: Settings) -> httpx.Client:
    """Return a cached synchronous httpx client."""
    key = _client_key(settings)
    client = _sync_clients.get(key)
    if client is None:
        client = httpx.Client(limits=_DEFAULT_LIMITS, timeout=60.0)
        _sync_clients[key] = client
    return client


def get_async_client(settings: Settings) -> httpx.AsyncClient:
    """Return a cached asynchronous httpx client."""
    key = _client_key(settings)
    client = _async_clients.get(key)
    if client is None:
        client = httpx.AsyncClient(limits=_DEFAULT_LIMITS, timeout=60.0)
        _async_clients[key] = client
    return client


def _close_all() -> None:
    """Close all cached clients on process exit."""
    for client in list(_sync_clients.values()):
        try:
            client.close()
        except Exception:
            pass
    for client in list(_async_clients.values()):
        try:
            client.aclose()
        except Exception:
            pass


atexit.register(_close_all)
