import os

import httpx

_TIMEOUT = 120.0
_MAX_CONNECTIONS = 100
_MAX_KEEPALIVE_CONNECTIONS = 20
_KEEPALIVE_EXPIRY = 30.0

# Two pooled clients, chosen per request by model config (allow_proxy flag).
# Default is direct (no proxy) so mihomo/Clash toggles don't break llm-proxy.
# See: model.allow_proxy in config.json + State.allow_proxy_map.

_direct_client: httpx.AsyncClient | None = None
_proxy_client: httpx.AsyncClient | None = None


def _build_env_proxy() -> httpx.Proxy | None:
    raw = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    if not raw:
        return None
    return httpx.Proxy(url=raw)


def _build_client(proxy: httpx.Proxy | None) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=_MAX_CONNECTIONS,
        max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=_KEEPALIVE_EXPIRY,
    )
    if proxy is None:
        return httpx.AsyncClient(timeout=_TIMEOUT, limits=limits, trust_env=False)
    return httpx.AsyncClient(timeout=_TIMEOUT, limits=limits, proxy=proxy, trust_env=False)


def get_client(*, direct: bool = True) -> httpx.AsyncClient:
    global _direct_client, _proxy_client
    if direct:
        if _direct_client is None:
            _direct_client = _build_client(proxy=None)
        return _direct_client
    if _proxy_client is None:
        _proxy_client = _build_client(proxy=_build_env_proxy())
    return _proxy_client


async def close_client():
    global _direct_client, _proxy_client
    for client in (_direct_client, _proxy_client):
        if client is not None:
            await client.aclose()
    _direct_client = None
    _proxy_client = None
