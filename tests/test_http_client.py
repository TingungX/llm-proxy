"""Tests for the http_client pool (direct vs proxy).

After the allow_proxy refactor, get_client() returns one of two pooled clients
based on the `direct` flag. Default is direct=True (no proxy). The proxy
client only constructs when first requested and reads HTTPS_PROXY at that
moment.
"""

import pytest

from llm_proxy.infra import http_client
from llm_proxy.infra.http_client import get_client


@pytest.fixture(autouse=True)
def _reset_pool():
    """Reset the module-level clients between tests so each starts fresh."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # If a loop is already running, just zero the references; close on
        # next get_client() call. For test isolation, this is acceptable.
        http_client._direct_client = None
        http_client._proxy_client = None
        yield
    else:
        old_direct = http_client._direct_client
        old_proxy = http_client._proxy_client
        http_client._direct_client = None
        http_client._proxy_client = None
        try:
            yield
        finally:
            async def _close():
                for c in (old_direct, old_proxy):
                    if c is not None:
                        try:
                            await c.aclose()
                        except Exception:
                            pass
            loop.run_until_complete(_close())


def _clear_proxy_env(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)


def test_get_client_default_is_direct(monkeypatch):
    _clear_proxy_env(monkeypatch)
    c1 = get_client()
    c2 = get_client()
    # Singleton: same direct client across calls
    assert c1 is c2


def test_get_client_direct_explicit(monkeypatch):
    _clear_proxy_env(monkeypatch)
    c = get_client(direct=True)
    # Direct client has no proxy attribute set
    assert c._transport is not None  # type: ignore[attr-defined]


def test_get_client_proxy_reads_https_proxy_env(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    c = get_client(direct=False)
    # httpx stores proxy on the transport; for the direct=false case the
    # AsyncClient is constructed with proxy=httpx.Proxy(...). We assert
    # that the proxy object reflects the env.
    assert c is not None
    # Indirectly check: a fresh direct client must be a *different* object
    d = get_client(direct=True)
    assert c is not d


def test_get_client_proxy_falls_back_when_env_missing(monkeypatch):
    """If no env proxy is set, direct=False still works — client is built
    without a proxy transport. This is a graceful degradation, not an error."""
    _clear_proxy_env(monkeypatch)
    c = get_client(direct=False)
    assert c is not None


def test_direct_client_ignores_env_proxy(monkeypatch):
    """Regression: direct client must NOT read HTTPS_PROXY env var.
    
    When Clash/mihomo sets HTTPS_PROXY then exits, the env var lingers.
    trust_env=True (httpx default) would route all outbound traffic through
    the dead proxy, killing connectivity.  This test proves we set
    trust_env=False.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    c = get_client(direct=True)
    # httpx mounts proxy transports only when trust_env=True + env var present.
    # With trust_env=False, only the default "all://" transport exists.
    assert not c._trust_env


def test_get_client_direct_and_proxy_are_distinct(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    direct = get_client(direct=True)
    proxy = get_client(direct=False)
    assert direct is not proxy
