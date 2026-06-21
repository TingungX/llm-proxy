"""Wire-format HTTP 客户端 — 发送 CLI 使用的真实线缆格式请求

模拟 Codex CLI (OpenAI Responses API) 和 Claude Code CLI (Anthropic Messages API)
发送的请求，走完整的 llm-proxy Pipeline。
"""

import json
import time
import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

Headers = dict[str, str]
Response = tuple[int, Headers, Any, float]  # status, headers, body, elapsed_ms


async def send_responses(
    base_url: str,
    body: dict,
    api_key: str | None = "test-key",
    stream: bool = False,
    timeout: float = 30,
) -> Response:
    """发送 OpenAI Responses API 请求（Codex CLI 线缆格式）

    POST /v1/responses

    Returns:
        (status_code, headers, response_body_or_events, elapsed_ms)
    """
    headers = _build_headers(api_key)
    payload = {**body}
    if stream:
        payload["stream"] = True

    url = f"{base_url.rstrip('/')}/v1/responses"
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        if stream:
            events = []
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        events.append(line[6:])
                elapsed = (time.perf_counter() - start) * 1000
                return (resp.status_code, dict(resp.headers), events, elapsed)
        else:
            r = await client.post(url, json=payload, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000
            body_data = _parse_body(r)
            return (r.status_code, dict(r.headers), body_data, elapsed)


async def send_messages(
    base_url: str,
    body: dict,
    api_key: str | None = "test-key",
    stream: bool = False,
    anthropic_version: str = "2023-06-01",
    timeout: float = 30,
) -> Response:
    """发送 Anthropic Messages API 请求（Claude Code CLI 线缆格式）

    POST /v1/messages

    Returns:
        (status_code, headers, response_body_or_events, elapsed_ms)
    """
    headers = _build_headers(api_key)
    headers["anthropic-version"] = anthropic_version

    payload = {**body}
    if stream:
        payload["stream"] = True

    url = f"{base_url.rstrip('/')}/v1/messages"
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        if stream:
            events = []
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                current_event = ""
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: "):
                        events.append({"event": current_event, "data": line[6:]})
                elapsed = (time.perf_counter() - start) * 1000
                return (resp.status_code, dict(resp.headers), events, elapsed)
        else:
            r = await client.post(url, json=payload, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000
            body_data = _parse_body(r)
            return (r.status_code, dict(r.headers), body_data, elapsed)


async def send_chat_completions(
    base_url: str,
    body: dict,
    api_key: str | None = "test-key",
    stream: bool = False,
    timeout: float = 30,
) -> Response:
    """发送 OpenAI Chat Completions API 请求

    POST /v1/chat/completions
    """
    headers = _build_headers(api_key)
    payload = {**body}
    if stream:
        payload["stream"] = True

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        if stream:
            events = []
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        events.append(line[6:])
                elapsed = (time.perf_counter() - start) * 1000
                return (resp.status_code, dict(resp.headers), events, elapsed)
        else:
            r = await client.post(url, json=payload, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000
            body_data = _parse_body(r)
            return (r.status_code, dict(r.headers), body_data, elapsed)


async def configure_endpoint(
    base_url: str,
    endpoint_body: dict,
    timeout: float = 10,
) -> Response:
    """通过 API 配置端点

    POST /api/endpoints
    """
    headers = {"Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/api/endpoints"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        r = await client.post(url, json=endpoint_body, headers=headers)
        return (r.status_code, dict(r.headers), _parse_body(r), 0)


async def get_endpoints(
    base_url: str,
    timeout: float = 10,
) -> Response:
    """获取端点列表

    GET /api/endpoints
    """
    url = f"{base_url.rstrip('/')}/api/endpoints"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        r = await client.get(url)
        return (r.status_code, dict(r.headers), _parse_body(r), 0)


def _build_headers(api_key: str | None) -> Headers:
    """构建请求头"""
    headers: Headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _parse_body(r: httpx.Response) -> Any:
    """解析响应 body"""
    content_type = r.headers.get("content-type", "")
    if "json" in content_type:
        try:
            return r.json()
        except Exception:
            return r.text
    return r.text
