"""协议探测器 - 自动检测上游 API 支持的协议"""

import logging

from llm_proxy.infra.http_client import get_client
from llm_proxy.infra.url_utils import normalize_api_base

logger = logging.getLogger(__name__)

# api_base → {protocol: working_path}
_path_cache: dict[str, dict[str, str]] = {}

_OPENAI_CHAT_PATHS = ["/v1/chat/completions", "/chat/completions", "/openai/v1/chat/completions"]
_OPENAI_RESPONSES_PATHS = ["/v1/responses", "/responses", "/openai/v1/responses"]
_ANTHROPIC_PATHS = ["/v1/messages", "/anthropic/v1/messages"]

_VALID_STATUSES = (200, 401, 403, 400, 422, 500)


async def detect_upstream_protocols(api_base: str, api_key: str) -> list[str]:
    """探测上游 API 支持的所有协议

    先用原始 api_base 探测；若某协议失败，尝试去掉常见的协议前缀
    （如 /anthropic、/openai）后重试，以覆盖"不同协议挂在不同路径"的场景。
    返回所有成功检测到的协议列表。

    Args:
        api_base: 上游 API 基础 URL（可能是某个协议专用路径）
        api_key: 上游 API 密钥

    Returns:
        支持的协议列表，如 ["anthropic"] / ["openai/chat-completions", "openai/responses"] / 空列表
    """
    cache_base = normalize_api_base(api_base)
    supported: list[str] = []
    cache: dict[str, str] = {}

    path = await _try_probe("anthropic", api_base, api_key, _ANTHROPIC_PATHS)
    if path:
        supported.append("anthropic")
        cache["anthropic"] = path

    path = await _try_probe("openai/chat-completions", api_base, api_key, _OPENAI_CHAT_PATHS)
    if path:
        supported.append("openai/chat-completions")
        cache["openai/chat-completions"] = path

    path = await _try_probe("openai/responses", api_base, api_key, _OPENAI_RESPONSES_PATHS)
    if path:
        supported.append("openai/responses")
        cache["openai/responses"] = path

    if cache:
        _path_cache[cache_base] = cache

    if supported:
        logger.info(f"Detected protocols for {api_base}: {supported}")
    else:
        logger.error(f"Failed to detect any protocol for {api_base}")
    return supported


def get_protocol_path(api_base: str, protocol: str) -> str | None:
    """获取协议的探测成功的路径"""
    base = normalize_api_base(api_base)
    cache = _path_cache.get(base)
    return cache.get(protocol) if cache else None


def _probe_headers(api_key: str, protocol: str) -> dict:
    if protocol == "anthropic":
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _probe_body(protocol: str) -> dict:
    if protocol == "anthropic":
        return {"model": "claude-3-haiku-20240307", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    if protocol == "openai/chat-completions":
        return {"model": "gpt-3.5-turbo", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    return {"model": "gpt-4", "input": "hi"}


def _is_valid_response(resp, protocol: str) -> bool:
    """检查响应是否表明端点存在"""
    s = resp.status_code
    if s in (401, 403, 400, 422, 500):
        return True
    if s != 200:
        return False
    try:
        data = resp.json()
    except Exception:
        return False
    if protocol == "anthropic":
        return "content" in data and ("type" in data or "id" in data)
    if protocol == "openai/chat-completions":
        return "choices" in data and "object" in data
    if protocol == "openai/responses":
        return "output" in data and "id" in data
    return False


async def _try_probe(protocol: str, api_base: str, api_key: str, paths: list[str]) -> str | None:
    """对一组路径进行探测，返回第一个成功的路径"""
    headers = _probe_headers(api_key, protocol)
    body = _probe_body(protocol)

    for path in paths:
        url = f"{api_base.rstrip('/')}{path}"
        try:
            client = get_client()
            resp = await client.post(url, json=body, headers=headers, timeout=30.0)
            if _is_valid_response(resp, protocol):
                logger.debug(f"Probed {protocol} OK at {path} for {api_base}")
                return path
        except Exception as e:
            logger.debug(f"Probe {protocol} at {path} failed: {e}")

    return None
