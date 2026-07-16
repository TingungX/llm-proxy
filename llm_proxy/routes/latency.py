"""POST /api/latency — 延迟测试"""

import asyncio
import time
import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.main import app
from llm_proxy.state import get_state, resolve_model
from llm_proxy.infra.http_client import get_client

logger = logging.getLogger(__name__)

# 协议 → (路径, 请求体模板, 请求头模板)
_PROTOCOL_TEMPLATES = {
    "anthropic": (
        "/v1/messages",
        {
            "model": "{actual_model}",
            "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 2,
        },
        {
            "x-api-key": "{api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    ),
    "openai/chat-completions": (
        "/v1/chat/completions",
        {
            "model": "{actual_model}",
            "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 2,
        },
        {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
    ),
    "openai/responses": (
        "/v1/responses",
        {
            "model": "{actual_model}",
            "input": "OK",
            "max_tokens": 2,
        },
        {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
    ),
}


def _select_protocol(config_entry: dict) -> str | None:
    """从模型配置中选择第一个启用的协议。"""
    for entry in config_entry.get("upstream_protocols", []):
        if entry.get("enabled"):
            return entry["protocol"]
    return None


def _build_request(proto: str, actual_model: str, api_key: str):
    """根据协议构建目标路径、请求体和请求头。"""
    path_template, body_template, headers_template = _PROTOCOL_TEMPLATES[proto]
    body = {k: v.format(actual_model=actual_model) if isinstance(v, str) else v
            for k, v in body_template.items()}
    headers = {k: v.format(api_key=api_key) for k, v in headers_template.items()}
    return path_template, body, headers


@app.post("/api/latency")
async def api_latency(request: Request):
    body = await request.json()
    model = body.get("model", "")
    rounds = body.get("rounds", 3)

    s = get_state()
    resolved = resolve_model(model, s.config, s.model_map)
    if not resolved:
        return JSONResponse({"error": f"Unknown model: {model}"}, status_code=400)

    api_base, api_key, actual_model, config_key, upstream_protocol = resolved

    # 从原始 config 获取模型配置，选择启用的协议
    config_entry = s.config.get("models", {}).get(config_key, {})
    proto = _select_protocol(config_entry)
    if proto is None:
        return JSONResponse({"error": f"Model {model} has no enabled protocol"}, status_code=400)
    if proto not in _PROTOCOL_TEMPLATES:
        return JSONResponse({"error": f"Protocol {proto} not supported for latency test"}, status_code=400)

    target_path, test_body, req_headers = _build_request(proto, actual_model, api_key)
    target_url = f"{api_base.rstrip('/')}{target_path}"

    # 根据模型 allow_proxy 配置选择是否走代理
    use_proxy = s.allow_proxy_map.get(config_key, False)

    async def do_one():
        t0 = time.perf_counter()
        try:
            client = get_client(direct=not use_proxy)
            await client.post(target_url, json=test_body, headers=req_headers, timeout=15.0)
            return time.perf_counter() - t0
        except Exception as e:
            logger.warning("Latency test round failed for %s: %s (%s)", model, type(e).__name__, e)
            return -1

    times = await asyncio.gather(*[do_one() for _ in range(rounds)])
    valid = [t for t in times if t > 0]
    return {
        "model": model,
        "protocol": proto,
        "target": target_url,
        "times": times,
        "avg": sum(valid) / max(1, len(valid)),
        "rounds": rounds,
    }
