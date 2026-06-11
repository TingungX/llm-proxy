"""首页、健康检查、模型列表、token 计数"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.main import app
from llm_proxy.state import get_state, resolve_model, resolve_model_for_endpoint
from llm_proxy.infra.http_client import get_client
from llm_proxy.infra import db

logger = logging.getLogger(__name__)


from llm_proxy.handlers.shared.auth import _extract_api_key


@app.get("/", include_in_schema=False)
async def index():
    """根路径重定向到管理界面（/static/ 由 Vite 构建产物提供服务）"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/static/")


@app.get("/health")
async def health():
    return {"status": "ok"}


_CODEX_MODEL_INFO_DEFAULTS = {
    "visibility": "list",
    "supported_in_api": True,
    "shell_type": "unified_exec",
    "apply_patch_tool_type": "freeform",
    "supports_parallel_tool_calls": True,
    "supports_search_tool": True,
    "supports_reasoning_summaries": True,
    "default_reasoning_level": "medium",
    "supported_reasoning_levels": [],
    "truncation_policy": {"mode": "bytes", "limit": 10000},
    "priority": 0,
    "base_instructions": "",
    "support_verbosity": False,
    "default_verbosity": None,
    "web_search_tool_type": "text",
    "supports_image_detail_original": False,
    "effective_context_window_percent": 95,
    "experimental_supported_tools": [],
    "input_modalities": ["text", "image"],
}


def _build_codex_model(slug: str, display_name: str, context_window: int) -> dict:
    m = dict(_CODEX_MODEL_INFO_DEFAULTS)
    m.update({
        "slug": slug,
        "display_name": display_name,
        "context_window": context_window,
        "max_context_window": context_window,
    })
    return m


@app.get("/v1/models")
async def list_models(request: Request):
    """返回端点可用的模型列表

    - mapping_enabled=true: 返回映射后的 Claude 模型名
    - mapping_enabled=false: 返回原始模型名
    - Codex Desktop (带 client_version 参数): 返回 Codex ModelInfo 格式
    """
    api_key = _extract_api_key(dict(request.headers))

    if not api_key:
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": "API Key is required"}},
            status_code=401,
        )

    endpoint = db.get_endpoint_by_api_key(api_key)
    if not endpoint:
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": "Unknown API Key"}},
            status_code=401,
        )

    allowed_models = [m.lower() for m in endpoint.get("models", [])]
    settings = endpoint.get("settings", {})
    mapping_enabled = settings.get("mapping_enabled", True)
    config = get_state().config

    is_codex = "client_version" in request.query_params

    endpoint_fr = endpoint.get("family_routing") or {}
    global_fr = config.get("family_routing", {})

    models = []
    if mapping_enabled:
        claude_families = {
            "claude-haiku-4-5": {"family_key": "haiku", "display_name": "Claude Haiku 4.5"},
            "claude-sonnet-4-6": {"family_key": "sonnet-4-6", "display_name": "Claude Sonnet 4.6"},
            "claude-sonnet-4-5": {"family_key": "sonnet-4-5", "display_name": "Claude Sonnet 4.5"},
            "claude-opus-4-7": {"family_key": "opus-4-7", "display_name": "Claude Opus 4.7"},
            "claude-opus-4-6": {"family_key": "opus-4-6", "display_name": "Claude Opus 4.6"},
        }
        fr = endpoint_fr or global_fr

        for model_id, info in claude_families.items():
            family_key = info["family_key"]
            if family_key in fr:
                target = fr[family_key]
                if target.lower() in allowed_models:
                    target_cfg = config["models"].get(target, {})
                    cw = target_cfg.get("context_window", 200000)
                    if is_codex:
                        models.append(_build_codex_model(model_id, info["display_name"], cw))
                    else:
                        models.append({
                            "id": model_id,
                            "object": "model",
                            "owned_by": "anthropic",
                            "display_name": info["display_name"],
                            "context_window": cw,
                        })

        non_claude_keys = [k for k in fr if not any(
            k == info["family_key"] for info in claude_families.values()
        )]
        for alias in non_claude_keys:
            target = fr[alias]
            if target.lower() in allowed_models:
                target_cfg = config["models"].get(target, {})
                display_name = target_cfg.get("display_name", alias)
                cw = target_cfg.get("context_window", 200000)
                if is_codex:
                    models.append(_build_codex_model(alias, display_name, cw))
                else:
                    models.append({
                        "id": alias,
                        "object": "model",
                        "owned_by": "proxy",
                        "display_name": display_name,
                        "context_window": cw,
                    })
    else:
        for model_id in endpoint.get("models", []):
            cfg = config["models"].get(model_id, {})
            display_name = cfg.get("display_name", model_id)
            cw = cfg.get("context_window", 200000)
            if is_codex:
                models.append(_build_codex_model(model_id, display_name, cw))
            else:
                item = {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "proxy",
                    "display_name": display_name,
                }
                if cfg.get("context_window") is not None:
                    item["context_window"] = cw
                models.append(item)

    if is_codex:
        return {"models": models}
    return {"data": models, "object": "list"}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.json()
    raw_model = body.get("model", "")

    s = get_state()

    # 使用端点级 routing 解析模型（和 ModelResolveStep 一致）
    api_key = _extract_api_key(dict(request.headers))
    endpoint = db.get_endpoint_by_api_key(api_key) if api_key else None
    if not endpoint:
        endpoint = db.get_endpoint_by_api_key("default")
    endpoint_family_routing = endpoint.get("family_routing") if endpoint else None
    resolved = resolve_model_for_endpoint(raw_model, s.config, s.model_map, endpoint_family_routing)
    if not resolved:
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": f"Unknown model: {raw_model}"}},
            status_code=400,
        )

    api_base, api_key, actual_model, model_id, upstream_protocol, _failover = resolved

    # OpenAI 协议上游没有 /v1/messages/count_tokens 端点，直接估算
    if upstream_protocol and upstream_protocol != "anthropic":
        messages = body.get("messages", [])
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated = max(1, total_chars // 4)
        return {"input_tokens": estimated}

    target_url = f"{api_base}/v1/messages/count_tokens"

    token_body = dict(body)
    token_body["model"] = actual_model

    req_headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    try:
        client = get_client()
        resp = await client.post(target_url, json=token_body, headers=req_headers, timeout=30.0)
        if resp.status_code == 200:
            return JSONResponse(resp.json())
        logger.info(f"count_tokens upstream returned {resp.status_code}, using estimate")
    except Exception as e:
        logger.info(f"count_tokens error: {e}, using estimate")

    messages = body.get("messages", [])
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    estimated = max(1, total_chars // 4)
    return {"input_tokens": estimated}
