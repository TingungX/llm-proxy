"""GET/PUT /api/config, PUT /api/models/{model_id}, DELETE /api/models/{model_id}, POST /api/providers/{model_id}/detect, POST /api/detect-protocol"""

import copy
import ipaddress
import logging
import socket
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.main import app
from llm_proxy.config_loader import save_config
from llm_proxy.protocol.detector import detect_upstream_protocols
from llm_proxy.state import get_state

logger = logging.getLogger(__name__)


def _entries_from_protocols(protocols: list[str]) -> list[dict]:
    """将探测到的协议列表转换为新版 entry 格式。"""
    path_defaults = {
        "anthropic": "/v1/messages",
        "openai/chat-completions": "/v1/chat/completions",
        "openai/responses": "/v1/responses",
    }
    return [
        {"protocol": p, "enabled": True, "path": path_defaults.get(p, "/v1/" + p.split("/")[-1])}
        for p in protocols
    ]


def _sanitize_config(config: dict) -> dict:
    """返回去除 api_key 的配置深拷贝，防止凭据泄露。"""
    safe = copy.deepcopy(config)
    for model_cfg in safe.get("models", {}).values():
        if isinstance(model_cfg, dict):
            model_cfg.pop("api_key", None)
    # 注入 admin_auth 状态（不泄露 key_hash）
    from llm_proxy.middleware.admin_auth import get_admin_auth_status
    safe["admin_auth"] = get_admin_auth_status()
    return safe


@app.get("/api/admin-auth")
async def api_get_admin_auth():
    """返回 admin auth 状态（不含密钥/哈希）。"""
    from llm_proxy.middleware.admin_auth import get_admin_auth_status
    return get_admin_auth_status()


@app.put("/api/admin-auth")
async def api_update_admin_auth(request: Request):
    """设置/修改/禁用 admin key。

    首次启用（当前 disabled）：只需提供 key
    修改/禁用（当前 enabled）：需提供 current_key 验证
    """
    import hashlib
    body = await request.json()
    enabled = body.get("enabled", False)
    new_key = (body.get("key") or "").strip()
    current_key = (body.get("current_key") or "").strip()

    # 检查是否被 env 强制覆盖
    import os
    if os.environ.get("LLM_PROXY_ADMIN_KEY", "").strip():
        return JSONResponse(
            {"error": "Admin auth is managed by LLM_PROXY_ADMIN_KEY environment variable"},
            status_code=403,
        )

    s = get_state()
    current_admin_auth = s.config.get("admin_auth", {})
    currently_enabled = isinstance(current_admin_auth, dict) and current_admin_auth.get("enabled")

    # 如果当前已启用，需要验证 current_key
    if currently_enabled:
        current_hash = current_admin_auth.get("key_hash", "")
        if not current_key or hashlib.sha256(current_key.encode()).hexdigest() != current_hash:
            return JSONResponse({"error": "Current key is required and must be correct"}, status_code=403)

    if not enabled:
        # 禁用
        s.config["admin_auth"] = {"enabled": False}
    else:
        # 启用或修改
        if not new_key or len(new_key) < 6:
            return JSONResponse({"error": "Key must be at least 6 characters"}, status_code=400)
        s.config["admin_auth"] = {
            "enabled": True,
            "key_hash": hashlib.sha256(new_key.encode()).hexdigest(),
        }

    from llm_proxy.config_loader import save_config
    save_config(s.config)
    await s.reload()
    logger.info("Admin auth %s", "enabled" if enabled else "disabled")
    return {"status": "ok", "enabled": enabled}


@app.get("/api/config")
async def api_get_config():
    return _sanitize_config(get_state().config)


@app.get("/api/provider-profiles")
async def api_get_provider_profiles():
    """返回所有厂商 profile，包含完整的 thinking format 详情。"""
    registry = get_state().provider_profiles
    return {
        key: {
            "display_name": p.display_name,
            "default_api_base": p.default_api_base,
            "default_thinking_effort_preset": p.default_thinking_effort_preset,
            "thinking_format": p.thinking_format,
            "default_thinking_type": p.default_thinking_type,
            "disable_thinking_value": p.disable_thinking_value,
            "effort_field": p.effort_field,
            "fixed_effort": p.fixed_effort,
            "effort_aliases": p.effort_aliases,
            "supports_reasoning_split": p.supports_reasoning_split,
            "preserve_reasoning_content": p.preserve_reasoning_content,
        }
        for key, p in registry.items()
    }


@app.get("/api/thinking-effort-defaults")
async def api_get_thinking_effort_defaults():
    """返回内置的默认 thinking effort mapping 配置，供前端展示系统默认规则。"""
    from llm_proxy.protocol.effort_mapping import get_default_mapping_config
    return get_default_mapping_config()


@app.put("/api/config")
async def api_update_config(request: Request):
    new_config = await request.json()
    new_config.pop("family_routing", None)
    new_config.pop("model_map", None)
    save_config(new_config)
    await get_state().reload()
    logger.info("Config reloaded")
    return {"status": "ok"}


@app.put("/api/models/{model_id}")
async def api_update_model(model_id: str, request: Request):
    """更新单个模型配置（增量合并，不覆盖其他模型）"""
    body = await request.json()
    s = get_state()
    if model_id not in s.config["models"]:
        s.config["models"][model_id] = {}
    model_cfg = s.config["models"][model_id]
    for key, value in body.items():
        if value is None:
            model_cfg.pop(key, None)
        else:
            model_cfg[key] = value
    if body.get("upstream_protocols") is not None:
        model_cfg.pop("upstream_protocol", None)
        model_cfg.pop("upstream_paths", None)
    save_config(s.config)
    await s.reload()
    logger.info(f"Model {model_id} updated")
    return {"status": "ok"}


@app.delete("/api/models/{model_id}")
async def api_delete_model(model_id: str):
    """删除单个模型配置"""
    s = get_state()
    if model_id not in s.config["models"]:
        return JSONResponse({"error": f"Model {model_id} not found"}, status_code=404)
    del s.config["models"][model_id]
    save_config(s.config)
    await s.reload()
    logger.info(f"Model {model_id} deleted")
    return {"status": "ok"}


# SSRF 防护：禁止访问的地址范围
_SSRF_BLOCKED_NETWORKS = [
    ipaddress.IPv4Network("10.0.0.0/8"),        # RFC1918
    ipaddress.IPv4Network("172.16.0.0/12"),      # RFC1918
    ipaddress.IPv4Network("192.168.0.0/16"),     # RFC1918
    ipaddress.IPv4Network("127.0.0.0/8"),        # Loopback
    ipaddress.IPv4Network("169.254.0.0/16"),     # Link-local / cloud metadata
    ipaddress.IPv4Network("0.0.0.0/8"),          # "This" network
    ipaddress.IPv6Network("::1/128"),            # IPv6 loopback
    ipaddress.IPv6Network("fe80::/10"),          # IPv6 link-local
]


def _validate_api_base(api_base: str) -> str | None:
    """验证 api_base 不为内网/敏感地址，返回错误消息或 None（通过）。"""
    try:
        parsed = urlparse(api_base)
    except Exception:
        return f"Invalid URL: {api_base}"

    if parsed.scheme not in ("http", "https"):
        return f"Unsupported scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return f"Missing hostname in URL: {api_base}"

    # DNS 解析 hostname → IP
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return f"Cannot resolve hostname: {hostname}"

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        for net in _SSRF_BLOCKED_NETWORKS:
            if ip in net:
                return f"Access to {ip_str} ({hostname}) is blocked for security reasons"

    return None  # 通过校验


@app.post("/api/detect-protocol")
async def api_detect_protocol(request: Request):
    """探测给定 api_base 的协议，不依赖已保存的模型"""
    body = await request.json()
    api_base = body.get("api_base", "").strip()
    api_key = body.get("api_key", "").strip()

    if not api_base or not api_key:
        return JSONResponse(
            {"error": "api_base and api_key are required"},
            status_code=400
        )

    # SSRF 防护：阻止访问内网地址
    ssrf_err = _validate_api_base(api_base)
    if ssrf_err:
        logger.warning("SSRF blocked: %s (api_base=%s)", ssrf_err, api_base)
        return JSONResponse({"error": ssrf_err}, status_code=400)

    protocols = await detect_upstream_protocols(api_base, api_key)

    if not protocols:
        return JSONResponse(
            {"error": "Failed to detect protocol. Please specify upstream_protocol manually."},
            status_code=400
        )

    return {
        "status": "ok",
        "upstream_protocol": protocols[0],
        "upstream_protocols": _entries_from_protocols(protocols),
    }


@app.post("/api/providers/{model_id}/detect")
async def api_detect_provider_protocol(model_id: str):
    """手动触发协议探测，记录所有支持的协议和路径"""
    model_id_lower = model_id.lower()
    
    if model_id_lower not in get_state().model_map:
        return JSONResponse(
            {"error": f"Model {model_id} not found in config"},
            status_code=404
        )
    
    api_base, api_key, _, _ = get_state().model_map[model_id_lower]
    
    protocols = await detect_upstream_protocols(api_base, api_key)
    
    if not protocols:
        return JSONResponse(
            {"error": "Failed to detect protocol. Please specify upstream_protocol manually."},
            status_code=400
        )
    
    # 更新 config.json：探测结果写入新版 entry 格式
    s = get_state()
    if model_id in s.config["models"]:
        entries = _entries_from_protocols(protocols)
        s.config["models"][model_id]["upstream_protocols"] = entries
        s.config["models"][model_id].pop("upstream_protocol", None)
        s.config["models"][model_id].pop("upstream_paths", None)
        save_config(s.config)
    await s.reload()

    return {
        "status": "ok",
        "model_id": model_id,
        "upstream_protocols": _entries_from_protocols(protocols),
    }
