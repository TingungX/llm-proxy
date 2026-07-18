"""Admin API 认证中间件 — 保护 /api/* 管理路由

三层优先级：
1. LLM_PROXY_ADMIN_KEY 环境变量 — 强制认证（部署级保护）
2. config.json admin_auth.enabled=true — 用户在前端手动启用"高级数据保护"
3. 默认 — 放行所有（零配置，个人部署最友好）

认证方式：X-Admin-Key 头 或 Authorization: Bearer <key>
"""

import hashlib
import logging
import os

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

ENV_ADMIN_KEY = os.environ.get("LLM_PROXY_ADMIN_KEY", "").strip()


async def admin_auth_middleware(request: Request, call_next):
    """对所有 /api/* 管理路由进行认证检查。"""
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    # Layer 1: 环境变量强制认证
    if ENV_ADMIN_KEY:
        provided = _extract_admin_key(request.headers)
        if provided != ENV_ADMIN_KEY:
            logger.warning(
                "Admin auth (env) rejected for %s", request.url.path
            )
            return _unauthorized()
        return await call_next(request)

    # Layer 2: config.json admin_auth
    from llm_proxy.state import get_state
    state = get_state()
    admin_auth = state.config.get("admin_auth")
    if isinstance(admin_auth, dict) and admin_auth.get("enabled"):
        key_hash = admin_auth.get("key_hash", "")
        if not key_hash:
            return await call_next(request)

        provided = _extract_admin_key(request.headers)
        if not provided:
            return _unauthorized()

        provided_hash = hashlib.sha256(provided.encode()).hexdigest()
        if provided_hash != key_hash:
            logger.warning(
                "Admin auth (config) rejected for %s", request.url.path
            )
            return _unauthorized()

    # Layer 3: 默认放行（零配置）
    return await call_next(request)


def get_admin_auth_status() -> dict:
    """返回当前 admin auth 状态（给 GET /api/admin-auth 和 GET /api/config 用）。"""
    if ENV_ADMIN_KEY:
        return {"enabled": True, "source": "env"}
    from llm_proxy.state import get_state
    admin_auth = get_state().config.get("admin_auth")
    if isinstance(admin_auth, dict) and admin_auth.get("enabled"):
        return {"enabled": True, "source": "config"}
    return {"enabled": False, "source": None}


def _unauthorized():
    return JSONResponse(
        status_code=401,
        content={"error": "Admin authentication required"},
    )


def _extract_admin_key(headers) -> str:
    """从请求头提取 admin key。"""
    key = headers.get("x-admin-key") or ""
    if not key:
        auth = headers.get("authorization") or ""
        if auth.startswith("Bearer "):
            key = auth[7:]
    return key.strip()
