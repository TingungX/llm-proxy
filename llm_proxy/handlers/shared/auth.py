"""认证步骤 — 提取 API Key，匹配端点，校验权限"""

import logging

from llm_proxy.handlers.base import PipelineContext, HandlerStep, PipelineStop
from llm_proxy.infra import db
from llm_proxy.protocol.errors import make_anthropic_error, make_openai_error

logger = logging.getLogger(__name__)


def _extract_api_key(headers: dict) -> str:
    """从请求头提取 API Key（公共函数，供 routes 和 handlers 共用）"""
    key = headers.get("x-api-key") or ""
    if not key:
        auth = headers.get("authorization") or ""
        if auth.startswith("Bearer "):
            key = auth[7:]
    if key.startswith("Bearer "):
        key = key[7:]
    return key


def _make_error(ctx: PipelineContext, message: str, error_type: str, status_code: int):
    """根据 error_protocol 生成对应格式的错误响应"""
    if ctx.error_protocol == "anthropic":
        return make_anthropic_error(message, error_type, status_code)
    return make_openai_error(message, error_type, status_code)


class AuthStep(HandlerStep):
    """认证步骤：提取 API Key → 匹配端点 → 校验启用状态"""

    async def execute(self, ctx: PipelineContext) -> None:
        client_api_key = _extract_api_key(ctx.headers)
        masked = f"{client_api_key[:4]}***" if len(client_api_key) > 4 else "***"
        logger.info(f"Auth: client_api_key={masked}")

        if not client_api_key:
            raise PipelineStop(_make_error(ctx, "API Key is required", "invalid_request_error", 401))

        endpoint = db.get_endpoint_by_api_key(client_api_key)
        if not endpoint:
            endpoint = db.get_endpoint_by_api_key("default")
        if not endpoint:
            raise PipelineStop(_make_error(ctx, "Unknown API Key", "invalid_request_error", 401))

        if not endpoint.get("enabled", True):
            raise PipelineStop(_make_error(ctx, "Endpoint is disabled", "invalid_request_error", 403))

        ctx.api_key = client_api_key
        ctx.endpoint = endpoint

