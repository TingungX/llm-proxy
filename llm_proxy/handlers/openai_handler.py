"""/v1/chat/completions 路由 Pipeline"""

import json
import logging

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.handlers.base import PipelineContext, HandlerStep, Pipeline, PipelineStop
from llm_proxy.handlers.shared.auth import AuthStep
from llm_proxy.handlers.shared.model_resolve import ModelResolveStep
from llm_proxy.handlers.shared.vision_fallback import VisionFallbackStep
from llm_proxy.handlers.shared.compression import CompressionStep
from llm_proxy.handlers.shared.proxy import ProxyStep
from llm_proxy.protocol.errors import make_openai_error
from llm_proxy.state import get_state
from llm_proxy.services.tool_call_fix import fix_orphaned_tool_calls

logger = logging.getLogger(__name__)


class OpenAIProtocolStep(HandlerStep):
    """确定 OpenAI 路由的上游协议 — 多协议选择 + 工具降级

    Chat Completions 路由：
      1. 从 protocols_map 读取该模型支持的上游协议集合
      2. 客户端 = "openai/chat-completions" 调 select_upstream
      3. 如果带 custom/namespace/web_search 工具，强制降级到 chat-completions
         （responses 上游不接受这些工具类型）
    """

    NON_CHAT_TOOL_TYPES = {"custom", "namespace", "web_search", "tool_search", "image_generation"}

    async def execute(self, ctx: PipelineContext) -> None:
        from llm_proxy.protocol.capabilities import (
            NoReachableProtocol,
            select_upstream,
        )

        _, _, _, model_id, _, _ = ctx.resolved
        s = get_state()
        body = ctx.body
        model_protocols = s.protocols_map.get(model_id.lower(), set())

        # 工具降级：custom/namespace/web_search 等只支持 chat-completions 上游
        has_non_chat_tools = any(
            t.get("type") in self.NON_CHAT_TOOL_TYPES for t in (body.get("tools") or [])
        )
        if has_non_chat_tools:
            logger.debug("Request has non-chat tool types, restricting to openai/chat-completions")
            model_protocols = model_protocols & {"openai", "openai/chat-completions"}

        try:
            upstream_protocol = select_upstream("openai/chat-completions", model_protocols)
        except NoReachableProtocol as e:
            logger.error(f"Protocol selection failed: {e}")
            raise PipelineStop(make_openai_error(str(e), "invalid_request_error", 400))

        ctx.upstream_protocol = upstream_protocol

        if upstream_protocol == "openai/chat-completions":
            ctx.converter = None  # 同协议透传
        elif upstream_protocol == "openai/responses":
            ctx.converter = "chat_to_responses"  # Chat → Responses 转换
        else:
            raise PipelineStop(make_openai_error(
                f"Unsupported upstream protocol: {upstream_protocol}",
                "invalid_request_error",
                400,
            ))

        logger.debug(f"Protocol: upstream={upstream_protocol}, converter={ctx.converter}")


class ToolCallFixStep(HandlerStep):
    """修复孤立 tool_call 后再发送到上游"""

    async def execute(self, ctx: PipelineContext) -> None:
        if isinstance(ctx.body.get("messages"), list):
            ctx.body["messages"] = fix_orphaned_tool_calls(ctx.body["messages"])
            logger.debug("Tool call fix applied")


class OpenAIHandler:
    def __init__(self):
        self.pipeline = Pipeline([
            AuthStep(),
            ModelResolveStep(),
            OpenAIProtocolStep(),
            ToolCallFixStep(),
            VisionFallbackStep(),
            CompressionStep(),
            ProxyStep(),
        ])

    async def handle(self, request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raw_body = await request.body()
            logger.warning(f"Empty or invalid JSON body: {raw_body[:100] if raw_body else '<empty>'}")
            return make_openai_error(
                "Request body must be valid JSON",
                "invalid_request_error",
                400,
            )

        ctx = PipelineContext(
            request=request,
            body=body,
            headers=dict(request.headers),
            error_protocol="openai",
        )
        return await self.pipeline.execute(ctx)
