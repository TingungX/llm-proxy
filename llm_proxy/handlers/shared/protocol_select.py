"""协议选择步骤 — 确定上游协议和转换方式

用于 /v1/responses 路由。
- 同协议透传 (openai/responses)
- 跨协议转换 (responses → chat-completions)
- custom/namespace/web_search 工具存在时降级走 chat-completions
  （这些工具类型上游 Responses API 不接受）
"""

import logging

from llm_proxy.handlers.base import PipelineContext, HandlerStep, PipelineStop
from llm_proxy.state import get_state
from llm_proxy.protocol.errors import make_openai_error

logger = logging.getLogger(__name__)


class ProtocolSelectStep(HandlerStep):
    """协议选择步骤：基于 protocols_map 用 select_upstream 选最优上游。

    优先级：
      1. 同协议透传（responses 客户端 → responses 上游）
      2. 跨协议转换（按 IMPLEMENTED_CONVERSIONS 顺序，responses → chat 优先）
      3. 工具降级：custom/namespace/web_search 等强制走 chat-completions
      4. 不可达 → 400 with 诊断信息
    """

    NON_RESPONSES_TOOL_TYPES = frozenset({
        "custom", "namespace", "web_search", "tool_search", "image_generation",
    })

    async def execute(self, ctx: PipelineContext) -> None:
        from llm_proxy.protocol.capabilities import (
            NoReachableProtocol,
            select_upstream,
        )

        _, _, _, model_id, _, _ = ctx.resolved
        s = get_state()
        body = ctx.body
        model_protocols = s.protocols_map.get(model_id.lower(), set())

        # 工具降级：把可用集合限制为 chat 上游
        needs_downgrade = any(
            t.get("type") in self.NON_RESPONSES_TOOL_TYPES
            for t in (body.get("tools") or [])
        )
        if needs_downgrade:
            logger.debug("Request has non-Responses tool types, restricting to openai/chat-completions")
            model_protocols = model_protocols & {"openai", "openai/chat-completions"}

        try:
            upstream_protocol = select_upstream("openai/responses", model_protocols)
        except NoReachableProtocol as e:
            logger.error(f"Protocol selection failed: {e}")
            raise PipelineStop(make_openai_error(str(e), "invalid_request_error", 400))

        ctx.upstream_protocol = upstream_protocol

        # 确定转换方式
        if upstream_protocol == "openai/responses":
            ctx.converter = None  # 同协议透传
        elif upstream_protocol in ("openai", "openai/chat-completions"):
            ctx.converter = "responses_to_chat"  # Responses → Chat Completions
        else:
            raise PipelineStop(make_openai_error(
                f"Unsupported upstream protocol: {upstream_protocol}",
                "invalid_request_error",
                400,
            ))

        logger.debug(f"Protocol: upstream={upstream_protocol}, converter={ctx.converter}")
