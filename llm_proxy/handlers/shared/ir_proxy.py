"""IRProxyStep — 基于 IR 抽象层的统一代理步骤。

当前为骨架，不接入任何 handler Pipeline。未来切换：
  - 在 endpoint 或 config 中添加 ir_enabled 标志
  - 当 ir_enabled=True 时，handler 用 IRProxyStep 替代 ProxyStep

本骨架的目的：
  1. 验证 IR 转换器在代理上下文中的接口
  2. 留好 stream/non-stream 两条调用链
  3. 文档化迁移路径
"""

from __future__ import annotations

import json
import logging
import time

from llm_proxy.handlers.base import HandlerStep, PipelineContext
from llm_proxy.protocol.capabilities import NoReachableProtocol, select_upstream
from llm_proxy.protocol.errors import make_openai_error
from llm_proxy.protocol.ir import REGISTRY, _resolve
from llm_proxy.protocol.sse import stream_response as sse_stream
from llm_proxy.state import get_state

logger = logging.getLogger(__name__)


class IRProxyStep(HandlerStep):
    """基于 IR 的统一代理步骤（骨架）。

    替换 ProxyStep 时：
    - handler 层调用 ctx.error_protocol 决定 client 协议（openai / anthropic）
    - 由 capability table 选 upstream 协议
    - client → IR → upstream 转换后转发
    - 上游响应 → IR → client 转换后返回
    """

    def __init__(self, client_protocol: str):
        """Args:
            client_protocol: 客户端使用的协议，"openai" | "anthropic"
        """
        self.client_protocol = client_protocol

    async def execute(self, ctx: PipelineContext) -> None:
        """IR-based 代理执行（待实现）。

        完整流程（不在本骨架中实现）：
        1. 解析 upstream_protocol（select_upstream）
        2. ctx.body → IRRequest（via REGISTRY[client].to_ir）
        3. IRRequest → upstream_body（via REGISTRY[upstream].to_upstream）
        4. 构造 target_url / req_headers
        5. 流式 / 非流式分支
        6. upstream response → IRResponse
        7. IRResponse → client_body
        8. ctx.response = JSONResponse/StreamingResponse
        """
        # ── 协议解析 ──
        _, _, _, model_id, _, _ = ctx.resolved
        s = get_state()
        available = s.protocols_map.get(model_id.lower(), set())

        try:
            upstream_protocol = select_upstream(self.client_protocol, available)
        except NoReachableProtocol as e:
            logger.error(f"IR proxy: protocol selection failed: {e}")
            # 让上层 catch_all_exceptions 处理
            raise

        # ── 路径解析 ──
        api_base, upstream_api_key, actual_model, _, _, _ = ctx.resolved
        from llm_proxy.handlers.shared.paths import resolve_path
        model_paths = s.paths_map.get(model_id.lower(), {})
        target_path = resolve_path(model_paths, _resolve(upstream_protocol))
        target_url = f"{api_base.rstrip('/')}{target_path}"

        logger.info(
            f"IRProxyStep (skeleton): client={self.client_protocol} → "
            f"upstream={upstream_protocol}, target={target_url}"
        )

        # ── IR 转换演示（不动 ctx.body）──
        try:
            ir_request = REGISTRY[_resolve(self.client_protocol)].to_ir(ctx.body)
            upstream_body = REGISTRY[_resolve(upstream_protocol)].to_upstream(
                ir_request, upstream_model=actual_model
            )
            logger.debug(
                f"IR: {len(ir_request.messages)} messages, "
                f"tools={len(ir_request.tools or [])}, "
                f"upstream_keys={sorted(upstream_body.keys())[:8]}"
            )
        except Exception as e:  # pragma: no cover - skeleton
            logger.error(f"IR conversion failed (skeleton, not used): {e}", exc_info=True)

        # 骨架不实际发出请求；保留原 ProxyStep 行为
        # 当真正启用时，下面会替换为流式/非流式代理逻辑
        return

    # ── 后续实现时将使用的方法（参考 ProxyStep）──

    async def _proxy_anthropic_same_protocol(
        self, ctx: PipelineContext, body: dict, target_url: str,
        upstream_api_key: str, actual_model: str, model_id: str,
        endpoint_id: str, downstream_headers: dict
    ):
        """Anthropic 同协议透传。骨架阶段不在此实现。"""
        raise NotImplementedError("IR proxy skeleton — streaming proxy not implemented yet")

    async def _proxy_cross_protocol_stream(
        self, ctx: PipelineContext, ir_request, upstream_protocol: str,
        target_url: str, upstream_api_key: str, actual_model: str,
        model_id: str, endpoint_id: str
    ):
        """跨协议流式代理。骨架阶段不在此实现。"""
        raise NotImplementedError("IR proxy skeleton — streaming cross-protocol not implemented yet")

    async def _proxy_non_stream(
        self, ctx: PipelineContext, upstream_body: dict, upstream_protocol: str,
        target_url: str, upstream_api_key: str, actual_model: str,
        model_id: str, endpoint_id: str
    ):
        """非流式代理。骨架阶段不在此实现。"""
        raise NotImplementedError("IR proxy skeleton — non-stream proxy not implemented yet")


# ── 文档化的迁移路径 ─────────────────────────────────────────────────
#
# 切换步骤（不在本次实现）：
#
# 1. 在 endpoint.settings / config 中加入 "ir_enabled": true
# 2. handler 顶层根据 ctx.endpoint.settings.get("ir_enabled") 决定 Pipeline：
#    - True  → [Auth, ModelResolve, VisionFallback, Compression, IRProxyStep(client)]
#    - False → [Auth, ModelResolve, ...old steps..., ProxyStep]  (现状)
# 3. 默认 False；逐个 endpoint 验证后切换
# 4. 全量切换后，删除 protocol/anthropic_openai/ 和 protocol/responses_chat/ 的 DeprecationWarning
#
# 关键约束：
# - IRProxyStep 必须完全复用现有 auth / model_resolve / vision_fallback / compression
#   这些步骤与协议无关，可以直接接入 IR 路径
# - 流式和非流式必须同时实现；流式可分阶段
# - 用量记录 / cache token 透传等细节需要从 ProxyStep 移植

