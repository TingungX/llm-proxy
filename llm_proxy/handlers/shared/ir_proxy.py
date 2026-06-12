"""IRProxyStep — 基于 IR 抽象层的统一代理步骤。

完整实现（非流式 + 流式）。当 `client_protocol` 是 "openai" 时，所有 OpenAI 客户端
(Chat/Responses) 都走 Chat 路径；按需扩展可分别处理。

迁移路径（不在本步骤内）：
1. endpoint.settings 加 "ir_enabled": true
2. handler 根据 flag 选 IRProxyStep 或 ProxyStep
3. 全量切换后删除 anthropic_openai/、responses_chat/ 通道
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.handlers.base import HandlerStep, PipelineContext
from llm_proxy.handlers.shared.paths import resolve_path
from llm_proxy.infra import db
from llm_proxy.infra.http_client import get_client
from llm_proxy.protocol.capabilities import NoReachableProtocol, select_upstream
from llm_proxy.protocol.errors import make_anthropic_error, make_openai_error
from llm_proxy.protocol.ir import REGISTRY, _resolve
from llm_proxy.protocol.ir._stream import keepalive_wrapper
from llm_proxy.protocol.ir.types import IRStreamEvent
from llm_proxy.state import get_state

logger = logging.getLogger(__name__)


def _make_error(ctx: PipelineContext, message: str, error_type: str = "api_error", status_code: int = 400):
    if ctx.error_protocol == "anthropic":
        return make_anthropic_error(message, error_type, status_code)
    return make_openai_error(message, error_type, status_code)


def _client_for(model_id: str | None):
    """Pick a pooled client based on whether the model is allow-listed to use
    the system HTTPS_PROXY. Default: direct (no proxy) for safety against
    mihomo/Clash toggles that leave 7897 listener up but refusing."""
    if model_id and get_state().allow_proxy_map.get(model_id.lower(), False):
        return get_client(direct=False)
    return get_client()


class IRProxyStep(HandlerStep):
    """基于 IR 的统一代理步骤。

    完整实现请求/响应/流式三种模式：
    - 解析 ctx.error_protocol 决定 client_protocol
    - 由 capabilities.select_upstream 选 upstream 协议
    - client → IR → upstream 转换后转发
    - 上游响应 → IR → client 转换后返回
    - 流式：upstream SSE → IR events → client SSE
    """

    def __init__(self, client_protocol: str):
        """Args:
            client_protocol: 客户端使用的协议，"anthropic" 或 "openai"（OpenAI Chat/Responses 共用）
        """
        self.client_protocol = client_protocol

    async def execute(self, ctx: PipelineContext) -> None:
        # ── 协议解析 ──
        _, _, _, model_id, _, _ = ctx.resolved
        s = get_state()
        available = s.protocols_map.get(model_id.lower(), set())

        try:
            upstream_protocol = select_upstream(self.client_protocol, available)
        except NoReachableProtocol as e:
            logger.error(f"IR proxy: protocol selection failed: {e}")
            self._record_usage(ctx, 0, 0, status="error", error_type="no_reachable_protocol")
            raise

        # ── 路径解析 ──
        api_base, upstream_api_key, actual_model, _, _, _ = ctx.resolved
        model_paths = s.paths_map.get(model_id.lower(), {})
        target_path = resolve_path(model_paths, _resolve(upstream_protocol))
        target_url = f"{api_base.rstrip('/')}{target_path}"

        # ── IR 转换 ──
        client_proto = _resolve(self.client_protocol)
        upstream_proto = _resolve(upstream_protocol)
        try:
            ir_request = REGISTRY[client_proto].to_ir(ctx.body)
            upstream_body = REGISTRY[upstream_proto].to_upstream(ir_request, upstream_model=actual_model)
        except Exception as e:
            logger.error(f"IR conversion failed: {e}", exc_info=True)
            self._record_usage(ctx, 0, 0, status="error", error_type="ir_conversion_error")
            raise

        # ── 请求头 ──
        if upstream_proto == "anthropic":
            req_headers = {
                "x-api-key": upstream_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            # 透传 anthropic-beta
            for k in ("anthropic-version", "anthropic-beta"):
                v = ctx.headers.get(k)
                if v:
                    req_headers[k] = v
        else:
            req_headers = {
                "Authorization": f"Bearer {upstream_api_key}",
                "Content-Type": "application/json",
            }

        logger.info(
            f"IRProxyStep: client={self.client_protocol} → upstream={upstream_protocol}, "
            f"target={target_url}, stream={ir_request.stream}"
        )

        # ── 分流：流式 / 非流式 ──
        if ir_request.stream:
            ctx.response = StreamingResponse(
                self._proxy_stream(
                    ctx, ir_request, upstream_body, upstream_protocol, upstream_proto,
                    target_url, req_headers, actual_model, model_id,
                ),
                media_type="text/event-stream",
            )
        else:
            await self._proxy_non_stream(
                ctx, ir_request, upstream_body, upstream_protocol, upstream_proto,
                target_url, req_headers, actual_model, model_id,
            )

    # ── 非流式 ───────────────────────────────────────────────────────

    async def _proxy_non_stream(
        self, ctx, ir_request, upstream_body, upstream_protocol,
        upstream_proto, target_url, req_headers, actual_model, model_id,
    ):
        client = _client_for(model_id)
        endpoint = ctx.endpoint
        endpoint_id = endpoint["endpoint_id"]
        client_proto = _resolve(self.client_protocol)

        try:
            resp = await client.post(target_url, json=upstream_body, headers=req_headers, timeout=120.0)
        except Exception as e:
            logger.error(f"IR proxy request error: {e}", exc_info=True)
            self._record_usage(ctx, 0, 0, status="error", error_type="proxy_error")
            raise

        try:
            upstream_resp_body = resp.json()
        except json.JSONDecodeError:
            raw = resp.text[:500]
            logger.error(f"Upstream returned non-JSON (status={resp.status_code}): {raw}")
            self._record_usage(ctx, 0, 0, status="error", error_type="non_json")
            raise

        if resp.status_code >= 400:
            self._record_usage(ctx, 0, 0, status="error", error_type=f"upstream_{resp.status_code}")
            # 直接透传错误
            raise  # 让上层 catch_all_exceptions 处理

        # IR → client
        ir_response = REGISTRY[upstream_proto].response_to_ir(upstream_resp_body)
        client_body = REGISTRY[client_proto].response_from_ir(ir_response)

        # 记录 usage
        usage = ir_response.usage or {}
        self._record_usage(
            ctx,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )

        ctx.response = JSONResponse(client_body, status_code=resp.status_code)

    # ── 流式 ─────────────────────────────────────────────────────────

    async def _proxy_stream(
        self, ctx, ir_request, upstream_body, upstream_protocol,
        upstream_proto, target_url, req_headers, actual_model, model_id,
    ):
        """跨协议流式：upstream SSE → IR events → client SSE。"""
        client = _client_for(model_id)
        endpoint = ctx.endpoint
        endpoint_id = endpoint["endpoint_id"]
        client_proto = _resolve(self.client_protocol)

        # 累积 usage 状态
        accumulated_input = 0
        accumulated_output = 0
        had_error = False

        async def _intercept_events(
            events: AsyncIterator[IRStreamEvent],
        ) -> AsyncIterator[IRStreamEvent]:
            """包装 IR events 流，提取 usage / 错误状态。"""
            nonlocal accumulated_input, accumulated_output, had_error
            async for ev in events:
                if ev.type == "usage":
                    data = ev.data or {}
                    accumulated_input = max(accumulated_input, int(data.get("input_tokens", 0)))
                    accumulated_output = max(accumulated_output, int(data.get("output_tokens", 0)))
                elif ev.type == "error":
                    had_error = True
                elif ev.type == "message_stop":
                    data = ev.data or {}
                    stop_reason = data.get("stop_reason", "end_turn")
                yield ev

        # 错误状态变量（用闭包）
        stop_reason = "end_turn"

        try:
            async with client.stream(
                "POST", target_url, json=upstream_body, headers=req_headers, timeout=120.0
            ) as resp:
                logger.info(f"IR proxy stream response status: {resp.status_code}")
                if resp.status_code >= 400:
                    # 错误处理：发 error event 给客户端
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    logger.error(f"Upstream stream error {resp.status_code}: {error_text[:500]}")
                    had_error = True
                    err_data: dict = {}
                    try:
                        err_data = json.loads(error_text)
                    except json.JSONDecodeError:
                        err_data = {"message": error_text}
                    # 发一个 error IR event
                    err_events = _err_event_gen(err_data, self.client_protocol)
                    async for chunk in REGISTRY[client_proto].format_ir_as_sse(
                        err_events,
                        actual_model,
                    ):
                        yield chunk
                    return

                # 上游 SSE → IR events（带拦截）
                raw_events = REGISTRY[upstream_proto].parse_stream_to_ir(resp, actual_model)
                intercepted_events = _intercept_events(raw_events)

                # 客户端 SSE formatter（带 reverse_tool_map / namespace_map）
                sse_bytes = REGISTRY[client_proto].format_ir_as_sse(
                    intercepted_events,
                    actual_model,
                    reverse_tool_map=ir_request.extensions.get("reverse_tool_map"),
                    namespace_map=ir_request.extensions.get("namespace_map"),
                )

                # keepalive 包装：每 15s 无数据发心跳
                async for sse_chunk in keepalive_wrapper(sse_bytes, interval=15.0):
                    yield sse_chunk

        except Exception as e:
            logger.error(f"IR stream error: {e}", exc_info=True)
            had_error = True
            err_chunk = _make_error_chunk(e, self.client_protocol)
            if err_chunk:
                yield err_chunk
        finally:
            # 记录 usage
            status = "error" if had_error else "success"
            estimated_input = accumulated_input or 1
            rctx = ctx.extra.get("_record_ctx", {}) if ctx.extra else {}
            try:
                db.record_usage(
                    endpoint_id=endpoint_id,
                    model_id=model_id,
                    input_tokens=estimated_input,
                    output_tokens=accumulated_output,
                    status=status,
                    error_type="stream_error" if had_error else None,
                    request_id=rctx.get("request_id", ""),
                    client_ip=rctx.get("client_ip", ""),
                    user_agent=rctx.get("user_agent", ""),
                )
            except Exception as e:  # pragma: no cover
                logger.debug(f"db.record_usage failed: {e}")

    # ── 工具 ─────────────────────────────────────────────────────────

    def _record_usage(
        self, ctx, input_tokens: int, output_tokens: int,
        *, status: str = "success", error_type: str | None = None,
    ):
        try:
            endpoint_id = ctx.endpoint["endpoint_id"]
        except (KeyError, TypeError):
            return
        _, _, _, model_id, _, _ = ctx.resolved
        rctx = ctx.extra.get("_record_ctx", {}) if ctx.extra else {}
        start_time = rctx.get("start_time")
        latency_ms = int((time.perf_counter() - start_time) * 1000) if start_time else None
        try:
            db.record_usage(
                endpoint_id=endpoint_id,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status=status,
                request_id=rctx.get("request_id", ""),
                latency_ms=latency_ms,
                error_type=error_type,
                client_ip=rctx.get("client_ip", ""),
                user_agent=rctx.get("user_agent", ""),
            )
        except Exception as e:  # pragma: no cover
            logger.debug(f"db.record_usage failed: {e}")


# ── 辅助：构造错误事件流 ─────────────────────────────────────────


async def _err_event_gen(err_data: dict, client_protocol: str):
    """生成一个错误 IR 事件。"""
    yield IRStreamEvent(
        type="error",
        data={
            "message": err_data.get("message") or err_data.get("error", {}).get("message", "Upstream error"),
            "code": err_data.get("error", {}).get("code", "api_error") if isinstance(err_data.get("error"), dict) else "api_error",
        },
    )


def _make_error_chunk(err: Exception, client_protocol: str) -> bytes | None:
    """为非流式错误流生成错误 SSE chunk。"""
    msg = f"Proxy error: {type(err).__name__}: {err}"
    if client_protocol == "anthropic":
        return (
            f"event: error\n"
            f'data: {{"type":"error","error":{{"type":"proxy_error","message":{json.dumps(msg)}}}}}\n\n'
        ).encode()
    # OpenAI 风格
    return (
        f'data: {{"error":{{"message":{json.dumps(msg)},"type":"proxy_error"}}}}\n\n'
    ).encode()
