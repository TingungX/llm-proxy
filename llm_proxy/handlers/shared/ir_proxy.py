"""IRProxyStep — 基于 IR 抽象层的统一代理步骤。

完整实现（非流式 + 流式）。当 `client_protocol` 是 "openai" 时，所有 OpenAI 客户端
(Chat/Responses) 都走 Chat 路径；按需扩展可分别处理。

支持 client_protocol="openai/responses"，用于 Responses 路由直接走 IR 转换。

迁移路径（不在本步骤内）：
1. Responses 路由已直接使用 IRProxyStep（替代 ProtocolSelect + ResponsesConvert + ProxyStep）
2. Chat/Anthropic 路由仍用 ProxyStep（待后续迁移）
3. 全量切换后删除 anthropic_openai/、responses_chat/ 通道
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

import socket

import httpx
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
from llm_proxy.logging_config import REQUEST_ID_CTX

logger = logging.getLogger(__name__)

# 单条请求重试：仅对 429/503，且仅在「未向下游 flush 任何 chunk」阶段重试。
# 指数退避：1s / 2s / 4s，最多 3 次（共 4 次请求）。
_RETRY_STATUSES = (429, 503)
_RETRY_MAX = 3


def _make_error(ctx: PipelineContext, message: str, error_type: str = "api_error", status_code: int = 400):
    if ctx.error_protocol == "anthropic":
        return make_anthropic_error(message, error_type, status_code)
    return make_openai_error(message, error_type, status_code)


def _classify_stream_error(e: Exception) -> str:
    """根据异常类型分类流式错误，返回细粒度的 error_type。

    分类优先级（从上到下匹配）：
    1. DNS 解析失败 → upstream_dns_error
    2. 连接超时 → upstream_timeout
    3. TCP/TLS 连接失败 → upstream_connect_error
    4. HTTP 超时 → upstream_timeout
    5. HTTP 错误（4xx/5xx 响应体解析失败等）→ upstream_http_error
    6. 其他 → stream_error
    """

    def _has_dns_error_in_chain(exc: BaseException | None) -> bool:
        """递归检查异常链中是否有 DNS 解析失败（gaierror 或 errno 为 -2/8）。"""
        if exc is None:
            return False
        if isinstance(exc, socket.gaierror):
            return True
        if isinstance(exc, OSError):
            errno = getattr(exc, "errno", None)
            # -2: Linux EAI_NONAME (Name or service not known)
            # 8:  macOS EAI_NONAME (nodename nor servname provided, or not known)
            if errno in (-2, 8):
                return True
        return _has_dns_error_in_chain(getattr(exc, "__cause__", None))

    # DNS 解析失败
    if isinstance(e, socket.gaierror):
        return "upstream_dns_error"

    # httpx 异常
    if isinstance(e, httpx.ConnectError):
        if _has_dns_error_in_chain(getattr(e, "__cause__", None)):
            return "upstream_dns_error"
        return "upstream_connect_error"

    if isinstance(e, httpx.TimeoutException):
        return "upstream_timeout"

    if isinstance(e, httpx.HTTPStatusError):
        return "upstream_http_error"

    # 底层 socket/OS 错误
    if isinstance(e, (socket.timeout, TimeoutError)):
        return "upstream_timeout"

    if isinstance(e, OSError):
        errno = getattr(e, "errno", None)
        # -2: Linux EAI_NONAME (Name or service not known)
        # 8:  macOS EAI_NONAME (nodename nor servname provided, or not known)
        if errno in (-2, 8):
            return "upstream_dns_error"
        # 连接被拒绝 / 网络不可达等
        if errno in (61, 64, 65, 111):  # ECONNREFUSED, EHOSTDOWN, EHOSTUNREACH, ECONNREFUSED
            return "upstream_connect_error"

    # 兜底
    return "stream_error"


def _estimate_input_tokens(body: dict) -> int:
    """从请求体估算 input tokens（上游未返回 usage 时的回退）。

    使用字符数 / 4 的粗略估算（英文约 4 chars/token，中文约 1.5 chars/token，
    取保守值 4 作为统一估算）。至少返回 1。
    """
    try:
        text = json.dumps(body, ensure_ascii=False)
        return max(1, len(text) // 4)
    except (TypeError, ValueError):
        return 1


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
            client_protocol: 客户端使用的协议：
                - "anthropic": Anthropic Messages API
                - "openai" / "openai/chat-completions": OpenAI Chat Completions
                - "openai/responses": OpenAI Responses API
        """
        self.client_protocol = client_protocol

    async def execute(self, ctx: PipelineContext) -> None:
        # ── 记录上下文（usage 记录需要） ──
        ctx.extra["_record_ctx"] = {
            "start_time": time.perf_counter(),
            "request_id": REQUEST_ID_CTX.get(""),
            "client_ip": ctx.request.client.host if ctx.request.client else "",
            "user_agent": ctx.request.headers.get("user-agent", ""),
        }

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
        # "anthropic" 是协议名，但 path key 需要 "anthropic/messages" 才能匹配
        # upstream_paths 和 DEFAULT_PATHS 中的完整路径键名。
        # openai/chat-completions 和 openai/responses 的协议名本身就是 path key。
        path_key = _resolve(upstream_protocol)
        if path_key == "anthropic":
            path_key = "anthropic/messages"
        target_path = resolve_path(model_paths, path_key)
        target_url = f"{api_base.rstrip('/')}/{target_path.lstrip('/')}"

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

        logger.debug(
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

        # ── 单条请求 retry：仅 429/503，仅在拿到响应头后（未向下游写任何字节）重试 ──
        resp = None
        for attempt in range(_RETRY_MAX + 1):
            try:
                client = _client_for(model_id)
                resp = await client.post(target_url, json=upstream_body, headers=req_headers, timeout=120.0)
            except Exception as e:
                # 连接级异常（DNS/超时/连接失败）在未 flush 阶段也可安全重试
                if attempt < _RETRY_MAX and not isinstance(e, (httpx.HTTPStatusError,)):
                    wait = 2 ** attempt
                    logger.warning(f"IR non-stream connect error: {e}, retrying in {wait}s (attempt {attempt + 1}/{_RETRY_MAX + 1})")
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"IR proxy request error: {e}", exc_info=True)
                self._record_usage(ctx, 0, 0, status="error", error_type=_classify_stream_error(e))
                raise
            # 429/503 重试
            if resp.status_code in _RETRY_STATUSES and attempt < _RETRY_MAX:
                wait = 2 ** attempt
                logger.warning(f"IR non-stream upstream {resp.status_code}, retrying in {wait}s (attempt {attempt + 1}/{_RETRY_MAX + 1})")
                await asyncio.sleep(wait)
                continue
            break

        try:
            upstream_resp_body = resp.json()
        except json.JSONDecodeError:
            raw = resp.text[:500]
            logger.error(f"Upstream returned non-JSON (status={resp.status_code}): {raw}")
            self._record_usage(ctx, 0, 0, status="error", error_type="non_json")
            raise

        if resp.status_code >= 300:
            self._record_usage(ctx, 0, 0, status="error", error_type=f"upstream_{resp.status_code}")
            # 透传上游错误/重定向响应
            ctx.response = JSONResponse(
                upstream_resp_body if isinstance(upstream_resp_body, dict) else {"error": {"message": str(upstream_resp_body)}},
                status_code=resp.status_code,
            )
            return

        # IR → client
        ir_response = REGISTRY[upstream_proto].response_to_ir(upstream_resp_body)
        client_body = REGISTRY[client_proto].response_from_ir(
            ir_response,
            reverse_tool_map=ir_request.extensions.get("reverse_tool_map"),
            tool_spec_map=ir_request.extensions.get("tool_spec_map"),
        )

        # 记录 usage
        usage = ir_response.usage or {}
        self._record_usage(
            ctx,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )

        if ctx.response_model:
            client_body["model"] = ctx.response_model
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
        error_type = "stream_error"  # 默认错误类型，会被 except 块覆盖

        # 预计算 input tokens 估算值（上游未返回 usage 时回退用）
        estimated_input_fallback = _estimate_input_tokens(upstream_body)

        async def _intercept_events(
            events: AsyncIterator[IRStreamEvent],
        ) -> AsyncIterator[IRStreamEvent]:
            """包装 IR events 流，提取 usage / 错误状态。"""
            nonlocal accumulated_input, accumulated_output, had_error, stop_reason
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

        # ── 单条请求 retry ──
        # 安全边界：429/503 或连接级异常一定出现在「首 chunk 之前」——
        # 此时还未向下游 yield 任何 chunk（下游 yield 在 status_code 检查通过之后），
        # 因此 retry 不会导致下游收到重复/截断的流。
        # 一旦进入 SSE body 遍历（下面的 async for）就不再 retry，任何异常走 except。
        #
        # 控制流：外层 try/finally 保证 usage 记录；for 内 try/except 处理单次请求；
        # 所有错误路径 yield error event 后 return（终止生成器，触发 finally）。
        try:
            for attempt in range(_RETRY_MAX + 1):
                try:
                    client = _client_for(model_id)
                    async with client.stream(
                        "POST", target_url, json=upstream_body, headers=req_headers, timeout=120.0
                    ) as resp:
                        logger.debug(f"IR proxy stream response status: {resp.status_code}")
                        # 429/503：在向下游 flush 任何东西之前，可安全重试
                        if resp.status_code in _RETRY_STATUSES and attempt < _RETRY_MAX:
                            # 先读完响应体以释放连接，再退避
                            await resp.aread()
                            wait = 2 ** attempt
                            logger.warning(f"IR stream upstream {resp.status_code}, retrying in {wait}s (attempt {attempt + 1}/{_RETRY_MAX + 1})")
                            await asyncio.sleep(wait)
                            continue

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
                                ctx.response_model or actual_model,
                            ):
                                yield chunk
                            return

                        # 上游 SSE → IR events（带拦截）
                        raw_events = REGISTRY[upstream_proto].parse_stream_to_ir(resp, actual_model)
                        intercepted_events = _intercept_events(raw_events)

                        # 客户端 SSE formatter（带 reverse_tool_map / tool_spec_map）
                        sse_bytes = REGISTRY[client_proto].format_ir_as_sse(
                            intercepted_events,
                            ctx.response_model or actual_model,
                            reverse_tool_map=ir_request.extensions.get("reverse_tool_map"),
                            tool_spec_map=ir_request.extensions.get("tool_spec_map"),
                        )

                        # keepalive 包装：每 15s 无数据发心跳
                        async for sse_chunk in keepalive_wrapper(sse_bytes, interval=15.0):
                            yield sse_chunk
                        return

                except Exception as e:
                    # 连接级异常（DNS/超时/连接失败）在未 flush 阶段可安全重试；
                    # 但若已经进入 SSE body 遍历后抛出的异常，则不应重试（下游可能已收到部分 chunk）。
                    # 这里无法精确区分，保守做法：仅当 attempt < _RETRY_MAX 且未向下游 yield 过时重试。
                    # 由于 yield 都发生在 status_code 检查通过之后，连接级异常（在拿到 resp 之前）
                    # 一定属于「未 flush」，可安全重试。
                    if attempt < _RETRY_MAX and isinstance(
                        e, (httpx.ConnectError, httpx.TimeoutException, socket.gaierror, socket.timeout, OSError)
                    ):
                        wait = 2 ** attempt
                        logger.warning(f"IR stream connect error: {e}, retrying in {wait}s (attempt {attempt + 1}/{_RETRY_MAX + 1})")
                        await asyncio.sleep(wait)
                        continue
                    # 不可重试：发 error event 给客户端后终止
                    error_type = _classify_stream_error(e)
                    logger.error(f"IR stream error ({error_type}): {e}", exc_info=True)
                    had_error = True
                    err_events = _err_event_gen(
                        {"message": f"Proxy error: {type(e).__name__}: {e}"},
                        self.client_protocol,
                    )
                    async for chunk in REGISTRY[client_proto].format_ir_as_sse(
                        err_events,
                        ctx.response_model or actual_model,
                    ):
                        yield chunk
                    return
        finally:
            # 记录 usage：上游未返回 usage 时，用请求体内容估算 input tokens
            status = "error" if had_error else "success"
            # 如果上游没有返回 usage，用预计算的估算值；否则用实际值
            final_input = accumulated_input if accumulated_input > 0 else estimated_input_fallback
            rctx = ctx.extra.get("_record_ctx", {}) if ctx.extra else {}
            try:
                db.record_usage(
                    endpoint_id=endpoint_id,
                    model_id=model_id,
                    input_tokens=final_input,
                    output_tokens=accumulated_output,
                    status=status,
                    error_type=error_type if had_error else None,
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


