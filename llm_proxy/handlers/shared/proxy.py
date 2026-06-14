"""代理转发步骤 — 根据协议和转换方式选择代理路径"""

import asyncio
import json
import logging
import time
import uuid

from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.handlers.base import PipelineContext, HandlerStep, PipelineStop
from llm_proxy.handlers.shared.paths import resolve_path
from llm_proxy.protocol.sse import stream_response as sse_stream
from llm_proxy.protocol.responses_chat.request import (
    to_responses_response,
    make_sse_event,
    make_response_completed_event,
    stream_chat_to_responses,
)
from llm_proxy.protocol.responses_chat.response import (
    convert_chat_to_responses_request,
    convert_responses_to_chat_response,
    stream_responses_to_chat,
)
from llm_proxy.protocol.errors import make_anthropic_error, make_openai_error
from llm_proxy.infra.http_client import get_client
from llm_proxy.infra import db
from llm_proxy.state import get_state, resolve_model_for_endpoint
from llm_proxy.logging_config import REQUEST_ID_CTX
from llm_proxy.services.vision_service import replace_images_in_anthropic_messages

logger = logging.getLogger(__name__)

_ANTHROPIC_FWD_HEADERS = {"anthropic-version", "anthropic-beta"}
_STATUS_CODE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "model_not_found",
    429: "rate_limit_exceeded",
    500: "server_error",
    502: "server_error",
    503: "server_error",
}
_RETRY_STATUSES = (429, 503)
_RETRY_MAX = 3


def _make_error(ctx: PipelineContext, message: str, error_type: str = "api_error", status_code: int = 400):
    if ctx.error_protocol == "anthropic":
        return make_anthropic_error(message, error_type, status_code)
    return make_openai_error(message, error_type, status_code)


def _status_to_code(status: int) -> str:
    return _STATUS_CODE_MAP.get(status, "api_error")


def _needs_reasoning_split(api_base: str) -> bool:
    """检测上游是否为 MiniMax（将 thinking 塞在 content 里），需要注入 reasoning_split=True"""
    return "minimaxi.com" in (api_base or "").lower()


def _estimate_tokens(body: dict) -> dict:
    messages = body.get("messages", [])
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return {"input_tokens": max(1, total_chars // 4)}


def _client_for(model_id: str | None) -> "httpx.AsyncClient":
    """Pick a pooled client based on whether the model is allow-listed to use
    the system HTTPS_PROXY. Default: direct (no proxy) for safety against
    mihomo/Clash toggles that leave 7897 listener up but refusing."""
    if model_id and get_state().allow_proxy_map.get(model_id.lower(), False):
        return get_client(direct=False)
    return get_client()


def _forward_anthropic_headers(headers: dict) -> dict:
    fwd = {}
    for key in _ANTHROPIC_FWD_HEADERS:
        val = headers.get(key)
        if val:
            fwd[key] = val
    return fwd


def _build_anthropic_upstream_headers(api_key: str, downstream_headers: dict) -> dict:
    fwd = _forward_anthropic_headers(downstream_headers)
    api_key = api_key.strip() if api_key else ""
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    headers.update(fwd)
    return headers


def _is_rate_error(resp_body: dict, status_code: int) -> bool:
    if status_code in (429, 503):
        return True
    return (
        resp_body.get("type") == "error"
        and resp_body.get("error", {}).get("type") in ("rate_limit_error", "overloaded_error")
    )


class ProxyStep(HandlerStep):
    """代理转发步骤 — 根据 upstream_protocol 选择代理路径"""

    def _record_usage(
        self, ctx: PipelineContext,
        endpoint_id: str, model_id: str,
        input_tokens: int, output_tokens: int,
        status: str = "success",
        error_type: str | None = None,
    ) -> None:
        """用 ctx 中的上下文信息补全 record_usage 调用"""
        rctx = ctx.extra.get("_record_ctx", {})
        start_time = rctx.get("start_time")
        latency = int((time.perf_counter() - start_time) * 1000) if start_time else None
        db.record_usage(
            endpoint_id, model_id, input_tokens, output_tokens, status,
            request_id=rctx.get("request_id", ""),
            latency_ms=latency,
            error_type=error_type,
            client_ip=rctx.get("client_ip", ""),
            user_agent=rctx.get("user_agent", ""),
        )

    async def execute(self, ctx: PipelineContext) -> None:
        ctx.extra["_record_ctx"] = {
            "start_time": time.perf_counter(),
            "request_id": REQUEST_ID_CTX.get(""),
            "client_ip": ctx.request.client.host if ctx.request.client else "",
            "user_agent": ctx.request.headers.get("user-agent", ""),
        }
        _, _, _, model_id, protocol, failover_family = ctx.resolved
        api_base, upstream_api_key, actual_model, _, _, _ = ctx.resolved

        endpoint = ctx.endpoint
        endpoint_id = endpoint["endpoint_id"]
        upstream_protocol = ctx.upstream_protocol
        raw_model = ctx.body.get("model", actual_model)  # SSE 事件中使用原始模型名
        ctx.extra["raw_model"] = raw_model

        if ctx.error_protocol == "anthropic":
            # Anthropic 格式：同协议透传 / 跨协议 Python 转换
            await self._handle_anthropic(ctx, api_base, upstream_api_key, actual_model, model_id,
                                          endpoint, endpoint_id, protocol, failover_family)
        elif upstream_protocol == "openai/responses":
            # Responses 同协议透传
            if ctx.converter is None:
                await self._proxy_responses_direct(ctx, api_base, upstream_api_key, actual_model,
                                                    model_id, endpoint_id)
            else:
                # Chat → Responses 转换
                await self._proxy_chat_to_responses(ctx, api_base, upstream_api_key, actual_model,
                                                     model_id, endpoint_id, endpoint)
        else:
            # Responses → Chat Completions 转换 / Chat 同协议透传
            await self._proxy_to_chat(ctx, api_base, upstream_api_key, actual_model, model_id,
                                      endpoint_id, endpoint)

    async def _handle_anthropic(
        self, ctx: PipelineContext, api_base: str, upstream_api_key: str,
        actual_model: str, model_id: str, endpoint: dict,
        endpoint_id: str, protocol: str | None, failover_family: str | None,
    ) -> None:
        """Anthropic 格式请求：同协议透传或跨协议 Python 转换"""
        from llm_proxy.protocol.capabilities import (
            NoReachableProtocol,
            select_upstream,
        )

        body = ctx.body
        downstream_protocol = "anthropic"

        # 多协议上游选择：同协议优先 → 否则按 IMPLEMENTED_CONVERSIONS 选
        s = get_state()
        available = s.protocols_map.get(model_id.lower(), set())
        try:
            final_upstream_protocol = select_upstream(downstream_protocol, available)
        except NoReachableProtocol as e:
            # 不可达：明确报 400（不静默 fallback）
            logger.error(f"Protocol selection failed: {e}")
            self._record_usage(ctx, endpoint_id, model_id, 0, 0,
                               status="error", error_type="no_reachable_protocol")
            raise PipelineStop(_make_error(
                ctx, str(e), "invalid_request_error", 400,
            ))

        endpoint_settings = endpoint.get("settings", {})
        global_eh = s.config.get("error_handling", {})
        failover_enabled = endpoint_settings.get("failover_enabled", global_eh.get("failover_enabled", False))
        no_retry = endpoint_settings.get("no_retry_enabled", global_eh.get("no_retry_enabled", False))

        def make_error_response(error_body: dict, status_code: int) -> JSONResponse:
            if "error" not in error_body:
                error_body = {"type": "error", "error": error_body}
            resp = JSONResponse(error_body, status_code=status_code)
            if no_retry:
                resp.headers["x-should-retry"] = "false"
            return resp

        if downstream_protocol == final_upstream_protocol:
            await self._anthropic_same_protocol(
                ctx, body, api_base, upstream_api_key, actual_model, model_id,
                endpoint, endpoint_id, endpoint_settings, failover_enabled, no_retry,
                failover_family, make_error_response,
            )
        else:
            await self._anthropic_cross_protocol(
                ctx, body, api_base, upstream_api_key, actual_model, model_id,
                endpoint_id, make_error_response,
            )

    async def _anthropic_same_protocol(
        self, ctx: PipelineContext, body: dict,
        api_base: str, api_key: str, actual_model: str, model_id: str,
        endpoint: dict, endpoint_id: str, endpoint_settings: dict,
        failover_enabled: bool, no_retry: bool, failover_family: str | None,
        make_error_response,
    ) -> None:
        """Anthropic 同协议透传"""
        out_body = dict(body)
        # vision fallback
        if not get_state().vision_map.get(model_id.lower(), False):
            if isinstance(out_body.get("messages"), list):
                out_body["messages"] = await replace_images_in_anthropic_messages(out_body["messages"])

        out_body["model"] = actual_model
        stream = out_body.get("stream", False)
        model_paths = get_state().paths_map.get(model_id.lower(), {})
        messages_path = resolve_path(model_paths, "anthropic/messages")
        target_url = f"{api_base.rstrip('/')}{messages_path}"
        req_headers = _build_anthropic_upstream_headers(api_key, ctx.headers)
        logger.info(f"Anthropic proxy: {target_url}, model={actual_model}, stream={stream}")

        if stream:
            ctx.response = StreamingResponse(
                self._anthropic_stream_gen(target_url, req_headers, out_body, body, endpoint_id, model_id,
                                            record_ctx=ctx.extra.get("_record_ctx", {})),
                media_type="text/event-stream",
            )
            return

        # Non-streaming
        resp_body, status_code = await self._anthropic_request(target_url, api_key, actual_model, out_body, ctx.headers, model_id)

        if _is_rate_error(resp_body, status_code) and failover_enabled and failover_family:
            current = failover_family
            endpoint_family_routing = endpoint.get("family_routing")
            s = get_state()
            while current:
                fr = resolve_model_for_endpoint(current, s.config, s.model_map, endpoint_family_routing)
                if not fr:
                    break
                f_api_base, f_api_key, f_upstream, f_model_id, _, next_failover = fr
                logger.info(f"Failover: trying {current} -> {f_model_id}")
                f_paths = s.paths_map.get(f_model_id.lower(), {})
                f_msg_path = resolve_path(f_paths, "anthropic/messages")
                f_target_url = f"{f_api_base.rstrip('/')}{f_msg_path}"
                fb, fs = await self._anthropic_request(f_target_url, f_api_key, f_upstream, out_body, ctx.headers, f_model_id)
                if fs < 400 or fb.get("type") != "error":
                    logger.info(f"Failover succeeded: {current} -> {f_model_id}")
                    resp_body, status_code = fb, fs
                    break
                if not _is_rate_error(fb, fs):
                    ctx.response = make_error_response(fb, fs)
                    return
                current = next_failover
            else:
                logger.warning("All failovers exhausted")

        if status_code >= 400:
            ctx.response = make_error_response(resp_body, status_code)
            return

        usage = resp_body.get("usage", {})
        self._record_usage(ctx, endpoint_id, model_id,
                           usage.get("input_tokens", 0),
                           usage.get("output_tokens", 0))
        ctx.response = JSONResponse(resp_body, status_code=status_code)

    async def _anthropic_request(
        self, target_url: str, api_key: str, actual_model: str,
        body: dict, headers: dict, model_id: str,
    ) -> tuple[dict, int]:
        """执行 Anthropic 非流式请求"""
        req_body = dict(body)
        req_body["model"] = actual_model
        h = _build_anthropic_upstream_headers(api_key, headers)
        try:
            client = _client_for(model_id)
            resp = await client.post(target_url, json=req_body, headers=h, timeout=120.0)
            try:
                return resp.json(), resp.status_code
            except json.JSONDecodeError:
                raw = resp.text[:500]
                logger.error(f"Upstream returned non-JSON response (status={resp.status_code}): {raw}")
                return {"error": {"type": "proxy_error", "message": f"Upstream returned non-JSON (status={resp.status_code}): {raw}"}}, 502
        except Exception as e:
            logger.error(f"Anthropic request error: {e}", exc_info=True)
            return {"error": {"type": "proxy_error", "message": str(e) or type(e).__name__}}, 500

    async def _anthropic_stream_gen(
        self, target_url: str, req_headers: dict, out_body: dict,
        original_body: dict, endpoint_id: str, model_id: str,
        record_ctx: dict | None = None,
    ):
        """Anthropic 流式生成器"""
        usage = {"input_tokens": 0, "output_tokens": 0}
        seen_stop = False

        def track_usage(event_type, data):
            nonlocal seen_stop
            if event_type == "message_start":
                u = data.get("message", {}).get("usage", {})
                if u.get("input_tokens") is not None:
                    usage["input_tokens"] = u["input_tokens"]
            elif event_type == "message_delta":
                u = data.get("usage", {})
                if u.get("input_tokens"):
                    usage["input_tokens"] = u["input_tokens"]
                if u.get("output_tokens"):
                    usage["output_tokens"] = u["output_tokens"]
            elif event_type == "message_stop":
                seen_stop = True

        try:
            client = _client_for(model_id)
            async with client.stream("POST", target_url, json=out_body, headers=req_headers, timeout=120.0) as resp:
                logger.info(f"Anthropic stream response status: {resp.status_code}")
                async for chunk in sse_stream(resp, on_event=track_usage):
                    yield chunk
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}", exc_info=True)
        finally:
            _rctx = record_ctx or {}
            if seen_stop and usage["input_tokens"]:
                db.record_usage(endpoint_id, model_id,
                                usage["input_tokens"], usage["output_tokens"], "success",
                                request_id=_rctx.get("request_id", ""),
                                client_ip=_rctx.get("client_ip", ""),
                                user_agent=_rctx.get("user_agent", ""))
            else:
                estimated = _estimate_tokens(original_body)
                db.record_usage(endpoint_id, model_id,
                                estimated.get("input_tokens", 0), 0, "success",
                                request_id=_rctx.get("request_id", ""),
                                client_ip=_rctx.get("client_ip", ""),
                                user_agent=_rctx.get("user_agent", ""))

    async def _anthropic_cross_protocol(
        self, ctx: PipelineContext, body: dict,
        api_base: str, api_key: str, actual_model: str, model_id: str,
        endpoint_id: str, make_error_response,
    ) -> None:
        """Anthropic 跨协议：转换为 OpenAI Chat Completions 格式发到上游"""
        from llm_proxy.protocol.anthropic_openai import (
            anthropic_to_chat,
            chat_to_anthropic,
            create_anthropic_sse_stream,
            should_rectify,
            rectify_request,
        )

        # 1. 转换请求
        chat_body = anthropic_to_chat(body)
        chat_body["model"] = actual_model
        if _needs_reasoning_split(api_base):
            chat_body["reasoning_split"] = True
        stream = chat_body.get("stream", False)

        state = get_state()
        model_paths = state.paths_map.get(model_id.lower(), {})
        chat_path = resolve_path(model_paths, "openai/chat-completions")
        target_url = f"{api_base.rstrip('/')}{chat_path}"
        req_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"Cross-protocol: {target_url}, model={actual_model}, stream={stream}")

        # 2. 流式请求
        if stream:
            ctx.response = StreamingResponse(
                self._cross_protocol_stream_gen(
                    target_url, req_headers, chat_body, body, actual_model, endpoint_id, model_id,
                    record_ctx=ctx.extra.get("_record_ctx", {}),
                ),
                media_type="text/event-stream",
            )
            return

        # 3. 非流式请求
        try:
            client = _client_for(model_id)
            resp = await client.post(target_url, json=chat_body, headers=req_headers, timeout=120.0)
            try:
                resp_body = resp.json()
            except json.JSONDecodeError:
                raw = resp.text[:500]
                logger.error(f"Cross-protocol upstream returned non-JSON (status={resp.status_code}): {raw}")
                ctx.response = make_error_response(
                    {"type": "proxy_error", "message": f"Upstream returned non-JSON (status={resp.status_code}): {raw}"},
                    502,
                )
                return
            status_code = resp.status_code
        except Exception as e:
            logger.error(f"Cross-protocol request error: {e}", exc_info=True)
            ctx.response = make_error_response({"type": "proxy_error", "message": str(e) or type(e).__name__}, 500)
            return

        # 4. Thinking 整流器：400 错误时检测并重试
        if status_code == 400:
            error_msg = None
            if isinstance(resp_body, dict):
                error_obj = resp_body.get("error", resp_body)
                if isinstance(error_obj, dict):
                    error_msg = error_obj.get("message", json.dumps(resp_body, ensure_ascii=False))
                else:
                    error_msg = str(error_obj)

            if should_rectify(error_msg):
                logger.info(f"Thinking rectifier triggered for model {model_id}, retrying...")
                rectified_body = dict(body)
                result = rectify_request(rectified_body)
                logger.info(
                    f"Rectify result: applied={result.applied}, "
                    f"removed_thinking={result.removed_thinking_blocks}, "
                    f"removed_redacted={result.removed_redacted_thinking_blocks}, "
                    f"removed_signatures={result.removed_signature_fields}"
                )

                if result.applied:
                    # 用整流后的 body 重新转换并重试
                    retry_chat_body = anthropic_to_chat(rectified_body)
                    retry_chat_body["model"] = actual_model
                    try:
                        retry_resp = await client.post(
                            target_url, json=retry_chat_body, headers=req_headers, timeout=120.0
                        )
                        try:
                            resp_body = retry_resp.json()
                        except json.JSONDecodeError:
                            raw = retry_resp.text[:500]
                            logger.error(f"Cross-protocol retry upstream returned non-JSON (status={retry_resp.status_code}): {raw}")
                            ctx.response = make_error_response(
                                {"type": "proxy_error", "message": f"Upstream returned non-JSON (status={retry_resp.status_code}): {raw}"},
                                502,
                            )
                            return
                        status_code = retry_resp.status_code
                    except Exception as e:
                        logger.error(f"Cross-protocol retry error: {e}", exc_info=True)
                        ctx.response = make_error_response({"type": "proxy_error", "message": str(e) or type(e).__name__}, 500)
                        return

        if status_code >= 400:
            ctx.response = make_error_response(resp_body, status_code)
            return

        # 5. 转换响应
        anthropic_response = chat_to_anthropic(resp_body)
        usage = anthropic_response.get("usage", {})
        self._record_usage(ctx, endpoint_id, model_id,
                           usage.get("input_tokens", 0),
                           usage.get("output_tokens", 0))
        ctx.response = JSONResponse(anthropic_response, status_code=status_code)

    async def _cross_protocol_stream_gen(
        self, target_url: str, req_headers: dict, chat_body: dict,
        original_body: dict, model: str, endpoint_id: str, model_id: str,
        record_ctx: dict | None = None,
    ):
        """跨协议流式生成器：Chat SSE → Anthropic SSE"""
        from llm_proxy.protocol.anthropic_openai import create_anthropic_sse_stream

        rctx = record_ctx or {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        seen_stop = False
        upstream_errored = False

        def track_usage(event_type: str, data: dict):
            nonlocal seen_stop
            if event_type == "message_start":
                u = data.get("message", {}).get("usage", {})
                if u.get("input_tokens") is not None:
                    usage["input_tokens"] = u["input_tokens"]
            elif event_type == "message_delta":
                u = data.get("usage", {})
                if u.get("input_tokens"):
                    usage["input_tokens"] = u["input_tokens"]
                if u.get("output_tokens"):
                    usage["output_tokens"] = u["output_tokens"]
            elif event_type == "message_stop":
                seen_stop = True

        try:
            client = _client_for(model_id)
            async with client.stream("POST", target_url, json=chat_body, headers=req_headers, timeout=120.0) as resp:
                logger.info(f"Cross-protocol stream response status: {resp.status_code}")

                if resp.status_code >= 400:
                    upstream_errored = True
                    error_body = await resp.aread()
                    error_msg = error_body.decode()
                    logger.error(f"Cross-protocol upstream error: {resp.status_code} {error_msg[:500]}")
                    err_event = {
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Upstream error ({resp.status_code}): {error_msg[:200]}"},
                    }
                    yield f"event: error\ndata: {json.dumps(err_event)}\n\n".encode()
                    return

                async for chunk in create_anthropic_sse_stream(resp, model, on_event=track_usage):
                    yield chunk

        except Exception as e:
            upstream_errored = True
            logger.error(f"Cross-protocol stream error: {e}", exc_info=True)
            err_body = {"type": "error", "error": {"type": "proxy_error", "message": str(e) or type(e).__name__}}
            yield f"event: error\ndata: {json.dumps(err_body)}\n\n".encode()
        finally:
            if upstream_errored:
                db.record_usage(endpoint_id, model_id, 0, 0, "error",
                                request_id=rctx.get("request_id", ""),
                                client_ip=rctx.get("client_ip", ""),
                                user_agent=rctx.get("user_agent", ""))
            elif seen_stop and usage["input_tokens"]:
                db.record_usage(endpoint_id, model_id,
                                usage["input_tokens"], usage["output_tokens"], "success",
                                request_id=rctx.get("request_id", ""),
                                client_ip=rctx.get("client_ip", ""),
                                user_agent=rctx.get("user_agent", ""))
            else:
                estimated = _estimate_tokens(original_body)
                db.record_usage(endpoint_id, model_id,
                                estimated.get("input_tokens", 0), 0, "success",
                                request_id=rctx.get("request_id", ""),
                                client_ip=rctx.get("client_ip", ""),
                                user_agent=rctx.get("user_agent", ""))

    async def _proxy_responses_direct(
        self, ctx: PipelineContext, api_base: str, upstream_api_key: str,
        actual_model: str, model_id: str, endpoint_id: str,
    ) -> None:
        """Responses 同协议透传"""
        body = ctx.body
        stream = body.get("stream", False)
        model_paths = get_state().paths_map.get(model_id.lower(), {})
        resp_path = resolve_path(model_paths, "openai/responses")
        target_url = f"{api_base.rstrip('/')}{resp_path}"
        req_headers = {
            "Authorization": f"Bearer {upstream_api_key}",
            "Content-Type": "application/json",
        }

        if stream:
            ctx.response = StreamingResponse(
                self._responses_direct_stream(target_url, req_headers, body, actual_model, endpoint_id, model_id,
                                              record_ctx=ctx.extra.get("_record_ctx", {})),
                media_type="text/event-stream",
            )
            return

        try:
            client = _client_for(model_id)
            resp = await client.post(target_url, json=body, headers=req_headers, timeout=120.0)
            try:
                resp_body = resp.json()
            except json.JSONDecodeError:
                raw = resp.text[:500]
                logger.error(f"Responses direct upstream returned non-JSON (status={resp.status_code}): {raw}")
                raise PipelineStop(_make_error(ctx, f"Upstream returned non-JSON (status={resp.status_code}): {raw}", "proxy_error", 502))
        except json.JSONDecodeError:
            raise
        except Exception as e:
            logger.error(f"Responses direct proxy failed: {e}", exc_info=True)
            raise PipelineStop(_make_error(ctx, f"Upstream error: {e}", "proxy_error", 502))

        if resp.status_code >= 400:
            upstream_error = resp_body.get("error")
            if isinstance(upstream_error, dict) and upstream_error.get("message"):
                error_obj = {"message": upstream_error["message"],
                             "type": upstream_error.get("type", "api_error")}
                if upstream_error.get("code"):
                    error_obj["code"] = upstream_error["code"]
                if upstream_error.get("param"):
                    error_obj["param"] = upstream_error["param"]
                raise PipelineStop(JSONResponse({"error": error_obj}, status_code=resp.status_code))
            raise PipelineStop(_make_error(ctx, resp_body.get("message", str(resp_body)),
                                           "api_error", resp.status_code))

        usage = resp_body.get("usage", {})
        self._record_usage(ctx, endpoint_id, model_id,
                           usage.get("input_tokens", 0),
                           usage.get("output_tokens", 0))
        ctx.response = JSONResponse(resp_body)

    async def _responses_direct_stream(
        self, target_url: str, headers: dict, body: dict,
        model: str, endpoint_id: str, model_id: str,
        record_ctx: dict | None = None,
    ):
        """Responses 直接透传流式"""
        rctx = record_ctx or {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        had_error = False

        try:
            client = _client_for(model_id)
            async with client.stream("POST", target_url, json=body, headers=headers, timeout=120.0) as resp:
                logger.info(f"Direct stream response status: {resp.status_code}")
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    error_msg = error_body.decode()
                    try:
                        err_data = json.loads(error_msg)
                        err_message = err_data.get("error", {}).get("message", error_msg)
                        err_code = err_data.get("error", {}).get("code") or _status_to_code(resp.status_code)
                    except json.JSONDecodeError:
                        err_message, err_code = error_msg, _status_to_code(resp.status_code)
                    yield make_sse_event(
                        {"error": {"code": err_code, "message": err_message}},
                        event_type="error",
                    )
                    yield make_response_completed_event(model, f"resp_{uuid.uuid4().hex[:16]}")
                    yield b"data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if line:
                        yield line.encode() if isinstance(line, str) else line
                        if line.startswith("data: ") and not line.endswith("[DONE]"):
                            try:
                                data = json.loads(line[6:])
                                if "usage" in data:
                                    u = data["usage"]
                                    usage["input_tokens"] = u.get("input_tokens", 0)
                                    usage["output_tokens"] = u.get("output_tokens", 0)
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            had_error = True
            logger.error(f"Direct stream error: {e}", exc_info=True)
            yield make_sse_event(
                {"error": {"code": "proxy_error", "message": str(e) or type(e).__name__}},
                event_type="error",
            )
            yield make_response_completed_event(model, f"resp_{uuid.uuid4().hex[:16]}")
            yield b"data: [DONE]\n\n"
        finally:
            status = "error" if had_error else "success"
            db.record_usage(endpoint_id, model_id,
                            usage["input_tokens"] or 1,
                            usage["output_tokens"] or 0, status,
                            request_id=rctx.get("request_id", ""),
                            client_ip=rctx.get("client_ip", ""),
                            user_agent=rctx.get("user_agent", ""))

    async def _proxy_to_chat(
        self, ctx: PipelineContext, api_base: str, upstream_api_key: str,
        actual_model: str, model_id: str, endpoint_id: str, endpoint: dict,
    ) -> None:
        """Responses → Chat Completions 转换代理（含流式重试逻辑）"""
        body = ctx.body
        # 用于 SSE 事件的原始模型名
        raw_model = ctx.extra.get("raw_model", actual_model)

        if ctx.converter == "responses_to_chat":
            # 已是转换后的 chat_body
            chat_body = body
            if _needs_reasoning_split(api_base):
                chat_body["reasoning_split"] = True
            reverse_tool_map = ctx.reverse_tool_map
            tool_spec_map = ctx.tool_spec_map
            model = raw_model  # SSE 事件中使用原始模型名

            model_paths = get_state().paths_map.get(model_id.lower(), {})
            chat_path = resolve_path(model_paths, "openai/chat-completions")
            target_url = f"{api_base.rstrip('/')}{chat_path}"
            req_headers = {
                "Authorization": f"Bearer {upstream_api_key}",
                "Content-Type": "application/json",
            }

            stream = chat_body.get("stream", False)

            if stream:
                ctx.response = StreamingResponse(
                    self._responses_to_chat_stream(target_url, req_headers, chat_body, model,
                                                   endpoint_id, model_id, reverse_tool_map,
                                                   tool_spec_map=tool_spec_map,
                                                   record_ctx=ctx.extra.get("_record_ctx", {})),
                    media_type="text/event-stream",
                )
                return

            try:
                client = _client_for(model_id)
                resp = await client.post(target_url, json=chat_body, headers=req_headers, timeout=120.0)
                try:
                    resp_body = resp.json()
                except json.JSONDecodeError:
                    raw = resp.text[:500]
                    logger.error(f"Responses→Chat upstream returned non-JSON (status={resp.status_code}): {raw}")
                    raise PipelineStop(_make_error(ctx, f"Upstream returned non-JSON (status={resp.status_code}): {raw}", "proxy_error", 502))
            except json.JSONDecodeError:
                raise
            except Exception as e:
                logger.error(f"Request to upstream failed: {e}", exc_info=True)
                raise PipelineStop(_make_error(ctx, f"Upstream error: {e}", "proxy_error", 502))

            if resp.status_code >= 400:
                upstream_error = resp_body.get("error")
                if isinstance(upstream_error, dict) and upstream_error.get("message"):
                    error_obj = {"message": upstream_error["message"],
                                 "type": upstream_error.get("type", "api_error")}
                    if upstream_error.get("code"): error_obj["code"] = upstream_error["code"]
                    if upstream_error.get("param"): error_obj["param"] = upstream_error["param"]
                    raise PipelineStop(JSONResponse({"error": error_obj}, status_code=resp.status_code))
                raise PipelineStop(_make_error(ctx, resp_body.get("message", str(resp_body)),
                                               "api_error", resp.status_code))

            self._record_usage(ctx, endpoint_id, model_id,
                               resp_body.get("usage", {}).get("prompt_tokens", 0),
                               resp_body.get("usage", {}).get("completion_tokens", 0))

            responses_body = to_responses_response(resp_body, actual_model, reverse_tool_map, tool_spec_map=tool_spec_map)
            ctx.response = JSONResponse(responses_body)
        else:
            # Chat completions 同协议透传
            await self._proxy_chat_completions(ctx, api_base, upstream_api_key, actual_model,
                                                model_id, endpoint_id)

    async def _responses_to_chat_stream(
        self, target_url: str, headers: dict, body: dict,
        model: str, endpoint_id: str, model_id: str,
        reverse_tool_map: dict | None = None,
        tool_spec_map: dict | None = None,
        record_ctx: dict | None = None,
    ):
        """Responses → Chat 流式 SSE 转换，含 429/503 重试"""
        for attempt in range(_RETRY_MAX + 1):
            try:
                client = _client_for(model_id)
                async with client.stream("POST", target_url, json=body, headers=headers, timeout=120.0) as resp:
                    status = resp.status_code
                    if status in _RETRY_STATUSES and attempt < _RETRY_MAX:
                        wait = 2 ** attempt
                        logger.warning(f"Upstream {status}, retrying in {wait}s (attempt {attempt + 1}/{_RETRY_MAX + 1})")
                        await asyncio.sleep(wait)
                        continue

                    if status >= 400:
                        error_body = await resp.aread()
                        error_msg = error_body.decode()
                        try:
                            err_data = json.loads(error_msg)
                            err_message = err_data.get("error", {}).get("message", error_msg)
                            err_code = err_data.get("error", {}).get("code") or _status_to_code(status)
                        except json.JSONDecodeError:
                            err_message, err_code = error_msg, _status_to_code(status)
                        yield make_sse_event(
                            {"error": {"code": err_code, "message": err_message}},
                            event_type="error",
                        )
                        yield make_response_completed_event(model, f"resp_{uuid.uuid4().hex[:16]}")
                        yield b"data: [DONE]\n\n"
                        return

                    stream_result = {"completed": False, "has_text": False}
                    async for event in stream_chat_to_responses(
                        resp, model, endpoint_id, model_id,
                        original_request=body, result=stream_result,
                        reverse_tool_map=reverse_tool_map,
                        tool_spec_map=tool_spec_map,
                        request_id=(record_ctx or {}).get("request_id", ""),
                        client_ip=(record_ctx or {}).get("client_ip", ""),
                        user_agent=(record_ctx or {}).get("user_agent", ""),
                    ):
                        yield event
                    return

            except Exception as e:
                if attempt < _RETRY_MAX:
                    wait = 2 ** attempt
                    logger.warning(f"Stream error: {e}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"Stream request error after {_RETRY_MAX + 1} attempts: {e}", exc_info=True)
                yield make_sse_event(
                    {"error": {"code": "proxy_error", "message": f"{type(e).__name__}: {e}"}},
                    event_type="error",
                )
                yield make_response_completed_event(model, f"resp_{uuid.uuid4().hex[:16]}")
                yield b"data: [DONE]\n\n"
                return

    async def _proxy_chat_completions(
        self, ctx: PipelineContext, api_base: str, upstream_api_key: str,
        actual_model: str, model_id: str, endpoint_id: str,
    ) -> None:
        """Chat Completions 同协议透传"""
        body = ctx.body
        out_body = dict(body)
        out_body["model"] = actual_model
        if _needs_reasoning_split(api_base):
            out_body["reasoning_split"] = True
        stream = out_body.get("stream", False)
        model_paths = get_state().paths_map.get(model_id.lower(), {})
        chat_path = resolve_path(model_paths, "openai/chat-completions")
        target_url = f"{api_base.rstrip('/')}{chat_path}"
        req_headers = {
            "Authorization": f"Bearer {upstream_api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"Chat completions proxy: {target_url}, model={actual_model}, stream={stream}")

        if stream:
            self._record_usage(ctx, endpoint_id, model_id, 0, 0)
            ctx.response = StreamingResponse(
                self._chat_stream_gen(target_url, req_headers, out_body, model_id),
                media_type="text/event-stream",
            )
            return

        try:
            client = _client_for(model_id)
            resp = await client.post(target_url, json=out_body, headers=req_headers, timeout=120.0)
            try:
                resp_body = resp.json()
            except json.JSONDecodeError:
                raw = resp.text[:500]
                logger.error(f"Chat completions upstream returned non-JSON (status={resp.status_code}): {raw}")
                self._record_usage(ctx, endpoint_id, model_id, 0, 0, status="error", error_type="proxy_error")
                raise PipelineStop(_make_error(ctx, f"Upstream returned non-JSON (status={resp.status_code}): {raw}", "proxy_error", 502))
        except json.JSONDecodeError:
            raise
        except Exception as e:
            logger.error(f"Request error: {e}", exc_info=True)
            self._record_usage(ctx, endpoint_id, model_id, 0, 0, status="error", error_type="proxy_error")
            raise PipelineStop(_make_error(ctx, str(e) or type(e).__name__, "proxy_error", 502))

        if resp.status_code >= 400:
            self._record_usage(ctx, endpoint_id, model_id, 0, 0, status="error",
                               error_type=_status_to_code(resp.status_code))
            raise PipelineStop(JSONResponse(
                make_openai_error(
                    resp_body.get("error", {}).get("message", "Upstream error"),
                    resp_body.get("error", {}).get("type", "api_error"),
                    resp.status_code,
                ).body,
                status_code=resp.status_code,
                media_type="application/json",
            ))

        usage = resp_body.get("usage", {})
        self._record_usage(ctx, endpoint_id, model_id,
                           usage.get("prompt_tokens", 0),
                           usage.get("completion_tokens", 0))
        ctx.response = JSONResponse(resp_body)

    async def _chat_stream_gen(self, target_url: str, headers: dict, body: dict, model_id: str):
        """Chat Completions 流式透传"""
        client = _client_for(model_id)
        try:
            async with client.stream("POST", target_url, json=body, headers=headers, timeout=120.0) as resp:
                logger.info(f"Chat stream response status: {resp.status_code}")
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield b"data: [DONE]\n\n"

    async def _proxy_chat_to_responses(
        self, ctx: PipelineContext, api_base: str, upstream_api_key: str,
        actual_model: str, model_id: str, endpoint_id: str, endpoint: dict,
    ) -> None:
        """Chat → Responses 转换模式"""
        body = ctx.body
        responses_body = convert_chat_to_responses_request(body)
        responses_body["model"] = actual_model

        stream = body.get("stream", False)
        if stream:
            responses_body["stream"] = True

        model_paths = get_state().paths_map.get(model_id.lower(), {})
        resp_path = resolve_path(model_paths, "openai/responses")
        target_url = f"{api_base.rstrip('/')}{resp_path}"
        req_headers = {
            "Authorization": f"Bearer {upstream_api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"Chat→Responses: {target_url}, model={actual_model}, stream={stream}")

        if stream:
            self._record_usage(ctx, endpoint_id, model_id, 0, 0)
            ctx.response = StreamingResponse(
                self._chat_to_responses_stream(target_url, req_headers, responses_body, actual_model,
                                               endpoint_id, model_id,
                                               record_ctx=ctx.extra.get("_record_ctx", {})),
                media_type="text/event-stream",
            )
            return

        try:
            client = _client_for(model_id)
            resp = await client.post(target_url, json=responses_body, headers=req_headers, timeout=120.0)
            try:
                resp_body = resp.json()
            except json.JSONDecodeError:
                raw = resp.text[:500]
                logger.error(f"Chat→Responses upstream returned non-JSON (status={resp.status_code}): {raw}")
                raise PipelineStop(_make_error(ctx, f"Upstream returned non-JSON (status={resp.status_code}): {raw}", "proxy_error", 502))
        except json.JSONDecodeError:
            raise
        except Exception as e:
            logger.error(f"Request error: {e}", exc_info=True)
            raise PipelineStop(_make_error(ctx, str(e) or type(e).__name__, "proxy_error", 502))

        if resp.status_code >= 400:
            raise PipelineStop(JSONResponse(
                make_openai_error(
                    resp_body.get("error", {}).get("message", "Upstream error"),
                    resp_body.get("error", {}).get("type", "api_error"),
                    resp.status_code,
                ).body,
                status_code=resp.status_code,
                media_type="application/json",
            ))

        usage = resp_body.get("usage", {})
        chat_response = convert_responses_to_chat_response(resp_body, actual_model)
        self._record_usage(ctx, endpoint_id, model_id,
                           usage.get("input_tokens", 0),
                           usage.get("output_tokens", 0))
        ctx.response = JSONResponse(chat_response)

    async def _chat_to_responses_stream(
        self, target_url: str, headers: dict, body: dict,
        model: str, endpoint_id: str, model_id: str,
        record_ctx: dict | None = None,
    ):
        """Chat→Responses 流式转换"""
        rctx = record_ctx or {}
        try:
            client = _client_for(model_id)
            async with client.stream("POST", target_url, json=body, headers=headers, timeout=120.0) as resp:
                logger.info(f"Chat→Responses stream status: {resp.status_code}")
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    error_chunk = {"error": {"message": error_body.decode(), "type": "api_error"}}
                    yield f"data: {json.dumps(error_chunk)}\n\n".encode()
                    return
                async for event in stream_responses_to_chat(resp, model, endpoint_id, model_id,
                                                           request_id=rctx.get("request_id", ""),
                                                           client_ip=rctx.get("client_ip", ""),
                                                           user_agent=rctx.get("user_agent", "")):
                    yield event
        except Exception as e:
            logger.error(f"Chat→Responses stream error: {e}", exc_info=True)
            error_chunk = {"error": {"message": str(e) or type(e).__name__, "type": "proxy_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n".encode()
