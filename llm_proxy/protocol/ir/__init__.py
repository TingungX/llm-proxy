"""统一协议 IR 抽象层。

三个协议（Anthropic、Chat Completions、Responses API）各自实现 to_ir / from_ir 转换器。
所有协议互转通过 IR 中转。增量迁移入口：旧 `anthropic_openai/` 和 `responses_chat/`
通道保持工作，新代码可走 IR 路径。

基本使用：
    from llm_proxy.protocol.ir import convert_request, convert_response

    # 请求方向：client → IR → upstream
    upstream_body = convert_request("anthropic", "openai/responses", body)

    # 响应方向：upstream → IR → client
    client_resp = convert_response("openai/responses", "anthropic", upstream_body)

流式：
    converter = REGISTRY["openai/chat-completions"]
    ir_events = converter.parse_stream_to_ir(upstream_resp, model)
    sse_bytes = converter.format_ir_as_sse(ir_events, model,
                                           reverse_tool_map=ir.extensions.get("reverse_tool_map"),
                                           tool_spec_map=ir.extensions.get("tool_spec_map"))
"""

from __future__ import annotations

from llm_proxy.protocol.ir.anthropic import (
    response_from_ir as anthropic_response_from_ir,
    response_to_ir as anthropic_response_to_ir,
    to_ir as anthropic_to_ir,
    to_upstream as anthropic_to_upstream,
    parse_stream_to_ir as anthropic_parse_stream,
    format_ir_as_sse as anthropic_format_sse,
)
from llm_proxy.protocol.ir.chat import (
    response_from_ir as chat_response_from_ir,
    response_to_ir as chat_response_to_ir,
    to_ir as chat_to_ir,
    to_upstream as chat_to_upstream,
    parse_stream_to_ir as chat_parse_stream,
    format_ir_as_sse as chat_format_sse,
)
from llm_proxy.protocol.ir.responses import (
    response_from_ir as responses_response_from_ir,
    response_to_ir as responses_response_to_ir,
    to_ir as responses_to_ir,
    to_upstream as responses_to_upstream,
    parse_stream_to_ir as responses_parse_stream,
    format_ir_as_sse as responses_format_sse,
)
from llm_proxy.protocol.ir.types import (
    IRContentBlock,
    IRImageBlock,
    IRMessage,
    IRRequest,
    IRResponse,
    IRStreamEvent,
    IRTextBlock,
    IRThinkingBlock,
    IRToolDef,
    IRToolResultBlock,
    IRToolUseBlock,
)


# ── 转换器注册表 ────────────────────────────────────────────────────


class ProtocolConverter:
    """协议转换器。每个协议提供此接口的具体实现。"""

    def to_ir(self, body: dict) -> IRRequest:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_upstream(self, ir: IRRequest, upstream_model: str | None = None) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def response_to_ir(self, body: dict) -> IRResponse:  # pragma: no cover - abstract
        raise NotImplementedError

    def response_from_ir(self, ir: IRResponse, *, reverse_tool_map: dict | None = None, tool_spec_map: dict | None = None) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    # ── 流式（异步）──

    async def parse_stream_to_ir(  # pragma: no cover - abstract
        self,
        resp,                # httpx 流式响应
        model: str,          # 上游模型名（id 兜底用）
    ):
        """解析上游 SSE 字节流 → IRStreamEvent 序列（异步生成器）。"""
        raise NotImplementedError

    async def format_ir_as_sse(  # pragma: no cover - abstract
        self,
        events,              # AsyncIterator[IRStreamEvent]
        model: str,          # 客户端模型名
        *,
        reverse_tool_map: dict | None = None,
        tool_spec_map: dict | None = None,
    ):
        """IRStreamEvent 序列 → 客户端 SSE 字节流（异步生成器）。

        reverse_tool_map / tool_spec_map 仅 Responses 协议使用：
        解析时存于 IRRequest.extensions，发送时透传过来。
        """
        raise NotImplementedError


class AnthropicConverter(ProtocolConverter):
    def to_ir(self, body):
        return anthropic_to_ir(body)

    def to_upstream(self, ir, upstream_model=None):
        return anthropic_to_upstream(ir, upstream_model)

    def response_to_ir(self, body):
        return anthropic_response_to_ir(body)

    def response_from_ir(self, ir, *, reverse_tool_map=None, tool_spec_map=None):
        return anthropic_response_from_ir(ir)

    async def parse_stream_to_ir(self, resp, model):
        async for event in anthropic_parse_stream(resp, model):
            yield event

    async def format_ir_as_sse(self, events, model, *, reverse_tool_map=None, tool_spec_map=None):
        async for chunk in anthropic_format_sse(events, model, reverse_tool_map=reverse_tool_map, tool_spec_map=tool_spec_map):
            yield chunk


class ChatConverter(ProtocolConverter):
    def to_ir(self, body):
        return chat_to_ir(body)

    def to_upstream(self, ir, upstream_model=None):
        return chat_to_upstream(ir, upstream_model)

    def response_to_ir(self, body):
        return chat_response_to_ir(body)

    def response_from_ir(self, ir, *, reverse_tool_map=None, tool_spec_map=None):
        return chat_response_from_ir(ir)

    async def parse_stream_to_ir(self, resp, model):
        async for event in chat_parse_stream(resp, model):
            yield event

    async def format_ir_as_sse(self, events, model, *, reverse_tool_map=None, tool_spec_map=None):
        async for chunk in chat_format_sse(events, model, reverse_tool_map=reverse_tool_map, tool_spec_map=tool_spec_map):
            yield chunk


class ResponsesConverter(ProtocolConverter):
    def to_ir(self, body):
        return responses_to_ir(body)

    def to_upstream(self, ir, upstream_model=None):
        return responses_to_upstream(ir, upstream_model)

    def response_to_ir(self, body):
        return responses_response_to_ir(body)

    def response_from_ir(self, ir, *, reverse_tool_map=None, tool_spec_map=None):
        return responses_response_from_ir(ir, reverse_tool_map=reverse_tool_map, tool_spec_map=tool_spec_map)

    async def parse_stream_to_ir(self, resp, model):
        async for event in responses_parse_stream(resp, model):
            yield event

    async def format_ir_as_sse(self, events, model, *, reverse_tool_map=None, tool_spec_map=None):
        async for chunk in responses_format_sse(events, model, reverse_tool_map=reverse_tool_map, tool_spec_map=tool_spec_map):
            yield chunk


# 协议名 → 转换器实例的注册表
REGISTRY: dict[str, ProtocolConverter] = {
    "anthropic": AnthropicConverter(),
    "openai/chat-completions": ChatConverter(),
    "openai/responses": ResponsesConverter(),
}


# 协议名 → 别名（兼容历史 "openai" = "openai/chat-completions"）
ALIASES: dict[str, str] = {
    "openai": "openai/chat-completions",
}


def _resolve(protocol: str) -> str:
    return ALIASES.get(protocol, protocol)


def convert_request(client_protocol: str, upstream_protocol: str, body: dict) -> dict:
    """client 协议 → IR → upstream 协议（一步到位）。"""
    ir = REGISTRY[_resolve(client_protocol)].to_ir(body)
    return REGISTRY[_resolve(upstream_protocol)].to_upstream(ir)


def convert_response(
    upstream_protocol: str, client_protocol: str, body: dict
) -> dict:
    """upstream 协议响应 → IR → client 协议响应。"""
    ir = REGISTRY[_resolve(upstream_protocol)].response_to_ir(body)
    return REGISTRY[_resolve(client_protocol)].response_from_ir(ir)


# 公开导出
__all__ = [
    "IRContentBlock",
    "IRImageBlock",
    "IRMessage",
    "IRRequest",
    "IRResponse",
    "IRStreamEvent",
    "IRTextBlock",
    "IRThinkingBlock",
    "IRToolDef",
    "IRToolResultBlock",
    "IRToolUseBlock",
    "ProtocolConverter",
    "AnthropicConverter",
    "ChatConverter",
    "ResponsesConverter",
    "REGISTRY",
    "convert_request",
    "convert_response",
    "ALIASES",
]
