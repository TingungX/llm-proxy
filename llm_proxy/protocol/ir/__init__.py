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
"""

from __future__ import annotations

from llm_proxy.protocol.ir.anthropic import (
    response_from_ir as anthropic_response_from_ir,
    response_to_ir as anthropic_response_to_ir,
    to_ir as anthropic_to_ir,
    to_upstream as anthropic_to_upstream,
)
from llm_proxy.protocol.ir.chat import (
    response_from_ir as chat_response_from_ir,
    response_to_ir as chat_response_to_ir,
    to_ir as chat_to_ir,
    to_upstream as chat_to_upstream,
)
from llm_proxy.protocol.ir.responses import (
    response_from_ir as responses_response_from_ir,
    response_to_ir as responses_response_to_ir,
    to_ir as responses_to_ir,
    to_upstream as responses_to_upstream,
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

    def response_from_ir(self, ir: IRResponse) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError


class AnthropicConverter(ProtocolConverter):
    def to_ir(self, body):
        return anthropic_to_ir(body)

    def to_upstream(self, ir, upstream_model=None):
        return anthropic_to_upstream(ir, upstream_model)

    def response_to_ir(self, body):
        return anthropic_response_to_ir(body)

    def response_from_ir(self, ir):
        return anthropic_response_from_ir(ir)


class ChatConverter(ProtocolConverter):
    def to_ir(self, body):
        return chat_to_ir(body)

    def to_upstream(self, ir, upstream_model=None):
        return chat_to_upstream(ir, upstream_model)

    def response_to_ir(self, body):
        return chat_response_to_ir(body)

    def response_from_ir(self, ir):
        return chat_response_from_ir(ir)


class ResponsesConverter(ProtocolConverter):
    def to_ir(self, body):
        return responses_to_ir(body)

    def to_upstream(self, ir, upstream_model=None):
        return responses_to_upstream(ir, upstream_model)

    def response_to_ir(self, body):
        return responses_response_to_ir(body)

    def response_from_ir(self, ir):
        return responses_response_from_ir(ir)


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

