"""Responses↔Chat 协议转换通道。

将 OpenAI Responses API 请求转换为 Chat Completions 格式，
并将上游 Chat Completions 响应转换回 Responses API 格式。

.. deprecated::
    旧通道。新代码请走 ``llm_proxy.protocol.ir``（IR 抽象层）。
    保留是为了不破坏现有 ProxyStep 的 if/elif 路由；后续可逐步迁移。
"""

import warnings as _warnings

_warnings.warn(
    "protocol.responses_chat is legacy; prefer protocol.ir for new code paths. "
    "See protocol/ir/__init__.py and AGENTS.md for migration notes.",
    PendingDeprecationWarning,
    stacklevel=2,
)

from llm_proxy.protocol.responses_chat.request import (
    convert_input_to_messages,
    convert_tools_to_chat,
    to_responses_response,
    convert_chunk_to_events,
    make_sse_event,
    make_response_completed_event,
    stream_chat_to_responses,
)
from llm_proxy.protocol.responses_chat.response import (
    convert_chat_to_responses_request,
    convert_responses_to_chat_response,
    stream_responses_to_chat,
)

__all__ = [
    # Responses → Chat
    "convert_input_to_messages",
    "convert_tools_to_chat",
    "to_responses_response",
    "convert_chunk_to_events",
    "make_sse_event",
    "make_response_completed_event",
    "stream_chat_to_responses",
    # Chat → Responses
    "convert_chat_to_responses_request",
    "convert_responses_to_chat_response",
    "stream_responses_to_chat",
]
