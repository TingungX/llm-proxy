"""Anthropic↔OpenAI Chat Completions 协议转换通道

当 Anthropic 格式请求需要转发到 OpenAI Chat Completions 上游时，
此模块负责双向协议转换，替代 anthropic-proxy-rs sidecar。

子模块:
  request  — Anthropic → Chat 请求转换
  response — Chat → Anthropic 响应转换
  stream   — Chat SSE → Anthropic SSE 流式转换
  rectifier — Thinking signature 整流器
"""

from llm_proxy.protocol.anthropic_openai.request import anthropic_to_chat
from llm_proxy.protocol.anthropic_openai.response import chat_to_anthropic
from llm_proxy.protocol.anthropic_openai.stream import create_anthropic_sse_stream
from llm_proxy.protocol.anthropic_openai.rectifier import should_rectify, rectify_request

__all__ = [
    "anthropic_to_chat",
    "chat_to_anthropic",
    "create_anthropic_sse_stream",
    "should_rectify",
    "rectify_request",
]

