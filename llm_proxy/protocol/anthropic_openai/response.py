"""OpenAI Chat Completions → Anthropic Messages 响应转换

将上游 OpenAI Chat Completions 格式的响应转换为 Anthropic Messages 格式，
用于跨协议代理返回给 Anthropic 格式客户端（Claude Code 等）。

参考: CCS (cc-switch) providers/transform.rs openai_to_anthropic()
"""

import json
import logging
import uuid

from llm_proxy.protocol.constants import STOP_REASON_MAP
from llm_proxy.protocol.think_tag import strip_think_tags

logger = logging.getLogger(__name__)


def _map_stop_reason(finish_reason: str | None) -> str:
    """将 OpenAI finish_reason 映射为 Anthropic stop_reason"""
    if finish_reason is None:
        return "end_turn"
    mapped = STOP_REASON_MAP.get(finish_reason)
    if mapped:
        return mapped
    logger.warning(f"Unknown finish_reason: {finish_reason}, mapping to end_turn")
    return "end_turn"


# ── Usage 映射 ────────────────────────────────────────────────────────

def _build_usage(raw_usage: dict) -> dict:
    """从 OpenAI usage 构建 Anthropic usage，含 cache token 支持

    OpenAI 字段名优先级:
      input_tokens (Anthropic 直接字段) > prompt_tokens (OpenAI 标准字段) > 0
      output_tokens > completion_tokens > 0

    Cache token 优先级:
      cache_read_input_tokens (直接字段) > prompt_tokens_details.cached_tokens > input_tokens_details.cached_tokens
      cache_creation_input_tokens — 仅直接字段
    """
    if not raw_usage or not isinstance(raw_usage, dict):
        return {"input_tokens": 0, "output_tokens": 0}

    # input_tokens
    input_tokens = raw_usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = raw_usage.get("prompt_tokens", 0)
    input_tokens = int(input_tokens) if input_tokens else 0

    # output_tokens
    output_tokens = raw_usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = raw_usage.get("completion_tokens", 0)
    output_tokens = int(output_tokens) if output_tokens else 0

    result = {"input_tokens": input_tokens, "output_tokens": output_tokens}

    # cache tokens — nested details 先作为 fallback
    cached_tokens = None
    # OpenAI Responses API: input_tokens_details.cached_tokens
    itd = raw_usage.get("input_tokens_details")
    if isinstance(itd, dict):
        ct = itd.get("cached_tokens")
        if ct and int(ct) > 0:
            cached_tokens = int(ct)
    # OpenAI Chat: prompt_tokens_details.cached_tokens
    ptd = raw_usage.get("prompt_tokens_details")
    if isinstance(ptd, dict) and cached_tokens is None:
        ct = ptd.get("cached_tokens")
        if ct and int(ct) > 0:
            cached_tokens = int(ct)

    if cached_tokens is not None:
        result["cache_read_input_tokens"] = cached_tokens

    # 直接字段覆盖（authoritative）
    if "cache_read_input_tokens" in raw_usage:
        result["cache_read_input_tokens"] = int(raw_usage["cache_read_input_tokens"])
    if "cache_creation_input_tokens" in raw_usage:
        result["cache_creation_input_tokens"] = int(raw_usage["cache_creation_input_tokens"])

    return result


# ── Content Block 构建 ────────────────────────────────────────────────

def _build_content_blocks(message: dict) -> list[dict]:
    """从 OpenAI Chat message 构建 Anthropic content blocks"""
    blocks = []

    # reasoning_content → thinking block
    reasoning = message.get("reasoning_content")

    # content → text block(s)
    content = message.get("content")

    # Strip think tags from string content
    extracted_reasoning = ""
    if content is not None and isinstance(content, str):
        extracted_reasoning, content = strip_think_tags(content)

    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    elif extracted_reasoning:
        blocks.append({"type": "thinking", "thinking": extracted_reasoning})

    if content is not None:
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")
                if part_type == "text":
                    text = part.get("text", "")
                    if text:
                        blocks.append({"type": "text", "text": text})
                elif part_type == "refusal":
                    # refusal → 转为 text block
                    refusal_text = part.get("refusal", "")
                    if refusal_text:
                        blocks.append({"type": "text", "text": refusal_text})

    # tool_calls → tool_use blocks
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        if tc.get("type") != "function":
            continue
        func = tc.get("function", {})
        name = func.get("name", "")
        arguments_str = func.get("arguments", "{}")
        try:
            input_data = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            input_data = {}

        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": name,
            "input": input_data,
        })

    return blocks


# ── 主转换函数 ────────────────────────────────────────────────────────

def chat_to_anthropic(body: dict) -> dict:
    """将 OpenAI Chat Completions 响应转换为 Anthropic Messages 响应。

    Args:
        body: OpenAI Chat Completions 格式响应体

    Returns:
        Anthropic Messages 格式响应体
    """
    choices = body.get("choices") or []
    if not choices:
        logger.warning("No choices in OpenAI response")
        return {
            "id": body.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "model": body.get("model", ""),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": _build_usage(body.get("usage", {})),
        }

    choice = choices[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason")

    content_blocks = _build_content_blocks(message)

    # 如果没有任何 content blocks，添加空 text block
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": body.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": body.get("model", ""),
        "stop_reason": _map_stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": _build_usage(body.get("usage", {})),
    }
