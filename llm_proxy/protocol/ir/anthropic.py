"""Anthropic Messages API ↔ IR 转换器。

请求方向：Anthropic 请求体 → IRRequest
响应方向：IRResponse → Anthropic 响应体
请求出方向：IRRequest → Anthropic 请求体（用于 Anthropic 客户端发到 Anthropic 上游）
响应入方向：Anthropic 响应体 → IRResponse
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from llm_proxy.protocol.constants import STOP_REASON_MAP
from llm_proxy.protocol.ir._common import (
    build_usage,
    clean_schema,
    map_tool_choice_to_chat,
    resolve_reasoning_effort,
    safe_json_loads,
    strip_leading_anthropic_billing_header,
    supports_reasoning_effort,
)
from llm_proxy.protocol.ir.types import (
    IRContentBlock,
    IRImageBlock,
    IRMessage,
    IRRequest,
    IRResponse,
    IRTextBlock,
    IRThinkingBlock,
    IRToolDef,
    IRToolResultBlock,
    IRToolUseBlock,
)
from llm_proxy.protocol.think_tag import strip_think_tags

logger = logging.getLogger(__name__)


# ── 请求：Anthropic → IR ──────────────────────────────────────────


def to_ir(body: dict[str, Any]) -> IRRequest:
    """Anthropic Messages API 请求体 → IRRequest。"""
    model = body.get("model", "")

    system_texts: list[str] = []
    system_cache_control: dict | None = None
    system = body.get("system")
    if isinstance(system, str):
        text = strip_leading_anthropic_billing_header(system)
        if text:
            system_texts.append(text)
    elif isinstance(system, list):
        for sys_part in system:
            if not isinstance(sys_part, dict):
                continue
            text = strip_leading_anthropic_billing_header(sys_part.get("text", ""))
            if not text:
                continue
            system_texts.append(text)
            if "cache_control" in sys_part:
                system_cache_control = sys_part["cache_control"]

    messages: list[IRMessage] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")

        converted_messages = _convert_message_to_ir(role, content)
        messages.extend(converted_messages)

    ir_request = IRRequest(
        model=model,
        messages=messages,
        system_prompt="\n".join(system_texts) if system_texts else None,
    )

    # 参数映射
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        ir_request.max_tokens = int(max_tokens)

    if "temperature" in body:
        ir_request.temperature = body["temperature"]
    if "top_p" in body:
        ir_request.top_p = body["top_p"]
    if "stream" in body:
        ir_request.stream = bool(body["stream"])

    if "stop_sequences" in body:
        ir_request.stop_sequences = body["stop_sequences"]

    # reasoning_effort
    if model and supports_reasoning_effort(model):
        effort = resolve_reasoning_effort(body)
        if effort:
            ir_request.reasoning_effort = effort

    # tools
    tools = body.get("tools") or []
    if tools:
        ir_tools: list[IRToolDef] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            # 过滤 BatchTool
            if tool.get("type") == "BatchTool":
                continue
            ir_tools.append(IRToolDef(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                parameters=clean_schema(tool.get("input_schema", {}) or {}),
                cache_control=tool.get("cache_control"),
            ))
        if ir_tools:
            ir_request.tools = ir_tools

    # tool_choice
    if "tool_choice" in body:
        ir_request.tool_choice = map_tool_choice_to_chat(body["tool_choice"])

    # extensions
    if system_cache_control is not None:
        ir_request.extensions["system_cache_control"] = system_cache_control
    if "thinking" in body:
        ir_request.extensions["thinking_config"] = body["thinking"]
    if "output_config" in body:
        ir_request.extensions["output_config"] = body["output_config"]
    if "metadata" in body:
        ir_request.extensions["metadata"] = body["metadata"]
    if "top_k" in body:
        ir_request.extensions["top_k"] = body["top_k"]

    return ir_request


def _convert_message_to_ir(role: str, content: Any) -> list[IRMessage]:
    """Anthropic 消息 → IRMessage 列表（可能产生多条，tool_result 独立）。"""
    if content is None:
        return [IRMessage(role=role, content="")]

    if isinstance(content, str):
        return [IRMessage(role=role, content=content)]

    if not isinstance(content, list):
        return [IRMessage(role=role, content=str(content))]

    # 数组 content — 按 block 类型分发
    blocks: list[IRContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        if block_type == "text":
            blocks.append(IRTextBlock(
                text=block.get("text", ""),
                cache_control=block.get("cache_control"),
            ))
        elif block_type == "image":
            source = block.get("source", {})
            blocks.append(IRImageBlock(
                base64_data=source.get("data", ""),
                media_type=source.get("media_type", "image/png"),
            ))
        elif block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                blocks.append(IRThinkingBlock(
                    thinking=thinking,
                    signature=block.get("signature"),
                ))
        elif block_type == "tool_use":
            blocks.append(IRToolUseBlock(
                id=block.get("id", ""),
                name=block.get("name", ""),
                input=block.get("input", {}) or {},
            ))
        elif block_type == "tool_result":
            # tool_result 产生独立的 tool role 消息
            tool_use_id = block.get("tool_use_id", "")
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                # 嵌套 blocks → 提取文本
                result_content = "\n".join(
                    b.get("text", "") for b in result_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            elif not isinstance(result_content, str):
                result_content = json.dumps(result_content, ensure_ascii=False)

            is_error = block.get("is_error", False)
            # 立即 yield 一条 tool 消息
            yield_msg = IRMessage(role="tool", content=[
                IRToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=result_content or "",
                    is_error=is_error,
                )
            ])
            yield_msg.name = None
            # 注入到 messages（处理方式：跳出循环再追加）
            return _merge_with_blocks(blocks, [yield_msg], role)

    if blocks:
        return [IRMessage(role=role, content=blocks)]
    return []


def _merge_with_blocks(
    existing_blocks: list[IRContentBlock],
    extra_messages: list[IRMessage],
    role: str,
) -> list[IRMessage]:
    """合并已收集的 blocks 与额外 messages。"""
    result: list[IRMessage] = []
    if existing_blocks:
        result.append(IRMessage(role=role, content=existing_blocks))
    result.extend(extra_messages)
    return result


# ── 响应：IR → Anthropic ──────────────────────────────────────────


def response_from_ir(ir: IRResponse) -> dict[str, Any]:
    """IRResponse → Anthropic Messages 响应体。"""
    content_blocks: list[dict] = []
    for block in ir.content_blocks:
        if isinstance(block, IRTextBlock):
            content_blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, IRThinkingBlock):
            tb: dict = {"type": "thinking", "thinking": block.thinking}
            if block.signature is not None:
                tb["signature"] = block.signature
            content_blocks.append(tb)
        elif isinstance(block, IRToolUseBlock):
            content_blocks.append({
                "type": "tool_use",
                "id": block.id or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": block.name,
                "input": block.input,
            })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": ir.id or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": ir.model,
        "stop_reason": ir.stop_reason,
        "stop_sequence": ir.stop_sequence,
        "usage": ir.usage,
    }


def response_to_ir(body: dict[str, Any]) -> IRResponse:
    """Anthropic Messages 响应体 → IRResponse。"""
    content_blocks: list[IRContentBlock] = []
    for block in body.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            content_blocks.append(IRTextBlock(text=block.get("text", "")))
        elif block_type == "thinking":
            content_blocks.append(IRThinkingBlock(
                thinking=block.get("thinking", ""),
                signature=block.get("signature"),
            ))
        elif block_type == "tool_use":
            content_blocks.append(IRToolUseBlock(
                id=block.get("id", ""),
                name=block.get("name", ""),
                input=block.get("input", {}) or {},
            ))
        elif block_type == "redacted_thinking":
            content_blocks.append(IRThinkingBlock(
                thinking="",
                signature=block.get("data", ""),
            ))

    if not content_blocks:
        content_blocks.append(IRTextBlock(text=""))

    return IRResponse(
        id=body.get("id", ""),
        model=body.get("model", ""),
        content_blocks=content_blocks,
        stop_reason=body.get("stop_reason", "end_turn"),
        stop_sequence=body.get("stop_sequence"),
        usage=build_usage(body.get("usage", {})),
    )


# ── 请求出：IR → Anthropic（用于客户端是 Anthropic、上游是 Anthropic）──


def to_upstream(ir: IRRequest, upstream_model: str | None = None) -> dict[str, Any]:
    """IRRequest → Anthropic Messages 请求体。"""
    result: dict[str, Any] = {}
    result["model"] = upstream_model or ir.model
    if result["model"] == "":
        result["model"] = ir.model

    # system
    if ir.system_prompt:
        system_cache = ir.extensions.get("system_cache_control")
        if system_cache is not None:
            result["system"] = [
                {"type": "text", "text": ir.system_prompt, "cache_control": system_cache}
            ]
        else:
            result["system"] = ir.system_prompt

    # messages
    result["messages"] = [_message_ir_to_anthropic(m) for m in ir.messages]

    # max_tokens — 必填字段（Anthropic API 要求）
    if ir.max_tokens is not None:
        result["max_tokens"] = int(ir.max_tokens)
    elif "max_tokens" not in result:
        # Anthropic API 强制要求 max_tokens；用安全默认 4096
        result["max_tokens"] = 4096

    if ir.temperature is not None:
        result["temperature"] = ir.temperature
    if ir.top_p is not None:
        result["top_p"] = ir.top_p
    if ir.stream:
        result["stream"] = True

    if ir.stop_sequences:
        result["stop_sequences"] = ir.stop_sequences

    # reasoning_effort → thinking / output_config
    if ir.reasoning_effort:
        output_config = ir.extensions.get("output_config")
        if isinstance(output_config, dict):
            result["output_config"] = {**output_config, "effort": ir.reasoning_effort}
        else:
            effort_to_budget = {
                "low": 2000,
                "medium": 8000,
                "high": 16000,
                "xhigh": 32000,
            }
            budget = effort_to_budget.get(ir.reasoning_effort, 16000)
            result["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # tools
    if ir.tools:
        anthropic_tools: list[dict] = []
        for tool in ir.tools:
            anthropic_tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            if tool.cache_control is not None:
                anthropic_tool["cache_control"] = tool.cache_control
            anthropic_tools.append(anthropic_tool)
        result["tools"] = anthropic_tools

    # tool_choice
    if ir.tool_choice is not None:
        tc = ir.tool_choice
        if tc in ("auto", "any", "none"):
            anthropic_tc = "any" if tc == "required" else tc
            result["tool_choice"] = anthropic_tc
        elif isinstance(tc, dict) and tc.get("type") == "function":
            result["tool_choice"] = {
                "type": "tool",
                "name": tc.get("function", {}).get("name", ""),
            }
        else:
            result["tool_choice"] = tc

    # metadata / top_k
    if "metadata" in ir.extensions:
        result["metadata"] = ir.extensions["metadata"]
    if "top_k" in ir.extensions:
        result["top_k"] = ir.extensions["top_k"]

    return result


def _message_ir_to_anthropic(msg: IRMessage) -> dict[str, Any]:
    """IRMessage → Anthropic message dict。"""
    role = msg.role
    content = msg.content

    if isinstance(content, str):
        return {"role": role, "content": content}

    if not isinstance(content, list):
        return {"role": role, "content": str(content)}

    blocks: list[dict] = []
    for block in content:
        if isinstance(block, IRTextBlock):
            tb: dict = {"type": "text", "text": block.text}
            if block.cache_control is not None:
                tb["cache_control"] = block.cache_control
            blocks.append(tb)
        elif isinstance(block, IRImageBlock):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.media_type,
                    "data": block.base64_data,
                },
            })
        elif isinstance(block, IRThinkingBlock):
            tb = {"type": "thinking", "thinking": block.thinking}
            if block.signature is not None:
                tb["signature"] = block.signature
            blocks.append(tb)
        elif isinstance(block, IRToolUseBlock):
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        elif isinstance(block, IRToolResultBlock):
            tr: dict = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
            }
            if block.is_error:
                tr["is_error"] = True
            blocks.append(tr)

    if not blocks:
        return {"role": role, "content": ""}

    return {"role": role, "content": blocks}

