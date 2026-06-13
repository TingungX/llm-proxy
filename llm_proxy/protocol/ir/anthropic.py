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
from typing import Any, AsyncIterator

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
from llm_proxy.protocol.ir._stream import (
    DONE_SENTINEL,
    IncrementalJSONParser,
    extract_usage_tokens,
    parse_sse_line,
    sse_format,
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
    IRRedactedThinkingBlock,
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
    # 收集非 tool_result blocks 和 tool_result messages，最后统一拆分
    # 原因：一条 Anthropic user message 可能包含多个 tool_result，
    # 必须全部处理完再拆分，不能遇到第一个就 return
    blocks: list[IRContentBlock] = []
    tool_messages: list[IRMessage] = []

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
            # tool_result 产生独立的 tool role 消息，先收集
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
            tool_msg = IRMessage(role="tool", content=[
                IRToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=result_content or "",
                    is_error=is_error,
                )
            ])
            tool_msg.name = None
            tool_messages.append(tool_msg)

    # 统一拆分：先放 blocks 消息（如有），再放 tool_messages
    if tool_messages:
        return _merge_with_blocks(blocks, tool_messages, role)

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
        elif isinstance(block, IRRedactedThinkingBlock):
            content_blocks.append({
                "type": "redacted_thinking",
                "data": block.data,
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
            content_blocks.append(IRRedactedThinkingBlock(
                data=block.get("data", ""),
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
        elif isinstance(block, IRRedactedThinkingBlock):
            blocks.append({
                "type": "redacted_thinking",
                "data": block.data,
            })
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


# ════════════════════════════════════════════════════════════════════
# 流式（Streaming）
# ════════════════════════════════════════════════════════════════════


async def parse_stream_to_ir(
    resp,
    model: str,
) -> AsyncIterator[IRStreamEvent]:
    """Anthropic Messages SSE → IRStreamEvent 序列。

    状态机职责：
    - 跟踪 message_start 是否已发
    - 跟踪当前 block index/type（Anthropic 风格：每个 block 有 index）
    - 累积 tool_use 的 input_json_delta 片段
    - 收集 message_delta 中的 stop_reason / usage
    """
    message_started = False
    message_id = ""
    message_model = model

    # 当前活跃 block 状态
    # block_index → {"type": "thinking"|"text"|"tool_use", "tool_id": str, "args": IncrementalJSONParser}
    blocks: dict[int, dict] = {}

    pending_stop_reason: str | None = None
    latest_usage: dict | None = None

    async for raw_line in resp.aiter_lines():
        # Anthropic SSE 用 "event: X" + "data: Y" 配对
        # parse_sse_line 不会处理 event 头（只返回 None），所以需要先缓存 event 名
        # 这里简化：read full event pair
        line = raw_line.rstrip("\r")
        if not line or line.startswith(":"):
            continue

        if line.startswith("event: "):
            # 把 event name 累积下来，与下一个 data 配对
            # 简化处理：用一个小 list 缓存单 event 流
            # 但 aiter_lines 是 line-by-line，所以我们需要手动累积
            # 改成：直接解析 data: 行（event 名通常也编码在 data 里）
            # 为简化，这里使用一个延迟策略：等下一行
            current_event = line[7:].strip()
            continue

        if line.startswith("data: "):
            data_str = line[6:].lstrip(" ")
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
        else:
            continue

        event_type = event.get("type", "")

        if event_type == "message_start":
            msg = event.get("message", {})
            message_id = msg.get("id", message_id)
            message_model = msg.get("model", message_model)
            if not message_started:
                yield IRStreamEvent(
                    type="message_start",
                    data={"id": message_id, "model": message_model},
                )
                message_started = True

        elif event_type == "content_block_start":
            block_index = event.get("index", 0)
            block = event.get("content_block", {})
            block_type = block.get("type", "")
            if block_type == "thinking":
                blocks[block_index] = {"type": "thinking", "open": True}
                yield IRStreamEvent(type="thinking_start", data={"index": block_index})
            elif block_type == "text":
                blocks[block_index] = {"type": "text", "open": True}
                yield IRStreamEvent(type="text_start", data={"index": block_index})
            elif block_type == "tool_use":
                blocks[block_index] = {
                    "type": "tool_use",
                    "open": True,
                    "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": block.get("name", ""),
                    "args": IncrementalJSONParser(),
                }
                yield IRStreamEvent(
                    type="tool_use_start",
                    data={
                        "id": blocks[block_index]["id"],
                        "name": blocks[block_index]["name"],
                    },
                )
            elif block_type == "redacted_thinking":
                # 不可读取，直接 emit end 即可
                blocks[block_index] = {"type": "thinking", "open": False, "redacted": True}
                yield IRStreamEvent(type="thinking_start", data={"index": block_index, "redacted": True})
                yield IRStreamEvent(type="thinking_end", data={"index": block_index})

        elif event_type == "content_block_delta":
            block_index = event.get("index", 0)
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")
            block = blocks.get(block_index)
            if not block:
                continue
            if delta_type == "thinking_delta":
                thinking_text = delta.get("thinking", "")
                if thinking_text:
                    yield IRStreamEvent(
                        type="thinking_delta",
                        data={"thinking": thinking_text, "index": block_index},
                    )
            elif delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    yield IRStreamEvent(
                        type="text_delta",
                        data={"text": text, "index": block_index},
                    )
            elif delta_type == "input_json_delta":
                fragment = delta.get("partial_json", "")
                if fragment and block["type"] == "tool_use":
                    block["args"].feed(fragment)
                    yield IRStreamEvent(
                        type="tool_use_delta",
                        data={"id": block["id"], "arguments_delta": fragment},
                    )

        elif event_type == "content_block_stop":
            block_index = event.get("index", 0)
            block = blocks.get(block_index)
            if not block:
                continue
            if block["type"] == "thinking" and block.get("open"):
                yield IRStreamEvent(type="thinking_end", data={"index": block_index})
                block["open"] = False
            elif block["type"] == "text" and block.get("open"):
                yield IRStreamEvent(type="text_end", data={"index": block_index})
                block["open"] = False
            elif block["type"] == "tool_use" and block.get("open"):
                final_input = block["args"].finalize()
                yield IRStreamEvent(
                    type="tool_use_end",
                    data={"id": block["id"], "input": final_input},
                )
                block["open"] = False

        elif event_type == "message_delta":
            delta = event.get("delta", {})
            if "stop_reason" in delta:
                pending_stop_reason = delta["stop_reason"]
            usage_raw = event.get("usage")
            if isinstance(usage_raw, dict) and usage_raw:
                latest_usage = usage_raw

        elif event_type == "message_stop":
            # 闭合所有仍 open 的 block（Anthropic 异常情况下可能没发 content_block_stop）
            for idx, b in blocks.items():
                if not b.get("open"):
                    continue
                if b["type"] == "thinking":
                    yield IRStreamEvent(type="thinking_end", data={"index": idx})
                elif b["type"] == "text":
                    yield IRStreamEvent(type="text_end", data={"index": idx})
                elif b["type"] == "tool_use":
                    final_input = b["args"].finalize()
                    yield IRStreamEvent(
                        type="tool_use_end",
                        data={"id": b["id"], "input": final_input},
                    )
                b["open"] = False

    # ── 流末尾：emit usage + message_stop ──
    if latest_usage:
        yield IRStreamEvent(type="usage", data=extract_usage_tokens(latest_usage))

    yield IRStreamEvent(
        type="message_stop",
        data={"stop_reason": pending_stop_reason or "end_turn"},
    )


async def format_ir_as_sse(
    events: AsyncIterator[IRStreamEvent],
    model: str,
    *,
    reverse_tool_map: dict | None = None,
    namespace_map: dict | None = None,
) -> AsyncIterator[bytes]:
    """IRStreamEvent 序列 → Anthropic Messages SSE 字节流。

    Anthropic 流式结构：
    - message_start
    - content_block_start ×N（每块一个）
    - content_block_delta ×N
    - content_block_stop ×N
    - message_delta（含 stop_reason, usage）
    - message_stop

    状态：message_id, current_model, block_index 计数, 各 block 的内容累积。
    """
    message_id = ""
    current_model = model
    seq = 0

    # 用于重建 text/thinking/tool_use 内容
    current_block_index = 0
    current_block_type: str | None = None  # "text" / "thinking" / "tool_use"
    current_block_id: str = ""  # for tool_use
    current_block_name: str = ""
    current_block_json: list[str] = []  # for tool_use: 累积 input_json 片段
    current_text_buf: str = ""
    current_thinking_buf: str = ""

    pending_stop_reason: str | None = None
    pending_usage: dict | None = None

    def next_block() -> int:
        nonlocal current_block_index
        idx = current_block_index
        current_block_index += 1
        return idx

    def close_current_block():
        """闭合当前 block（如有 open），返回是否 emit 了 stop 事件。"""
        nonlocal current_block_type
        if current_block_type is None:
            return False
        idx = current_block_index - 1
        if current_block_type == "text":
            # text block stop（无需 content，因为 message 里已累加）
            return False  # 已在 text_delta 中自然结束
        elif current_block_type == "thinking":
            return False
        elif current_block_type == "tool_use":
            # tool_use 块：通过 tool_use_end 事件触发 stop
            return False
        current_block_type = None
        return False

    async for event in events:
        etype = event.type
        data = event.data or {}

        if etype == "message_start":
            message_id = data.get("id") or f"msg_{uuid.uuid4().hex[:24]}"
            current_model = data.get("model") or model
            yield sse_format("message_start", {
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": current_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })

        elif etype == "thinking_start":
            idx = next_block()
            current_block_type = "thinking"
            current_thinking_buf = ""
            yield sse_format("content_block_start", {
                "index": idx,
                "content_block": {"type": "thinking", "thinking": ""},
            })

        elif etype == "thinking_delta":
            text = data.get("thinking", "")
            if text:
                idx = current_block_index - 1
                current_thinking_buf += text
                yield sse_format("content_block_delta", {
                    "index": idx,
                    "delta": {"type": "thinking_delta", "thinking": text},
                })

        elif etype == "thinking_end":
            if current_block_type == "thinking":
                idx = current_block_index - 1
                yield sse_format("content_block_stop", {"index": idx})
                current_block_type = None

        elif etype == "text_start":
            idx = next_block()
            current_block_type = "text"
            current_text_buf = ""
            yield sse_format("content_block_start", {
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            })

        elif etype == "text_delta":
            text = data.get("text", "")
            if text:
                idx = current_block_index - 1
                current_text_buf += text
                yield sse_format("content_block_delta", {
                    "index": idx,
                    "delta": {"type": "text_delta", "text": text},
                })

        elif etype == "text_end":
            if current_block_type == "text":
                idx = current_block_index - 1
                yield sse_format("content_block_stop", {"index": idx})
                current_block_type = None

        elif etype == "tool_use_start":
            idx = next_block()
            current_block_type = "tool_use"
            current_block_id = data.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
            current_block_name = data.get("name", "")
            current_block_json = []
            yield sse_format("content_block_start", {
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": current_block_id,
                    "name": current_block_name,
                    "input": {},
                },
            })

        elif etype == "tool_use_delta":
            args_delta = data.get("arguments_delta", "")
            if args_delta:
                idx = current_block_index - 1
                current_block_json.append(args_delta)
                yield sse_format("content_block_delta", {
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": args_delta},
                })

        elif etype == "tool_use_end":
            if current_block_type == "tool_use":
                idx = current_block_index - 1
                yield sse_format("content_block_stop", {"index": idx})
                current_block_type = None

        elif etype == "usage":
            pending_usage = extract_usage_tokens(data)

        elif etype == "message_stop":
            pending_stop_reason = data.get("stop_reason", "end_turn")
            # 发 message_delta（含 stop_reason + usage）
            delta_payload: dict[str, Any] = {
                "delta": {
                    "stop_reason": pending_stop_reason,
                    "stop_sequence": None,
                },
                "usage": pending_usage or {"input_tokens": 0, "output_tokens": 0},
            }
            yield sse_format("message_delta", delta_payload)
            yield sse_format("message_stop", {"type": "message_stop"})

        elif etype == "error":
            err = data or {}
            yield sse_format("error", {
                "type": "error",
                "error": {
                    "type": err.get("code", "api_error"),
                    "message": err.get("message", "Stream error"),
                },
            })

        elif etype == "keepalive":
            yield b": keepalive\n\n"
