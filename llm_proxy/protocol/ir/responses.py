"""OpenAI Responses API ↔ IR 转换器。

透传模式：apply_patch 不再展开为 4 个标准文件工具，直接降级为 function tool。
"""

from __future__ import annotations

import json
import logging
import uuid
import time
from typing import Any, AsyncIterator

from llm_proxy.protocol.ir._common import (
    build_usage,
    clean_schema,
    is_openai_o_series,
    safe_json_dumps,
    safe_json_loads,
    supports_reasoning_effort,
)
from llm_proxy.protocol.ir._stream import (
    DONE_SENTINEL,
    extract_usage_tokens,
    map_finish_to_stop_reason,
    map_stop_to_finish_reason,
    map_stop_to_responses_status,
    parse_sse_line,
    sse_format,
)
from llm_proxy.protocol.responses_chat.tool_replacement import repair_apply_patch_dsl, APPLY_PATCH_TOOL_DESCRIPTION
from llm_proxy.protocol.ir.types import (
    IRStreamEvent,
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

logger = logging.getLogger(__name__)


# ── 请求：Responses → IR ──────────────────────────────────────────


def to_ir(body: dict[str, Any]) -> IRRequest:
    """OpenAI Responses API 请求体 → IRRequest。"""
    model = body.get("model", "")

    # instructions → system_prompt
    instructions = body.get("instructions")
    input_data = body.get("input", [])

    messages: list[IRMessage] = []
    if isinstance(input_data, str):
        # 短输入：单条 user 消息
        messages.append(IRMessage(role="user", content=input_data))
    elif isinstance(input_data, list):
        messages = _convert_input_to_ir_messages(input_data)

    ir_request = IRRequest(
        model=model,
        messages=messages,
        system_prompt=instructions if isinstance(instructions, str) else None,
    )

    # 参数映射
    if "max_output_tokens" in body:
        ir_request.max_tokens = int(body["max_output_tokens"])
    elif "max_tokens" in body:
        ir_request.max_tokens = int(body["max_tokens"])

    if "temperature" in body:
        ir_request.temperature = body["temperature"]
    if "top_p" in body:
        ir_request.top_p = body["top_p"]
    if "stream" in body:
        ir_request.stream = bool(body["stream"])

    if "stream_options" in body:
        ir_request.extensions["stream_options"] = body["stream_options"]

    # reasoning.effort → reasoning_effort
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort:
            effort_map = {
                "none": "none",
                "auto": "auto",
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
            }
            ir_request.reasoning_effort = effort_map.get(effort, "auto")

    # tools — 处理 apply_patch / namespace / web_search 等
    tools = body.get("tools") or []
    if tools:
        ir_tools, reverse_tool_map, tool_spec_map = _convert_tools_to_ir(tools)
        if ir_tools:
            ir_request.tools = ir_tools
        if reverse_tool_map:
            ir_request.extensions["reverse_tool_map"] = reverse_tool_map
        if tool_spec_map:
            ir_request.extensions["tool_spec_map"] = tool_spec_map

    if "tool_choice" in body:
        ir_request.tool_choice = body["tool_choice"]

    # 透传 Responses 特有字段
    for key in ("parallel_tool_calls", "truncation", "store", "user"):
        if key in body:
            ir_request.extensions[key] = body[key]

    return ir_request


def _convert_input_to_ir_messages(input_data: list) -> list[IRMessage]:
    """Responses input 数组 → IRMessage 列表。"""
    messages: list[IRMessage] = []

    # 收集同一 assistant turn 的 reasoning / tool_calls 一起拼
    pending_reasoning: list[str] = []
    pending_tool_calls: list[IRToolUseBlock] = []
    pending_assistant_text: str | None = None

    def flush_assistant():
        nonlocal pending_assistant_text
        blocks: list[IRContentBlock] = []
        if pending_reasoning:
            blocks.append(IRThinkingBlock(thinking="\n".join(pending_reasoning)))
            pending_reasoning.clear()
        if pending_assistant_text is not None:
            blocks.append(IRTextBlock(text=pending_assistant_text))
            pending_assistant_text = None
        elif pending_tool_calls:
            pass  # content 留空（tool calls）
        for tc in pending_tool_calls:
            blocks.append(tc)
        if blocks:
            messages.append(IRMessage(role="assistant", content=blocks))
        pending_tool_calls.clear()

    for item in input_data:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")

        if item_type == "message":
            role = item.get("role", "user")
            content = item.get("content")
            text = _extract_message_text(content)
            if text is not None:
                if role == "assistant":
                    if pending_assistant_text is None:
                        pending_assistant_text = ""
                    pending_assistant_text += text
                else:
                    flush_assistant()
                    messages.append(IRMessage(role=role, content=text))

        elif item_type == "reasoning":
            # 提取 reasoning 文本（Responses 通常在 summary[] 或 encrypted_content）
            text = _extract_reasoning_text(item)
            if text:
                pending_reasoning.append(text)

        elif item_type == "function_call":
            flush_assistant()
            call_id = item.get("call_id") or item.get("id", "")
            name = item.get("name", "")
            arguments = safe_json_loads(item.get("arguments", "{}"), default={})
            pending_tool_calls.append(IRToolUseBlock(
                id=call_id,
                name=name,
                input=arguments if isinstance(arguments, dict) else {},
            ))

        elif item_type == "function_call_output":
            flush_assistant()
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            if not isinstance(output, str):
                output = safe_json_dumps(output, default="")
            messages.append(IRMessage(role="tool", content=[
                IRToolResultBlock(
                    tool_use_id=call_id,
                    content=output,
                )
            ]))

        else:
            logger.debug(f"Skipping unknown input item type: {item_type}")

    flush_assistant()
    return messages


def _extract_message_text(content) -> str | None:
    """从 Responses message.content 数组中提取文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type", "")
        if ptype in ("input_text", "output_text"):
            text = p.get("text", "")
            if text:
                parts.append(text)
        elif ptype == "input_image":
            url = p.get("image_url", "")
            if url:
                parts.append(f"[image: {url}]")
        # 其他类型（refusal 等）忽略
    return "\n".join(parts) if parts else None


def _extract_reasoning_text(item: dict) -> str:
    """从 Responses reasoning item 提取文本。"""
    summary = item.get("summary", [])
    if isinstance(summary, list):
        for s in summary:
            if isinstance(s, dict) and s.get("type") == "summary_text":
                return s.get("text", "")
    return ""


def _convert_tools_to_ir(tools: list) -> tuple[list[IRToolDef], dict[str, str], dict[str, dict]]:
    """Responses tools → (IRToolDef list, reverse_tool_map, tool_spec_map)。

    透传模式：apply_patch 与其他 custom 工具一致，直接降级为 function tool。
    tool_spec_map 的 value 为 {"kind": "...", "name": "...", "namespace": "..."}。
    """
    ir_tools: list[IRToolDef] = []
    reverse_tool_map: dict[str, str] = {}
    tool_spec_map: dict[str, dict] = {}

    # 客户端侧工具：跳过
    CLIENT_SIDE_TOOLS = {"tool_search", "web_search"}

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "")

        if tool_type in CLIENT_SIDE_TOOLS:
            logger.debug(f"Skipping client-side tool: {tool_type}")
            continue

        if tool_type == "custom":
            name = tool.get("name", "")
            # 透传：apply_patch 与其他 custom 工具一致，直接降级为单个 function tool
            params = _normalize_params(tool.get("parameters"))
            if not params.get("properties"):
                # apply_patch 用完整 DSL 描述让上游模型理解格式，其他工具用简短描述
                input_desc = APPLY_PATCH_TOOL_DESCRIPTION if name == "apply_patch" else "Tool input"
                params = {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": input_desc}},
                    "required": [],
                }
            ir_tools.append(IRToolDef(
                name=name,
                description=tool.get("description") or (APPLY_PATCH_TOOL_DESCRIPTION if name == "apply_patch" else ""),
                parameters=clean_schema(params),
            ))
            reverse_tool_map[name] = name
            tool_spec_map[name] = {"kind": "custom", "name": name}
            continue

        if tool_type == "namespace":
            ns_name = tool.get("name", "")
            for sub in tool.get("tools") or []:
                if not isinstance(sub, dict) or sub.get("type") != "function":
                    continue
                sub_name = sub.get("name", "")
                chat_name = f"{ns_name}__{sub_name}"
                ir_tools.append(IRToolDef(
                    name=chat_name,
                    description=sub.get("description", ""),
                    parameters=clean_schema(_normalize_params(sub.get("parameters"))),
                ))
                tool_spec_map[chat_name] = {
                    "kind": "namespace", "name": sub_name, "namespace": ns_name,
                }
            continue

        if tool_type == "web_search":
            # 降级为 function with query 参数
            ir_tools.append(IRToolDef(
                name="web_search",
                description="Search the web for a query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string.",
                        },
                    },
                    "required": ["query"],
                },
            ))
            continue

        # function 类型
        if tool_type == "function":
            name = tool.get("name", "")
            ir_tools.append(IRToolDef(
                name=name,
                description=tool.get("description", ""),
                parameters=clean_schema(_normalize_params(tool.get("parameters"))),
            ))
            continue

        # 未知类型：尝试降级为 function
        name = tool.get("name") or tool_type
        ir_tools.append(IRToolDef(
            name=name,
            description=tool.get("description", f"Tool of type {tool_type}"),
            parameters=clean_schema(_normalize_params(tool.get("parameters"))),
        ))

    return ir_tools, reverse_tool_map, tool_spec_map


def _normalize_params(params) -> dict:
    """补全缺失的 JSON schema 字段。"""
    if not isinstance(params, dict):
        params = {}
    if not params.get("type"):
        params["type"] = "object"
    params.setdefault("properties", {})
    params.setdefault("required", [])
    return params


# ── 响应：IR → Responses ──────────────────────────────────────────


def response_from_ir(ir: IRResponse, *, reverse_tool_map: dict | None = None, tool_spec_map: dict | None = None) -> dict[str, Any]:
    """IRResponse → OpenAI Responses API 响应体。"""
    output: list[dict] = []
    _reverse = reverse_tool_map or {}
    _spec = tool_spec_map or {}
    # 收集所有 text blocks 合并到一个 message item（Responses API 规范：
    # 同一 response 的 output 中，assistant role 的 message 应只有一个）
    text_parts: list[dict] = []
    for block in ir.content_blocks:
        if isinstance(block, IRTextBlock):
            if block.text:
                text_parts.append({"type": "output_text", "text": block.text, "annotations": []})
        elif isinstance(block, IRThinkingBlock):
            # thinking 必须先于 text 输出，先 flush text parts
            if text_parts:
                output.append({
                    "type": "message",
                    "role": "assistant",
                    "content": text_parts,
                })
                text_parts = []
            if block.thinking:
                output.append({
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "summary": [{"type": "summary_text", "text": block.thinking}],
                })
        elif isinstance(block, IRToolUseBlock):
            # tool_use 也需要先 flush text parts
            if text_parts:
                output.append({
                    "type": "message",
                    "role": "assistant",
                    "content": text_parts,
                })
                text_parts = []

            downstream_name = _reverse.get(block.name)
            spec = _spec.get(block.name)

            if downstream_name is not None:
                # 1) reverse_tool_map 命中 → custom_tool_call
                args_raw = safe_json_dumps(block.input, default="{}")
                if block.name == "apply_patch" and isinstance(block.input, dict):
                    dsl = block.input.get("input", "")
                    if isinstance(dsl, str):
                        args_raw = repair_apply_patch_dsl(dsl).dsl
                output.append({
                    "type": "custom_tool_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "name": downstream_name,
                    "status": "completed",
                    "call_id": block.id or f"call_{uuid.uuid4().hex[:24]}",
                    "input": args_raw,
                })
            elif spec is not None and spec.get("kind") == "namespace" and spec.get("namespace"):
                # 2) namespace 子工具 → function_call + 原始名称 + namespace
                output.append({
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "name": spec["name"],
                    "namespace": spec["namespace"],
                    "arguments": safe_json_dumps(block.input, default="{}"),
                    "status": "completed",
                    "call_id": block.id or f"call_{uuid.uuid4().hex[:24]}",
                })
            else:
                # 3) 普通工具 → function_call
                output.append({
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": block.id or f"call_{uuid.uuid4().hex[:24]}",
                    "name": block.name,
                    "arguments": safe_json_dumps(block.input, default="{}"),
                    "status": "completed",
                })

    # flush 剩余的 text parts
    if text_parts:
        output.append({
            "type": "message",
            "role": "assistant",
            "content": text_parts,
        })

    # status
    status = "completed"
    incomplete_reason = None
    if ir.stop_reason == "max_tokens":
        status = "incomplete"
        incomplete_reason = "max_output_tokens"
    elif ir.stop_reason == "tool_use":
        status = "completed"  # 工具调用后正常完成

    usage = ir.usage or {}
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    cached = int(usage.get("cache_read_input_tokens", 0))

    responses_usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cached:
        responses_usage["input_tokens_details"] = {"cached_tokens": cached}

    result: dict[str, Any] = {
        "id": ir.id or f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created": int(time.time()),
        "model": ir.model,
        "status": status,
        "output": output,
        "usage": responses_usage,
    }
    if incomplete_reason:
        result["incomplete_details"] = {"reason": incomplete_reason}

    return result


def response_to_ir(body: dict[str, Any]) -> IRResponse:
    """OpenAI Responses API 响应体 → IRResponse。"""
    blocks: list[IRContentBlock] = []

    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")

        if item_type == "message":
            content = item.get("content", [])
            for part in (content if isinstance(content, list) else []):
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "output_text":
                    if part.get("text"):
                        blocks.append(IRTextBlock(text=part["text"]))
                elif ptype == "refusal":
                    if part.get("refusal"):
                        blocks.append(IRTextBlock(text=part["refusal"]))

        elif item_type == "function_call":
            arguments = safe_json_loads(item.get("arguments", "{}"), default={})
            blocks.append(IRToolUseBlock(
                id=item.get("call_id") or item.get("id", ""),
                name=item.get("name", ""),
                input=arguments if isinstance(arguments, dict) else {},
            ))

        elif item_type == "reasoning":
            text = _extract_reasoning_text(item)
            if text:
                blocks.append(IRThinkingBlock(thinking=text))

    if not blocks:
        blocks.append(IRTextBlock(text=""))

    # status → stop_reason
    status = body.get("status", "completed")
    incomplete_details = body.get("incomplete_details", {}) or {}
    if status == "incomplete":
        reason = incomplete_details.get("reason", "max_output_tokens")
        if reason == "max_output_tokens":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"
    else:
        # 检查是否包含 tool_use
        has_tool = any(isinstance(b, IRToolUseBlock) for b in blocks)
        stop_reason = "tool_use" if has_tool else "end_turn"

    return IRResponse(
        id=body.get("id", ""),
        model=body.get("model", ""),
        content_blocks=blocks,
        stop_reason=stop_reason,
        usage=build_usage(body.get("usage", {})),
    )


# ── 请求出：IR → Responses ──────────────────────────────────────


def to_upstream(ir: IRRequest, upstream_model: str | None = None) -> dict[str, Any]:
    """IRRequest → OpenAI Responses API 请求体。"""
    result: dict[str, Any] = {"model": upstream_model or ir.model}

    # instructions
    if ir.system_prompt:
        result["instructions"] = ir.system_prompt
    elif "instructions" in ir.extensions:
        result["instructions"] = ir.extensions["instructions"]

    # messages → input
    input_items = _messages_ir_to_responses_input(ir.messages)
    if len(input_items) == 1 and input_items[0].get("type") == "message" and input_items[0].get("role") == "user":
        # 短输入优化：单条 user 消息
        content = input_items[0].get("content", "")
        if isinstance(content, str):
            result["input"] = content
        else:
            result["input"] = input_items
    else:
        result["input"] = input_items

    # 参数
    if ir.max_tokens is not None:
        result["max_output_tokens"] = int(ir.max_tokens)

    if ir.temperature is not None:
        result["temperature"] = ir.temperature
    if ir.top_p is not None:
        result["top_p"] = ir.top_p
    if ir.stream:
        result["stream"] = True
    if "stream_options" in ir.extensions:
        result["stream_options"] = ir.extensions["stream_options"]

    if ir.reasoning_effort is not None:
        result["reasoning"] = {"effort": ir.reasoning_effort}

    # tools
    if ir.tools:
        result["tools"] = _tools_ir_to_responses(ir.tools, ir.extensions.get("reverse_tool_map"))

    if ir.tool_choice is not None:
        result["tool_choice"] = ir.tool_choice

    # 透传 Responses 特有字段
    for key in ("parallel_tool_calls", "truncation", "store", "user"):
        if key in ir.extensions:
            result[key] = ir.extensions[key]

    return result


def _messages_ir_to_responses_input(messages: list[IRMessage]) -> list[dict]:
    """IRMessage 列表 → Responses input 数组。"""
    items: list[dict] = []
    for msg in messages:
        role = msg.role
        content = msg.content

        if role == "system":
            # system 消息在 Responses 中应转为 instructions（在 to_upstream 顶部处理）
            # 这里如果出现独立 system message，转换为 user role（不理想但兼容）
            text = content if isinstance(content, str) else _extract_text(content)
            if text:
                items.append({"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": text}
                ]})
            continue

        if role == "tool":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, IRToolResultBlock):
                        items.append({
                            "type": "function_call_output",
                            "call_id": block.tool_use_id,
                            "output": block.content,
                        })
            continue

        if role == "assistant":
            if isinstance(content, list):
                # 可能有 thinking + text + tool_use
                for block in content:
                    if isinstance(block, IRThinkingBlock):
                        if block.thinking:
                            items.append({
                                "type": "reasoning",
                                "summary": [{"type": "summary_text", "text": block.thinking}],
                            })
                    elif isinstance(block, IRTextBlock):
                        if block.text:
                            items.append({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": block.text}],
                            })
                    elif isinstance(block, IRToolUseBlock):
                        items.append({
                            "type": "function_call",
                            "id": block.id,
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": safe_json_dumps(block.input, default="{}"),
                        })
            elif isinstance(content, str) and content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
            continue

        # user (default)
        if isinstance(content, str):
            items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            })
        elif isinstance(content, list):
            resp_content: list[dict] = []
            for block in content:
                if isinstance(block, IRTextBlock):
                    if block.text:
                        resp_content.append({"type": "input_text", "text": block.text})
                elif isinstance(block, IRImageBlock):
                    resp_content.append({
                        "type": "input_image",
                        "image_url": f"data:{block.media_type};base64,{block.base64_data}",
                    })
            if resp_content:
                items.append({"type": "message", "role": "user", "content": resp_content})

    return items


def _extract_text(content) -> str:
    """从 IRMessage content 中提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, IRTextBlock):
                parts.append(block.text)
        return "\n".join(parts)
    return ""


def _tools_ir_to_responses(
    ir_tools: list[IRToolDef], reverse_tool_map: dict[str, str] | None
) -> list[dict]:
    """IRToolDef list → Responses tools 列表。

    透传模式：每个 IRToolDef 直接还原为 Responses function tool。
    """
    if not ir_tools:
        return []

    result: list[dict] = []
    for tool in ir_tools:
        result.append({
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        })
    return result





# ════════════════════════════════════════════════════════════════════
# 流式（Streaming）
# ════════════════════════════════════════════════════════════════════


async def parse_stream_to_ir(
    resp,
    model: str,
) -> AsyncIterator[IRStreamEvent]:
    """OpenAI Responses API SSE → IRStreamEvent 序列。

    状态机职责：
    - 跟踪 message_start 是否已发
    - 跟踪 reasoning / text / function_call item 的开闭
    - 累积 function_call 的 arguments_delta 片段
    - 收集 response.completed 中的 usage
    """
    message_started = False
    response_id = ""
    current_model = model

    # item_id → {"type": "reasoning"|"message"|"function_call", "open": bool, "args": str}
    items: dict[str, dict] = {}

    latest_usage: dict | None = None
    pending_stop_reason: str | None = None

    async for raw_line in resp.aiter_lines():
        line = raw_line.rstrip("\r")
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data: "):
            continue
        data_str = line[6:].lstrip(" ")
        if data_str == "[DONE]":
            break
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type", "")

        # Usage 提取（多个事件可能携带 usage）
        if isinstance(event.get("usage"), dict) and event["usage"]:
            latest_usage = event["usage"]

        if event_type == "response.created":
            resp_obj = event.get("response", {})
            response_id = resp_obj.get("id", response_id)
            current_model = resp_obj.get("model", current_model)
            if not message_started:
                yield IRStreamEvent(
                    type="message_start",
                    data={"id": response_id, "model": current_model},
                )
                message_started = True

        elif event_type == "response.in_progress":
            # 中间进度事件，忽略
            if not message_started and event.get("response"):
                resp_obj = event["response"]
                response_id = resp_obj.get("id", response_id)
                current_model = resp_obj.get("model", current_model)
                yield IRStreamEvent(
                    type="message_start",
                    data={"id": response_id, "model": current_model},
                )
                message_started = True

        elif event_type == "response.output_item.added":
            item = event.get("item", {})
            item_id = item.get("id", "")
            item_type = item.get("type", "")
            output_index = event.get("output_index", 0)
            if item_type == "reasoning":
                items[item_id] = {"type": "reasoning", "open": True, "output_index": output_index}
                yield IRStreamEvent(type="thinking_start", data={"id": item_id, "index": output_index})
            elif item_type == "message":
                items[item_id] = {"type": "message", "open": True, "output_index": output_index}
                yield IRStreamEvent(type="text_start", data={"id": item_id, "index": output_index})
            elif item_type == "function_call":
                items[item_id] = {
                    "type": "function_call",
                    "open": True,
                    "output_index": output_index,
                    "call_id": item.get("call_id", item_id),
                    "name": item.get("name", ""),
                    "args": "",
                }
                yield IRStreamEvent(
                    type="tool_use_start",
                    data={
                        "id": items[item_id]["call_id"],
                        "name": items[item_id]["name"],
                    },
                )

        elif event_type == "response.reasoning_summary_text.delta":
            text = event.get("delta", "")
            if text:
                item_id = event.get("item_id", "")
                yield IRStreamEvent(
                    type="thinking_delta",
                    data={"thinking": text, "id": item_id},
                )

        elif event_type == "response.output_text.delta":
            text = event.get("delta", "")
            if text:
                item_id = event.get("item_id", "")
                yield IRStreamEvent(
                    type="text_delta",
                    data={"text": text, "id": item_id},
                )

        elif event_type == "response.function_call_arguments.delta":
            args_delta = event.get("delta", "")
            if args_delta:
                item_id = event.get("item_id", "")
                item = items.get(item_id)
                if item and item["type"] == "function_call":
                    item["args"] += args_delta
                    yield IRStreamEvent(
                        type="tool_use_delta",
                        data={
                            "id": item["call_id"],
                            "arguments_delta": args_delta,
                        },
                    )

        elif event_type == "response.reasoning_summary_text.done":
            pass  # 在 output_item.done 中关闭

        elif event_type == "response.output_item.done":
            item = event.get("item", {})
            item_id = item.get("id", "")
            item_type = item.get("type", "")
            tracked = items.get(item_id)
            if not tracked or not tracked.get("open"):
                continue
            if item_type == "reasoning":
                yield IRStreamEvent(type="thinking_end", data={"id": item_id})
                tracked["open"] = False
            elif item_type == "message":
                yield IRStreamEvent(type="text_end", data={"id": item_id})
                tracked["open"] = False
            elif item_type == "function_call":
                args_raw = item.get("arguments", tracked.get("args", ""))
                final_input = safe_json_loads(args_raw, default={})
                if not isinstance(final_input, dict):
                    final_input = {"_raw": args_raw}
                yield IRStreamEvent(
                    type="tool_use_end",
                    data={"id": tracked["call_id"], "input": final_input},
                )
                tracked["open"] = False

        elif event_type in ("response.completed", "response.done"):
            resp_obj = event.get("response", {})
            if isinstance(resp_obj, dict):
                status = resp_obj.get("status", "completed")
                if status == "incomplete":
                    pending_stop_reason = "max_tokens"
                # 检查 output 中是否有 tool_call
                if not pending_stop_reason:
                    has_tool = any(
                        i.get("type") == "function_call" for i in resp_obj.get("output", [])
                    )
                    pending_stop_reason = "tool_use" if has_tool else "end_turn"
                # usage（response.completed 通常携带完整 usage）
                resp_usage = resp_obj.get("usage")
                if isinstance(resp_usage, dict) and resp_usage:
                    latest_usage = resp_usage
            if not pending_stop_reason:
                pending_stop_reason = "end_turn"
            break

        elif event_type == "response.error":
            err = event.get("error", {})
            yield IRStreamEvent(
                type="error",
                data={"message": err.get("message", "Upstream error"), "code": err.get("code", "api_error")},
            )
            return

    # 流末尾：闭合所有未关闭 item
    for item_id, item in items.items():
        if not item.get("open"):
            continue
        if item["type"] == "reasoning":
            yield IRStreamEvent(type="thinking_end", data={"id": item_id})
        elif item["type"] == "message":
            yield IRStreamEvent(type="text_end", data={"id": item_id})
        elif item["type"] == "function_call":
            args_raw = item.get("args", "")
            final_input = safe_json_loads(args_raw, default={"_raw": args_raw})
            if not isinstance(final_input, dict):
                final_input = {"_raw": args_raw}
            yield IRStreamEvent(
                type="tool_use_end",
                data={"id": item["call_id"], "input": final_input},
            )

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
    tool_spec_map: dict | None = None,
) -> AsyncIterator[bytes]:
    """IRStreamEvent 序列 → OpenAI Responses API SSE 字节流。

    Responses 流式结构（每条 message/reasoning/function_call item 拆为多事件）：
    - response.created
    - response.output_item.added (reasoning / message / function_call)
    - response.reasoning_summary_part.added（仅 reasoning）
    - response.reasoning_summary_text.delta ×N（仅 reasoning）
    - response.reasoning_summary_text.done（仅 reasoning）
    - response.reasoning_summary_part.done（仅 reasoning）
    - response.content_part.added（仅 message）
    - response.output_text.delta ×N（仅 message）
    - response.output_text.done（仅 message）
    - response.content_part.done（仅 message）
    - response.function_call_arguments.delta ×N（仅 function_call, 不含 reverse）
    - response.function_call_arguments.done（仅 function_call, 不含 reverse）
    - response.custom_tool_call_input.delta（reverse_tool_map 命中的）
    - response.output_item.done ×N
    - response.completed（含 status, output 数组, usage）

    reverse_tool_map 命中时（透传模式）：
    - tool_use_end 的 input 原样作为 custom_tool_call.input
    - 走 custom_tool_call 事件序列（不是 function_call）
    tool_spec_map 命中且子工具时：
    - 普通 function_call，但使用原始 sub-tool 名，附加 namespace 字段
    """
    response_id = ""
    seq = 0
    created_ts = int(time.time())
    reverse_tool_map = reverse_tool_map or {}
    tool_spec_map = tool_spec_map or {}

    # 当前 item 状态
    current_item_type: str | None = None  # "reasoning" / "message" / "function_call" / "custom_tool_call"
    current_item_id: str = ""
    current_output_index: int = 0
    current_text_index: int = 0
    current_reasoning_index: int = 0
    reasoning_buf: str = ""
    text_buf: str = ""
    tool_args_buf: dict = {}  # 累积所有 tool call 的 input（仅用于 reverse 时拼装）
    current_tool_call_id: str = ""
    current_tool_name: str = ""
    current_tool_downstream_name: str = ""  # reverse_tool_map[name] 后的名
    current_tool_spec: dict = {}  # tool_spec_map[name] 的 spec 副本

    has_reasoning = False
    has_text = False
    has_function_calls = False
    function_call_count = 0  # 每个 function_call 独立 output_index

    def next_seq() -> int:
        nonlocal seq
        s = seq
        seq += 1
        return s

    def item_output_index() -> int:
        # output_index = 已闭合的 item 数量（即 all_output_items 的当前长度）
        return len(all_output_items)

    def new_item_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:24]}"

    def close_current_item():
        """闭合当前活跃 item 的所有 part，返回 yield 的事件列表。

        返回 list[bytes] 而不是 yield，方便外层 async for 迭代。
        """
        nonlocal current_item_type, current_item_id, reasoning_buf, text_buf
        nonlocal current_tool_call_id, current_tool_name
        nonlocal current_tool_downstream_name, current_tool_spec
        events_out: list[bytes] = []
        if current_item_type is None:
            return events_out
        output_index = item_output_index()

        if current_item_type == "reasoning":
            # reasoning_summary_text.done
            events_out.append(sse_format("response.reasoning_summary_text.done", {
                "item_id": current_item_id,
                "output_index": output_index,
                "summary_index": 0,
                "sequence_number": next_seq(),
                "delta": reasoning_buf,
            }))
            events_out.append(sse_format("response.reasoning_summary_part.done", {
                "item_id": current_item_id,
                "output_index": output_index,
                "summary_index": 0,
                "sequence_number": next_seq(),
                "part": {"type": "summary_text", "text": reasoning_buf},
            }))
            events_out.append(sse_format("response.output_item.done", {
                "output_index": output_index,
                "sequence_number": next_seq(),
                "item": {
                    "id": current_item_id,
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": reasoning_buf}],
                },
            }))
            # 保存到 all_output_items（必须在清空 buf 前完成）
            all_output_items.append({
                "id": current_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning_buf}],
            })
            reasoning_buf = ""
        elif current_item_type == "message":
            events_out.append(sse_format("response.output_text.done", {
                "item_id": current_item_id,
                "output_index": output_index,
                "content_index": 0,
                "sequence_number": next_seq(),
                "delta": text_buf,
            }))
            events_out.append(sse_format("response.content_part.done", {
                "item_id": current_item_id,
                "output_index": output_index,
                "content_index": 0,
                "sequence_number": next_seq(),
                "part": {"type": "output_text", "text": text_buf},
            }))
            events_out.append(sse_format("response.output_item.done", {
                "output_index": output_index,
                "sequence_number": next_seq(),
                "item": {
                    "id": current_item_id,
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text_buf}],
                    "role": "assistant",
                },
            }))
            # 保存到 all_output_items（必须在清空 buf 前完成）
            all_output_items.append({
                "id": current_item_id,
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text_buf}],
                "role": "assistant",
            })
            text_buf = ""  # 立即清空，避免下一轮累积
        elif current_item_type == "function_call":
            # 普通 function_call
            args_raw = tool_args_buf.get(current_tool_call_id, "")
            events_out.append(sse_format("response.function_call_arguments.done", {
                "item_id": current_item_id,
                "output_index": output_index,
                "sequence_number": next_seq(),
                "arguments": args_raw,
            }))
            item_payload: dict[str, Any] = {
                "id": current_item_id,
                "type": "function_call",
                "status": "completed",
                "arguments": args_raw,
                "call_id": current_tool_call_id,
                "name": current_tool_name,
            }
            if current_tool_spec.get("kind") == "namespace" and current_tool_spec.get("namespace"):
                item_payload["name"] = current_tool_spec["name"]
                item_payload["namespace"] = current_tool_spec["namespace"]
            events_out.append(sse_format("response.output_item.done", {
                "output_index": output_index,
                "sequence_number": next_seq(),
                "item": item_payload,
            }))
            all_output_items.append(item_payload)
        elif current_item_type == "custom_tool_call":
            # custom_tool_call（apply_patch 反向）
            args_raw = tool_args_buf.get(current_tool_call_id, "")
            ctc_payload: dict[str, Any] = {
                "id": current_item_id,
                "type": "custom_tool_call",
                "name": current_tool_downstream_name,
                "status": "completed",
                "call_id": current_tool_call_id,
                "input": args_raw,
            }
            events_out.append(sse_format("response.output_item.done", {
                "output_index": output_index,
                "sequence_number": next_seq(),
                "item": ctc_payload,
            }))
            all_output_items.append(ctc_payload)

        current_item_type = None
        current_item_id = ""
        current_tool_call_id = ""
        current_tool_name = ""
        current_tool_downstream_name = ""
        current_tool_spec = {}
        return events_out

    # ── 起始：response.created ──
    yield sse_format("response.created", {
        "sequence_number": next_seq(),
        "response": {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "created": created_ts,
            "model": model,
            "status": "in_progress",
            "output": [],
        },
    })

    # 累计所有输出 item
    all_output_items: list[dict] = []
    pending_stop_reason: str = "end_turn"
    pending_usage: dict | None = None
    last_text_id: str = ""
    last_reasoning_id: str = ""
    last_function_call_id: str = ""

    async for event in events:
        etype = event.type
        data = event.data or {}

        if etype == "message_start":
            response_id = data.get("id") or response_id
            # Anthropic model → Responses model
            if data.get("model"):
                yield sse_format("response.in_progress", {
                    "sequence_number": next_seq(),
                    "response": {
                        "id": response_id,
                        "model": data["model"],
                        "status": "in_progress",
                        "output": [],
                    },
                })
            else:
                # 已在 response.created 中发过 model，跳过
                pass

        elif etype == "thinking_start":
            # 闭合前一个 item
            for ev in close_current_item():
                yield ev
            current_item_type = "reasoning"
            current_item_id = new_item_id("rs")
            last_reasoning_id = current_item_id
            has_reasoning = True
            reasoning_buf = ""
            output_index = item_output_index()
            yield sse_format("response.output_item.added", {
                "output_index": output_index,
                "sequence_number": next_seq(),
                "item": {
                    "id": current_item_id,
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                },
            })
            yield sse_format("response.reasoning_summary_part.added", {
                "item_id": current_item_id,
                "output_index": output_index,
                "summary_index": 0,
                "sequence_number": next_seq(),
                "part": {"type": "summary_text", "text": ""},
            })

        elif etype == "thinking_delta":
            text = data.get("thinking", "")
            if text and current_item_type == "reasoning":
                reasoning_buf += text
                output_index = item_output_index()
                yield sse_format("response.reasoning_summary_text.delta", {
                    "item_id": current_item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "sequence_number": next_seq(),
                    "delta": text,
                })

        elif etype == "thinking_end":
            if current_item_type == "reasoning":
                for ev in close_current_item():
                    yield ev

        elif etype == "text_start":
            for ev in close_current_item():
                yield ev
            current_item_type = "message"
            current_item_id = new_item_id("msg")
            last_text_id = current_item_id
            has_text = True
            text_buf = ""
            output_index = item_output_index()
            yield sse_format("response.output_item.added", {
                "output_index": output_index,
                "sequence_number": next_seq(),
                "item": {
                    "id": current_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "content": [],
                    "role": "assistant",
                },
            })
            yield sse_format("response.content_part.added", {
                "item_id": current_item_id,
                "output_index": output_index,
                "content_index": 0,
                "sequence_number": next_seq(),
                "part": {"type": "output_text", "text": ""},
            })

        elif etype == "text_delta":
            text = data.get("text", "")
            if text and current_item_type == "message":
                text_buf += text
                output_index = item_output_index()
                yield sse_format("response.output_text.delta", {
                    "item_id": current_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "sequence_number": next_seq(),
                    "delta": text,
                })

        elif etype == "text_end":
            if current_item_type == "message":
                for ev in close_current_item():
                    yield ev

        elif etype == "tool_use_start":
            for ev in close_current_item():
                yield ev

            upstream_name = data.get("name", "")
            tool_call_id = data.get("id") or f"call_{uuid.uuid4().hex[:24]}"
            current_item_id = new_item_id("fc")
            last_function_call_id = current_item_id
            current_tool_call_id = tool_call_id
            current_tool_name = upstream_name
            tool_args_buf[tool_call_id] = ""
            has_function_calls = True

            # 决定是 custom_tool_call 还是 function_call
            downstream_name = reverse_tool_map.get(upstream_name)
            spec = tool_spec_map.get(upstream_name, {})
            current_tool_spec = spec

            if downstream_name is not None:
                # apply_patch 反向：custom_tool_call（参数一次性 emit）
                current_item_type = "custom_tool_call"
                current_tool_downstream_name = downstream_name
                output_index = item_output_index()
                yield sse_format("response.output_item.added", {
                    "output_index": output_index,
                    "sequence_number": next_seq(),
                    "item": {
                        "id": current_item_id,
                        "type": "custom_tool_call",
                        "status": "in_progress",
                        "call_id": tool_call_id,
                        "name": downstream_name,
                        "input": "",
                    },
                })
            else:
                # 普通 function_call（可流式发 arguments delta）
                current_item_type = "function_call"
                current_tool_spec = spec
                output_index = item_output_index()
                added_item: dict[str, Any] = {
                    "id": current_item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": tool_call_id,
                    "name": spec.get("name", upstream_name),
                    "arguments": "",
                }
                if spec.get("kind") == "namespace" and spec.get("namespace"):
                    added_item["namespace"] = spec["namespace"]
                yield sse_format("response.output_item.added", {
                    "output_index": output_index,
                    "sequence_number": next_seq(),
                    "item": added_item,
                })

        elif etype == "tool_use_delta":
            args_delta = data.get("arguments_delta", "")
            if args_delta and current_item_type == "function_call":
                tool_args_buf[current_tool_call_id] = tool_args_buf.get(current_tool_call_id, "") + args_delta
                output_index = item_output_index()
                yield sse_format("response.function_call_arguments.delta", {
                    "item_id": current_item_id,
                    "output_index": output_index,
                    "sequence_number": next_seq(),
                    "delta": args_delta,
                })
            # custom_tool_call 不发流式 delta（参数一次性在 end 给出）

        elif etype == "tool_use_end":
            if current_item_type == "function_call":
                # 流的 arguments 可能有遗漏（end 时填入完整 input）
                final_input = data.get("input", {})
                if not isinstance(final_input, dict):
                    final_input = {}
                # 若之前没流过 arguments delta，end 时一次写完
                existing = tool_args_buf.get(current_tool_call_id, "")
                if not existing:
                    full_args = safe_json_dumps(final_input, default="{}")
                    tool_args_buf[current_tool_call_id] = full_args
                for ev in close_current_item():
                    yield ev
            elif current_item_type == "custom_tool_call":
                # 透传模式：final_input 原样作为 custom_tool_call.input
                final_input = data.get("input", {})
                if isinstance(final_input, dict):
                    if current_tool_downstream_name == "apply_patch":
                        # 解包 {"input": "<DSL>"} → 裸 DSL 文本 + DSL 修复
                        dsl = final_input.get("input", "")
                        input_text = repair_apply_patch_dsl(dsl).dsl if isinstance(dsl, str) else safe_json_dumps(final_input, default="{}")
                    else:
                        input_text = safe_json_dumps(final_input, default="{}")
                else:
                    input_text = str(final_input)
                tool_args_buf[current_tool_call_id] = input_text
                # emit custom_tool_call_input.delta（一次性给完 input）
                output_index = item_output_index()
                yield sse_format("response.custom_tool_call_input.delta", {
                    "item_id": current_item_id,
                    "call_id": current_tool_call_id,
                    "delta": input_text,
                })
                for ev in close_current_item():
                    yield ev

        elif etype == "usage":
            pending_usage = data

        elif etype == "message_stop":
            pending_stop_reason = data.get("stop_reason", "end_turn")

        elif etype == "error":
            err = data or {}
            yield sse_format("response.error", {
                "sequence_number": next_seq(),
                "error": {
                    "code": err.get("code", "api_error"),
                    "message": err.get("message", "Stream error"),
                },
            })

        elif etype == "keepalive":
            yield b": keepalive\n\n"

    # ── 流末尾：闭合所有 + 发 response.completed ──
    for ev in close_current_item():
        yield ev

    # 计算最终 status
    status, incomplete_reason = map_stop_to_responses_status(pending_stop_reason)
    input_tokens = int((pending_usage or {}).get("input_tokens", 0))
    output_tokens = int((pending_usage or {}).get("output_tokens", 0))

    responses_usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if input_tokens or output_tokens:
        # 保留 cache token 字段（IR 已透传）
        cached = (pending_usage or {}).get("cache_read_input_tokens", 0)
        if cached:
            responses_usage["input_tokens_details"] = {"cached_tokens": cached}

    completed_payload: dict[str, Any] = {
        "id": response_id or f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created": created_ts,
        "model": model,
        "status": status,
        "output": all_output_items,
        "usage": responses_usage,
    }
    if incomplete_reason:
        completed_payload["incomplete_details"] = {"reason": incomplete_reason}

    yield sse_format("response.completed", {
        "sequence_number": next_seq(),
        "response": completed_payload,
    })
    yield b"data: [DONE]\n\n"
