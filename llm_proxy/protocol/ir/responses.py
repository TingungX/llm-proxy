"""OpenAI Responses API ↔ IR 转换器。

复用 responses_chat/tool_replacement.py 中的 apply_patch 解析与反向构造。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from llm_proxy.protocol.ir._common import (
    build_usage,
    clean_schema,
    is_openai_o_series,
    safe_json_dumps,
    safe_json_loads,
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
from llm_proxy.protocol.responses_chat.tool_replacement import (
    APPEND_TOOL_DEF,
    DELETE_TOOL_DEF,
    REPLACE_TOOL_DEF,
    WRITE_TOOL_DEF,
    build_reverse_tool_map,
    parse_apply_patch_to_simple,
    reverse_tool_args_to_apply_patch,
    ReverseConversionError,
)

logger = logging.getLogger(__name__)

# 标准文件工具名（apply_patch 展开后的 4 个）
_STANDARD_FILE_TOOL_NAMES = frozenset({"write_to_file", "replace_in_file", "delete_file", "append_to_file"})


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
        ir_tools, reverse_tool_map, namespace_map = _convert_tools_to_ir(tools)
        if ir_tools:
            ir_request.tools = ir_tools
        if reverse_tool_map:
            ir_request.extensions["reverse_tool_map"] = reverse_tool_map
        if namespace_map:
            ir_request.extensions["namespace_map"] = namespace_map

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


def _convert_tools_to_ir(tools: list) -> tuple[list[IRToolDef], dict[str, str], dict[str, str]]:
    """Responses tools → (IRToolDef list, reverse_tool_map, namespace_map)。

    处理 custom / namespace / web_search / function 等类型。
    apply_patch 展开为 4 个标准文件工具。
    """
    ir_tools: list[IRToolDef] = []
    reverse_tool_map: dict[str, str] = {}
    namespace_map: dict[str, str] = {}

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
            if name == "apply_patch":
                # 展开为 4 个标准文件工具
                _add_standard_file_tools(ir_tools, reverse_tool_map)
                continue
            # 其他 custom 工具 → 降级为 function
            ir_tools.append(IRToolDef(
                name=name,
                description=tool.get("description", ""),
                parameters=clean_schema(_normalize_params(tool.get("parameters"))),
            ))
            reverse_tool_map[name] = name
            continue

        if tool_type == "namespace":
            ns_name = tool.get("name", "")
            for sub in tool.get("tools") or []:
                if not isinstance(sub, dict) or sub.get("type") != "function":
                    continue
                sub_name = sub.get("name", "")
                ir_tools.append(IRToolDef(
                    name=sub_name,
                    description=sub.get("description", ""),
                    parameters=clean_schema(_normalize_params(sub.get("parameters"))),
                ))
                namespace_map[sub_name] = ns_name
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

    return ir_tools, reverse_tool_map, namespace_map


def _normalize_params(params) -> dict:
    """补全缺失的 JSON schema 字段。"""
    if not isinstance(params, dict):
        params = {}
    if not params.get("type"):
        params["type"] = "object"
    params.setdefault("properties", {})
    params.setdefault("required", [])
    return params


def _add_standard_file_tools(ir_tools: list[IRToolDef], reverse_tool_map: dict[str, str]):
    """添加 4 个标准文件工具到 IR tools 列表。"""
    for tool_def in (WRITE_TOOL_DEF, REPLACE_TOOL_DEF, DELETE_TOOL_DEF, APPEND_TOOL_DEF):
        func = tool_def.get("function", {})
        ir_tools.append(IRToolDef(
            name=func.get("name", ""),
            description=func.get("description", ""),
            parameters=clean_schema(func.get("parameters", {})),
        ))
    reverse_tool_map.update(build_reverse_tool_map())


# ── 响应：IR → Responses ──────────────────────────────────────────


def response_from_ir(ir: IRResponse) -> dict[str, Any]:
    """IRResponse → OpenAI Responses API 响应体。"""
    output: list[dict] = []
    for block in ir.content_blocks:
        if isinstance(block, IRTextBlock):
            if block.text:
                output.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": block.text, "annotations": []}
                    ],
                })
        elif isinstance(block, IRThinkingBlock):
            if block.thinking:
                output.append({
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "summary": [{"type": "summary_text", "text": block.thinking}],
                })
        elif isinstance(block, IRToolUseBlock):
            output.append({
                "type": "function_call",
                "id": block.id or f"call_{uuid.uuid4().hex[:24]}",
                "call_id": block.id or f"call_{uuid.uuid4().hex[:24]}",
                "name": block.name,
                "arguments": safe_json_dumps(block.input, default="{}"),
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
        "created": 0,
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

    如果 reverse_tool_map 包含标准文件工具，反向还原为 apply_patch custom tool。
    """
    if not ir_tools:
        return []

    # 检测是否所有标准文件工具都存在 → 还原为 apply_patch
    tool_names = {t.name for t in ir_tools}
    if _STANDARD_FILE_TOOL_NAMES.issubset(tool_names) and reverse_tool_map:
        # 所有 4 个标准工具都存在 → 还原为 apply_patch
        return [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch to files in the workspace.",
        }]

    result: list[dict] = []
    for tool in ir_tools:
        if tool.name in _STANDARD_FILE_TOOL_NAMES and reverse_tool_map:
            # 单个标准文件工具：作为 custom tool（apply_patch 的一部分）
            result.append({
                "type": "custom",
                "name": reverse_tool_map.get(tool.name, tool.name),
                "description": tool.description,
                "format": {"type": "grammar", "syntax": "lark"},
            })
            continue

        result.append({
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        })
    return result


# 反向构造 apply_patch 工具调用的辅助（暴露给流式层用）
_ = (parse_apply_patch_to_simple, reverse_tool_args_to_apply_patch, ReverseConversionError)

