"""Responses API → Chat Completions 协议转换

将 OpenAI Responses API 格式转换为 Chat Completions 格式，
用于代理到上游 OpenAI 兼容端点。
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from llm_proxy.infra import db

from llm_proxy.protocol.responses_chat.stream import StreamState, _unwrap_input_arg
from llm_proxy.protocol.responses_chat.tool_replacement import (
    APPLY_PATCH_TOOL_DESCRIPTION,
    repair_apply_patch_dsl,
)
from llm_proxy.protocol.think_tag import strip_think_tags

logger = logging.getLogger(__name__)


_ROLE_MAP = {
    "developer": "system",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


@dataclass
class CodexToolSpec:
    """Tracks original Responses API tool info for reverse conversion.

    Records the original tool kind, name, and optional namespace so the return
    path can restore correct event types and names (e.g., namespace sub-tools
    get their original name and namespace field back).
    """
    kind: str       # "function" | "custom" | "namespace" | "tool_search" | "web_search" | ...
    name: str       # original name as sent by Codex
    namespace: Optional[str] = None  # parent namespace name (only for namespace tools)


def _normalize_params(params) -> dict:
    if params is None or not isinstance(params, dict):
        params = {}
    params = dict(params)
    if not params.get("type"):
        params["type"] = "object"
    params.setdefault("properties", {})
    params.setdefault("required", [])
    return params


def _make_chat_function_tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def convert_tools_to_chat(tools: list) -> tuple[list, dict[str, str], dict[str, CodexToolSpec]]:
    """Convert Responses API tools to Chat Completions format.

    Returns:
        (chat_tools, reverse_tool_map, tool_spec_map) —
        reverse_tool_map maps upstream tool names back to downstream
        custom tool names (apply_patch).
        tool_spec_map maps upstream chat names → CodexToolSpec for
        namespace sub-tools, custom tools, and degraded tools so the
        return path can restore correct event types and names.
    """
    result = []
    reverse_tool_map: dict[str, str] = {}
    tool_spec_map: dict[str, CodexToolSpec] = {}

    # 客户端侧内置工具，不应发给上游模型
    # tool_search: Codex 客户端搜索延迟加载工具，客户端执行
    # web_search: Codex 客户端执行搜索，客户端执行
    CLIENT_SIDE_TOOLS = {"tool_search", "web_search"}

    for tool in tools:
        if not isinstance(tool, dict):
            logger.warning(f"Unsupported tool type: {type(tool).__name__}, expected dict")
            continue

        tool_type = tool.get("type", "")

        # 客户端侧工具：跳过，不发给上游
        if tool_type in CLIENT_SIDE_TOOLS:
            logger.debug(f"Skipping client-side tool: {tool_type}")
            continue

        # --- custom 工具（透传模式） ---
        # apply_patch 与其他 custom 工具一致，直接降级为单个 function tool
        # 不再展开为 4 个标准文件工具，反向也不再做 DSL 转换
        if tool_type == "custom":
            name = tool.get("name", "")
            params = _normalize_params(tool.get("parameters"))
            if not params.get("properties"):
                params = {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": APPLY_PATCH_TOOL_DESCRIPTION}},
                    "required": [],
                }
            result.append(_make_chat_function_tool(
                name, tool.get("description") or APPLY_PATCH_TOOL_DESCRIPTION, params
            ))
            reverse_tool_map[name] = name
            tool_spec_map[name] = CodexToolSpec(kind="custom", name=name)
            logger.debug(f"Passthrough custom tool as function: {name}")
            continue

        # --- namespace 工具：展开子工具，使用限定的全名以避免同名冲突 ---
        # 注意：分隔符使用 "__"（双下划线）而非 "."，因为部分上游（如 DeepSeek）
        # 要求 tool name 匹配 ^[a-zA-Z0-9_-]+$，不接受点号。
        if tool_type == "namespace":
            ns_name = tool.get("name", "")
            for sub in (tool.get("tools") or []):
                if not isinstance(sub, dict) or sub.get("type") != "function":
                    continue
                sub_name = sub.get("name", "")
                chat_name = f"{ns_name}__{sub_name}"
                result.append(_make_chat_function_tool(
                    chat_name,
                    sub.get("description", ""),
                    _normalize_params(sub.get("parameters")),
                ))
                tool_spec_map[chat_name] = CodexToolSpec(
                    kind="namespace", name=sub_name, namespace=ns_name
                )
                logger.debug(f"Expanded namespace tool: {ns_name}.{sub_name} → upstream name={chat_name}")
            continue

        # --- 其他非 function 类型：尝试降级为 function ---
        # web_search / tool_search / image_generation 等都走此分支
        # 优先用工具自带的 name/parameters，没有则生成占位定义
        if tool_type != "function":
            name = tool.get("name", tool_type)
            params = _normalize_params(tool.get("parameters"))
            if not params.get("properties") and tool_type == "web_search":
                name = "web_search"
                params = {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string.",
                        },
                    },
                    "required": ["query"],
                }
            desc = tool.get("description", f"Tool of type {tool_type}")
            result.append(_make_chat_function_tool(name, desc, params))
            tool_spec_map[name] = CodexToolSpec(kind=tool_type, name=name)
            logger.debug(f"Degraded non-function tool type={tool_type} name={name} to function")
            continue

        func = tool.get("function", tool)
        name = func.get("name", "")
        result.append(_make_chat_function_tool(
            name, func.get("description", ""), _normalize_params(func.get("parameters"))
        ))
    return result, reverse_tool_map, tool_spec_map


def _extract_reasoning_text(item: dict) -> str:
    for s in item.get("summary", []):
        if isinstance(s, dict) and s.get("type") == "summary_text":
            return s.get("text", "")
    return ""


def convert_input_to_messages(input_data, instructions: str | None = None) -> list[dict]:
    """将 Responses API input 转为 Chat Completions messages 格式。

    Args:
        input_data: Responses API 的 input 字段（字符串或消息列表）
        instructions: 可选的系统指令

    Returns:
        Chat Completions 格式的 messages 列表
    """
    messages = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages

    if isinstance(input_data, list):
        pending_reasoning: list[str] = []
        pending_tool_calls: list[dict] = []
        pending_assistant_content: str | None = None
        split_call_ids: dict[str, list[str]] = {}  # original_id → [derived_ids] for multi-file patches

        def _flush_assistant_turn():
            nonlocal pending_assistant_content
            has_content = pending_assistant_content is not None
            has_tc = bool(pending_tool_calls)
            has_rc = bool(pending_reasoning)
            if not has_content and not has_tc and not has_rc:
                return
            msg: dict = {"role": "assistant"}
            if has_rc:
                msg["reasoning_content"] = "\n".join(pending_reasoning)
                pending_reasoning.clear()
            if has_content:
                msg["content"] = pending_assistant_content
                pending_assistant_content = None
            elif has_tc:
                msg["content"] = None
            else:
                msg["content"] = ""
            if has_tc:
                msg["tool_calls"] = list(pending_tool_calls)
                pending_tool_calls.clear()
            messages.append(msg)

        for item in input_data:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")

            if item_type == "reasoning":
                text = _extract_reasoning_text(item)
                if text:
                    pending_reasoning.append(text)
                continue

            if item_type == "function_call":
                pending_tool_calls.append({
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                })
                continue

            if item_type == "function_call_output":
                _flush_assistant_turn()
                tool_call_id = item.get("call_id", "")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": item.get("output", ""),
                })
                continue

            # mcp_call：转为 Chat tool_calls（McpCall 用 id 字段，非 call_id）
            if item_type == "mcp_call":
                mcp_call_id = item.get("call_id") or item.get("id", "")
                logger.debug(
                    "mcp_call input: id=%s, call_id=%s, name=%s, server_label=%s, "
                    "has_output=%s, has_error=%s, status=%s",
                    item.get("id", "-"), item.get("call_id", "-"),
                    item.get("name", "-"), item.get("server_label", "-"),
                    item.get("output") is not None, item.get("error") is not None,
                    item.get("status", "-"),
                )
                pending_tool_calls.append({
                    "id": mcp_call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                })
                # 若有 output/error 字段，生成对应的 tool message
                output_text = item.get("output") or item.get("error")
                if output_text is not None:
                    _flush_assistant_turn()
                    messages.append({
                        "role": "tool",
                        "tool_call_id": mcp_call_id,
                        "content": output_text,
                    })
                continue

            if item_type in ("custom", "custom_tool_call"):
                name = item.get("name", "")
                input_text = item.get("input", "")
                if not isinstance(input_text, str):
                    input_text = json.dumps(input_text, ensure_ascii=False)
                call_id = item.get("call_id", "") or item.get("id", "")
                try:
                    parsed = json.loads(input_text)
                    if isinstance(parsed, dict):
                        arguments = input_text
                    else:
                        arguments = json.dumps({"input": input_text}, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    arguments = json.dumps({"input": input_text}, ensure_ascii=False)
                pending_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                })
                logger.debug(f"Passthrough custom_tool_call to function: {name}")
                continue

            if item_type in ("custom_tool_call_output", "custom_output"):
                _flush_assistant_turn()
                tool_call_id = item.get("call_id", "")
                output = item.get("output", "")
                if isinstance(output, str):
                    content = output
                elif isinstance(output, dict) and "text" in output:
                    content = output["text"]
                else:
                    content = str(output)
                # 多文件拆分：为每个派生 ID 生成 tool 消息
                derived_ids = split_call_ids.get(tool_call_id)
                if derived_ids:
                    for did in derived_ids:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": did,
                            "content": content,
                        })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    })
                continue

            role = _ROLE_MAP.get(item.get("role", "user"), "user")
            content = item.get("content", "")

            if isinstance(content, str):
                msg_content = content
            elif isinstance(content, list):
                parts = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    ctype = c.get("type", "")
                    if ctype in ("input_text", "text"):
                        parts.append({"type": "text", "text": c.get("text", "")})
                    elif ctype == "input_image":
                        img_url = c.get("image_url") or c.get("image", {}).get("url", "")
                        parts.append({"type": "image_url", "image_url": {"url": img_url}})
                    elif ctype == "output_text":
                        parts.append({"type": "text", "text": c.get("text", "")})
                if parts:
                    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                    if text:
                        msg_content = text
                    else:
                        msg_content = parts
                elif content:
                    msg_content = str(content)
                else:
                    msg_content = None
            else:
                msg_content = None

            if role == "assistant":
                if msg_content is not None:
                    pending_assistant_content = msg_content
            else:
                _flush_assistant_turn()
                if msg_content is not None:
                    messages.append({"role": role, "content": msg_content})

        _flush_assistant_turn()

    from llm_proxy.services.tool_call_fix import fix_orphaned_tool_calls
    messages = fix_orphaned_tool_calls(messages)

    for msg in messages:
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            content = msg.get("content")
            if content is None or content == "" or (isinstance(content, list) and all(p.get("type") == "text" and not p.get("text", "") for p in content)):
                if "reasoning_content" not in msg:
                    msg["reasoning_content"] = ""
                if content is None or content == "":
                    msg["content"] = ""
                elif isinstance(content, list):
                    text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
                    msg["content"] = text if text else ""
        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            content = msg.get("content")
            if content is None:
                msg["content"] = None

    return messages


def to_responses_response(chat_body: dict, original_model: str, reverse_tool_map: dict[str, str] | None = None, tool_spec_map: dict[str, CodexToolSpec] | None = None) -> dict:
    choices = chat_body.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})

    output = []
    content = message.get("content", "")
    reasoning_content = message.get("reasoning_content", "")

    # Strip think tags from string content
    extracted_reasoning = ""
    if content:
        extracted_reasoning, content = strip_think_tags(content)

    if reasoning_content:
        output.append({
            "id": f"rs_{uuid.uuid4().hex[:16]}",
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": reasoning_content}],
        })
    elif extracted_reasoning:
        output.append({
            "id": f"rs_{uuid.uuid4().hex[:16]}",
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": extracted_reasoning}],
        })

    if content:
        output.append({
            "id": f"msg_{uuid.uuid4().hex[:16]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })

    _reverse = reverse_tool_map or {}
    _spec = tool_spec_map or {}
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        if tc.get("type") == "function":
            func = tc.get("function", {})
            name = func.get("name", "")
            downstream_name = _reverse.get(name)
            spec = _spec.get(name)
            if downstream_name is not None:
                # 1) reverse_tool_map 命中 → custom_tool_call（透传：arguments 原样作为 input）
                args_str = func.get("arguments", "{}")
                if not isinstance(args_str, str):
                    args_str = json.dumps(args_str, ensure_ascii=False)
                # 解包 {"input": "..."} 为原始 DSL 文本
                args_str = repair_apply_patch_dsl(_unwrap_input_arg(args_str)).dsl
                output.append({
                    "id": f"fc_{tc.get('id', '')}",
                    "type": "custom_tool_call",
                    "name": downstream_name,
                    "status": "completed",
                    "call_id": tc.get("id", ""),
                    "input": args_str,
                })
            elif spec is not None and spec.kind == "namespace" and spec.namespace:
                # 2) namespace 子工具 → function_call + 原始名称 + namespace
                args = func.get("arguments", "")
                if not args:
                    args = "{}"
                output.append({
                    "id": f"fc_{tc.get('id', '')}",
                    "type": "function_call",
                    "name": spec.name,
                    "namespace": spec.namespace,
                    "arguments": args,
                    "status": "completed",
                    "call_id": tc.get("id", ""),
                })
            else:
                # 3) 普通函数 → function_call
                args = func.get("arguments", "")
                if not args:
                    args = "{}"
                output.append({
                    "id": f"fc_{tc.get('id', '')}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tc.get("id", ""),
                    "name": name,
                    "arguments": args,
                })

    raw_usage = chat_body.get("usage", {})
    from llm_proxy.protocol.responses_chat.usage import extract_usage_metrics
    usage = extract_usage_metrics(raw_usage)
    if not usage:
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    return {
        "id": f"resp_{uuid.uuid4().hex[:16]}",
        "object": "response",
        "created": int(time.time()),
        "model": original_model,
        "output": output,
        "usage": usage,
    }


def convert_chunk_to_events(
    chunk: dict,
    model: str,
    state: StreamState,
) -> list[bytes]:
    events = []
    choices = chunk.get("choices", [])

    if not choices:
        logger.debug(f"Chunk has no choices: {json.dumps(chunk)[:200]}")

    for choice in choices:
        if not choice:
            logger.warning(f"Empty choice in chunk")
            continue
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        reasoning_text = delta.get("reasoning_content")
        if reasoning_text:
            events.extend(state.handle_reasoning_delta(reasoning_text))

        content = delta.get("content")
        if content:
            if state.think.state != "done":
                reasoning_parts, content_parts = state.think.feed(content)
                for rp in reasoning_parts:
                    events.extend(state.handle_reasoning_delta(rp))
                content = "".join(content_parts)

            if content:
                events.extend(state.handle_content_delta(content))

        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            idx = tc.get("index", 0)
            tc_id = tc.get("id")
            func = tc.get("function", {})

            if tc_id and idx not in state.func_call_ids:
                name = func.get("name", "")
                events.extend(state.handle_tool_call_id(idx, tc_id, name))

            args_delta = func.get("arguments", "")
            if args_delta and idx in state.func_args_buf:
                events.extend(state.handle_tool_call_args_delta(idx, args_delta))

        if finish_reason and finish_reason != "null":
            state.finish_reason = finish_reason
            events.extend(state.flush_think_tag_buf())
            if state.reasoning_active:
                events.extend(state.close_reasoning_block())
            if state.in_text_block:
                events.extend(state.close_text_block())
            if state.in_func_block:
                events.extend(state.close_func_blocks())

    return events


def make_sse_event(data: dict, event_type: str | None = None) -> bytes:
    """将字典转换为 SSE 事件格式

    Args:
        data: 事件数据字典
        event_type: 显式指定 SSE `event:` 头（OpenAI Responses API 错误流需要
                    显式 `event: error` 头 + `data: {"error": {...}}`，
                    不能从 data 字段反推）

    Returns:
        SSE 格式的字节串，形如 b'event: xxx\\ndata: {...}\\n\\n' 或 b'data: {...}\\n\\n'
    """
    # 显式 event_type 优先；否则从 data["type"] 兜底（向后兼容）
    resolved_type = event_type if event_type is not None else data.get("type", "")
    payload = json.dumps(data, ensure_ascii=False)
    if resolved_type:
        return f"event: {resolved_type}\ndata: {payload}\n\n".encode()
    return f"data: {payload}\n\n".encode()


def make_response_completed_event(
    model: str,
    response_id: str,
    output: list | None = None,
    usage: dict | None = None,
) -> bytes:
    """生成 response.completed 事件，包含完整 output 和 usage。

    Args:
        model: 原始模型名
        response_id: 响应 ID
        output: output 数组（reasoning + message + function_call items）
        usage: usage 字典（input_tokens, output_tokens, total_tokens）

    Returns:
        SSE 格式的 response.completed 事件
    """
    response_obj = {
        "id": response_id,
        "object": "response",
        "model": model,
        "status": "completed",
    }
    if output:
        response_obj["output"] = output
    if usage:
        response_obj["usage"] = usage
    event_data = {
        "type": "response.completed",
        "response": response_obj,
    }
    return make_sse_event(event_data)


async def stream_chat_to_responses(
    resp,
    model: str,
    endpoint_id: str,
    model_id: str,
    original_request: dict | None = None,
    result: dict | None = None,
    reverse_tool_map: dict[str, str] | None = None,
    tool_spec_map: dict[str, CodexToolSpec] | None = None,
    request_id: str = "",
    client_ip: str = "",
    user_agent: str = "",
):
    state = StreamState(reverse_tool_map=reverse_tool_map, tool_spec_map=tool_spec_map)
    response_id = state.response_id
    usage = {"input_tokens": 0, "output_tokens": 0}
    had_error = False
    completed = False

    _KEEPALIVE_INTERVAL = 15

    try:
        yield make_sse_event({
            "type": "response.created",
            "sequence_number": state._next_seq(),
            "response": {
                "id": response_id,
                "object": "response",
                "model": model,
                "status": "in_progress",
            },
        })
        yield make_sse_event({
            "type": "response.in_progress",
            "sequence_number": state._next_seq(),
            "response": {
                "id": response_id,
                "object": "response",
                "model": model,
                "status": "in_progress",
            },
        })

        line_queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream_done = False

        async def _read_lines():
            nonlocal stream_done
            line_count = 0
            try:
                async for line in resp.aiter_lines():
                    line_count += 1
                    await line_queue.put(line)
            except Exception as e:
                logger.warning(f"Error reading upstream stream lines: {e}", exc_info=True)
            finally:
                logger.debug(f"_read_lines finished, stream_done=True, lines={line_count}")
                stream_done = True
                await line_queue.put(None)

        read_task = asyncio.create_task(_read_lines())

        try:
            while True:
                try:
                    line = await asyncio.wait_for(line_queue.get(), timeout=_KEEPALIVE_INTERVAL)
                except asyncio.TimeoutError:
                    if stream_done:
                        break
                    yield make_sse_event({
                        "type": "response.keepalive",
                        "message": "waiting for upstream data",
                    })
                    continue

                if line is None:
                    break

                logger.debug(f"Raw line: {line[:100] if len(line) > 100 else line}")
                if not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    logger.debug(f"Stream completed for model {model}")
                    completed = True
                    for event in state.generate_completed_events(model, response_id, original_request, usage):
                        yield event
                    yield b"data: [DONE]\n\n"
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse chunk: {data[:100]}")
                    continue

                if not chunk:
                    logger.warning(f"Empty chunk")
                    continue

                # DEBUG级别：记录每个chunk
                if (chunk.get("choices") or [{}])[0].get("delta", {}).get("tool_calls"):
                    logger.debug(f"Received chunk with tool_calls: {json.dumps(chunk, ensure_ascii=False)}")
                else:
                    logger.debug(f"Received chunk: {json.dumps(chunk)}")

                u = chunk.get("usage")
                if u:
                    from llm_proxy.protocol.responses_chat.usage import extract_usage_metrics
                    extracted = extract_usage_metrics(u)
                    usage.update(extracted)

                try:
                    chunk_events = list(convert_chunk_to_events(chunk, model, state))
                except Exception as e:
                    # 单个 chat chunk 转换失败不能让整个 stream 挂掉——
                    # 跳过这个 chunk 并继续，OpenAI 客户端会看到 partial 响应
                    logger.warning(
                        f"convert_chunk_to_events failed for chunk (skipping): "
                        f"{type(e).__name__}: {e}; chunk={json.dumps(chunk)[:200]}"
                    )
                    continue
                for event in chunk_events:
                    yield event
        finally:
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass

        if result is not None:
            result["completed"] = completed
            result["has_text"] = bool(state.text_buf) or state.in_text_block

        if completed:
            output_items = state._build_output_items()
            logger.debug(
                f"Stream completed: output_items={len(output_items)}, "
                f"item_types={[o.get('type') for o in output_items]}, "
                f"text_buf_len={len(state.text_buf)}, "
                f"reasoning_buf_len={len(state.reasoning_buf)}, "
                f"in_func_block={state.in_func_block}"
            )

        if not completed:
            logger.warning(f"Stream ended without [DONE] marker for model {model}")
            for event in state.generate_completed_events(model, response_id, original_request, usage):
                yield event
            yield b"data: [DONE]\n\n"

    except Exception as e:
        had_error = True
        logger.error(f"Stream error: {e}", exc_info=True)
        yield make_sse_event({
            "type": "error",
            "code": "proxy_error",
            "message": str(e) or type(e).__name__,
        })
        yield make_response_completed_event(model, response_id)
        yield b"data: [DONE]\n\n"
    finally:
        status = "error" if had_error else "success"
        if usage["input_tokens"] > 0:
            db.record_usage(
                endpoint_id=endpoint_id,
                model_id=model_id,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                status=status,
                error_type=status if status == "error" else None,
                request_id=request_id,
                client_ip=client_ip,
                user_agent=user_agent,
            )
        else:
            db.record_usage(
                endpoint_id=endpoint_id,
                model_id=model_id,
                input_tokens=1,
                output_tokens=0,
                status=status,
                error_type=status if status == "error" else None,
                request_id=request_id,
                client_ip=client_ip,
                user_agent=user_agent,
            )
