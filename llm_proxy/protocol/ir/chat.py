"""OpenAI Chat Completions ↔ IR 转换器。"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from llm_proxy.protocol.constants import STOP_REASON_MAP
from llm_proxy.protocol.ir._common import (
    build_usage,
    clean_schema,
    is_openai_o_series,
    safe_json_dumps,
    safe_json_loads,
    supports_reasoning_effort,
)
from llm_proxy.protocol.think_tag import strip_think_tags
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

logger = logging.getLogger(__name__)


# ── 请求：Chat → IR ──────────────────────────────────────────────


def to_ir(body: dict[str, Any]) -> IRRequest:
    """OpenAI Chat Completions 请求体 → IRRequest。"""
    model = body.get("model", "")

    messages: list[IRMessage] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")

        # 提取 reasoning_content
        reasoning_content = msg.get("reasoning_content")
        converted = _convert_message_to_ir(role, content, reasoning_content)
        messages.extend(converted)

    # 提取 system_prompt（保留 Chat 的 system 角色约定）
    system_prompt: str | None = None
    if messages and messages[0].role == "system":
        first = messages[0]
        if isinstance(first.content, str):
            system_prompt = first.content
        elif isinstance(first.content, list) and first.content and isinstance(first.content[0], IRTextBlock):
            system_prompt = first.content[0].text
        messages = messages[1:]

    ir_request = IRRequest(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
    )

    # 参数映射
    if is_openai_o_series(model) and "max_completion_tokens" in body:
        ir_request.max_tokens = int(body["max_completion_tokens"])
    elif "max_tokens" in body:
        ir_request.max_tokens = int(body["max_tokens"])

    if "temperature" in body:
        ir_request.temperature = body["temperature"]
    if "top_p" in body:
        ir_request.top_p = body["top_p"]
    if "stream" in body:
        ir_request.stream = bool(body["stream"])

    if "reasoning_effort" in body:
        ir_request.reasoning_effort = body["reasoning_effort"]

    if "stop" in body:
        stop_val = body["stop"]
        if isinstance(stop_val, str):
            ir_request.stop_sequences = [stop_val]
        elif isinstance(stop_val, list):
            ir_request.stop_sequences = list(stop_val)

    # tools
    tools = body.get("tools") or []
    if tools:
        ir_tools: list[IRToolDef] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function") or tool
            ir_tools.append(IRToolDef(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=clean_schema(func.get("parameters", {}) or {}),
                cache_control=tool.get("cache_control"),
            ))
        if ir_tools:
            ir_request.tools = ir_tools

    # tool_choice
    if "tool_choice" in body:
        ir_request.tool_choice = body["tool_choice"]

    # extensions
    for key in ("stream_options", "response_format", "logprobs", "n",
                "presence_penalty", "frequency_penalty", "seed", "user"):
        if key in body:
            ir_request.extensions[key] = body[key]

    return ir_request


def _convert_message_to_ir(
    role: str, content: Any, reasoning_content: str | None = None
) -> list[IRMessage]:
    """Chat 消息 → IRMessage。"""
    # 处理 tool_calls
    tool_calls = None
    if isinstance(content, dict) and content.get("type") == "tool_calls":
        # 罕见格式
        tool_calls = content.get("tool_calls")

    if tool_calls is None and isinstance(content, list):
        # 部分 Chat 格式把 tool_calls 和 content 一起放在 content list（极少见）
        pass

    blocks: list[IRContentBlock] = []

    # content
    if content is None:
        pass  # tool_calls-only message
    elif isinstance(content, str):
        text = content
        if role == "assistant" and text:
            # 提取 <think> 标签
            extracted_reasoning, clean_text = strip_think_tags(text)
            if reasoning_content is None:
                reasoning_content = extracted_reasoning
            else:
                if extracted_reasoning:
                    reasoning_content = (reasoning_content or "") + "\n" + extracted_reasoning
            text = clean_text
        if text:
            blocks.append(IRTextBlock(text=text))
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                blocks.append(IRTextBlock(text=part.get("text", "")))
            elif part_type == "image_url":
                url = part.get("image_url", {}).get("url", "")
                # data:URL → 拆出 base64 + media_type
                if url.startswith("data:"):
                    media_type, data = _parse_data_url(url)
                    blocks.append(IRImageBlock(base64_data=data, media_type=media_type))
                else:
                    # 非 data URL：原样保留在 extensions（IR 不能表达 URL）
                    blocks.append(IRTextBlock(text=f"[image: {url}]"))
            elif part_type == "refusal":
                # refusal 降级为 text
                blocks.append(IRTextBlock(text=part.get("refusal", "")))
    elif isinstance(content, dict):
        # 嵌套 dict 当 text
        blocks.append(IRTextBlock(text=str(content)))

    # reasoning_content
    if reasoning_content:
        blocks.insert(0, IRThinkingBlock(thinking=reasoning_content))

    # 独立 tool_call_id / name（Chat tool 消息）
    if role == "tool":
        tool_call_id = None
        # 工具消息通常没有 content 块，而是纯字符串
        if isinstance(content, str):
            return [IRMessage(role="tool", content=[IRToolResultBlock(
                tool_use_id="",  # Chat 工具消息 id 在 message["tool_call_id"] 上，不在 content
                content=content,
            )])]
        return [IRMessage(role="tool", content=blocks)]

    return [IRMessage(role=role, content=blocks if blocks else "")]


def _parse_data_url(url: str) -> tuple[str, str]:
    """data:URL → (media_type, base64_data)"""
    if not url.startswith("data:"):
        return "image/png", ""
    rest = url[5:]
    if ";" in rest:
        media_type, rest = rest.split(";", 1)
    else:
        media_type = "image/png"
    if rest.startswith("base64,"):
        data = rest[7:]
    else:
        data = rest
    return media_type, data


# ── 响应：IR → Chat ──────────────────────────────────────────────


def response_from_ir(ir: IRResponse) -> dict[str, Any]:
    """IRResponse → Chat Completions 响应体。"""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    reasoning_text: str | None = None

    for block in ir.content_blocks:
        if isinstance(block, IRTextBlock):
            if block.text:
                text_parts.append(block.text)
        elif isinstance(block, IRThinkingBlock):
            if block.thinking:
                reasoning_text = block.thinking
        elif isinstance(block, IRToolUseBlock):
            tool_calls.append({
                "id": block.id or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": safe_json_dumps(block.input, default="{}"),
                },
            })

    # content
    if tool_calls and not text_parts:
        content_value = None
    elif text_parts:
        content_value = "\n".join(text_parts)
    else:
        content_value = ""

    message: dict[str, Any] = {"role": "assistant", "content": content_value}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_text:
        message["reasoning_content"] = reasoning_text

    # finish_reason
    finish_reason = _map_stop_reason_to_chat(ir.stop_reason)

    # usage
    raw_usage = ir.usage
    chat_usage: dict[str, int] = {}
    if raw_usage.get("input_tokens") is not None:
        chat_usage["prompt_tokens"] = int(raw_usage["input_tokens"])
    if raw_usage.get("output_tokens") is not None:
        chat_usage["completion_tokens"] = int(raw_usage["output_tokens"])

    # cache tokens → prompt_tokens_details.cached_tokens
    cached = raw_usage.get("cache_read_input_tokens", 0)
    if cached:
        chat_usage["prompt_tokens_details"] = {"cached_tokens": int(cached)}

    total = chat_usage.get("prompt_tokens", 0) + chat_usage.get("completion_tokens", 0)
    chat_usage["total_tokens"] = total

    return {
        "id": ir.id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": 0,
        "model": ir.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": chat_usage,
    }


def _map_stop_reason_to_chat(stop_reason: str) -> str:
    """Anthropic stop_reason → Chat finish_reason。"""
    inv = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "refusal": "content_filter",
    }
    return inv.get(stop_reason, "stop")


def response_to_ir(body: dict[str, Any]) -> IRResponse:
    """Chat Completions 响应体 → IRResponse。"""
    choices = body.get("choices") or []
    if not choices:
        return IRResponse(
            id=body.get("id", ""),
            model=body.get("model", ""),
            content_blocks=[IRTextBlock(text="")],
            stop_reason="end_turn",
            usage=build_usage(body.get("usage", {})),
        )

    choice = choices[0]
    message = choice.get("message", {}) or {}
    finish_reason = choice.get("finish_reason")

    blocks: list[IRContentBlock] = []

    reasoning = message.get("reasoning_content")
    content = message.get("content")

    if content is not None and isinstance(content, str):
        extracted_reasoning, clean_text = strip_think_tags(content)
        if reasoning is None:
            reasoning = extracted_reasoning
        elif extracted_reasoning:
            reasoning = (reasoning or "") + "\n" + extracted_reasoning
        content = clean_text

    if reasoning:
        blocks.append(IRThinkingBlock(thinking=reasoning))

    if content is not None:
        if isinstance(content, str):
            if content:
                blocks.append(IRTextBlock(text=content))
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    if part.get("text"):
                        blocks.append(IRTextBlock(text=part["text"]))
                elif ptype == "refusal":
                    if part.get("refusal"):
                        blocks.append(IRTextBlock(text=part["refusal"]))

    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict) or tc.get("type") != "function":
            continue
        func = tc.get("function", {})
        arguments = safe_json_loads(func.get("arguments", "{}"), default={})
        blocks.append(IRToolUseBlock(
            id=tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
            name=func.get("name", ""),
            input=arguments if isinstance(arguments, dict) else {},
        ))

    if not blocks:
        blocks.append(IRTextBlock(text=""))

    return IRResponse(
        id=body.get("id", ""),
        model=body.get("model", ""),
        content_blocks=blocks,
        stop_reason=_map_stop_reason_to_ir(finish_reason),
        usage=build_usage(body.get("usage", {})),
    )


def _map_stop_reason_to_ir(finish_reason: str | None) -> str:
    """Chat finish_reason → IR stop_reason。"""
    if finish_reason is None:
        return "end_turn"
    inv = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
    }
    return inv.get(finish_reason, "end_turn")


# 重新导出避免未使用警告
_ = STOP_REASON_MAP


# ── 请求出：IR → Chat ─────────────────────────────────────────────


def to_upstream(ir: IRRequest, upstream_model: str | None = None) -> dict[str, Any]:
    """IRRequest → OpenAI Chat Completions 请求体。"""
    result: dict[str, Any] = {"model": upstream_model or ir.model}

    # messages — 顶部追加 system（如果有 system_prompt）
    chat_messages: list[dict] = []
    if ir.system_prompt:
        sys_msg: dict = {"role": "system", "content": ir.system_prompt}
        if "system_cache_control" in ir.extensions:
            sys_msg["cache_control"] = ir.extensions["system_cache_control"]
        chat_messages.append(sys_msg)

    for msg in ir.messages:
        chat_messages.extend(_message_ir_to_chat(msg))

    result["messages"] = chat_messages

    # max_tokens
    if ir.max_tokens is not None:
        if is_openai_o_series(result["model"]):
            result["max_completion_tokens"] = int(ir.max_tokens)
        else:
            result["max_tokens"] = int(ir.max_tokens)

    if ir.temperature is not None:
        result["temperature"] = ir.temperature
    if ir.top_p is not None:
        result["top_p"] = ir.top_p
    if ir.stream:
        result["stream"] = True
        # 流式自动加 stream_options
        result["stream_options"] = ir.extensions.get("stream_options", {"include_usage": True})

    if ir.reasoning_effort is not None:
        result["reasoning_effort"] = ir.reasoning_effort

    if ir.stop_sequences is not None:
        if len(ir.stop_sequences) == 1:
            result["stop"] = ir.stop_sequences[0]
        else:
            result["stop"] = ir.stop_sequences

    # tools
    if ir.tools:
        chat_tools: list[dict] = []
        for tool in ir.tools:
            ct: dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            if tool.cache_control is not None:
                ct["cache_control"] = tool.cache_control
            chat_tools.append(ct)
        result["tools"] = chat_tools

    # tool_choice
    if ir.tool_choice is not None:
        result["tool_choice"] = ir.tool_choice

    # 透传 extensions 中的 Chat 字段
    for key in ("response_format", "logprobs", "n",
                "presence_penalty", "frequency_penalty", "seed", "user"):
        if key in ir.extensions:
            result[key] = ir.extensions[key]

    return result


def _message_ir_to_chat(msg: IRMessage) -> list[dict[str, Any]]:
    """IRMessage → Chat messages 列表（tool_result 独立）。"""
    role = msg.role
    content = msg.content

    if role == "tool":
        # tool 消息
        if isinstance(content, list):
            for block in content:
                if isinstance(block, IRToolResultBlock):
                    return [{"role": "tool", "tool_call_id": block.tool_use_id, "content": block.content}]
        return [{"role": "tool", "content": str(content) if not isinstance(content, str) else content}]

    if isinstance(content, str):
        if role == "system":
            return [{"role": "system", "content": content}]
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": str(content)}]

    # 数组 content
    content_parts: list[dict] = []
    tool_calls: list[dict] = []
    reasoning_parts: list[str] = []

    for block in content:
        if isinstance(block, IRTextBlock):
            content_parts.append({"type": "text", "text": block.text})
        elif isinstance(block, IRImageBlock):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{block.media_type};base64,{block.base64_data}"},
            })
        elif isinstance(block, IRThinkingBlock):
            if block.thinking:
                reasoning_parts.append(block.thinking)
        elif isinstance(block, IRToolUseBlock):
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": safe_json_dumps(block.input, default="{}"),
                },
            })
        elif isinstance(block, IRToolResultBlock):
            # tool_result 应该是独立 tool 消息（to_ir 阶段已分离）
            pass

    result: list[dict[str, Any]] = []
    msg_dict: dict[str, Any] = {"role": role}

    if content_parts:
        if len(content_parts) == 1 and content_parts[0].get("type") == "text" and "cache_control" not in content_parts[0]:
            msg_dict["content"] = content_parts[0]["text"]
        else:
            msg_dict["content"] = content_parts
    elif tool_calls:
        msg_dict["content"] = None
    else:
        msg_dict["content"] = ""

    if tool_calls:
        msg_dict["tool_calls"] = tool_calls

    if reasoning_parts and role == "assistant":
        msg_dict["reasoning_content"] = "\n".join(reasoning_parts)

    result.append(msg_dict)
    return result
