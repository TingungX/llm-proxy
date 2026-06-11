"""Anthropic Messages → OpenAI Chat Completions 请求转换

将 Anthropic Messages API 格式的请求体转换为 OpenAI Chat Completions 格式，
用于跨协议代理到 OpenAI 兼容上游端点。

参考: CCS (cc-switch) providers/transform.rs
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ── Billing Header 清理 ──────────────────────────────────────────────

_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


def strip_leading_anthropic_billing_header(text: str) -> str:
    """移除 system prompt 开头的 billing header 行。

    Claude Code 会在 system 开头插入动态 billing header（含 rotating cch= 值），
    如果转发到 OpenAI Chat 的 system message，每次请求的 prompt prefix 都不同，
    导致 prefix cache 复用失效。只移除最开头的首行，后续出现的保留。
    """
    if not text.startswith(_BILLING_HEADER_PREFIX):
        return text

    # 找到第一行结尾
    line_end = len(text)
    for i, ch in enumerate(text):
        if ch == "\n" or ch == "\r":
            line_end = i
            break

    # 如果整段就是一行 billing header，返回空
    if line_end == len(text):
        return ""

    rest_start = line_end + 1
    # 处理 \r\n
    if text[line_end] == "\r" and rest_start < len(text) and text[rest_start] == "\n":
        rest_start += 1

    rest = text[rest_start:]
    # 去掉紧跟的换行
    for prefix in ("\r\n", "\n", "\r"):
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
            break

    return rest


# ── Reasoning Effort 映射 ────────────────────────────────────────────

_O_SERIES_RE = re.compile(r"^o\d")


def is_openai_o_series(model: str) -> bool:
    """检测 OpenAI o-series 推理模型（o1, o3, o4-mini 等）

    这些模型需要 max_completion_tokens 而非 max_tokens，
    且支持 reasoning_effort 参数。
    """
    if not model or len(model) < 2:
        return False
    if model[0] != "o":
        return False
    return model[1].isdigit()


def supports_reasoning_effort(model: str) -> bool:
    """检测是否支持 reasoning_effort 参数

    支持: o-series (o1, o3, o4-mini) 和 GPT-5+ (gpt-5, gpt-5.1, gpt-5-codex)
    不支持: gpt-4o, gpt-3.5-turbo 等
    """
    if is_openai_o_series(model):
        return True
    lower = model.lower()
    if lower.startswith("gpt-"):
        rest = lower[4:]
        first_char = rest[0] if rest else ""
        if first_char.isdigit() and int(first_char) >= 5:
            return True
    return False


def resolve_reasoning_effort(body: dict) -> str | None:
    """从 Anthropic 请求体解析出 OpenAI reasoning_effort 值。

    Priority 1: output_config.effort — 用户显式指定
      low/medium/high → 直接映射; max → xhigh; 未知值 → 不注入

    Priority 2: thinking.type + budget_tokens — 从 Anthropic thinking 配置推导
      adaptive → xhigh
      enabled + budget < 4000 → low
      enabled + budget < 16000 → medium
      enabled + budget >= 16000 → high
      enabled 无 budget → high
      disabled / 缺失 → 不注入
    """
    # Priority 1: explicit output_config.effort
    effort = (body.get("output_config") or {}).get("effort")
    if effort is not None:
        effort_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "xhigh",
        }
        return effort_map.get(effort)

    # Priority 2: thinking fallback
    thinking = body.get("thinking")
    if not thinking or not isinstance(thinking, dict):
        return None

    thinking_type = thinking.get("type")
    if thinking_type == "adaptive":
        return "xhigh"
    if thinking_type == "enabled":
        budget = thinking.get("budget_tokens")
        if budget is None:
            return "high"
        budget = int(budget)
        if budget < 4000:
            return "low"
        if budget < 16000:
            return "medium"
        return "high"

    # disabled 或其他 → 不注入
    return None


# ── Schema 清理 ──────────────────────────────────────────────────────

def clean_schema(schema: dict) -> dict:
    """清理 JSON Schema，移除 OpenAI 不支持的字段

    - 移除 format: "uri"（OpenAI 不支持）
    - 递归清理嵌套的 properties / items
    """
    if not isinstance(schema, dict):
        return schema

    result = dict(schema)

    # 移除 format: "uri"
    if result.get("format") == "uri":
        result.pop("format", None)

    # 递归清理 properties
    properties = result.get("properties")
    if isinstance(properties, dict):
        result["properties"] = {k: clean_schema(v) for k, v in properties.items()}

    # 递归清理 items
    items = result.get("items")
    if isinstance(items, dict):
        result["items"] = clean_schema(items)

    return result


# ── Tool Choice 映射 ────────────────────────────────────────────────

def map_tool_choice_to_chat(tool_choice) -> dict | str:
    """将 Anthropic tool_choice 映射为 OpenAI Chat Completions 格式

    Anthropic:
      "auto" / "any" / "none"           — 字符串枚举
      {"type": "auto" | "any" | "none"} — 对象形式
      {"type": "tool", "name": "<X>"}   — 强制调用特定工具

    OpenAI Chat:
      "auto" / "none" / "required"      — 字符串（注意：没有 "any"，用 "required"）
      {"type": "function", "function": {"name": "<X>"}} — 强制调用（嵌套形式）
    """
    if isinstance(tool_choice, str):
        if tool_choice == "any":
            return "required"
        return tool_choice

    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type")
        if tc_type == "any":
            return "required"
        if tc_type in ("auto", "none"):
            return tc_type
        if tc_type == "tool":
            name = tool_choice.get("name", "")
            return {"type": "function", "function": {"name": name}}

    # 未知格式 — 尽量透传
    return tool_choice


# ── System Messages 合并 ────────────────────────────────────────────

def _normalize_system_messages(messages: list[dict]) -> None:
    """将多条 system messages 合并为一条，放在 messages[0]

    OpenAI Chat 只需要一个 system message；
    Anthropic 可以有多条（含 cache_control）。
    合并时保留 cache_control（如果一致）。
    """
    system_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "system"
    ]

    if len(system_indices) == 0:
        return

    if len(system_indices) == 1:
        # 单条 system — 移到 [0] 位置
        idx = system_indices[0]
        if idx > 0:
            messages.insert(0, messages.pop(idx))
        return

    # 多条 system — 合并文本，保留 cache_control（如果一致）
    parts = []
    inherited_cc = None
    cc_conflict = False
    saw_cc = False
    saw_missing_cc = False

    for idx in system_indices:
        msg = messages[idx]
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            text = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
            if text:
                parts.append(text)

        cc = msg.get("cache_control")
        if cc is not None:
            saw_cc = True
            if inherited_cc is None:
                inherited_cc = cc
            elif inherited_cc != cc:
                cc_conflict = True
        else:
            saw_missing_cc = True

    # 移除所有 system messages
    for idx in reversed(system_indices):
        messages.pop(idx)

    if parts:
        merged = {"role": "system", "content": "\n".join(parts)}
        if not cc_conflict and not (saw_cc and saw_missing_cc) and inherited_cc:
            merged["cache_control"] = inherited_cc
        messages.insert(0, merged)


# ── 单条消息转换 ────────────────────────────────────────────────────

def _convert_content_block_to_openai(block: dict) -> list[dict]:
    """将 Anthropic content block 转为 OpenAI 格式片段

    tool_result 会产生独立的 tool role 消息（不在此函数处理），
    此函数只产生可合并到 assistant/user 消息的内容部件。
    """
    block_type = block.get("type", "")

    if block_type == "text":
        part = {"type": "text", "text": block.get("text", "")}
        if "cache_control" in block:
            part["cache_control"] = block["cache_control"]
        return [part]

    if block_type == "image":
        source = block.get("source", {})
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return [{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}]

    # thinking, redacted_thinking, tool_use, tool_result 不在此处理
    return []


def _convert_message_to_openai(role: str, content) -> list[dict]:
    """将一条 Anthropic 消息转换为 OpenAI Chat 消息列表

    可能产生多条消息（tool_result 变成独立的 tool role 消息）。
    """
    result = []

    if content is None:
        result.append({"role": role, "content": None})
        return result

    if isinstance(content, str):
        result.append({"role": role, "content": content})
        return result

    if not isinstance(content, list):
        result.append({"role": role, "content": str(content)})
        return result

    # 数组内容 — 多模态 / 工具调用
    content_parts = []
    tool_calls = []
    reasoning_parts = []
    has_tool_result = False

    for block in content:
        block_type = block.get("type", "")

        if block_type == "text":
            part = {"type": "text", "text": block.get("text", "")}
            if "cache_control" in block:
                part["cache_control"] = block["cache_control"]
            content_parts.append(part)

        elif block_type == "image":
            source = block.get("source", {})
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"}
            })

        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            })

        elif block_type == "tool_result":
            # tool_result → 独立的 tool role 消息
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

            result.append({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": result_content or "",
            })
            has_tool_result = True

        elif block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                reasoning_parts.append(thinking)

        elif block_type == "redacted_thinking":
            # redacted_thinking 不可读取，跳过
            pass

    # 组合 assistant/user 消息
    if content_parts or tool_calls:
        msg = {"role": role}

        if content_parts:
            if len(content_parts) == 1:
                # 单 text part → 简化为字符串（除非有 cache_control）
                part = content_parts[0]
                if part.get("type") == "text" and "cache_control" not in part:
                    msg["content"] = part.get("text", "")
                else:
                    msg["content"] = content_parts
            else:
                msg["content"] = content_parts
        elif tool_calls:
            msg["content"] = None
        else:
            msg["content"] = ""

        if tool_calls:
            msg["tool_calls"] = tool_calls

        if reasoning_parts and role == "assistant":
            msg["reasoning_content"] = "\n".join(reasoning_parts)

        result.append(msg)

    return result


# ── 主转换函数 ────────────────────────────────────────────────────────

def anthropic_to_chat(body: dict) -> dict:
    """将 Anthropic Messages API 请求转换为 OpenAI Chat Completions 请求。

    Args:
        body: Anthropic 格式请求体

    Returns:
        OpenAI Chat Completions 格式请求体
    """
    result = {}

    # 模型名 — 直接透传（映射由 resolver 处理）
    model = body.get("model", "")
    if model:
        result["model"] = model

    messages = []

    # system → system role message
    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            text = strip_leading_anthropic_billing_header(system)
            if text:
                messages.append({"role": "system", "content": text})
        elif isinstance(system, list):
            for msg in system:
                text = msg.get("text", "")
                text = strip_leading_anthropic_billing_header(text)
                if not text:
                    continue
                sys_msg = {"role": "system", "content": text}
                if "cache_control" in msg:
                    sys_msg["cache_control"] = msg["cache_control"]
                messages.append(sys_msg)

    # messages → 转换
    for msg in (body.get("messages") or []):
        role = msg.get("role", "user")
        content = msg.get("content")
        converted = _convert_message_to_openai(role, content)
        messages.extend(converted)

    _normalize_system_messages(messages)
    result["messages"] = messages

    # ── 参数映射 ──

    # max_tokens → max_completion_tokens (o-series) / max_tokens (其他)
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        if is_openai_o_series(model):
            result["max_completion_tokens"] = max_tokens
        else:
            result["max_tokens"] = max_tokens

    # 透传参数
    for key in ("temperature", "top_p", "stream"):
        if key in body:
            result[key] = body[key]

    # stop_sequences → stop
    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    # reasoning_effort — 仅对支持模型注入
    if model and supports_reasoning_effort(model):
        effort = resolve_reasoning_effort(body)
        if effort:
            result["reasoning_effort"] = effort

    # tools → 过滤 BatchTool + 转换
    tools = body.get("tools") or []
    if tools:
        openai_tools = []
        for tool in tools:
            # 过滤 BatchTool
            if tool.get("type") == "BatchTool":
                continue

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": clean_schema(tool.get("input_schema", {})),
                },
            }
            if "cache_control" in tool:
                openai_tool["cache_control"] = tool["cache_control"]
            openai_tools.append(openai_tool)

        if openai_tools:
            result["tools"] = openai_tools

    # tool_choice → 映射
    if "tool_choice" in body:
        result["tool_choice"] = map_tool_choice_to_chat(body["tool_choice"])

    return result
