"""IR 转换器共享工具。

从旧转换器复用的纯函数，不涉及协议特有逻辑：
- schema 清理（移除 OpenAI 不支持的 format）
- o-series 模型检测
- reasoning_effort 解析
- billing header 剥离
- usage 三级 fallback
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


# ── Schema 清理 ────────────────────────────────────────────────────────


def clean_schema(schema: dict) -> dict:
    """递归清理 JSON schema，移除 OpenAI 不支持的字段。

    当前规则：移除 `format: "uri"`，递归处理 properties 和 items。
    返回新 dict（shallow copy），不修改原 schema。
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, value in schema.items():
        if key == "format" and value == "uri":
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {k: clean_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = clean_schema(value)
        else:
            result[key] = value
    return result


# ── Reasoning effort 解析 ─────────────────────────────────────────────


_O_SERIES_RE = re.compile(r"^o\d")


def is_openai_o_series(model: str) -> bool:
    """检测 OpenAI o-series 推理模型（o1, o3, o4-mini 等）"""
    if not model or len(model) < 2:
        return False
    if model[0] != "o":
        return False
    return model[1].isdigit()


def supports_reasoning_effort(model: str) -> bool:
    """检测模型是否支持 reasoning_effort 参数。"""
    if is_openai_o_series(model):
        return True
    # GPT-5 系列
    if model.startswith("gpt-5"):
        return True
    return False


_EFFORT_MAP_OUTPUT_CONFIG = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
}


def resolve_reasoning_effort(body: dict) -> str | None:
    """从 Anthropic 请求中提取 reasoning_effort 值。

    Priority 1: body["output_config"]["effort"]
    Priority 2: body["thinking"] 配置 fallback
    """
    output_config = body.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if effort:
            mapped = _EFFORT_MAP_OUTPUT_CONFIG.get(effort)
            if mapped:
                return mapped

    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        budget = thinking.get("budget_tokens")
        if thinking_type == "adaptive":
            return "xhigh"
        if thinking_type == "enabled":
            if budget is None:
                return "high"
            if budget < 4000:
                return "low"
            if budget < 16000:
                return "medium"
            return "high"

    return None


# ── Billing header 剥离 ──────────────────────────────────────────────


_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


def strip_leading_anthropic_billing_header(text: str) -> str:
    """移除 system prompt 开头的 billing header 行。

    Claude Code 在 system 开头插入动态 billing header（含 rotating cch= 值），
    如果转发到 OpenAI Chat 的 system message，每次请求的 prompt prefix 都不同，
    导致 prefix cache 复用失效。只移除最开头的首行，后续出现的保留。
    """
    if not text.startswith(_BILLING_HEADER_PREFIX):
        return text

    line_end = len(text)
    for i, ch in enumerate(text):
        if ch == "\n" or ch == "\r":
            line_end = i
            break

    if line_end == len(text):
        return ""

    rest_start = line_end + 1
    if text[line_end] == "\r" and rest_start < len(text) and text[rest_start] == "\n":
        rest_start += 1

    rest = text[rest_start:]
    for prefix in ("\r\n", "\n", "\r"):
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
            break

    return rest


# ── Usage 字段映射 ──────────────────────────────────────────────────


def build_usage(raw_usage: dict) -> dict:
    """从任意协议的 usage 字段构建统一的 IR usage 字段。

    cache_read_input_tokens 三级 fallback：
    1. raw["cache_read_input_tokens"]（直接字段，最权威）
    2. raw["input_tokens_details"]["cached_tokens"]（Responses）
    3. raw["prompt_tokens_details"]["cached_tokens"]（Chat）
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

    # cache tokens fallback
    cached_tokens = None
    itd = raw_usage.get("input_tokens_details")
    if isinstance(itd, dict):
        ct = itd.get("cached_tokens")
        if ct and int(ct) > 0:
            cached_tokens = int(ct)
    ptd = raw_usage.get("prompt_tokens_details")
    if isinstance(ptd, dict) and cached_tokens is None:
        ct = ptd.get("cached_tokens")
        if ct and int(ct) > 0:
            cached_tokens = int(ct)

    if cached_tokens is not None:
        result["cache_read_input_tokens"] = cached_tokens

    # 直接字段覆盖
    if "cache_read_input_tokens" in raw_usage:
        result["cache_read_input_tokens"] = int(raw_usage["cache_read_input_tokens"])

    if "cache_creation_input_tokens" in raw_usage:
        result["cache_creation_input_tokens"] = int(raw_usage["cache_creation_input_tokens"])

    return result


# ── Tool choice 映射（Anthropic → Chat）─────────────────────────────


_TOOL_CHOICE_MAP = {
    "auto": "auto",
    "any": "required",
    "none": "none",
}


def map_tool_choice_to_chat(tool_choice) -> dict | str:
    """Anthropic tool_choice → Chat tool_choice。

    - "any" (str) → "required"（OpenAI 无 "any" 概念）
    - {"type": "tool", "name": "X"} → {"type": "function", "function": {"name": "X"}}
    """
    if isinstance(tool_choice, str):
        return _TOOL_CHOICE_MAP.get(tool_choice, tool_choice)
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type")
        if tc_type in _TOOL_CHOICE_MAP:
            return _TOOL_CHOICE_MAP[tc_type]
        if tc_type == "tool":
            name = tool_choice.get("name", "")
            return {"type": "function", "function": {"name": name}}
        return tool_choice
    return tool_choice


# ── JSON 辅助 ──────────────────────────────────────────────────────


def safe_json_loads(s: str | dict, default=None) -> dict | list | None:
    """解析 JSON 字符串，失败时返回 default。"""
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return default
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return default


def safe_json_dumps(obj, default: str = "{}") -> str:
    """序列化对象为 JSON 字符串，失败时返回 default。"""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return default

