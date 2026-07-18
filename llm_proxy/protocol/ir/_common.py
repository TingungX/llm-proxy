"""IR 转换器共享工具。

从旧转换器复用的纯函数，不涉及协议特有逻辑：
- schema 清理（移除 OpenAI 不支持的 format）
- o-series 模型检测
- reasoning_effort 解析
- billing header 剥离
- usage 三级 fallback
- base64 图片检测（tool result 中嵌入的图片数据）
"""

from __future__ import annotations

import base64
import json
import logging
import re
from io import BytesIO
from typing import Any

from PIL import Image

from llm_proxy.protocol.effort_mapping import apply_effort_mapping

logger = logging.getLogger(__name__)


# ── Schema 清理 ────────────────────────────────────────────────────────


_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "minLength", "maxLength",
    "minItems", "maxItems",
    "pattern", "format",
    "default", "examples",
})


def clean_schema(schema: dict) -> dict:
    """递归清理 JSON schema，移除上游不支持的约束字段。

    移除规则：
    - format: "uri" 及其他 format 值（部分上游不支持）
    - minLength / maxLength / minItems / maxItems / pattern / default / examples
    - 递归处理 properties 和 items
    返回新 dict（shallow copy），不修改原 schema。
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
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


def resolve_reasoning_effort(body: dict, mapping_config: dict | None = None) -> str | None:
    """从 Anthropic 请求中提取 reasoning_effort 值。

    Priority 1: body["output_config"]["effort"]
    Priority 2: body["thinking"] 配置 fallback

    Args:
        body: Anthropic 格式请求体
        mapping_config: 全局 thinking_effort_mapping 配置；None 时使用默认兜底
    """
    output_config = body.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if effort:
            mapped = apply_effort_mapping(effort, mapping_config)
            if mapped:
                return mapped

    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        budget = thinking.get("budget_tokens")
        if thinking_type == "adaptive":
            return apply_effort_mapping("xhigh", mapping_config)
        if thinking_type == "enabled":
            if budget is None:
                return apply_effort_mapping("high", mapping_config)
            if budget < 4000:
                return apply_effort_mapping("low", mapping_config)
            if budget < 16000:
                return apply_effort_mapping("medium", mapping_config)
            return apply_effort_mapping("high", mapping_config)

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
    """任意协议 tool_choice → IR 规范形式（与 Chat 形式一致）。

    IR 规范形式：
    - str: "auto" | "required" | "none"
    - dict: {"type": "function", "function": {"name": "X"}}

    转换规则：
    - "any" (str/dict.type) → "required"（OpenAI 无 "any" 概念）
    - {"type": "tool", "name": "X"} (Anthropic) → {"type": "function", "function": {"name": "X"}}
    - {"type": "function", "name": "X"} (Responses 扁平形式) → {"type": "function", "function": {"name": "X"}}
    - {"type": "function", "function": {...}} (Chat 嵌套形式) → 透传
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
        if tc_type == "function":
            # 同时接受 Chat 嵌套形式与 Responses 扁平形式，统一到 Chat 嵌套形式
            if "function" in tool_choice and isinstance(tool_choice["function"], dict):
                return tool_choice
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


# ── base64 图片检测 ──────────────────────────────────────────────────


_B64_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)


def detect_base64_image(text: str) -> tuple[str | None, str | None]:
    """检测文本中的 base64 编码图片。

    工具调用的结果可能包含 base64 编码的图片数据（如模型通过 exec/read 读取图片后返回）。
    如果不处理，这些 base64 数据会被上游当作普通文本计入 token，导致 token 暴涨。
    此函数检测并提取图片数据，返回 (media_type, base64_data) 或 (None, None)。

    Args:
        text: 可能包含 base64 图片的文本

    Returns:
        (media_type, base64_data): 如果检测到图片
        (None, None): 如果不是图片
    """
    if not text or len(text) < 500:
        return None, None

    raw = text.strip()

    # 1. 检查 data:image/...;base64,... 格式（已包装的 data URL）
    if raw.startswith("data:image/"):
        try:
            header, _, b64_data = raw.partition(",")
            if not b64_data:
                return None, None
            media_type = header.replace("data:", "").split(";")[0]
            padding = 4 - len(b64_data) % 4
            if padding != 4:
                b64_data += "=" * padding
            data = base64.b64decode(b64_data)
            img = Image.open(BytesIO(data))
            img.verify()
            return media_type, b64_data.rstrip("=")
        except Exception:
            return None, None

    # 2. 检查纯 base64 字符串（无包装）
    b64_chars = sum(1 for c in raw if c in _B64_ALPHABET)
    if b64_chars < len(raw) * 0.85 or len(raw) < 1000:
        return None, None

    try:
        b64_data = raw
        padding = 4 - len(b64_data) % 4
        if padding != 4:
            b64_data += "=" * padding
        data = base64.b64decode(b64_data)
        img = Image.open(BytesIO(data))
        fmt = img.format or "PNG"
        media_type = f"image/{fmt.lower().replace('jpeg', 'jpg')}"
        return media_type, raw
    except Exception:
        return None, None


def extract_tool_result_image(
    content: str,
) -> tuple[str | None, str | None, str]:
    """从 tool result content 中检测 base64 图片，返回占位符。

    Returns:
        (media_type, b64_data, replacement_content):
            media_type/b64_data 为 None 时，replacement_content 为原始 content
            检测到图片时，replacement_content 为 "[image: {media_type}]"
    """
    media_type, b64_data = detect_base64_image(content)
    if b64_data:
        return media_type, b64_data, f"[image: {media_type}]"
    return None, None, content

