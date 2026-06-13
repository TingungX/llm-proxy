"""协议无关的中间表示（IR）类型定义。

IR 是 Anthropic / Chat Completions / Responses 三种协议间的统一抽象层。
- 请求方向: client_body → IRRequest → upstream_body
- 响应方向: upstream_body → IRResponse → client_body
- 流式方向: upstream SSE → IRStreamEvent → client SSE

所有类型用 dataclass，零外部依赖。协议特有字段通过 `extensions: dict` 透传，
不在核心字段里加特例字段（如 apply_patch 的 reverse_tool_map）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


# ────────────────────────────────────────────────────────────────────
# Content Blocks
# ────────────────────────────────────────────────────────────────────


@dataclass
class IRTextBlock:
    type: str = "text"
    text: str = ""
    cache_control: dict | None = None


@dataclass
class IRImageBlock:
    type: str = "image"
    base64_data: str = ""
    media_type: str = "image/png"


@dataclass
class IRThinkingBlock:
    type: str = "thinking"
    thinking: str = ""
    signature: str | None = None


@dataclass
class IRRedactedThinkingBlock:
    type: str = "redacted_thinking"
    data: str = ""


@dataclass
class IRToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class IRToolResultBlock:
    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False


# Discriminated union
IRContentBlock = Union[
    IRTextBlock,
    IRImageBlock,
    IRThinkingBlock,
    IRRedactedThinkingBlock,
    IRToolUseBlock,
    IRToolResultBlock,
]


# ────────────────────────────────────────────────────────────────────
# Messages
# ────────────────────────────────────────────────────────────────────


@dataclass
class IRMessage:
    role: str = ""  # "system" | "user" | "assistant" | "tool"
    # str 表示纯文本消息（兼容单 text block 简化形式）；
    # list 表示带 content block 数组的复杂消息（tool_use/thinking 等）。
    content: Union[str, list[IRContentBlock]] = ""
    name: str | None = None  # function 调用返回结果的来源工具名（Responses 特定）


# ────────────────────────────────────────────────────────────────────
# Tool Definitions
# ────────────────────────────────────────────────────────────────────


@dataclass
class IRToolDef:
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    cache_control: dict | None = None
    # 协议特有扩展：
    #   - Anthropic: "strict": bool
    #   - Responses: "defer_loading": bool


# ────────────────────────────────────────────────────────────────────
# Request
# ────────────────────────────────────────────────────────────────────


@dataclass
class IRRequest:
    """统一的请求 IR。

    协议特有字段通过 `extensions` 透传，不在核心字段里硬编码。已知扩展键：

    Anthropic 特有：
      - "thinking_config": {"type": "enabled"|"adaptive"|"disabled", "budget_tokens": int}
      - "output_config": {"effort": "low"|"medium"|"high"|"max"}
      - "metadata": dict  # Anthropic 顶层 metadata 字段
      - "top_k": int
      - "system_cache_control": dict  # Anthropic system 字段带 cache_control 时
      - "raw_tool_choice": dict  # Anthropic tool_choice 的原始 dict 形式

    Chat 特有：
      - "stream_options": dict  # e.g. {"include_usage": true}
      - "response_format": dict
      - "logprobs": bool
      - "n": int
      - "presence_penalty": float
      - "frequency_penalty": float
      - "seed": int
      - "user": str

    Responses 特有：
      - "reverse_tool_map": dict[str, str]  # 工具名 → 原始 custom tool 名（apply_patch 用）
      - "namespace_map": dict[str, str]  # 子工具名 → namespace 名
      - "instructions": str  # Responses 顶层 instructions（与 system_prompt 类似但语义不同）
      - "parallel_tool_calls": bool
      - "truncation": str
      - "store": bool
    """

    model: str = ""
    messages: list[IRMessage] = field(default_factory=list)
    system_prompt: str | None = None
    tools: list[IRToolDef] | None = None
    tool_choice: Union[str, dict, None] = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    reasoning_effort: str | None = None
    stop_sequences: list[str] | None = None

    extensions: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Response
# ────────────────────────────────────────────────────────────────────


@dataclass
class IRResponse:
    """统一的非流式响应 IR。"""

    id: str = ""
    model: str = ""
    content_blocks: list[IRContentBlock] = field(default_factory=list)
    # "end_turn" | "tool_use" | "max_tokens" | "refusal"
    stop_reason: str = "end_turn"
    stop_sequence: str | None = None
    # {"input_tokens": int, "output_tokens": int, "cache_read_input_tokens"?: int, "cache_creation_input_tokens"?: int}
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    extensions: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Stream Events
# ────────────────────────────────────────────────────────────────────


@dataclass
class IRStreamEvent:
    """统一的流式事件 IR。

    type 取值与 data 字段约定：
      - "message_start":   {"id": str, "model": str}
      - "text_delta":      {"text": str}
      - "thinking_delta":  {"thinking": str}
      - "tool_use_start":  {"id": str, "name": str}
      - "tool_use_delta":  {"id": str, "arguments_delta": str}
      - "tool_use_end":    {"id": str, "input": dict}  # 累计完成的 input（可选）
      - "message_stop":    {"stop_reason": str}
      - "usage":           {"input_tokens": int, "output_tokens": int, ...}
      - "error":           {"message": str, "code"?: str}
      - "keepalive":       {}  # 心跳
    """

    type: str = ""
    data: dict = field(default_factory=dict)

