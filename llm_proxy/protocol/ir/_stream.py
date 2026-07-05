"""IR 流式层共享工具。

从 anthropic_openai/stream.py 和 responses_chat/stream.py 提取协议无关的辅助：
- SSE 行解析（event / data / [DONE]）
- SSE 序列化（带 event 头 / 仅 data 行 / keepalive 注释）
- chunk → usage 提取
- keepalive 包装器（15s 无数据插入心跳）
- 反向解析 SSE buffer（处理多行 data 累积）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# 特殊 sentinel：[DONE] 标记
DONE_SENTINEL = object()


def parse_sse_line(line: str) -> dict | None | object:
    """解析单行 SSE，返回 dict（普通事件）、None（注释/空行）、DONE_SENTINEL。

    输入行可能是：
    - "data: {...json...}" → 返回 dict 或 DONE_SENTINEL
    - "data: [DONE]" → 返回 DONE_SENTINEL
    - "event: xxx" → 返回 None（event 头）
    - ": keepalive" → 返回 None（注释）
    - "" → 返回 None
    """
    line = line.rstrip("\r")
    if not line:
        return None
    if line.startswith(":"):
        return None  # 注释
    if line.startswith("data:"):
        data_str = line[5:].lstrip(" ")
        if data_str == "[DONE]":
            return DONE_SENTINEL
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse SSE data: {data_str[:200]}")
            return None
    # event: 行、id: 行等都忽略（外层 caller 处理 event header）
    return None


def sse_format(event_type: str, data: dict) -> bytes:
    """构造 Anthropic / Responses 风格的 SSE 事件（带 event 头）。

    规范要求：每个 data payload 顶层必须带 `type` 字段，且与 `event:` 头镜像。
    Codex / Claude Code 等客户端只解析 data 的 `type`，不认 `event:` 头；
    缺失 `type` 会被判为"未收到该事件"（如 stream closed before response.completed）。
    因此这里无条件把 `type` 注入到 data 顶层（data 已提供同名 key 时不覆盖）。
    """
    payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n".encode()


def sse_format_data_only(data: dict) -> bytes:
    """构造 Chat 风格的 SSE（仅 data 行，无 event 头）。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()


def sse_comment(line: str = "keepalive") -> bytes:
    """构造 SSE 注释行（`: keepalive\\n\\n`），用于心跳。"""
    return f": {line}\n\n".encode()


def chunk_to_usage(chunk: dict) -> dict | None:
    """从 Chat/Responses chunk 中提取 usage 字段。

    返回 {input_tokens, output_tokens, ...} 或 None。
    """
    return chunk.get("usage") or None


def extract_usage_tokens(usage: dict) -> dict:
    """从异构 usage dict 提取标准 token 计数。

    支持 Chat (prompt_tokens/completion_tokens) 和 Responses (input_tokens/output_tokens)
    和 Anthropic (input_tokens/output_tokens)。

    保留 Anthropic 的 cache token 字段（cache_read_input_tokens / cache_creation_input_tokens），
    避免流式 usage 事件丢失 cache 统计。
    """
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}

    # Responses / Anthropic 风格
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens", 0)

    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens", 0)

    result: dict = {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens)}

    # 保留 cache token（Anthropic 特有），非零时才加入避免噪声
    cache_read = usage.get("cache_read_input_tokens")
    if cache_read is not None and int(cache_read) > 0:
        result["cache_read_input_tokens"] = int(cache_read)
    cache_creation = usage.get("cache_creation_input_tokens")
    if cache_creation is not None and int(cache_creation) > 0:
        result["cache_creation_input_tokens"] = int(cache_creation)

    return result


# ── Keepalive 包装器 ────────────────────────────────────────────────


# 持有正在运行的泵任务，防止 finally 中 cancel 后因未 await 被 GC 时
# 触发 "Task was destroyed but it is pending" 警告。
# pump 在下一个事件循环 tick 收到 CancelledError 后会自然退出并触发
# done_callback 清理自身。
_running_pumps: set[asyncio.Task] = set()


async def keepalive_wrapper(
    source: AsyncIterator[bytes],
    interval: float = 15.0,
) -> AsyncIterator[bytes]:
    """包装 SSE byte 流，每 interval 秒无新数据则插入 `:keepalive\\n\\n`。

    用于反向长连接：避免 Codex / Claude Code 客户端因空闲超时断开。

    实现：后台 Task 持续从 source 读数据到 Queue；外层轮询 Queue。
    这样 wait_for 不会取消源 coroutine。
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    source_done = False

    async def _pump():
        nonlocal source_done
        try:
            async for chunk in source:
                await queue.put(("data", chunk))
        except Exception as e:  # pragma: no cover
            await queue.put(("error", e))
        finally:
            source_done = True
            await queue.put(("done", None))

    pump_task = asyncio.create_task(_pump())
    _running_pumps.add(pump_task)
    pump_task.add_done_callback(_running_pumps.discard)
    try:
        while True:
            if source_done and queue.empty():
                return
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield sse_comment("keepalive")
                continue
            if kind == "data":
                yield payload
            elif kind == "error":
                logger.error(f"keepalive_wrapper: source error: {payload}", exc_info=payload)
                return
            elif kind == "done":
                return
    finally:
        # 只 cancel，不 await。
        #
        # 当客户端断开，GeneratorExit 在此 yield 处抛出，进入 finally。
        # 如果在这里 await pump_task，CancelledError 会同步传播到 pump
        # 的 async for chunk in source → source.close() → ... →
        # resp.aiter_lines() → httpx 试图关闭 resp。但此时 async with
        # client.stream() 上下文管理器尚未退出（它在 _proxy_stream 中
        # 包裹 keepalive_wrapper 的 async for），resp 仍在活跃状态 →
        # "aclose(): asynchronous generator is already running"，
        # 且上游连接悬挂 → 30s 后 minimax timeout 返回 500。
        #
        # 只 cancel 不 await，CancelledError 在下一个事件循环 tick 才
        # 传播到 pump，此时 GeneratorExit 已经沿 _proxy_stream 的
        # async with 传播 → resp.close() 已经先执行了。httpx 对已关闭
        # 连接的 aiter_lines 清理是安全的（不操作底层 socket）。
        # pump_task 被 _running_pumps 持有生命期，完成后通过 done_
        # callback 自动清理。
        if not pump_task.done():
            pump_task.cancel()


# ── Stop reason 映射（与 _common.py 保持一致，但 IR 中心化）──


# OpenAI Chat finish_reason → IR stop_reason
FINISH_TO_STOP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}

# IR stop_reason → OpenAI Chat finish_reason
STOP_TO_FINISH = {v: k for k, v in FINISH_TO_STOP.items()}
STOP_TO_FINISH["tool_use"] = "tool_calls"  # 反向纠正（同 v）

# IR stop_reason → Anthropic stop_reason（直通）
# Anthropic 已是 IR 规范：end_turn / tool_use / max_tokens / refusal

# IR stop_reason → Responses status
STOP_TO_STATUS = {
    "end_turn": "completed",
    "tool_use": "completed",
    "max_tokens": "incomplete",
    "refusal": "incomplete",
}

INCOMPLETE_REASON = {
    "max_tokens": "max_output_tokens",
    "refusal": "content_filter",
}


def map_finish_to_stop_reason(finish_reason: str | None) -> str:
    return FINISH_TO_STOP.get(finish_reason or "", "end_turn")


def map_stop_to_finish_reason(stop_reason: str) -> str:
    return STOP_TO_FINISH.get(stop_reason, "stop")


def map_stop_to_responses_status(stop_reason: str) -> tuple[str, str | None]:
    """IR stop_reason → (status, incomplete_reason)。"""
    status = STOP_TO_STATUS.get(stop_reason, "completed")
    incomplete = INCOMPLETE_REASON.get(stop_reason) if status == "incomplete" else None
    return status, incomplete


# ── 增量输入合并 ──────────────────────────────────────────────────


class IncrementalJSONParser:
    """累积 JSON 字符串片段直到完整可解析。

    用于 tool_use arguments 分片到达的场景。
    """

    def __init__(self):
        self._buf = ""

    def feed(self, fragment: str) -> dict | None:
        self._buf += fragment
        if not self._buf.strip():
            return None
        try:
            return json.loads(self._buf)
        except json.JSONDecodeError:
            return None

    def finalize(self) -> dict:
        """流结束时调用，返回累积的 JSON（解析失败则返回 _raw 字段保留原文）。"""
        try:
            return json.loads(self._buf) if self._buf.strip() else {}
        except json.JSONDecodeError:
            return {"_raw": self._buf}

    @property
    def buffer_length(self) -> int:
        return len(self._buf)


# ── Multi-line 累积解析（data: 行可能跨多次 aiter）──


class SSELineAccumulator:
    """SSE 协议在 httpx 流中可能单次产出多字节不完整行（罕见）或单次跨多行。

    本类只关心按行输出，自动处理 \\r\\n / \\n。
    """

    _LINE_SPLIT_RE = re.compile(r"\r\n|\n|\r")

    def __init__(self):
        self._buf = ""

    def feed(self, chunk: bytes) -> list[str]:
        text = (self._buf + chunk.decode("utf-8", errors="replace"))
        parts = self._LINE_SPLIT_RE.split(text)
        # 最后一个是不完整行，保留为 buffer
        self._buf = parts[-1]
        return parts[:-1]

    def flush(self) -> list[str]:
        """流结束时返回残留 buffer（视为最后一行）。"""
        if self._buf:
            rest = [self._buf]
            self._buf = ""
            return rest
        return []
