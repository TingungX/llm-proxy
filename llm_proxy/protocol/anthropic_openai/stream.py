"""OpenAI Chat Completions SSE → Anthropic Messages SSE 流式转换

将上游 OpenAI Chat Completions SSE 流转换为 Anthropic Messages SSE 流，
用于跨协议代理的流式响应路径。

状态机跟踪: message_id, current_model, content_index, block 类型,
tool call 累积器, message_delta 去重。

参考: CCS (cc-switch) providers/streaming.rs
      anthropic-proxy-rs src/translate/stream.rs
"""

import json
import logging
import uuid

from llm_proxy.protocol.constants import STOP_REASON_MAP
from llm_proxy.protocol.anthropic_openai.response import _build_usage
from llm_proxy.protocol.think_tag import ThinkTagStateMachine

logger = logging.getLogger(__name__)


# ── SSE 事件构建 ────────────────────────────────────────────────────

def _sse_event(event_type: str, data: dict) -> bytes:
    """构建 Anthropic SSE 事件"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n".encode()


def _default_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0}


# ── 流式转换核心 ─────────────────────────────────────────────────────

async def create_anthropic_sse_stream(resp, model: str, on_event=None) -> bytes:
    """将 OpenAI Chat Completions SSE 流转换为 Anthropic SSE 流。

    Args:
        resp: httpx 流式响应对象（已打开的 stream）
        model: 原始模型名（用于 SSE 事件）

    Returns:
        AsyncGenerator[bytes] — 逐个产出 Anthropic SSE 事件字节串
    """
    # 状态
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    current_model = model
    has_sent_message_start = False
    next_content_index = 0

    # Think tag 状态机（处理 content 字段中的 <think> 标签）
    think = ThinkTagStateMachine()

    # 当前活跃 block
    current_block_type = None   # "thinking" | "text" | None
    current_block_index = None  # int

    # Tool call 状态
    tool_blocks: dict[int, dict] = {}  # index → {id, name, started, pending_args, anthropic_index}
    open_tool_indices: set[int] = set()  # Anthropic content indices of open tool blocks

    # message_delta 去重
    has_emitted_message_delta = False
    pending_message_delta: tuple[str | None, dict | None] | None = None  # (stop_reason, usage_json)

    # Usage 跟踪
    latest_usage: dict | None = None

    try:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue

            data_str = line[6:]

            # [DONE] 标记
            if data_str.strip() == "[DONE]":
                logger.debug("Chat SSE: [DONE] received")

                # 发出 pending message_delta（含完整 usage）
                if pending_message_delta is not None:
                    stop_reason, usage_json = pending_message_delta
                    delta_data = {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": usage_json or _default_usage(),
                    }
                    if on_event:
                        on_event("message_delta", delta_data)
                    yield _sse_event("message_delta", delta_data)

                if on_event:
                    on_event("message_stop", {})
                yield _sse_event("message_stop", {"type": "message_stop"})
                return

            # 解析 chunk
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse SSE chunk: {data_str[:100]}")
                continue

            # 提取 chunk 元数据
            chunk_id = chunk.get("id", "")
            chunk_model = chunk.get("model", "")
            if chunk_id:
                message_id = chunk_id
            if chunk_model:
                current_model = chunk_model

            # Usage 处理
            raw_usage = chunk.get("usage")
            if raw_usage:
                usage_json = _build_usage(raw_usage)
                latest_usage = usage_json
                # 更新 pending delta 的 usage
                if pending_message_delta is not None:
                    _, old_usage = pending_message_delta
                    pending_message_delta = (pending_message_delta[0], usage_json)

            # 处理 choices
            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # ── message_start ──
            if not has_sent_message_start:
                start_usage = _default_usage()
                if raw_usage:
                    start_usage = _build_usage(raw_usage)

                if on_event:
                    on_event("message_start", {
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": current_model,
                            "usage": start_usage,
                        },
                    })

                yield _sse_event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": current_model,
                        "usage": start_usage,
                    },
                })
                has_sent_message_start = True

            # ── reasoning_content → thinking block ──
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                if current_block_type != "thinking":
                    # 关闭当前 block
                    if current_block_index is not None:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": current_block_index,
                        })

                    index = next_content_index
                    next_content_index += 1
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                    current_block_type = "thinking"
                    current_block_index = index

                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": current_block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                })

            # ── content → text block（先过 think tag 状态机）──
            content = delta.get("content")
            if content:
                if think.state != "done":
                    reasoning_parts, content_parts = think.feed(content)
                    for rp in reasoning_parts:
                        if rp:
                            if current_block_type != "thinking":
                                if current_block_index is not None:
                                    yield _sse_event("content_block_stop", {
                                        "type": "content_block_stop",
                                        "index": current_block_index,
                                    })
                                index = next_content_index
                                next_content_index += 1
                                yield _sse_event("content_block_start", {
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                })
                                current_block_type = "thinking"
                                current_block_index = index

                            yield _sse_event("content_block_delta", {
                                "type": "content_block_delta",
                                "index": current_block_index,
                                "delta": {"type": "thinking_delta", "thinking": rp},
                            })
                    content = "".join(content_parts)

                if content:
                    if current_block_type != "text":
                        if current_block_index is not None:
                            yield _sse_event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": current_block_index,
                            })
                        index = next_content_index
                        next_content_index += 1
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        current_block_type = "text"
                        current_block_index = index

                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": current_block_index,
                        "delta": {"type": "text_delta", "text": content},
                    })

            # ── tool_calls → tool_use blocks ──
            tool_calls = delta.get("tool_calls") or []
            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                tc_id = tc.get("id")
                func = tc.get("function", {})

                # 新 tool call（有 id + name）→ 开始 block
                if tc_id and func.get("name"):
                    # 关闭当前非 tool block
                    if current_block_type is not None:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": current_block_index,
                        })
                        current_block_type = None
                        current_block_index = None

                    anthropic_index = next_content_index
                    next_content_index += 1
                    tool_name = func.get("name", "")

                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": anthropic_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tool_name,
                        },
                    })

                    tool_blocks[tc_index] = {
                        "id": tc_id,
                        "name": tool_name,
                        "started": True,
                        "pending_args": "",
                        "anthropic_index": anthropic_index,
                    }
                    open_tool_indices.add(anthropic_index)

                # arguments delta → input_json_delta
                args_delta = func.get("arguments", "")
                if args_delta:
                    state = tool_blocks.get(tc_index)
                    if state:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": state["anthropic_index"],
                            "delta": {"type": "input_json_delta", "partial_json": args_delta},
                        })

            # ── finish_reason → message_delta ──
            if finish_reason and finish_reason != "null" and not has_emitted_message_delta:
                # 先排干 think tag 缓冲
                remaining, to_reasoning = think.drain()
                if remaining:
                    if to_reasoning:
                        if current_block_type != "thinking":
                            if current_block_index is not None:
                                yield _sse_event("content_block_stop", {
                                    "type": "content_block_stop",
                                    "index": current_block_index,
                                })
                            index = next_content_index
                            next_content_index += 1
                            yield _sse_event("content_block_start", {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "thinking", "thinking": ""},
                            })
                            current_block_type = "thinking"
                            current_block_index = index
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {"type": "thinking_delta", "thinking": remaining},
                        })
                    else:
                        if current_block_type != "text":
                            if current_block_index is not None:
                                yield _sse_event("content_block_stop", {
                                    "type": "content_block_stop",
                                    "index": current_block_index,
                                })
                            index = next_content_index
                            next_content_index += 1
                            yield _sse_event("content_block_start", {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "text", "text": ""},
                            })
                            current_block_type = "text"
                            current_block_index = index
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {"type": "text_delta", "text": remaining},
                        })

                # 关闭所有活跃 blocks
                if current_block_type is not None and current_block_index is not None:
                    yield _sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": current_block_index,
                    })
                    current_block_type = None
                    current_block_index = None

                for anthropic_idx in list(open_tool_indices):
                    yield _sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": anthropic_idx,
                    })
                open_tool_indices.clear()

                stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

                # 缓存 message_delta，延迟到 [DONE] 发送以确保 usage 完整
                # 但如果后续没有更多数据，也需要在 [DONE] 时发出
                usage_json = latest_usage or _default_usage()
                pending_message_delta = (stop_reason, usage_json)
                # 不立即发出 — 等 [DONE] 或流结束时发出
                has_emitted_message_delta = True  # 标记已处理，后续 finish_reason 不再重复处理

    except Exception as e:
        logger.error(f"Stream conversion error: {e}", exc_info=True)
        yield _sse_event("error", {
            "type": "error",
            "error": {"type": "stream_error", "message": str(e) or type(e).__name__},
        })

    # 流异常结束 — 发出 pending delta + message_stop
    if pending_message_delta is not None:
        stop_reason, usage_json = pending_message_delta
        delta_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage_json or _default_usage(),
        }
        if on_event:
            on_event("message_delta", delta_data)
        yield _sse_event("message_delta", delta_data)
        if on_event:
            on_event("message_stop", {})
        yield _sse_event("message_stop", {"type": "message_stop"})
