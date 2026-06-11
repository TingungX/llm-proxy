import json
import logging
import uuid

from llm_proxy.protocol.responses_chat.tool_replacement import reverse_tool_args_to_apply_patch, ReverseConversionError
from llm_proxy.protocol.think_tag import ThinkTagStateMachine

logger = logging.getLogger(__name__)


def _make_sse_event(data: dict) -> bytes:
    event_type = data.get("type", "")
    payload = json.dumps(data, ensure_ascii=False)
    if event_type:
        return f"event: {event_type}\ndata: {payload}\n\n".encode()
    return f"data: {payload}\n\n".encode()


def _gen_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:24]


class StreamState:
    def __init__(self, response_id: str | None = None, reverse_tool_map: dict[str, str] | None = None, namespace_map: dict[str, str] | None = None):
        self.response_id: str = response_id or _gen_id("resp_")
        self.seq: int = 0
        self.reasoning_active: bool = False
        self.reasoning_item_id: str = _gen_id("rs_")
        self.reasoning_part_added: bool = False
        self.reasoning_buf: str = ""
        self.reasoning_index: int = 0
        self.in_text_block: bool = False
        self.current_msg_id: str = _gen_id("msg_")
        self.text_buf: str = ""
        self.in_func_block: bool = False
        self.func_args_buf: dict[int, str] = {}
        self.func_names: dict[int, str] = {}
        self.func_call_ids: dict[int, str] = {}
        self.func_item_added: dict[int, bool] = {}
        self.reverse_tool_map: dict[str, str] = reverse_tool_map or {}
        self.namespace_map: dict[str, str] = namespace_map or {}
        self.think: ThinkTagStateMachine = ThinkTagStateMachine()
        self.finish_reason: str | None = None

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _text_output_index(self) -> int:
        return 1 if self.reasoning_part_added else 0

    def _func_output_index(self, tc_index: int) -> int:
        offset = 1 if self.reasoning_part_added else 0
        offset += 1 if (self.in_text_block or self.text_buf) else 0
        return tc_index + offset

    def handle_reasoning_delta(self, text: str) -> list[bytes]:
        events: list[bytes] = []
        if not self.reasoning_active:
            self.reasoning_active = True
            events.append(_make_sse_event({
                "type": "response.output_item.added",
                "output_index": self.reasoning_index,
                "sequence_number": self._next_seq(),
                "item": {
                    "id": self.reasoning_item_id,
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                },
            }))
        if not self.reasoning_part_added:
            self.reasoning_part_added = True
            events.append(_make_sse_event({
                "type": "response.reasoning_summary_part.added",
                "item_id": self.reasoning_item_id,
                "output_index": self.reasoning_index,
                "summary_index": 0,
                "sequence_number": self._next_seq(),
                "part": {"type": "summary_text", "text": ""},
            }))
        self.reasoning_buf += text
        events.append(_make_sse_event({
            "type": "response.reasoning_summary_text.delta",
            "item_id": self.reasoning_item_id,
            "output_index": self.reasoning_index,
            "summary_index": 0,
            "sequence_number": self._next_seq(),
            "delta": text,
        }))
        return events

    def close_reasoning_block(self) -> list[bytes]:
        if not self.reasoning_active:
            return []
        events: list[bytes] = []
        events.append(_make_sse_event({
            "type": "response.reasoning_summary_text.done",
            "item_id": self.reasoning_item_id,
            "output_index": self.reasoning_index,
            "summary_index": 0,
            "sequence_number": self._next_seq(),
            "delta": self.reasoning_buf,
        }))
        events.append(_make_sse_event({
            "type": "response.reasoning_summary_part.done",
            "item_id": self.reasoning_item_id,
            "output_index": self.reasoning_index,
            "summary_index": 0,
            "sequence_number": self._next_seq(),
            "part": {"type": "summary_text", "text": self.reasoning_buf},
        }))
        events.append(_make_sse_event({
            "type": "response.output_item.done",
            "output_index": self.reasoning_index,
            "sequence_number": self._next_seq(),
            "item": {
                "id": self.reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": self.reasoning_buf}],
            },
        }))
        self.reasoning_active = False
        return events

    def handle_content_delta(self, text: str) -> list[bytes]:
        events: list[bytes] = []
        if self.reasoning_active:
            events.extend(self.close_reasoning_block())
        if not self.in_text_block:
            self.in_text_block = True
            output_index = self._text_output_index()
            self.current_msg_id = f"msg_{self.response_id}_{output_index}"
            events.append(_make_sse_event({
                "type": "response.output_item.added",
                "output_index": output_index,
                "sequence_number": self._next_seq(),
                "item": {
                    "id": self.current_msg_id,
                    "type": "message",
                    "status": "in_progress",
                    "content": [],
                    "role": "assistant",
                },
            }))
            events.append(_make_sse_event({
                "type": "response.content_part.added",
                "item_id": self.current_msg_id,
                "output_index": output_index,
                "content_index": 0,
                "sequence_number": self._next_seq(),
                "part": {"type": "output_text", "text": ""},
            }))
        output_index = self._text_output_index()
        self.text_buf += text
        events.append(_make_sse_event({
            "type": "response.output_text.delta",
            "item_id": self.current_msg_id,
            "output_index": output_index,
            "content_index": 0,
            "sequence_number": self._next_seq(),
            "delta": text,
        }))
        return events

    def close_text_block(self) -> list[bytes]:
        if not self.in_text_block:
            return []
        events: list[bytes] = []
        output_index = self._text_output_index()
        events.append(_make_sse_event({
            "type": "response.output_text.done",
            "item_id": self.current_msg_id,
            "output_index": output_index,
            "content_index": 0,
            "sequence_number": self._next_seq(),
            "delta": self.text_buf,
        }))
        events.append(_make_sse_event({
            "type": "response.content_part.done",
            "item_id": self.current_msg_id,
            "output_index": output_index,
            "content_index": 0,
            "sequence_number": self._next_seq(),
            "part": {"type": "output_text", "text": self.text_buf},
        }))
        events.append(_make_sse_event({
            "type": "response.output_item.done",
            "output_index": output_index,
            "sequence_number": self._next_seq(),
            "item": {
                "id": self.current_msg_id,
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": self.text_buf}],
                "role": "assistant",
            },
        }))
        self.in_text_block = False
        return events

    def handle_tool_call_id(self, idx: int, tc_id: str, name: str) -> list[bytes]:
        events: list[bytes] = []
        if self.reasoning_active:
            events.extend(self.close_reasoning_block())
        if self.in_text_block:
            events.extend(self.close_text_block())
        self.func_call_ids[idx] = tc_id
        self.func_names[idx] = name
        self.in_func_block = True
        self.func_args_buf.setdefault(idx, "")
        output_index = self._func_output_index(idx)
        downstream_name = self.reverse_tool_map.get(name)
        server_label = self.namespace_map.get(name)

        if downstream_name is not None:
            # 1) reverse_tool_map 命中 → custom_tool_call
            item = {
                "id": f"fc_{tc_id}",
                "type": "custom_tool_call",
                "status": "in_progress",
                "call_id": tc_id,
                "name": downstream_name,
                "input": "",
            }
        elif server_label is not None:
            # 2) namespace 子工具 → function_call + namespace
            item = {
                "id": f"fc_{tc_id}",
                "type": "function_call",
                "status": "in_progress",
                "call_id": tc_id,
                "name": name,
                "namespace": server_label,
                "arguments": "",
            }
        else:
            # 3) 普通函数 → function_call
            item = {
                "id": f"fc_{tc_id}",
                "type": "function_call",
                "status": "in_progress",
                "call_id": tc_id,
                "name": name,
                "arguments": "",
            }

        events.append(_make_sse_event({
            "type": "response.output_item.added",
            "output_index": output_index,
            "sequence_number": self._next_seq(),
            "item": item,
        }))
        self.func_item_added[idx] = True
        return events

    def handle_tool_call_args_delta(self, idx: int, args_delta: str) -> list[bytes]:
        self.func_args_buf[idx] = self.func_args_buf.get(idx, "") + args_delta
        name = self.func_names.get(idx, "")
        if name in self.reverse_tool_map:
            return []  # custom_tool_call 不流式 delta
        output_index = self._func_output_index(idx)
        item_id = f"fc_{self.func_call_ids[idx]}"
        server_label = self.namespace_map.get(name)
        # namespace 子工具也使用标准 function_call_arguments 事件（Codex 不认 mcp_call 事件）
        event_type = "response.function_call_arguments.delta"
        return [_make_sse_event({
            "type": event_type,
            "item_id": item_id,
            "output_index": output_index,
            "sequence_number": self._next_seq(),
            "delta": args_delta,
        })]

    def close_func_blocks(self) -> list[bytes]:
        if not self.in_func_block:
            return []
        events: list[bytes] = []
        for idx in sorted(self.func_args_buf.keys()):
            args = self.func_args_buf[idx] or "{}"
            call_id = self.func_call_ids[idx]
            name = self.func_names[idx]
            output_index = self._func_output_index(idx)
            downstream_name = self.reverse_tool_map.get(name)

            if downstream_name is not None:
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    # arguments 被截断（如上游模型输出 token 超限）
                    # 不发 custom_tool_call，改发 text message，避免 Codex 执行无效 apply_patch 死循环
                    logger.warning(
                        "Truncated tool call arguments for %s (call_id=%s), "
                        "falling back to text message",
                        name, call_id,
                    )
                    truncated_msg = (
                        f"[Output truncated] The tool call {name} was interrupted "
                        f"because the arguments were too long and got truncated. "
                        f"Consider using append_to_file to write content in smaller chunks."
                    )
                    # 发 text message 而非 custom_tool_call，这样 Codex 不会执行 apply_patch
                    events.append(_make_sse_event({
                        "type": "response.output_text.delta",
                        "item_id": f"msg_{call_id}",
                        "output_index": output_index,
                        "delta": truncated_msg,
                    }))
                    events.append(_make_sse_event({
                        "type": "response.output_text.done",
                        "item_id": f"msg_{call_id}",
                        "output_index": output_index,
                        "sequence_number": self._next_seq(),
                    }))
                    events.append(_make_sse_event({
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "sequence_number": self._next_seq(),
                        "item": {
                            "id": f"msg_{call_id}",
                            "type": "message",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": truncated_msg}],
                        },
                    }))
                    continue

                if downstream_name == "apply_patch":
                    # apply_patch：需要将标准文件工具参数转换为 apply_patch DSL
                    try:
                        input_text = reverse_tool_args_to_apply_patch(name, parsed_args)
                    except ReverseConversionError as exc:
                        # 反向转换失败 → 发 text message 而非 custom_tool_call，避免 Codex 执行无效 apply_patch
                        logger.warning("Reverse conversion failed: %s", exc)
                        error_msg = (
                            f"Tool call {name} failed: {exc.reason}. {exc.detail}"
                        )
                        # 发 text message，Codex 不会执行 apply_patch，模型收到错误文本后可修正重试
                        events.append(_make_sse_event({
                            "type": "response.output_text.delta",
                            "item_id": f"msg_{call_id}",
                            "output_index": output_index,
                            "delta": error_msg,
                        }))
                        events.append(_make_sse_event({
                            "type": "response.output_text.done",
                            "item_id": f"msg_{call_id}",
                            "output_index": output_index,
                            "sequence_number": self._next_seq(),
                        }))
                        events.append(_make_sse_event({
                            "type": "response.output_item.done",
                            "output_index": output_index,
                            "sequence_number": self._next_seq(),
                            "item": {
                                "id": f"msg_{call_id}",
                                "type": "message",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": error_msg}],
                            },
                        }))
                        continue
                else:
                    # 其他 custom 工具（spawn_agent, view_image 等）：
                    # 参数本身就是 JSON，直接序列化为 input
                    input_text = json.dumps(parsed_args, ensure_ascii=False)

                events.append(_make_sse_event({
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": f"fc_{call_id}",
                    "call_id": call_id,
                    "delta": input_text,
                }))
                events.append(_make_sse_event({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "sequence_number": self._next_seq(),
                    "item": {
                        "id": f"fc_{call_id}",
                        "type": "custom_tool_call",
                        "name": downstream_name,
                        "status": "completed",
                        "call_id": call_id,
                        "input": input_text,
                    },
                }))
            else:
                server_label = self.namespace_map.get(name)
                if server_label is not None:
                    # 2) namespace 子工具 → function_call + namespace
                    events.append(_make_sse_event({
                        "type": "response.function_call_arguments.done",
                        "item_id": f"fc_{call_id}",
                        "output_index": output_index,
                        "sequence_number": self._next_seq(),
                        "arguments": args,
                    }))
                    events.append(_make_sse_event({
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "sequence_number": self._next_seq(),
                        "item": {
                            "id": f"fc_{call_id}",
                            "type": "function_call",
                            "name": name,
                            "namespace": server_label,
                            "arguments": args,
                            "status": "completed",
                            "call_id": call_id,
                        },
                    }))
                else:
                    # 3) 普通函数 → function_call
                    events.append(_make_sse_event({
                        "type": "response.function_call_arguments.done",
                        "item_id": f"fc_{call_id}",
                        "output_index": output_index,
                        "sequence_number": self._next_seq(),
                        "arguments": args,
                    }))
                    events.append(_make_sse_event({
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "sequence_number": self._next_seq(),
                        "item": {
                            "id": f"fc_{call_id}",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": args,
                            "call_id": call_id,
                            "name": name,
                        },
                    }))
        self.in_func_block = False
        return events

    def flush_think_tag_buf(self) -> list[bytes]:
        remaining, to_reasoning = self.think.drain()
        if not remaining and not to_reasoning:
            return []
        if to_reasoning:
            return self.handle_reasoning_delta(remaining)
        return self.handle_content_delta(remaining)

    def generate_completed_events(
        self,
        model: str,
        response_id: str,
        original_request: dict | None = None,
        usage: dict | None = None,
    ) -> list[bytes]:
        events: list[bytes] = []
        events.extend(self.flush_think_tag_buf())
        if self.reasoning_active:
            events.extend(self.close_reasoning_block())
        if self.in_text_block:
            events.extend(self.close_text_block())
        if self.in_func_block:
            events.extend(self.close_func_blocks())
        output_items = self._build_output_items()
        response_obj: dict = {
            "id": response_id,
            "object": "response",
            "model": model,
            "status": "completed",
        }
        if output_items:
            response_obj["output"] = output_items
        if usage:
            u = dict(usage)
            if "total_tokens" not in u:
                u["total_tokens"] = u.get("input_tokens", 0) + u.get("output_tokens", 0)
            response_obj["usage"] = u

        if original_request:
            _ECHO_FIELDS = [
                "instructions", "max_output_tokens", "model",
                "parallel_tool_calls", "previous_response_id",
                "reasoning", "temperature", "tool_choice",
                "tools", "top_p", "metadata",
            ]
            for field in _ECHO_FIELDS:
                if field in original_request and original_request[field] is not None:
                    response_obj[field] = original_request[field]

        events.append(_make_sse_event({
            "type": "response.completed",
            "sequence_number": self._next_seq(),
            "response": response_obj,
        }))
        return events

    def _build_output_items(self) -> list[dict]:
        items: list[dict] = []
        if self.reasoning_active or self.reasoning_part_added or self.reasoning_buf:
            items.append({
                "id": self.reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": self.reasoning_buf}],
            })
        if self.in_text_block or self.text_buf:
            items.append({
                "id": self.current_msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.text_buf}],
            })
        for idx in sorted(self.func_args_buf.keys()):
            call_id = self.func_call_ids[idx]
            name = self.func_names[idx]
            args = self.func_args_buf[idx] or "{}"
            downstream_name = self.reverse_tool_map.get(name)
            if downstream_name is not None:
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    # arguments 被截断 → 发 message 而非 custom_tool_call
                    logger.warning(
                        "Truncated tool call arguments for %s (call_id=%s), "
                        "falling back in output items",
                        name, call_id,
                    )
                    truncated_msg = (
                        f"[Output truncated] The tool call {name} was interrupted "
                        f"because the arguments were too long and got truncated. "
                        f"Consider using append_to_file to write content in smaller chunks."
                    )
                    items.append({
                        "id": f"msg_{call_id}",
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": truncated_msg}],
                    })
                    continue
                if downstream_name == "apply_patch":
                    try:
                        input_text = reverse_tool_args_to_apply_patch(name, parsed_args)
                    except ReverseConversionError as exc:
                        logger.warning("Reverse conversion failed: %s", exc)
                        error_msg = f"Tool call {name} failed: {exc.reason}. {exc.detail}"
                        items.append({
                            "id": f"msg_{call_id}",
                            "type": "message",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": error_msg}],
                        })
                        continue
                else:
                    # 其他 custom 工具：参数直接 JSON 序列化
                    input_text = json.dumps(parsed_args, ensure_ascii=False)
                items.append({
                    "id": f"fc_{call_id}",
                    "type": "custom_tool_call",
                    "name": downstream_name,
                    "status": "completed",
                    "call_id": call_id,
                    "input": input_text,
                })
            else:
                server_label = self.namespace_map.get(name)
                if server_label is not None:
                    # 2) namespace 子工具 → function_call + namespace
                    items.append({
                        "id": f"fc_{call_id}",
                        "type": "function_call",
                        "name": name,
                        "namespace": server_label,
                        "arguments": args,
                        "status": "completed",
                        "call_id": call_id,
                    })
                else:
                    # 3) 普通函数 → function_call
                    items.append({
                        "id": f"fc_{call_id}",
                        "type": "function_call",
                        "status": "completed",
                        "arguments": args,
                        "call_id": call_id,
                        "name": name,
                    })
        return items
