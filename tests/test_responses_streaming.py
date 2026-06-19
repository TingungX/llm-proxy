"""Tests for Responses API streaming SSE conversion"""
import json
import pytest
from llm_proxy.protocol.responses_chat.stream import StreamState
from llm_proxy.protocol.responses_chat.request import (
    CodexToolSpec,
    convert_chunk_to_events,
    make_response_completed_event,
    make_sse_event,
)


def _parse_events(events: list[bytes]) -> list[dict]:
    result = []
    for evt in events:
        decoded = evt.decode()
        for line in decoded.split("\n"):
            if line.startswith("data: "):
                result.append(json.loads(line[6:]))
    return result


def _event_types(events: list[bytes]) -> list[str]:
    return [p.get("type", "") for p in _parse_events(events)]


class TestConvertChunkToEvents:

    def test_text_delta_conversion(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": "hel"}}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        assert len(events) >= 1
        assert b'"type": "response.output_text.delta"' in events[-1]
        assert b'"hel"' in events[-1]

    def test_finish_reason_closes_blocks(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        types = _event_types(events)
        assert "response.output_text.done" in types
        assert "response.content_part.done" in types
        assert "response.output_item.done" in types

    def test_empty_delta_no_events(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {}}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        assert len(events) == 0

    def test_tool_call_lifecycle(self):
        state = StreamState()
        chunk1 = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_abc", "function": {"name": "get_weather"}}]
            }}]
        }
        events1 = convert_chunk_to_events(chunk1, "gpt-4", state)
        types1 = _event_types(events1)
        assert "response.output_item.added" in types1

        chunk2 = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]
            }}]
        }
        events2 = convert_chunk_to_events(chunk2, "gpt-4", state)
        types2 = _event_types(events2)
        assert "response.function_call_arguments.delta" in types2

    def test_tool_call_finish_reason(self):
        state = StreamState()
        state.handle_tool_call_id(0, "call_abc", "get_weather")
        state.handle_tool_call_args_delta(0, '{"city":"NYC"}')
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        types = _event_types(events)
        assert "response.function_call_arguments.done" in types
        assert "response.output_item.done" in types


class TestMakeResponseCompletedEvent:

    def test_completed_event_format(self):
        event = make_response_completed_event("gpt-4", "resp_test123")
        assert b"data: " in event
        assert b'"type": "response.completed"' in event
        assert b'"model": "gpt-4"' in event
        assert b'"id": "resp_test123"' in event
        assert b'"status": "completed"' in event


class TestReasoningEvents:

    def test_first_reasoning_delta_sends_output_item_added(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"reasoning_content": "I need to think"}}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        types = _event_types(events)
        assert "response.output_item.added" in types
        assert "response.reasoning_summary_part.added" in types
        assert "response.reasoning_summary_text.delta" in types
        assert state.reasoning_active is True
        assert state.reasoning_part_added is True

    def test_subsequent_reasoning_delta_only_text_delta(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"reasoning_content": " more thought"}}]
        }
        state = StreamState()
        state.reasoning_active = True
        state.reasoning_part_added = True
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        for evt in events:
            payload = _parse_events([evt])[0]
            assert payload.get("type") == "response.reasoning_summary_text.delta"
            break

    def test_no_reasoning_content_produces_no_reasoning_events(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": "just text"}}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        for evt in events:
            payload = _parse_events([evt])[0]
            assert "reasoning" not in payload.get("type", "")
            break


class TestEventSequence:

    def test_response_created_sent_at_start(self):
        event = make_sse_event({
            "type": "response.created",
            "response": {"id": "resp_test", "model": "gpt-4", "status": "in_progress"},
        })
        decoded = event.decode()
        payload = {}
        for line in decoded.split("\n"):
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                break
        assert payload["type"] == "response.created"

    def test_output_item_added_on_first_text(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        types = _event_types(events)
        assert "response.output_item.added" in types
        assert "response.content_part.added" in types
        assert "response.output_text.delta" in types
        assert state.in_text_block is True

    def test_subsequent_text_no_output_item_added(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": " world"}}]
        }
        state = StreamState()
        state.in_text_block = True
        state.text_buf = "Hello"
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        for evt in events:
            payload = _parse_events([evt])[0]
            assert payload.get("type") == "response.output_text.delta"
            break


class TestThinkTagWithStreamState:
    def test_think_tag_extracted_as_reasoning(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": "<think>I need to calculate</think>def fib(n):"}}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        types = [(p.get("type"), p.get("delta", "")) for p in _parse_events(events)]
        assert any(t[0] == "response.reasoning_summary_text.delta" and "I need to calculate" in t[1] for t in types)
        assert any(t[0] == "response.output_text.delta" and "def fib(n):" in t[1] for t in types)

    def test_think_with_text_state(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": "<think>reason</think>text"
}}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        types = set(p.get("type") for p in _parse_events(events))
        assert "response.reasoning_summary_text.delta" in types
        assert "response.output_text.delta" in types
        assert "response.content_part.added" in types

    def test_think_tag_in_middle_not_recognized(self):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": "text <think>not think</think> more"}}]
        }
        state = StreamState()
        events = convert_chunk_to_events(chunk, "gpt-4", state)
        types = _event_types(events)
        assert "response.reasoning_summary_text.delta" not in types
        assert "response.output_text.delta" in types


class TestConvertToolsToChatCustomPassthrough:
    """测试 convert_tools_to_chat 对非 apply_patch 的 custom 工具透传"""

    def test_spawn_agent_passthrough(self):
        """spawn_agent 应透传为 function 类型，保留参数 schema"""
        from llm_proxy.protocol.responses_chat.request import convert_tools_to_chat
        tools = [
            {"type": "custom", "name": "spawn_agent", "description": "Spawn a sub-agent",
             "parameters": {
                 "type": "object",
                 "properties": {
                     "task_name": {"type": "string"},
                     "message": {"type": "string"},
                 },
                 "required": ["task_name"],
             }},
        ]
        result, reverse_map, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "spawn_agent"
        assert result[0]["function"]["description"] == "Spawn a sub-agent"
        assert "task_name" in result[0]["function"]["parameters"]["properties"]
        assert reverse_map["spawn_agent"] == "spawn_agent"

    def test_apply_patch_passthrough(self):
        """透传：apply_patch 转为单个 function tool + reverse_tool_map 自映射"""
        from llm_proxy.protocol.responses_chat.request import convert_tools_to_chat
        tools = [
            {"type": "custom", "name": "apply_patch", "description": "Apply a patch"},
        ]
        result, reverse_map, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "apply_patch"
        assert reverse_map["apply_patch"] == "apply_patch"

    def test_mixed_custom_and_apply_patch(self):
        """同时有 apply_patch 和 spawn_agent(namespace) 时，各自走不同路径"""
        from llm_proxy.protocol.responses_chat.request import convert_tools_to_chat
        tools = [
            {"type": "custom", "name": "apply_patch", "description": "Apply a patch"},
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Agent tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn agent",
                        "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                    }
                ],
            },
        ]
        result, reverse_map, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 2  # apply_patch + namespace 子工具
        names = [t["function"]["name"] for t in result]
        assert "apply_patch" in names
        assert "multi_agent_v1__spawn_agent" in names
        assert reverse_map["apply_patch"] == "apply_patch"
        # namespace 子工具以 function_call 格式返回给客户端，不写入 reverse_tool_map
        assert "spawn_agent" not in reverse_map

    def test_custom_tool_without_parameters(self):
        """没有 parameters 的 custom 工具应自动补全空 schema"""
        from llm_proxy.protocol.responses_chat.request import convert_tools_to_chat
        tools = [
            {"type": "custom", "name": "view_image", "description": "View image"},
        ]
        result, reverse_map, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert reverse_map["view_image"] == "view_image"

    def test_custom_tool_with_function_tools(self):
        """custom 工具和 function 工具混合时，function 工具正常处理"""
        from llm_proxy.protocol.responses_chat.request import convert_tools_to_chat
        tools = [
            {"type": "custom", "name": "spawn_agent", "description": "Spawn",
             "parameters": {"type": "object", "properties": {"task_name": {"type": "string"}}}},
            {"type": "function", "function": {
                "name": "exec_command",
                "description": "Run command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            }},
        ]
        result, reverse_map, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 2
        names = [t["function"]["name"] for t in result]
        assert "spawn_agent" in names
        assert "exec_command" in names
        assert "spawn_agent" in reverse_map
        assert "exec_command" not in reverse_map


class TestConvertInputCustomToolPassthrough:
    """测试 convert_input_to_messages 对非 apply_patch 的 custom_tool_call 历史消息处理"""

    def test_spawn_agent_custom_tool_call_in_history(self):
        """历史消息中的 spawn_agent custom_tool_call 应转为 function tool call"""
        from llm_proxy.protocol.responses_chat.request import convert_input_to_messages
        input_data = [
            {"type": "message", "role": "user", "content": "spawn an agent"},
            {"type": "custom_tool_call", "name": "spawn_agent", "call_id": "call_1",
             "input": '{"task_name": "task1", "message": "do work"}'},
            {"type": "custom_tool_call_output", "call_id": "call_1",
             "output": "Agent spawned successfully"},
        ]
        messages = convert_input_to_messages(input_data)
        # 应该有 user + assistant(tool_calls) + tool 三条消息
        assert len(messages) >= 3
        # 找 assistant 消息中的 tool_calls
        assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistant_msgs) == 1
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "spawn_agent"
        args = json.loads(tc["function"]["arguments"])
        assert args["task_name"] == "task1"
        # 找 tool 消息
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "Agent spawned successfully" in tool_msgs[0]["content"]

    def test_view_image_custom_tool_call_in_history(self):
        """历史消息中的 view_image custom_tool_call 应转为 function tool call"""
        from llm_proxy.protocol.responses_chat.request import convert_input_to_messages
        input_data = [
            {"type": "custom_tool_call", "name": "view_image", "call_id": "call_2",
             "input": '{"path": "/tmp/test.png", "detail": "high"}'},
            {"type": "custom_tool_call_output", "call_id": "call_2",
             "output": "Image loaded"},
        ]
        messages = convert_input_to_messages(input_data)
        assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistant_msgs) == 1
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "view_image"
        args = json.loads(tc["function"]["arguments"])
        assert args["path"] == "/tmp/test.png"

    def test_apply_patch_custom_tool_call_passthrough(self):
        """透传：apply_patch 的 custom_tool_call input 原样作为 function arguments"""
        from llm_proxy.protocol.responses_chat.request import convert_input_to_messages
        input_data = [
            {"type": "custom_tool_call", "name": "apply_patch", "call_id": "call_3",
             "input": "*** Begin Patch\n*** Add File: /tmp/test.txt\n+hello\n*** End Patch"},
            {"type": "custom_tool_call_output", "call_id": "call_3",
             "output": "File created"},
        ]
        messages = convert_input_to_messages(input_data)
        assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistant_msgs) == 1
        tc = assistant_msgs[0]["tool_calls"][0]
        # 透传：function name 保持 apply_patch，arguments 是原始 DSL 字符串
        assert tc["function"]["name"] == "apply_patch"
        assert "*** Begin Patch" in tc["function"]["arguments"]
        assert "/tmp/test.txt" in tc["function"]["arguments"]

    def test_custom_tool_call_with_non_string_input(self):
        """custom_tool_call 的 input 是非字符串时，应 JSON 序列化"""
        from llm_proxy.protocol.responses_chat.request import convert_input_to_messages
        input_data = [
            {"type": "custom_tool_call", "name": "spawn_agent", "call_id": "call_4",
             "input": {"task_name": "task1", "message": "hi"}},
            {"type": "custom_tool_call_output", "call_id": "call_4",
             "output": "ok"},
        ]
        messages = convert_input_to_messages(input_data)
        assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistant_msgs) == 1
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "spawn_agent"
        args = json.loads(tc["function"]["arguments"])
        assert args == {"task_name": "task1", "message": "hi"}


class TestToResponsesResponseCustomPassthrough:
    """测试 to_responses_response 对非 apply_patch 的 custom 工具反向转换"""

    def test_spawn_agent_custom_tool_call_in_response(self):
        """上游返回 spawn_agent 调用时，应生成 custom_tool_call，input 为 JSON"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_spawn1",
                        "type": "function",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": '{"task_name": "task1", "message": "hello"}',
                        },
                    }],
                },
            }],
        }
        reverse_map = {"spawn_agent": "spawn_agent"}
        result = to_responses_response(chat_body, "gpt-4", reverse_map)
        output = result.get("output", [])
        custom_items = [o for o in output if o.get("type") == "custom_tool_call"]
        assert len(custom_items) == 1
        item = custom_items[0]
        assert item["name"] == "spawn_agent"
        parsed = json.loads(item["input"])
        assert parsed["task_name"] == "task1"

    def test_apply_patch_passthrough_in_response(self):
        """透传：上游返回 apply_patch 调用，arguments 原样作为 custom_tool_call.input"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_patch1",
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "arguments": '{"input": "*** Begin Patch\\n*** Add File: /tmp/x.txt\\n+hello\\n*** End Patch"}',
                        },
                    }],
                },
            }],
        }
        reverse_map = {"apply_patch": "apply_patch"}
        result = to_responses_response(chat_body, "gpt-4", reverse_map)
        output = result.get("output", [])
        custom_items = [o for o in output if o.get("type") == "custom_tool_call"]
        assert len(custom_items) == 1
        item = custom_items[0]
        assert item["name"] == "apply_patch"
        assert "*** Begin Patch" in item["input"]
        assert "*** Add File: /tmp/x.txt" in item["input"]

    def test_mixed_apply_patch_and_custom_in_response(self):
        """透传：apply_patch 和 spawn_agent 都走 JSON 透传"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_patch1",
                            "type": "function",
                            "function": {
                                "name": "apply_patch",
                                "arguments": '{"input": "*** Begin Patch\\n*** Add File: a.txt\\n+hi\\n*** End Patch"}',
                            },
                        },
                        {
                            "id": "call_spawn1",
                            "type": "function",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": '{"task_name": "task1"}',
                            },
                        },
                    ],
                },
            }],
        }
        reverse_map = {"apply_patch": "apply_patch", "spawn_agent": "spawn_agent"}
        result = to_responses_response(chat_body, "gpt-4", reverse_map)
        output = result.get("output", [])
        custom_items = [o for o in output if o.get("type") == "custom_tool_call"]
        assert len(custom_items) == 2
        names = [i["name"] for i in custom_items]
        assert "apply_patch" in names
        assert "spawn_agent" in names
        apply_item = [i for i in custom_items if i["name"] == "apply_patch"][0]
        assert "*** Begin Patch" in apply_item["input"]
        assert "*** Add File: a.txt" in apply_item["input"]
        spawn_item = [i for i in custom_items if i["name"] == "spawn_agent"][0]
        parsed = json.loads(spawn_item["input"])
        assert parsed["task_name"] == "task1"


class TestStreamChatToResponsesEmptyChoices:
    """stream_chat_to_responses 必须容忍上游返回的 choices: [] chunk"""

    @staticmethod
    def _make_mock_resp(sse_lines: list[str]):
        """构造一个 mock httpx.Response，提供 aiter_lines"""
        import httpx

        class _MockResp:
            def __init__(self, lines):
                self._lines = iter(lines)

            async def aiter_lines(self):
                for line in self._lines:
                    yield line

        return _MockResp(sse_lines)

    def test_empty_choices_chunk_does_not_crash(self):
        import asyncio
        from llm_proxy.protocol.responses_chat.request import stream_chat_to_responses

        sse_lines = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[]}',
            '',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}',
            '',
            'data: [DONE]',
            '',
        ]
        resp = self._make_mock_resp(sse_lines)

        async def collect():
            events = []
            async for ev in stream_chat_to_responses(resp, "gpt-4", "ep-1", "gpt-4"):
                events.append(ev)
            return events

        events = asyncio.run(collect())
        assert any(b'"hi"' in e for e in events)


class TestMcpCallStreaming:
    """namespace 子工具在流式场景下应产出 function_call+namespace 事件"""

    def _ns_spec(self, name: str, namespace: str) -> CodexToolSpec:
        return CodexToolSpec(kind="namespace", name=name, namespace=namespace)

    def test_namespace_tool_call_produces_function_call_with_namespace(self):
        """namespace 子工具应产出 function_call + namespace 字段的 output_item.added"""
        nsmap = {"web_search__search": self._ns_spec("search", "web_search")}
        state = StreamState(tool_spec_map=nsmap)

        # 1) tool call id → output_item.added (type: function_call, namespace: web_search)
        events1 = state.handle_tool_call_id(0, "call_mcp1", "web_search__search")
        parsed1 = _parse_events(events1)
        added = [e for e in parsed1 if e["type"] == "response.output_item.added"]
        assert len(added) == 1
        item = added[0]["item"]
        assert item["type"] == "function_call"
        assert item["name"] == "search"  # 恢复原始子工具名
        assert item["namespace"] == "web_search"
        assert item["status"] == "in_progress"

        # 2) arguments delta → function_call_arguments.delta
        events2 = state.handle_tool_call_args_delta(0, '{"query":')
        parsed2 = _parse_events(events2)
        delta_types = [e["type"] for e in parsed2]
        assert "response.function_call_arguments.delta" in delta_types

        # 3) finish_reason → close_func_blocks → done
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }
        events3 = convert_chunk_to_events(chunk, "gpt-4", state)
        parsed3 = _parse_events(events3)
        done_types = [e["type"] for e in parsed3]
        assert "response.function_call_arguments.done" in done_types
        assert "response.output_item.done" in done_types
        done_item = [e for e in parsed3 if e["type"] == "response.output_item.done"][0]
        assert done_item["item"]["type"] == "function_call"
        assert done_item["item"]["name"] == "search"
        assert done_item["item"]["namespace"] == "web_search"
        assert done_item["item"]["status"] == "completed"

    def test_mixed_namespace_and_plain_function_call_streaming(self):
        """混合 namespace 子工具和普通 function 工具时，各自产出正确事件类型"""
        nsmap = {"web_search__search": self._ns_spec("search", "web_search")}
        state = StreamState(tool_spec_map=nsmap)

        # namespace 子工具 (index 0) — 使用限定的上游名
        events1 = state.handle_tool_call_id(0, "call_mcp1", "web_search__search")
        parsed1 = _parse_events(events1)
        added_ns = [e for e in parsed1 if e["type"] == "response.output_item.added"][0]
        assert added_ns["item"]["type"] == "function_call"
        assert added_ns["item"]["name"] == "search"
        assert added_ns["item"]["namespace"] == "web_search"

        # 普通 function 工具 (index 1)
        events2 = state.handle_tool_call_id(1, "call_fn1", "get_weather")
        parsed2 = _parse_events(events2)
        added_fn = [e for e in parsed2 if e["type"] == "response.output_item.added"][0]
        assert added_fn["item"]["type"] == "function_call"
        assert "namespace" not in added_fn["item"]

        events3 = state.handle_tool_call_args_delta(0, '{"q":')
        events4 = state.handle_tool_call_args_delta(1, '{"city":')
        events5 = state.close_func_blocks()
        parsed5 = _parse_events(events5)
        done_items = [e for e in parsed5 if e["type"] == "response.output_item.done"]
        items_with_ns = [e for e in done_items if e["item"].get("namespace")]
        items_without_ns = [e for e in done_items if not e["item"].get("namespace")]
        assert len(items_with_ns) == 1
        assert items_with_ns[0]["item"]["name"] == "search"
        assert items_with_ns[0]["item"]["namespace"] == "web_search"
        assert len(items_without_ns) == 1
        assert items_without_ns[0]["item"]["name"] == "get_weather"

    def test_namespace_with_mcp_prefix(self):
        """mcp__web_search namespace 的子工具应保留原始 namespace 名"""
        from llm_proxy.protocol.responses_chat.request import convert_tools_to_chat
        tools = [{
            "type": "namespace",
            "name": "mcp__web_search",
            "tools": [
                {"type": "function", "name": "search", "description": "Search web",
                 "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}},
                {"type": "function", "name": "fetch_url", "description": "Fetch URL",
                 "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}},
            ],
        }]
        _, _, nsmap = convert_tools_to_chat(tools)
        assert nsmap["mcp__web_search__search"].kind == "namespace"
        assert nsmap["mcp__web_search__search"].name == "search"
        assert nsmap["mcp__web_search__search"].namespace == "mcp__web_search"
        assert nsmap["mcp__web_search__fetch_url"].kind == "namespace"
        assert nsmap["mcp__web_search__fetch_url"].name == "fetch_url"
        assert nsmap["mcp__web_search__fetch_url"].namespace == "mcp__web_search"

        state = StreamState(tool_spec_map=nsmap)
        events = state.handle_tool_call_id(0, "call_ws1", "mcp__web_search__search")
        parsed = _parse_events(events)
        added = [e for e in parsed if e["type"] == "response.output_item.added"][0]
        assert added["item"]["name"] == "search"
        assert added["item"]["namespace"] == "mcp__web_search"

    def test_reverse_tool_map_takes_priority_over_tool_spec_map(self):
        """reverse_tool_map 命中时应优先走 custom_tool_call"""
        state = StreamState(
            reverse_tool_map={"search": "search"},
            tool_spec_map={"search": self._ns_spec("search", "web_search")},
        )
        events = state.handle_tool_call_id(0, "call_1", "search")
        parsed = _parse_events(events)
        added = [e for e in parsed if e["type"] == "response.output_item.added"][0]
        assert added["item"]["type"] == "custom_tool_call"

    def test_build_output_items_namespace_function_call(self):
        """_build_output_items 应为 namespace 子工具生成 function_call+namespace 的 item"""
        nsmap = {"web_search__search": self._ns_spec("search", "web_search")}
        state = StreamState(tool_spec_map=nsmap)
        state.handle_tool_call_id(0, "call_mcp1", "web_search__search")
        state.handle_tool_call_args_delta(0, '{"query": "weather"}')
        items = state._build_output_items()
        assert len(items) == 1
        assert items[0]["type"] == "function_call"
        assert items[0]["name"] == "search"
        assert items[0]["namespace"] == "web_search"
        assert items[0]["status"] == "completed"
        assert items[0]["call_id"] == "call_mcp1"
