"""Tests for Chat Completions → Responses API conversion"""
import asyncio
import json
import pytest
from llm_proxy.protocol.responses_chat.response import (
    convert_chat_to_responses_request,
    convert_responses_to_chat_response,
)
from llm_proxy.protocol.responses_chat.request import (
    CodexToolSpec,
    convert_input_to_messages,
    convert_tools_to_chat,
)


class TestConvertChatToResponsesRequest:
    """Test Chat Completions → Responses request conversion"""

    def test_basic_messages_conversion(self):
        """Test basic messages to input conversion"""
        chat_body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }

        result = convert_chat_to_responses_request(chat_body)

        assert result["model"] == "gpt-4"
        assert result["input"] == "Hello"
        assert "instructions" not in result

    def test_system_message_becomes_instructions(self):
        """Test system message is extracted as instructions"""
        chat_body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"}
            ]
        }

        result = convert_chat_to_responses_request(chat_body)

        assert result["instructions"] == "You are helpful"
        assert result["input"] == "Hi"

    def test_max_tokens_to_max_output_tokens(self):
        """Test max_tokens is renamed to max_output_tokens"""
        chat_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100
        }

        result = convert_chat_to_responses_request(chat_body)

        assert result["max_output_tokens"] == 100
        assert "max_tokens" not in result

    def test_temperature_and_top_p_preserved(self):
        """Test temperature and top_p are preserved"""
        chat_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9
        }

        result = convert_chat_to_responses_request(chat_body)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_stream_preserved(self):
        """Test stream flag is preserved"""
        chat_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        }

        result = convert_chat_to_responses_request(chat_body)

        assert result["stream"] is True

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation conversion"""
        chat_body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        }

        result = convert_chat_to_responses_request(chat_body)

        # 应该保留完整对话历史
        assert isinstance(result["input"], list)
        assert len(result["input"]) == 3


class TestConvertResponsesToChatResponse:
    """Test Responses → Chat Completions response conversion"""

    def test_basic_response_conversion(self):
        """Test basic output to choices conversion"""
        responses_body = {
            "id": "resp_123",
            "model": "gpt-4",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}]
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }
        }

        result = convert_responses_to_chat_response(responses_body, "gpt-4")

        assert result["id"] == "resp_123"
        assert result["model"] == "gpt-4"
        assert result["object"] == "chat.completion"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_empty_output(self):
        """Test empty output handling"""
        responses_body = {
            "id": "resp_123",
            "model": "gpt-4",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": []
            }],
            "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}
        }

        result = convert_responses_to_chat_response(responses_body, "gpt-4")

        assert result["choices"][0]["message"]["content"] == ""


class TestConvertToolsToChat:
    """Test Responses tools -> Chat tools conversion"""

    def test_basic_function_tool(self):
        tools = [{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }]
        result, custom, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["parameters"]["required"] == ["city"]

    def test_missing_required_field(self):
        tools = [{
            "type": "function",
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}}
            }
        }]
        result, custom, nsmap = convert_tools_to_chat(tools)
        assert result[0]["function"]["parameters"]["required"] == []

    def test_missing_properties_field(self):
        tools = [{
            "type": "function",
            "name": "ping",
            "parameters": {"type": "object", "required": []}
        }]
        result, custom, nsmap = convert_tools_to_chat(tools)
        assert result[0]["function"]["parameters"]["properties"] == {}

    def test_fully_missing_parameters(self):
        tools = [{"type": "function", "name": "ping"}]
        result, custom, nsmap = convert_tools_to_chat(tools)
        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}, "required": []}

    def test_converts_web_search_tool(self):
        tools = [
            {"type": "web_search"},
            {"type": "function", "name": "read_file"}
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "read_file"

    def test_expands_namespace_tool(self):
        tools = [
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Agent tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn a sub-agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"}
                            },
                            "required": ["message"]
                        }
                    }
                ]
            }
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "multi_agent_v1__spawn_agent"
        # namespace 子工具不写入 reverse_tool_map
        assert "multi_agent_v1__spawn_agent" not in rmap
        # namespace 子工具写入 tool_spec_map，含 kind/name/namespace
        assert nsmap["multi_agent_v1__spawn_agent"].kind == "namespace"
        assert nsmap["multi_agent_v1__spawn_agent"].name == "spawn_agent"
        assert nsmap["multi_agent_v1__spawn_agent"].namespace == "multi_agent_v1"

    def test_namespace_mcp_prefix_preserves_original_name(self):
        """namespace name 为 mcp__web_search 时，namespace_map 保留原始名"""
        tools = [
            {
                "type": "namespace",
                "name": "mcp__web_search",
                "tools": [
                    {"type": "function", "name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}, "required": []}},
                    {"type": "function", "name": "fetch_url", "description": "Fetch URL", "parameters": {"type": "object", "properties": {}, "required": []}},
                ],
            }
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "mcp__web_search__search"
        assert result[1]["function"]["name"] == "mcp__web_search__fetch_url"
        assert nsmap["mcp__web_search__search"].kind == "namespace"
        assert nsmap["mcp__web_search__search"].name == "search"
        assert nsmap["mcp__web_search__search"].namespace == "mcp__web_search"
        assert nsmap["mcp__web_search__fetch_url"].kind == "namespace"
        assert nsmap["mcp__web_search__fetch_url"].name == "fetch_url"
        assert nsmap["mcp__web_search__fetch_url"].namespace == "mcp__web_search"
        assert "mcp__web_search__search" not in rmap
        assert "mcp__web_search__fetch_url" not in rmap

    def test_web_search_with_top_level_config_fields(self):
        """web_search 带 user_location/search_context_size 顶级字段时，
        当前实现会丢弃这些字段（它们不在 parameters 里），
        但 parameters 内的字段应保留。此测试锁定当前行为。"""
        tools = [
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                },
                "search_context_size": "medium",
            }
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 0

    def test_web_search_with_parameters_preserved(self):
        """web_search 自带 parameters 时，保留其内容而非覆盖为默认 query schema。"""
        tools = [
            {
                "type": "web_search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "region": {"type": "string"},
                    },
                    "required": ["query"],
                },
            }
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 0

    def test_tool_search_degraded_to_function(self):
        """tool_search 降级为普通 function，当前行为锁定。"""
        tools = [
            {"type": "tool_search"},
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 0

    def test_image_generation_degraded_to_function(self):
        """image_generation 降级为普通 function，当前行为锁定。"""
        tools = [
            {"type": "image_generation"},
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "image_generation"
        assert "image_generation" not in rmap

    def test_namespace_with_defer_loading_subtool(self):
        """namespace 含 defer_loading 子工具时，当前行为：忽略 defer_loading，
        仍然展开为 function。此测试锁定当前行为。"""
        tools = [
            {
                "type": "namespace",
                "name": "crm",
                "tools": [
                    {
                        "type": "function",
                        "name": "list_open_orders",
                        "description": "List open orders",
                        "parameters": {
                            "type": "object",
                            "properties": {"status": {"type": "string"}},
                            "required": [],
                        },
                        "defer_loading": True,
                    }
                ],
            }
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "crm__list_open_orders"
        # defer_loading 被忽略，工具仍被展开
        assert "defer_loading" not in result[0]["function"]
        # namespace 子工具不写入 reverse_tool_map
        assert "crm__list_open_orders" not in rmap
        # namespace 子工具写入 tool_spec_map
        assert nsmap["crm__list_open_orders"].kind == "namespace"
        assert nsmap["crm__list_open_orders"].name == "list_open_orders"
        assert nsmap["crm__list_open_orders"].namespace == "crm"

    def test_namespace_with_mixed_subtool_types(self):
        """namespace 中非 function 类型的子工具应被跳过。"""
        tools = [
            {
                "type": "namespace",
                "name": "mixed",
                "tools": [
                    {
                        "type": "function",
                        "name": "valid_func",
                        "description": "A valid function",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                    {"type": "web_search"},  # 非函数子工具，应跳过
                    "not_a_dict",  # 非字典，应跳过
                ],
            }
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "mixed__valid_func"

    def test_namespace_with_empty_tools(self):
        """namespace 的 tools 为空列表时，不产生任何工具。"""
        tools = [
            {
                "type": "namespace",
                "name": "empty_ns",
                "tools": [],
            }
        ]
        result, rmap, nsmap = convert_tools_to_chat(tools)
        assert len(result) == 0

    def test_nested_function_field(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_data",
                "description": "Fetch data",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }]
        result, custom, nsmap = convert_tools_to_chat(tools)
        assert result[0]["function"]["name"] == "get_data"
        assert result[0]["function"]["description"] == "Fetch data"


class TestStreamResponsesToChatToolCalls:
    """Test Responses SSE function_call events -> Chat tool_calls delta"""

    def test_function_call_started_event(self):
        """function_call.started -> tool_calls delta with id and name"""
        from llm_proxy.protocol.responses_chat.response import stream_responses_to_chat

        event_data = {
            "type": "response.function_call.started",
            "id": "call_abc123",
            "name": "get_weather",
        }
        sse_line = f"data: {json.dumps(event_data)}\n\n"

        class FakeResponse:
            def __init__(self):
                self.lines = [sse_line.encode(), b"data: [DONE]\n\n"]

            async def aiter_lines(self):
                for line in self.lines:
                    yield line.decode()

        async def collect():
            results = []
            async for chunk in stream_responses_to_chat(FakeResponse(), "gpt-4", "ep1", "m1"):
                results.append(chunk)
            return results

        results = asyncio.run(collect())

        found = False
        for chunk in results:
            if chunk.startswith(b"data: [DONE]"):
                continue
            payload = json.loads(chunk.decode().replace("data: ", ""))
            choices = payload.get("choices", [])
            for c in choices:
                delta = c.get("delta", {})
                tc = delta.get("tool_calls", [])
                if tc:
                    assert tc[0]["id"] == "call_abc123"
                    assert tc[0]["function"]["name"] == "get_weather"
                    found = True
        assert found, "Should have found tool_calls in stream output"

    def test_function_call_arguments_delta_event(self):
        """function_call_arguments.delta -> tool_calls delta with arguments"""
        from llm_proxy.protocol.responses_chat.response import stream_responses_to_chat

        event_data = {
            "type": "response.function_call_arguments.delta",
            "delta": '{"city":"',
        }
        sse_line = f"data: {json.dumps(event_data)}\n\n"

        class FakeResponse:
            def __init__(self):
                self.lines = [sse_line.encode(), b"data: [DONE]\n\n"]

            async def aiter_lines(self):
                for line in self.lines:
                    yield line.decode()

        async def collect():
            results = []
            async for chunk in stream_responses_to_chat(FakeResponse(), "gpt-4", "ep1", "m1"):
                results.append(chunk)
            return results

        results = asyncio.run(collect())

        found = False
        for chunk in results:
            if chunk.startswith(b"data: [DONE]"):
                continue
            payload = json.loads(chunk.decode().replace("data: ", ""))
            choices = payload.get("choices", [])
            for c in choices:
                delta = c.get("delta", {})
                tc = delta.get("tool_calls", [])
                if tc:
                    assert tc[0]["function"]["arguments"] == '{"city":"'
                    found = True
        assert found

    def test_function_call_done_event(self):
        """function_call.done -> tool_calls delta with complete arguments"""
        from llm_proxy.protocol.responses_chat.response import stream_responses_to_chat

        event_data = {
            "type": "response.function_call.done",
            "id": "call_abc123",
            "name": "get_weather",
            "arguments": '{"city":"NYC"}',
        }
        sse_line = f"data: {json.dumps(event_data)}\n\n"

        class FakeResponse:
            def __init__(self):
                self.lines = [sse_line.encode(), b"data: [DONE]\n\n"]

            async def aiter_lines(self):
                for line in self.lines:
                    yield line.decode()

        async def collect():
            results = []
            async for chunk in stream_responses_to_chat(FakeResponse(), "gpt-4", "ep1", "m1"):
                results.append(chunk)
            return results

        results = asyncio.run(collect())

        found = False
        for chunk in results:
            if chunk.startswith(b"data: [DONE]"):
                continue
            payload = json.loads(chunk.decode().replace("data: ", ""))
            choices = payload.get("choices", [])
            for c in choices:
                delta = c.get("delta", {})
                tc = delta.get("tool_calls", [])
                if tc:
                    assert tc[0]["function"]["arguments"] == '{"city":"NYC"}'
                    found = True
        assert found


class TestConvertInputFunctionCall:
    """Test Responses API function_call/function_call_output input → Chat messages"""

    def test_function_call_input_converted_to_tool_calls(self):
        input_data = [{
            "type": "function_call",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": '{"city":"NYC"}',
        }]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["tool_calls"][0]["id"] == "call_abc"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"city":"NYC"}'
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_abc"
        assert messages[1]["content"] == "[Tool call was interrupted]"

    def test_function_call_output_converted_to_tool_message(self):
        input_data = [{
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": '{"temperature":72}',
        }]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "call_abc" in messages[0]["content"]

    def test_function_call_without_arguments(self):
        input_data = [{
            "type": "function_call",
            "call_id": "call_xyz",
            "name": "ping",
        }]
        messages = convert_input_to_messages(input_data)
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == ""

    def test_function_call_output_without_call_id(self):
        input_data = [{"type": "function_call_output", "output": "ok"}]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Function call output" in messages[0]["content"]

    def test_mixed_input_with_regular_messages(self):
        input_data = [
            {"role": "user", "content": "check weather"},
            {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {"role": "user", "content": "thanks"},
        ]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 1
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "user"


class TestConvertInputReasoning:
    """Test Responses API reasoning item → Chat messages (DeepSeek reasoning_content)"""

    def test_reasoning_merged_into_next_assistant(self):
        input_data = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "result"}]},
        ]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["reasoning_content"] == "thinking..."
        assert messages[0]["content"] == "result"

    def test_reasoning_merged_into_assistant_with_tool_calls(self):
        input_data = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "planning"}]},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        ]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["reasoning_content"] == "planning"
        assert messages[0]["content"] is None
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "c1"

    def test_reasoning_at_end_creates_assistant(self):
        input_data = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "pondering"}]},
        ]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["reasoning_content"] == "pondering"
        assert messages[1]["content"] == ""

    def test_multiple_reasoning_items_merged(self):
        input_data = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "step1"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "step2"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 1
        assert messages[0]["reasoning_content"] == "step1\nstep2"
        assert messages[0]["content"] == "done"

    def test_reasoning_text_fallback(self):
        input_data = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "from_summary"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
        ]
        messages = convert_input_to_messages(input_data)
        assert messages[0]["reasoning_content"] == "from_summary"


class TestConvertInputToolCallsMerge:
    def test_consecutive_function_calls_merged_into_one_assistant(self):
        input_data = [
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "write", "arguments": "{}"},
        ]
        messages = convert_input_to_messages(input_data)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["tool_calls"]) == 2
        assert assistant_msgs[0]["tool_calls"][0]["id"] == "c1"
        assert assistant_msgs[0]["tool_calls"][1]["id"] == "c2"
        assert assistant_msgs[0]["content"] is None

    def test_function_call_then_output_separates_messages(self):
        input_data = [
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "data"},
        ]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[1]["role"] == "tool"

    def test_reasoning_then_function_call_merged(self):
        input_data = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "planning"}]},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        ]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["reasoning_content"] == "planning"
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "c1"

    def test_function_call_then_regular_message_flushes(self):
        input_data = [
            {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"q":"weather"}'},
            {"role": "user", "content": "thanks"},
        ]
        messages = convert_input_to_messages(input_data)
        assert len(messages) == 3
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "c1"
        assert messages[2]["role"] == "user"


class TestStreamResponsesToChatMultiToolCall:
    def test_multiple_tool_calls_have_correct_indices(self):
        from llm_proxy.protocol.responses_chat.response import stream_responses_to_chat

        events = [
            {"type": "response.function_call.started", "id": "call_1", "name": "read"},
            {"type": "response.function_call.started", "id": "call_2", "name": "write"},
            {"type": "response.function_call_arguments.delta", "id": "call_1", "delta": '{"a":'},
            {"type": "response.function_call_arguments.delta", "id": "call_2", "delta": '{"b":'},
        ]
        lines = [f"data: {json.dumps(e)}\n\n" for e in events]
        lines.append("data: [DONE]\n\n")

        class FakeResponse:
            async def aiter_lines(self):
                for line in lines:
                    yield line

        async def collect():
            results = []
            async for chunk in stream_responses_to_chat(FakeResponse(), "gpt-4", "ep1", "m1"):
                results.append(chunk)
            return results

        results = asyncio.run(collect())
        indices_found = set()
        for chunk in results:
            if chunk.startswith(b"data: [DONE]"):
                continue
            try:
                payload = json.loads(chunk.decode().replace("data: ", ""))
            except json.JSONDecodeError:
                continue
            for c in payload.get("choices", []):
                for tc in c.get("delta", {}).get("tool_calls", []):
                    idx = tc.get("index")
                    if idx is not None:
                        indices_found.add(idx)

        assert 0 in indices_found
        assert 1 in indices_found


class TestConvertInputOrphanToolDowngrade:
    def test_orphan_function_call_output_downgraded(self):
        input_data = [
            {"type": "function_call_output", "call_id": "call_orphan", "output": "orphan result"},
            {"role": "user", "content": "hello"},
        ]
        messages = convert_input_to_messages(input_data)
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 0
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any("call_orphan" in m.get("content", "") for m in user_msgs)

    def test_tool_response_without_matching_call_gets_placeholder(self):
        input_data = [
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
            {"role": "user", "content": "interrupted"},
        ]
        messages = convert_input_to_messages(input_data)
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "c1"


class TestToResponsesResponseFixes:
    def test_tool_calls_become_function_call_items(self):
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = to_responses_response(chat_body, "gpt-4")
        output = result.get("output", [])
        fc_items = [o for o in output if o.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["call_id"] == "call_abc"
        assert fc_items[0]["name"] == "get_weather"
        assert fc_items[0]["arguments"] == '{"city":"NYC"}'

    def test_reasoning_content_becomes_reasoning_item(self):
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "result",
                    "reasoning_content": "I thought about it",
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = to_responses_response(chat_body, "gpt-4")
        output = result.get("output", [])
        reasoning_items = [o for o in output if o.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["summary"][0]["text"] == "I thought about it"

    def test_empty_content_no_message_item(self):
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "ping", "arguments": "{}"},
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = to_responses_response(chat_body, "gpt-4")
        output = result.get("output", [])
        msg_items = [o for o in output if o.get("type") == "message"]
        assert len(msg_items) == 0
        fc_items = [o for o in output if o.get("type") == "function_call"]
        assert len(fc_items) == 1

    def test_empty_tool_call_arguments_uses_empty_object(self):
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "result",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "ping"},
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = to_responses_response(chat_body, "gpt-4")
        fc_items = [o for o in result.get("output", []) if o.get("type") == "function_call"]
        assert fc_items[0]["arguments"] == "{}"

    def test_tool_calls_null_does_not_crash(self):
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": None,
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = to_responses_response(chat_body, "gpt-4")
        output = result.get("output", [])
        msg_items = [o for o in output if o.get("type") == "message"]
        assert len(msg_items) == 1
        assert msg_items[0]["content"][0]["text"] == "ok"

    def test_tool_calls_missing_key_does_not_crash(self):
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "ok",
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = to_responses_response(chat_body, "gpt-4")
        output = result.get("output", [])
        msg_items = [o for o in output if o.get("type") == "message"]
        assert len(msg_items) == 1


class TestMcpCallResponse:
    """测试 namespace 子工具的 function_call+namespace 响应格式"""

    def test_namespace_tool_call_uses_function_call_with_namespace(self):
        """namespace 子工具的 tool call 应转为 function_call + namespace 字段"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_mcp1",
                        "type": "function",
                        "function": {"name": "mcp__web_search__search", "arguments": '{"query":"weather"}'},
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        tool_spec_map = {
            "mcp__web_search__search": CodexToolSpec(kind="namespace", name="search", namespace="mcp__web_search"),
            "mcp__web_search__fetch_url": CodexToolSpec(kind="namespace", name="fetch_url", namespace="mcp__web_search"),
        }
        result = to_responses_response(chat_body, "gpt-4", tool_spec_map=tool_spec_map)
        output = result.get("output", [])
        ns_items = [o for o in output if o.get("type") == "function_call" and o.get("namespace")]
        assert len(ns_items) == 1
        assert ns_items[0]["name"] == "search"
        assert ns_items[0]["namespace"] == "mcp__web_search"
        assert ns_items[0]["arguments"] == '{"query":"weather"}'
        assert ns_items[0]["call_id"] == "call_mcp1"
        # 不应生成无 namespace 的 function_call
        plain_fc = [o for o in output if o.get("type") == "function_call" and not o.get("namespace")]
        assert len(plain_fc) == 0

    def test_mixed_namespace_and_plain_function_call(self):
        """namespace 子工具和普通函数共存时，各自走不同路径"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "mcp__web_search__search", "arguments": '{"q":"test"}'}},
                        {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}},
                    ],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        tool_spec_map = {
            "mcp__web_search__search": CodexToolSpec(kind="namespace", name="search", namespace="mcp__web_search"),
        }
        result = to_responses_response(chat_body, "gpt-4", tool_spec_map=tool_spec_map)
        output = result.get("output", [])
        ns_items = [o for o in output if o.get("type") == "function_call" and o.get("namespace")]
        fc_items = [o for o in output if o.get("type") == "function_call" and not o.get("namespace")]
        assert len(ns_items) == 1
        assert ns_items[0]["name"] == "search"
        assert ns_items[0]["namespace"] == "mcp__web_search"
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "get_weather"

    def test_reverse_tool_map_takes_priority_over_tool_spec_map(self):
        """reverse_tool_map 优先于 tool_spec_map"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "write_to_file", "arguments": '{"filePath":"/tmp/f","content":"hi"}'},
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        reverse_tool_map = {"write_to_file": "apply_patch"}
        tool_spec_map = {"write_to_file": CodexToolSpec(kind="namespace", name="write_to_file", namespace="some_ns")}
        result = to_responses_response(chat_body, "gpt-4",
                                       reverse_tool_map=reverse_tool_map,
                                       tool_spec_map=tool_spec_map)
        output = result.get("output", [])
        custom_items = [o for o in output if o.get("type") == "custom_tool_call"]
        ns_items = [o for o in output if o.get("type") == "function_call" and o.get("namespace")]
        assert len(custom_items) == 1
        assert custom_items[0]["name"] == "apply_patch"
        assert len(ns_items) == 0

    def test_no_tool_spec_map_falls_back_to_function_call(self):
        """无 tool_spec_map 时，工具仍走 function_call"""
        from llm_proxy.protocol.responses_chat.request import to_responses_response
        chat_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "unknown_tool", "arguments": '{}'},
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = to_responses_response(chat_body, "gpt-4")
        output = result.get("output", [])
        fc_items = [o for o in output if o.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "unknown_tool"


class TestMcpCallInputConversion:
    """测试 mcp_call 输入项的转换"""

    def test_mcp_call_converted_to_tool_call(self):
        """mcp_call 输入项应转为 Chat tool_calls"""
        from llm_proxy.protocol.responses_chat.request import convert_input_to_messages
        input_data = [
            {"type": "message", "role": "user", "content": "search for weather"},
            {"type": "mcp_call", "call_id": "call_1", "name": "search",
             "arguments": '{"query":"weather"}'},
            {"type": "function_call_output", "call_id": "call_1",
             "output": "Sunny, 72F"},
        ]
        messages = convert_input_to_messages(input_data)
        # 应有 user + assistant(tool_calls) + tool
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "search"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"

    def test_mcp_call_with_output_field(self):
        """mcp_call 带 output 字段时，应生成 tool message"""
        from llm_proxy.protocol.responses_chat.request import convert_input_to_messages
        input_data = [
            {"type": "message", "role": "user", "content": "search"},
            {"type": "mcp_call", "call_id": "call_1", "name": "search",
             "arguments": '{"q":"test"}', "output": "result text"},
        ]
        messages = convert_input_to_messages(input_data)
        # user + assistant(tool_calls) + tool
        assert len(messages) == 3
        assert messages[2]["role"] == "tool"
        assert messages[2]["content"] == "result text"
