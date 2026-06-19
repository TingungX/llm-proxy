"""Tests for StreamState — streaming conversion state machine"""
import json
import pytest
from llm_proxy.protocol.responses_chat.stream import StreamState


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


class TestStreamStateReasoningLifecycle:
    def test_first_reasoning_delta_opens_block(self):
        st = StreamState()
        events = st.handle_reasoning_delta("thinking...")
        types = _event_types(events)
        assert "response.output_item.added" in types
        assert "response.reasoning_summary_part.added" in types
        assert "response.reasoning_summary_text.delta" in types
        assert st.reasoning_active is True

    def test_subsequent_reasoning_delta_only_sends_delta(self):
        st = StreamState()
        st.handle_reasoning_delta("first")
        events = st.handle_reasoning_delta(" second")
        types = _event_types(events)
        assert types == ["response.reasoning_summary_text.delta"]
        assert " second" in _parse_events(events)[0].get("delta", "")

    def test_close_reasoning_block_sends_three_events(self):
        st = StreamState()
        st.handle_reasoning_delta("thinking")
        events = st.close_reasoning_block()
        types = _event_types(events)
        assert "response.reasoning_summary_text.done" in types
        assert "response.reasoning_summary_part.done" in types
        assert "response.output_item.done" in types
        assert st.reasoning_active is False


class TestStreamStateTextLifecycle:
    def test_first_content_delta_opens_text_block(self):
        st = StreamState()
        events = st.handle_content_delta("hello")
        types = _event_types(events)
        assert "response.output_item.added" in types
        assert "response.content_part.added" in types
        assert "response.output_text.delta" in types
        assert st.in_text_block is True

    def test_subsequent_content_only_sends_delta(self):
        st = StreamState()
        st.handle_content_delta("hello")
        events = st.handle_content_delta(" world")
        types = _event_types(events)
        assert types == ["response.output_text.delta"]

    def test_close_text_block_sends_three_events(self):
        st = StreamState()
        st.handle_content_delta("hello")
        events = st.close_text_block()
        types = _event_types(events)
        assert "response.output_text.done" in types
        assert "response.content_part.done" in types
        assert "response.output_item.done" in types
        assert st.in_text_block is False


class TestStreamStateFunctionCallLifecycle:
    def test_tool_call_id_opens_function_block(self):
        st = StreamState()
        events = st.handle_tool_call_id(0, "call_abc", "get_weather")
        types = _event_types(events)
        assert "response.output_item.added" in types
        assert st.in_func_block is True

    def test_tool_call_arguments_sends_delta(self):
        st = StreamState()
        st.handle_tool_call_id(0, "call_abc", "get_weather")
        events = st.handle_tool_call_args_delta(0, '{"city":')
        types = _event_types(events)
        assert "response.function_call_arguments.delta" in types

    def test_close_func_blocks_sends_done_events(self):
        st = StreamState()
        st.handle_tool_call_id(0, "call_abc", "get_weather")
        st.handle_tool_call_args_delta(0, '{"city":"NYC"}')
        events = st.close_func_blocks()
        types = _event_types(events)
        assert "response.function_call_arguments.done" in types
        assert "response.output_item.done" in types
        assert st.in_func_block is False


class TestStreamStateBlockTransitions:
    def test_reasoning_to_content_closes_reasoning_first(self):
        st = StreamState()
        st.handle_reasoning_delta("thinking")
        events = st.handle_content_delta("result")
        types = _event_types(events)
        assert "response.reasoning_summary_text.done" in types
        assert "response.reasoning_summary_part.done" in types
        assert "response.output_item.done" in types
        assert "response.output_item.added" in types
        assert "response.content_part.added" in types
        assert "response.output_text.delta" in types

    def test_reasoning_to_tool_call_closes_reasoning_and_text(self):
        st = StreamState()
        st.handle_reasoning_delta("planning")
        events = st.handle_tool_call_id(0, "call_abc", "read")
        types = _event_types(events)
        assert "response.reasoning_summary_text.done" in types
        assert "response.reasoning_summary_part.done" in types
        assert "response.output_item.done" in types
        assert "response.output_item.added" in types

    def test_text_to_tool_call_closes_text(self):
        st = StreamState()
        st.handle_content_delta("some text")
        events = st.handle_tool_call_id(0, "call_abc", "read")
        types = _event_types(events)
        assert "response.output_text.done" in types
        assert "response.content_part.done" in types
        assert "response.output_item.done" in types
        assert "response.output_item.added" in types


class TestStreamStateOutputIndex:
    def test_output_index_no_reasoning(self):
        st = StreamState()
        events = st.handle_content_delta("hello")
        added = _parse_events(events)[0]
        assert added.get("output_index") == 0

    def test_output_index_with_reasoning(self):
        st = StreamState()
        st.handle_reasoning_delta("think")
        st.close_reasoning_block()
        events = st.handle_content_delta("hello")
        added = _parse_events(events)[0]
        assert added.get("output_index") == 1


class TestStreamStateFlushAndCompleted:
    def test_flush_think_tag_buf(self):
        st = StreamState()
        from llm_proxy.protocol.think_tag import ThinkTagStateMachine, OPEN_TAG
        st.think = ThinkTagStateMachine()
        st.think.feed(OPEN_TAG + "still thinking")
        events = st.flush_think_tag_buf()
        assert len(events) > 0
        assert st.reasoning_active is True

    def test_generate_completed_events_closes_all_blocks(self):
        st = StreamState()
        st.handle_reasoning_delta("think")
        st.handle_content_delta("result")
        events = st.generate_completed_events("gpt-4", "resp_test123")
        types = _event_types(events)
        assert "response.output_text.done" in types
        assert "response.completed" in types
        parsed = _parse_events(events)
        completed = [e for e in parsed if e["type"] == "response.completed"][0]
        assert completed["response"]["status"] == "completed"


class TestStreamStateEdgeCases:
    def test_only_reasoning_no_content(self):
        st = StreamState()
        events = st.handle_reasoning_delta("thinking only")
        completed = st.generate_completed_events("gpt-4", "resp_test")
        all_events = _parse_events(events + completed)
        all_types = [e.get("type") for e in all_events]
        assert "response.reasoning_summary_text.done" in all_types
        assert "response.completed" in all_types

    def test_only_tool_calls_no_text(self):
        st = StreamState()
        st.handle_tool_call_id(0, "call_abc", "read")
        st.handle_tool_call_args_delta(0, '{}')
        completed = st.generate_completed_events("gpt-4", "resp_test")
        types = _event_types(completed)
        assert "response.function_call_arguments.done" in types
        assert "response.output_item.done" in types
        assert "response.completed" in types

    def test_multiple_tool_calls(self):
        st = StreamState()
        st.handle_tool_call_id(0, "call_1", "read")
        st.handle_tool_call_args_delta(0, '{"a":1}')
        st.handle_tool_call_id(1, "call_2", "write")
        st.handle_tool_call_args_delta(1, '{"b":2}')
        completed = st.generate_completed_events("gpt-4", "resp_test")
        types = _event_types(completed)
        assert types.count("response.function_call_arguments.done") == 2
        assert types.count("response.output_item.done") == 2

    def test_empty_stream_completed(self):
        st = StreamState()
        completed = st.generate_completed_events("gpt-4", "resp_test")
        types = _event_types(completed)
        assert "response.completed" in types

    def test_reasoning_then_tool_call(self):
        from llm_proxy.protocol.responses_chat.stream import StreamState
        st = StreamState()
        st.handle_reasoning_delta("think")
        st.handle_tool_call_id(0, "call_abc", "read")
        st.handle_tool_call_args_delta(0, '{}')
        completed = st.generate_completed_events("gpt-4", "resp_test")
        types = _event_types(completed)
        assert "response.completed" in types


class TestStreamStateDrainIntegration:
    def test_unclosed_think_tag_drained_on_completed(self):
        from llm_proxy.protocol.think_tag import ThinkTagStateMachine
        st = StreamState()
        st.think = ThinkTagStateMachine()
        st.think.feed("<think>still thinking without close")
        completed = st.generate_completed_events("gpt-4", "resp_test")
        types = _event_types(completed)
        assert "response.reasoning_summary_text.delta" in types
        assert "response.completed" in types

    def test_partial_tag_buffer_drained(self):
        from llm_proxy.protocol.think_tag import ThinkTagStateMachine
        st = StreamState()
        st.think = ThinkTagStateMachine()
        st.think.feed("normal text<thi")
        completed = st.generate_completed_events("gpt-4", "resp_test")
        types = _event_types(completed)
        assert "response.completed" in types


class TestSSEEventTypeLine:
    def test_event_has_event_type_line(self):
        from llm_proxy.protocol.responses_chat.stream import _make_sse_event
        evt = _make_sse_event({"type": "response.created", "response": {}})
        decoded = evt.decode()
        lines = decoded.strip().split("\n")
        assert any(line.startswith("event: ") for line in lines)
        event_line = [l for l in lines if l.startswith("event: ")][0]
        assert "response.created" in event_line

    def test_event_has_data_line(self):
        from llm_proxy.protocol.responses_chat.stream import _make_sse_event
        evt = _make_sse_event({"type": "response.output_text.delta", "delta": "hi"})
        decoded = evt.decode()
        assert "data: " in decoded

    def test_completed_event_echoes_original_request_fields(self):
        st = StreamState()
        st.handle_content_delta("hello")
        original_request = {
            "instructions": "Be helpful",
            "model": "gpt-4",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 100,
            "tool_choice": "auto",
        }
        events = st.generate_completed_events("gpt-4", "resp_test", original_request=original_request)
        completed = [e for e in events if b"response.completed" in e]
        assert len(completed) == 1
        decoded = completed[0].decode()
        for line in decoded.split("\n"):
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                response = payload.get("response", {})
                assert response.get("instructions") == "Be helpful"
                assert response.get("temperature") == 0.7
                assert response.get("max_output_tokens") == 100
                break


class TestCustomToolPassthrough:
    """测试非 apply_patch 的 custom 工具（spawn_agent, view_image 等）透传"""

    def test_spawn_agent_tool_call_produces_custom_tool_call(self):
        """spawn_agent 工具调用应生成 custom_tool_call 事件，参数为 JSON"""
        st = StreamState(reverse_tool_map={"spawn_agent": "spawn_agent"})
        events = st.handle_tool_call_id(0, "call_spawn1", "spawn_agent")
        types = _event_types(events)
        # 应该生成 custom_tool_call 类型的 item
        added = [e for e in _parse_events(events) if e.get("type") == "response.output_item.added"]
        assert len(added) == 1
        item = added[0].get("item", {})
        assert item.get("type") == "custom_tool_call"
        assert item.get("name") == "spawn_agent"
        assert item.get("input") == ""

    def test_spawn_agent_args_not_emitted_as_delta(self):
        """spawn_agent 的参数 delta 不应发送 function_call_arguments.delta"""
        st = StreamState(reverse_tool_map={"spawn_agent": "spawn_agent"})
        st.handle_tool_call_id(0, "call_spawn1", "spawn_agent")
        events = st.handle_tool_call_args_delta(0, '{"task_name":')
        types = _event_types(events)
        # 参数被缓存，不发送 delta
        assert "response.function_call_arguments.delta" not in types

    def test_spawn_agent_close_produces_custom_tool_call_with_json_input(self):
        """关闭 spawn_agent 工具块时，应生成 custom_tool_call，input 为 JSON 字符串"""
        st = StreamState(reverse_tool_map={"spawn_agent": "spawn_agent"})
        st.handle_tool_call_id(0, "call_spawn1", "spawn_agent")
        st.handle_tool_call_args_delta(0, '{"task_name": "task1", "message": "hello"}')
        events = st.close_func_blocks()
        types = _event_types(events)
        assert "response.custom_tool_call_input.delta" in types
        assert "response.output_item.done" in types
        # 检查 input 内容
        done_events = [e for e in _parse_events(events) if e.get("type") == "response.output_item.done"]
        assert len(done_events) == 1
        item = done_events[0].get("item", {})
        assert item.get("type") == "custom_tool_call"
        assert item.get("name") == "spawn_agent"
        input_text = item.get("input", "")
        parsed = json.loads(input_text)
        assert parsed["task_name"] == "task1"
        assert parsed["message"] == "hello"

    def test_view_image_tool_call_produces_custom_tool_call(self):
        """view_image 工具调用应生成 custom_tool_call 事件"""
        st = StreamState(reverse_tool_map={"view_image": "view_image"})
        st.handle_tool_call_id(0, "call_view1", "view_image")
        st.handle_tool_call_args_delta(0, '{"path": "/tmp/test.png"}')
        events = st.close_func_blocks()
        done_events = [e for e in _parse_events(events) if e.get("type") == "response.output_item.done"]
        assert len(done_events) == 1
        item = done_events[0].get("item", {})
        assert item.get("type") == "custom_tool_call"
        assert item.get("name") == "view_image"
        input_text = item.get("input", "")
        parsed = json.loads(input_text)
        assert parsed["path"] == "/tmp/test.png"

    def test_apply_patch_passthrough(self):
        """透传：上游 apply_patch 调用的 arguments 原样作为 custom_tool_call.input"""
        st = StreamState(reverse_tool_map={"apply_patch": "apply_patch"})
        st.handle_tool_call_id(0, "call_patch1", "apply_patch")
        st.handle_tool_call_args_delta(0, '{"input": "*** Begin Patch\\n*** Add File: /tmp/x.txt\\n+hello\\n*** End Patch"}')
        events = st.close_func_blocks()
        done_events = [e for e in _parse_events(events) if e.get("type") == "response.output_item.done"]
        assert len(done_events) == 1
        item = done_events[0].get("item", {})
        assert item.get("type") == "custom_tool_call"
        assert item.get("name") == "apply_patch"
        input_text = item.get("input", "")
        assert "*** Begin Patch" in input_text
        assert "*** Add File: /tmp/x.txt" in input_text

    def test_mixed_apply_patch_and_custom_tools(self):
        """透传：apply_patch 和 spawn_agent 都走 JSON 透传"""
        st = StreamState(reverse_tool_map={
            "apply_patch": "apply_patch",
            "spawn_agent": "spawn_agent",
        })
        st.handle_tool_call_id(0, "call_patch1", "apply_patch")
        st.handle_tool_call_args_delta(0, '{"input": "*** Begin Patch\\n*** Add File: a.txt\\n+hi\\n*** End Patch"}')
        st.handle_tool_call_id(1, "call_spawn1", "spawn_agent")
        st.handle_tool_call_args_delta(1, '{"task_name": "task1"}')
        events = st.close_func_blocks()
        done_events = [e for e in _parse_events(events) if e.get("type") == "response.output_item.done"]
        assert len(done_events) == 2
        patch_item = done_events[0].get("item", {})
        assert patch_item.get("name") == "apply_patch"
        patch_input = patch_item.get("input", "")
        assert "*** Begin Patch" in patch_input
        assert "*** Add File: a.txt" in patch_input
        spawn_item = done_events[1].get("item", {})
        assert spawn_item.get("name") == "spawn_agent"
        parsed = json.loads(spawn_item.get("input", ""))
        assert parsed["task_name"] == "task1"

    def test_build_output_items_spawn_agent(self):
        """非流式路径：spawn_agent 工具调用应生成 custom_tool_call item"""
        st = StreamState(reverse_tool_map={"spawn_agent": "spawn_agent"})
        st.handle_tool_call_id(0, "call_spawn1", "spawn_agent")
        st.handle_tool_call_args_delta(0, '{"task_name": "task1"}')
        items = st._build_output_items()
        assert len(items) == 1
        item = items[0]
        assert item.get("type") == "custom_tool_call"
        assert item.get("name") == "spawn_agent"
        parsed = json.loads(item.get("input", ""))
        assert parsed["task_name"] == "task1"
