"""IR 流式层单元测试。

覆盖：
1. Chat ↔ IR 流式
2. Anthropic ↔ IR 流式
3. Responses ↔ IR 流式（含 apply_patch 反向、namespace）
4. 跨协议流式
5. _stream 工具函数
"""

import asyncio
import json

import pytest

from llm_proxy.protocol.ir._stream import (
    DONE_SENTINEL,
    IncrementalJSONParser,
    SSELineAccumulator,
    chunk_to_usage,
    extract_usage_tokens,
    keepalive_wrapper,
    map_finish_to_stop_reason,
    map_stop_to_finish_reason,
    map_stop_to_responses_status,
    parse_sse_line,
    sse_comment,
    sse_format,
    sse_format_data_only,
)
from llm_proxy.protocol.ir.anthropic import (
    format_ir_as_sse as anthropic_format,
    parse_stream_to_ir as anthropic_parse,
)
from llm_proxy.protocol.ir.chat import (
    format_ir_as_sse as chat_format,
    parse_stream_to_ir as chat_parse,
)
from llm_proxy.protocol.ir.responses import (
    format_ir_as_sse as responses_format,
    parse_stream_to_ir as responses_parse,
)
from llm_proxy.protocol.ir.types import IRStreamEvent


# ── Mock HTTP 响应 ──────────────────────────────────────────────


class MockResp:
    """模拟 httpx 流式响应。aiter_lines() 逐行产出原始字节。"""

    def __init__(self, lines):
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


async def _collect(aiter) -> list:
    """收集异步生成器的所有结果。"""
    out = []
    async for item in aiter:
        out.append(item)
    return out


async def _collect_bytes(aiter) -> list:
    out = []
    async for item in aiter:
        out.append(item)
    return out


# ── _stream 工具函数 ──────────────────────────────────────────────


class TestSSELineParser:
    @pytest.mark.parametrize("line,expected", [
        ('data: {"foo": 1}', {"foo": 1}),
        ('data:{"foo":1}', {"foo": 1}),
        ('data: [DONE]', DONE_SENTINEL),
        ('', None),
        (': keepalive', None),
        ('event: message_start', None),
        ('id: 1', None),
    ])
    def test_parse_sse_line(self, line, expected):
        result = parse_sse_line(line)
        if expected is DONE_SENTINEL:
            assert result is DONE_SENTINEL
        else:
            assert result == expected


class TestSSEFormat:
    def test_sse_format(self):
        b = sse_format("message_start", {"id": "x"})
        assert b == b'event: message_start\ndata: {"id": "x"}\n\n'

    def test_sse_format_data_only(self):
        b = sse_format_data_only({"delta": {"content": "hi"}})
        assert b == b'data: {"delta": {"content": "hi"}}\n\n'

    def test_sse_comment(self):
        b = sse_comment("keepalive")
        assert b == b": keepalive\n\n"


class TestExtractUsageTokens:
    def test_chat_style(self):
        u = extract_usage_tokens({"prompt_tokens": 10, "completion_tokens": 5})
        assert u == {"input_tokens": 10, "output_tokens": 5}

    def test_responses_style(self):
        u = extract_usage_tokens({"input_tokens": 10, "output_tokens": 5})
        assert u == {"input_tokens": 10, "output_tokens": 5}

    def test_empty(self):
        u = extract_usage_tokens({})
        assert u == {"input_tokens": 0, "output_tokens": 0}


class TestStopReasonMapping:
    def test_finish_to_stop(self):
        assert map_finish_to_stop_reason("stop") == "end_turn"
        assert map_finish_to_stop_reason("tool_calls") == "tool_use"
        assert map_finish_to_stop_reason("length") == "max_tokens"
        assert map_finish_to_stop_reason("content_filter") == "refusal"
        assert map_finish_to_stop_reason(None) == "end_turn"
        assert map_finish_to_stop_reason("unknown") == "end_turn"

    def test_stop_to_finish(self):
        assert map_stop_to_finish_reason("end_turn") == "stop"
        assert map_stop_to_finish_reason("tool_use") == "tool_calls"
        assert map_stop_to_finish_reason("max_tokens") == "length"
        assert map_stop_to_finish_reason("refusal") == "content_filter"

    def test_stop_to_responses_status(self):
        assert map_stop_to_responses_status("end_turn") == ("completed", None)
        assert map_stop_to_responses_status("tool_use") == ("completed", None)
        assert map_stop_to_responses_status("max_tokens") == ("incomplete", "max_output_tokens")
        assert map_stop_to_responses_status("refusal") == ("incomplete", "content_filter")


class TestIncrementalJSONParser:
    def test_complete_json(self):
        p = IncrementalJSONParser()
        assert p.feed('{"a": 1') is None
        assert p.feed('}') == {"a": 1}

    def test_finalize_partial(self):
        p = IncrementalJSONParser()
        p.feed('{"a": 1, "b":')
        result = p.finalize()
        assert result == {"_raw": '{"a": 1, "b":'}

    def test_finalize_empty(self):
        p = IncrementalJSONParser()
        result = p.finalize()
        assert result == {}


class TestSSELineAccumulator:
    def test_split_lines(self):
        a = SSELineAccumulator()
        lines = a.feed(b"data: line1\ndata: line2\n")
        assert lines == ["data: line1", "data: line2"]
        # 第三行不完整，留在 buffer
        lines = a.feed(b"data: line3")
        assert lines == []
        # 完成后 flush
        assert a.flush() == ["data: line3"]

    def test_crlf_split(self):
        a = SSELineAccumulator()
        lines = a.feed(b"data: line1\r\ndata: line2\r\n")
        assert lines == ["data: line1", "data: line2"]


# ── Chat ↔ IR 流式 ────────────────────────────────────────────────


class TestChatParseStream:
    async def test_text_only(self):
        sse = [
            'data: {"id":"cmpl-1","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}',
            '',
            'data: {"choices":[{"delta":{"content":"Hello "}}]}',
            '',
            'data: {"choices":[{"delta":{"content":"world!"}}]}',
            '',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            '',
            'data: [DONE]',
        ]
        resp = MockResp(sse)
        events = await _collect(chat_parse(resp, model='gpt-5'))
        types = [e.type for e in events]
        assert types == [
            "message_start", "text_start", "text_delta", "text_delta", "text_end",
            "message_stop",
        ]
        assert events[0].data["id"] == "cmpl-1"
        assert events[2].data["text"] == "Hello "
        assert events[3].data["text"] == "world!"
        assert events[5].data["stop_reason"] == "end_turn"

    async def test_thinking_tag_extraction(self):
        sse = [
            'data: {"id":"cmpl-1","model":"gpt-5","choices":[{"delta":{"role":"assistant","content":""}}]}',
            '',
            'data: {"choices":[{"delta":{"content":"<think>I should respond</think>Hello"}}]}',
            '',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            '',
            'data: [DONE]',
        ]
        resp = MockResp(sse)
        events = await _collect(chat_parse(resp, model='gpt-5'))
        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types
        # text block 也应存在
        text_events = [e for e in events if e.type == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0].data["text"] == "Hello"

    async def test_reasoning_content_field(self):
        """DeepSeek/o1 原生 reasoning_content 字段。"""
        sse = [
            'data: {"id":"cmpl-1","model":"deepseek","choices":[{"delta":{"role":"assistant","content":""}}]}',
            '',
            'data: {"choices":[{"delta":{"reasoning_content":"Let me think"}}]}',
            '',
            'data: {"choices":[{"delta":{"content":"OK"}}]}',
            '',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            '',
            'data: [DONE]',
        ]
        resp = MockResp(sse)
        events = await _collect(chat_parse(resp, model='deepseek'))
        thinking_deltas = [e for e in events if e.type == "thinking_delta"]
        assert any(d.data.get("thinking") == "Let me think" for d in thinking_deltas)

    async def test_tool_call_full_lifecycle(self):
        sse = [
            'data: {"id":"cmpl-1","model":"gpt-5","choices":[{"delta":{"role":"assistant","content":""}}]}',
            '',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":""}}]}}]}',
            '',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":"}}]}}]}',
            '',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"test\\"}"}}]}}]}',
            '',
            'data: {"choices":[{"finish_reason":"tool_calls"}]}',
            '',
            'data: [DONE]',
        ]
        resp = MockResp(sse)
        events = await _collect(chat_parse(resp, model='gpt-5'))
        starts = [e for e in events if e.type == "tool_use_start"]
        deltas = [e for e in events if e.type == "tool_use_delta"]
        ends = [e for e in events if e.type == "tool_use_end"]
        assert len(starts) == 1
        assert starts[0].data["name"] == "search"
        assert len(deltas) == 2
        assert len(ends) == 1
        assert ends[0].data["input"] == {"q": "test"}
        # stop_reason 应为 tool_use
        stops = [e for e in events if e.type == "message_stop"]
        assert stops[0].data["stop_reason"] == "tool_use"

    async def test_usage_extracted(self):
        sse = [
            'data: {"id":"cmpl-1","model":"gpt-5","choices":[{"delta":{"content":"hi"}}]}',
            '',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            '',
            'data: {"usage":{"prompt_tokens":15,"completion_tokens":3}}',
            '',
            'data: [DONE]',
        ]
        resp = MockResp(sse)
        events = await _collect(chat_parse(resp, model='gpt-5'))
        usage = [e for e in events if e.type == "usage"]
        assert len(usage) == 1
        assert usage[0].data == {"input_tokens": 15, "output_tokens": 3}


class TestChatFormat:
    async def test_text_to_chat_sse(self):
        async def events_aiter():
            for e in [
                IRStreamEvent("message_start", {"id": "cmpl-1", "model": "gpt-5"}),
                IRStreamEvent("text_start", {}),
                IRStreamEvent("text_delta", {"text": "Hello"}),
                IRStreamEvent("text_end", {}),
                IRStreamEvent("message_stop", {"stop_reason": "end_turn"}),
            ]:
                yield e
        events_aiter = events_aiter()
        chunks = []
        async for b in chat_format(events_aiter, model='gpt-5'):
            chunks.append(b.decode())
        # 解析所有 data 行（去掉 [DONE]）
        data_chunks = []
        for c in chunks:
            if c.startswith('data: '):
                payload_str = c[6:].strip()
                if payload_str.startswith('['):  # [DONE]
                    continue
                try:
                    data_chunks.append(json.loads(payload_str))
                except json.JSONDecodeError:
                    pass
        # 首个 chunk 是 role delta
        assert data_chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        # text delta chunk 存在
        text_chunks = [
            d for d in data_chunks
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content") == "Hello"
        ]
        assert len(text_chunks) == 1
        # 末尾的 finish_reason 是 stop
        finish_chunks = [
            d for d in data_chunks
            if d.get("choices") and d["choices"][0].get("finish_reason") == "stop"
        ]
        assert len(finish_chunks) == 1
        # 末尾有 [DONE]
        assert any('[DONE]' in c for c in chunks)

    async def test_thinking_to_reasoning_content(self):
        async def events_aiter():
            for e in [
            IRStreamEvent("message_start", {"id": "x", "model": "gpt-5"}),
            IRStreamEvent("thinking_start", {}),
            IRStreamEvent("thinking_delta", {"thinking": "reasoning"}),
            IRStreamEvent("text_delta", {"text": "answer"}),
            IRStreamEvent("message_stop", {"stop_reason": "end_turn"}),
        ]:
                yield e
        events_aiter = events_aiter()
        chunks = []
        async for b in chat_format(events_aiter, model='gpt-5'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        # reasoning_content 字段
        assert "reasoning_content" in text
        # 末尾有 [DONE]
        assert '[DONE]' in text

    async def test_tool_use_to_tool_calls(self):
        async def events_aiter():
            for e in [
            IRStreamEvent("message_start", {"id": "x", "model": "gpt-5"}),
            IRStreamEvent("tool_use_start", {"id": "call_1", "name": "search"}),
            IRStreamEvent("tool_use_delta", {"id": "call_1", "arguments_delta": '{"q":'}),
            IRStreamEvent("tool_use_delta", {"id": "call_1", "arguments_delta": '"x"}'}),
            IRStreamEvent("message_stop", {"stop_reason": "tool_use"}),
        ]:
                yield e
        events_aiter = events_aiter()
        chunks = []
        async for b in chat_format(events_aiter, model='gpt-5'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        # tool_calls 在 chunk 中
        assert "tool_calls" in text
        # 末尾的 finish_reason 是 tool_calls
        assert "finish_reason" in text and "tool_calls" in text


# ── Anthropic ↔ IR 流式 ───────────────────────────────────────────


class TestAnthropicParseStream:
    async def test_text_block_lifecycle(self):
        sse = [
            'event: message_start',
            'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}',
            '',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            '',
            'event: content_block_stop',
            'data: {"type":"content_block_stop","index":0}',
            '',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":10,"output_tokens":5}}',
            '',
            'event: message_stop',
            'data: {"type":"message_stop"}',
            '',
        ]
        resp = MockResp(sse)
        events = await _collect(anthropic_parse(resp, model='claude'))
        types = [e.type for e in events]
        assert types == [
            "message_start", "text_start", "text_delta", "text_end",
            "usage", "message_stop",
        ]
        assert events[0].data["id"] == "msg_1"
        assert events[2].data["text"] == "Hello"
        assert events[4].data == {"input_tokens": 10, "output_tokens": 5}
        assert events[5].data["stop_reason"] == "end_turn"

    async def test_thinking_block(self):
        sse = [
            'event: message_start',
            'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}',
            '',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning"}}',
            '',
            'event: content_block_stop',
            'data: {"type":"content_block_stop","index":0}',
            '',
            'event: message_stop',
            'data: {"type":"message_stop"}',
            '',
        ]
        resp = MockResp(sse)
        events = await _collect(anthropic_parse(resp, model='claude'))
        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types
        # 查 thinking_delta 的内容
        td = [e for e in events if e.type == "thinking_delta"]
        assert td[0].data["thinking"] == "reasoning"

    async def test_tool_use_input_json_accumulation(self):
        """input_json_delta 片段累积到 tool_use_end 时一次 parse。"""
        sse = [
            'event: message_start',
            'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}',
            '',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"search","input":{}}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":"}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"x\\"}"}}',
            '',
            'event: content_block_stop',
            'data: {"type":"content_block_stop","index":0}',
            '',
            'event: message_stop',
            'data: {"type":"message_stop"}',
            '',
        ]
        resp = MockResp(sse)
        events = await _collect(anthropic_parse(resp, model='claude'))
        ends = [e for e in events if e.type == "tool_use_end"]
        assert len(ends) == 1
        # input 应被累积并解析为 dict
        assert ends[0].data["id"] == "toolu_1"
        assert ends[0].data["input"] == {"q": "x"}


class TestAnthropicFormat:
    async def test_ir_to_anthropic_sse(self):
        async def events_aiter():
            for e in [
            IRStreamEvent("message_start", {"id": "msg_1", "model": "claude"}),
            IRStreamEvent("thinking_start", {}),
            IRStreamEvent("thinking_delta", {"thinking": "reasoning"}),
            IRStreamEvent("thinking_end", {}),
            IRStreamEvent("text_start", {}),
            IRStreamEvent("text_delta", {"text": "Hello"}),
            IRStreamEvent("text_end", {}),
            IRStreamEvent("message_stop", {"stop_reason": "end_turn"}),
        ]:
                yield e
        events_aiter = events_aiter()
        chunks = []
        async for b in anthropic_format(events_aiter, model='claude'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        # 必须含 Anthropic 标准 event
        assert "event: message_start" in text
        assert "event: content_block_start" in text
        assert "event: content_block_delta" in text
        assert "event: content_block_stop" in text
        assert "event: message_delta" in text
        assert "event: message_stop" in text
        # 含 thinking 块
        assert "thinking_delta" in text
        # text 块
        assert "text_delta" in text
        # stop_reason
        assert "end_turn" in text

    async def test_tool_use_to_anthropic(self):
        async def events_aiter():
            for e in [
            IRStreamEvent("message_start", {"id": "msg_1", "model": "claude"}),
            IRStreamEvent("tool_use_start", {"id": "toolu_1", "name": "search"}),
            IRStreamEvent("tool_use_delta", {"id": "toolu_1", "arguments_delta": '{"q":"x"}'}),
            IRStreamEvent("tool_use_end", {"id": "toolu_1", "input": {"q": "x"}}),
            IRStreamEvent("message_stop", {"stop_reason": "tool_use"}),
        ]:
                yield e
        events_aiter = events_aiter()
        chunks = []
        async for b in anthropic_format(events_aiter, model='claude'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        assert "tool_use" in text
        assert "input_json_delta" in text
        assert "stop_reason" in text


# ── Responses ↔ IR 流式 ───────────────────────────────────────────


class TestResponsesParseStream:
    async def test_text_item_lifecycle(self):
        sse = [
            'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5","status":"in_progress","output":[]}}',
            '',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","content":[],"role":"assistant"}}',
            '',
            'data: {"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}',
            '',
            'data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hi"}',
            '',
            'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","text":"Hi"}],"role":"assistant"}}',
            '',
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5}}}',
            '',
        ]
        resp = MockResp(sse)
        events = await _collect(responses_parse(resp, model='gpt-5'))
        types = [e.type for e in events]
        assert types == [
            "message_start", "text_start", "text_delta", "text_end",
            "usage", "message_stop",
        ]

    async def test_reasoning_item_lifecycle(self):
        sse = [
            'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5","output":[]}}',
            '',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress","summary":[]}}',
            '',
            'data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"reasoning text"}',
            '',
            'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"reasoning text"}]}}',
            '',
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}',
        ]
        resp = MockResp(sse)
        events = await _collect(responses_parse(resp, model='gpt-5'))
        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types

    async def test_function_call_lifecycle(self):
        sse = [
            'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5","output":[]}}',
            '',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"search","arguments":""}}',
            '',
            'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\\"q\\":\\"x\\"}"}',
            '',
            'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"search","arguments":"{\\"q\\":\\"x\\"}"}}',
            '',
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}',
        ]
        resp = MockResp(sse)
        events = await _collect(responses_parse(resp, model='gpt-5'))
        starts = [e for e in events if e.type == "tool_use_start"]
        deltas = [e for e in events if e.type == "tool_use_delta"]
        ends = [e for e in events if e.type == "tool_use_end"]
        assert len(starts) == 1
        assert starts[0].data["name"] == "search"
        assert len(deltas) == 1
        assert len(ends) == 1
        assert ends[0].data["input"] == {"q": "x"}


class TestResponsesFormat:
    async def test_basic_text_to_responses_sse(self):
        async def events_aiter():
            for e in [
            IRStreamEvent("message_start", {"id": "resp_1", "model": "gpt-5"}),
            IRStreamEvent("text_start", {}),
            IRStreamEvent("text_delta", {"text": "Hello"}),
            IRStreamEvent("text_end", {}),
            IRStreamEvent("usage", {"input_tokens": 10, "output_tokens": 5}),
            IRStreamEvent("message_stop", {"stop_reason": "end_turn"}),
        ]:
                yield e
        events_aiter = events_aiter()
        chunks = []
        async for b in responses_format(events_aiter, model='gpt-5'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        # 应有 Responses 标准 event
        assert "event: response.created" in text
        assert "event: response.output_item.added" in text
        assert "event: response.content_part.added" in text
        assert "event: response.output_text.delta" in text
        assert "event: response.output_text.done" in text
        assert "event: response.output_item.done" in text
        assert "event: response.completed" in text
        assert "data: [DONE]" in text
        # usage
        assert '"input_tokens": 10' in text

    async def test_apply_patch_reverse_to_custom_tool_call(self):
        """standard file tool → apply_patch custom_tool_call via reverse_tool_map。"""
        async def events_aiter():
            for e in [
            IRStreamEvent("message_start", {"id": "resp_1", "model": "gpt-5"}),
            IRStreamEvent("tool_use_start", {"id": "call_1", "name": "write_to_file"}),
            IRStreamEvent("tool_use_end", {
                "id": "call_1",
                "input": {"filePath": "/tmp/x.py", "content": "print('hi')"},
            }),
            IRStreamEvent("message_stop", {"stop_reason": "tool_use"}),
        ]:
                yield e
        events_aiter = events_aiter()
        reverse_tool_map = {"write_to_file": "apply_patch"}
        chunks = []
        async for b in responses_format(
            events_aiter, model='gpt-5', reverse_tool_map=reverse_tool_map,
        ):
            chunks.append(b)
        text = b''.join(chunks).decode()
        # 必须含 custom_tool_call
        assert "custom_tool_call" in text
        # 必须含 apply_patch 名字
        assert '"name": "apply_patch"' in text
        # 必须含反向构造的 apply_patch DSL
        assert "*** Add File" in text
        assert "/tmp/x.py" in text
        assert "+print" in text

    async def test_namespace_map_adds_namespace_field(self):
        """namespace_map 命中的 tool → function_call + namespace 字段。"""
        async def events_aiter():
            for e in [
            IRStreamEvent("message_start", {"id": "resp_1", "model": "gpt-5"}),
            IRStreamEvent("tool_use_start", {"id": "call_1", "name": "search"}),
            IRStreamEvent("tool_use_delta", {"id": "call_1", "arguments_delta": '{"q":"x"}'}),
            IRStreamEvent("tool_use_end", {"id": "call_1", "input": {"q": "x"}}),
            IRStreamEvent("message_stop", {"stop_reason": "tool_use"}),
        ]:
                yield e
        events_aiter = events_aiter()
        namespace_map = {"search": "mcp__web_search"}
        chunks = []
        async for b in responses_format(
            events_aiter, model='gpt-5', namespace_map=namespace_map,
        ):
            chunks.append(b)
        text = b''.join(chunks).decode()
        # 应是 function_call（不是 custom_tool_call）
        assert "function_call" in text
        assert "custom_tool_call" not in text
        # 含 namespace 字段
        assert '"namespace": "mcp__web_search"' in text


# ── 跨协议流式 ────────────────────────────────────────────────────


class TestCrossProtocolStream:
    async def test_chat_to_responses_basic(self):
        """Chat 风格 SSE → Responses 风格 SSE。"""
        chat_sse = [
            'data: {"id":"cmpl-1","model":"gpt-5","choices":[{"delta":{"role":"assistant","content":""}}]}',
            '',
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            '',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            '',
            'data: [DONE]',
        ]
        chat_resp = MockResp(chat_sse)

        # Chat → IR
        ir_events = await _collect(chat_parse(chat_resp, model='gpt-5'))

        # IR → Responses
        async def event_gen():
            for e in ir_events:
                yield e
        chunks = []
        async for b in responses_format(event_gen(), model='gpt-5'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        assert "response.created" in text
        assert "output_text.delta" in text
        assert "response.completed" in text
        assert "Hello" in text

    async def test_responses_to_chat_basic(self):
        """Responses 风格 SSE → Chat 风格 SSE。"""
        responses_sse = [
            'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5","output":[]}}',
            '',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","content":[],"role":"assistant"}}',
            '',
            'data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hi"}',
            '',
            'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","text":"Hi"}],"role":"assistant"}}',
            '',
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}',
        ]
        resp = MockResp(responses_sse)
        ir_events = await _collect(responses_parse(resp, model='gpt-5'))

        # IR → Chat
        async def event_gen():
            for e in ir_events:
                yield e
        chunks = []
        async for b in chat_format(event_gen(), model='gpt-5'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        assert "Hi" in text and "content" in text
        assert "[DONE]" in text

    async def test_anthropic_to_responses_basic(self):
        """Anthropic 风格 → Responses 风格。"""
        anthropic_sse = [
            'event: message_start',
            'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}',
            '',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            '',
            'event: content_block_stop',
            'data: {"type":"content_block_stop","index":0}',
            '',
            'event: message_stop',
            'data: {"type":"message_stop"}',
            '',
        ]
        resp = MockResp(anthropic_sse)
        ir_events = await _collect(anthropic_parse(resp, model='claude'))

        # IR → Responses
        async def event_gen():
            for e in ir_events:
                yield e
        chunks = []
        async for b in responses_format(event_gen(), model='gpt-5'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        assert "response.created" in text
        assert "output_text.delta" in text
        assert "Hello" in text

    async def test_chat_to_anthropic_basic(self):
        """Chat 风格 → Anthropic 风格。"""
        chat_sse = [
            'data: {"id":"cmpl-1","model":"gpt-5","choices":[{"delta":{"role":"assistant","content":""}}]}',
            '',
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            '',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            '',
            'data: [DONE]',
        ]
        resp = MockResp(chat_sse)
        ir_events = await _collect(chat_parse(resp, model='gpt-5'))

        async def event_gen():
            for e in ir_events:
                yield e
        chunks = []
        async for b in anthropic_format(event_gen(), model='claude'):
            chunks.append(b)
        text = b''.join(chunks).decode()
        assert "event: message_start" in text
        assert "content_block_delta" in text
        assert "Hello" in text
        assert "event: message_stop" in text


# ── Keepalive 包装器 ─────────────────────────────────────────────


class TestKeepaliveWrapper:
    async def test_data_passes_through(self):
        async def src():
            yield b"data: hello\n\n"
            yield b"data: world\n\n"

        out = []
        async for chunk in keepalive_wrapper(src(), interval=0.1):
            out.append(chunk)
        assert b"data: hello" in b"".join(out)
        assert b"data: world" in b"".join(out)

    async def test_keepalive_inserted_on_idle(self):
        async def src():
            yield b"data: first\n\n"
            await asyncio.sleep(0.3)  # 超过 interval
            yield b"data: second\n\n"

        out = []
        async for chunk in keepalive_wrapper(src(), interval=0.1):
            out.append(chunk)
        text = b"".join(out).decode()
        # 第一个数据后应有 keepalive
        assert "data: first" in text
        assert ": keepalive" in text
        assert "data: second" in text


# ── 回归测试（review 修复点）───────────────────────────────────────


class TestRegressionReviewFixes:
    """针对第二轮 review 找出的问题的回归测试。"""

    async def test_responses_streaming_output_items_populated(self):
        """S2: response.completed 的 output 数组必须包含已闭合的 items。"""
        sse = [
            'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5","status":"in_progress","output":[]}}',
            '',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","content":[],"role":"assistant"}}',
            '',
            'data: {"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}',
            '',
            'data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hi"}',
            '',
            'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","text":"Hi"}],"role":"assistant"}}',
            '',
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5}}}',
            '',
        ]
        resp = MockResp(sse)
        events = responses_parse(resp, model="gpt-5")

        chunks = []
        async for b in responses_format(events, model="gpt-5"):
            chunks.append(b)
        text = b"".join(chunks).decode()

        # 找到 response.completed 事件
        import re
        m = re.search(r"event: response\.completed\ndata: (.+)\n\n", text)
        assert m is not None, "response.completed event not found"
        payload = json.loads(m.group(1))
        completed = payload.get("response", payload)
        # output 不应该是空
        assert len(completed.get("output", [])) >= 1
        msg_item = completed["output"][0]
        assert msg_item["type"] == "message"
        assert msg_item["content"][0]["text"] == "Hi"

    async def test_responses_function_calls_have_unique_output_index(self):
        """S5: 多个 function_call 的 output_index 必须递增，不能共享同一 index。"""
        sse = [
            'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5","status":"in_progress","output":[]}}',
            '',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"c1","name":"a","arguments":""}}',
            '',
            'data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_2","type":"function_call","status":"in_progress","call_id":"c2","name":"b","arguments":""}}',
            '',
            'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{}"}',
            '',
            'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{}","call_id":"c1","name":"a"}}',
            '',
            'data: {"type":"response.function_call_arguments.done","item_id":"fc_2","output_index":1,"arguments":"{}"}',
            '',
            'data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_2","type":"function_call","status":"completed","arguments":"{}","call_id":"c2","name":"b"}}',
            '',
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5}}}',
            '',
        ]
        resp = MockResp(sse)
        events = responses_parse(resp, model="gpt-5")

        chunks = []
        async for b in responses_format(events, model="gpt-5"):
            chunks.append(b)
        text = b"".join(chunks).decode()

        import re
        # 找到所有 function_call 的 output_index
        added_matches = re.findall(
            r'event: response\.output_item\.added\ndata: (\{.+?\})\n\n', text
        )
        output_indexes = []
        for payload_str in added_matches:
            payload = json.loads(payload_str)
            if payload.get("item", {}).get("type") == "function_call":
                output_indexes.append(payload["output_index"])
        assert output_indexes == [0, 1], f"Expected [0, 1], got {output_indexes}"

    async def test_chat_tool_call_indices_are_incremented(self):
        """N3: Chat 流式多个 tool_call 的 index 必须递增。"""
        async def events_aiter():
            for e in [
                IRStreamEvent("message_start", {"id": "x", "model": "gpt-5"}),
                IRStreamEvent("tool_use_start", {"id": "call_1", "name": "search"}),
                IRStreamEvent("tool_use_delta", {"id": "call_1", "arguments_delta": '{"q":'}),
                IRStreamEvent("tool_use_end", {"id": "call_1", "input": {}}),
                IRStreamEvent("tool_use_start", {"id": "call_2", "name": "fetch"}),
                IRStreamEvent("tool_use_delta", {"id": "call_2", "arguments_delta": '{"u":'}),
                IRStreamEvent("tool_use_end", {"id": "call_2", "input": {}}),
                IRStreamEvent("message_stop", {"stop_reason": "tool_use"}),
            ]:
                yield e

        chunks = []
        async for b in chat_format(events_aiter(), model="gpt-5"):
            chunks.append(b)
        text = b"".join(chunks).decode()

        # 解析所有 tool_calls chunks
        tool_indexes = []
        for line in text.split("\n\n"):
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                tcs = payload.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
                if tcs:
                    for tc in tcs:
                        if "index" in tc:
                            tool_indexes.append(tc["index"])
        # 第一次 tool_use_start 是 0，第二次是 1
        assert 0 in tool_indexes
        assert 1 in tool_indexes

    async def test_chat_created_field_is_unix_timestamp(self):
        """N4: Chat 流式 chunk 的 created 字段必须是 Unix 时间戳（>0）。"""
        async def events_aiter():
            for e in [
                IRStreamEvent("message_start", {"id": "x", "model": "gpt-5"}),
                IRStreamEvent("text_delta", {"text": "hi"}),
                IRStreamEvent("message_stop", {"stop_reason": "end_turn"}),
            ]:
                yield e

        chunks = []
        async for b in chat_format(events_aiter(), model="gpt-5"):
            chunks.append(b)
        text = b"".join(chunks).decode()

        # 所有 chunk 的 created 字段 > 0
        for line in text.split("\n\n"):
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if "created" in payload:
                    assert payload["created"] > 0, f"created should be > 0, got {payload['created']}"
