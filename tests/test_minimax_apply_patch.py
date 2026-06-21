"""Test for MiniMax M3 apply_patch with literal newlines in args.

Reproduces: Codex→minimax-m3 stream_error after first tool call.

Root cause: MiniMax upstream may send tool_call arguments as a JSON string with
literal newlines (e.g. `{"input": "*** Begin Patch\n*** Add File: ..."}`), which
is technically invalid JSON. The IR Chat parser's IncrementalJSONParser fails to
parse this, returning `{"_raw": "..."}`. The Responses IR custom_tool_call handler
then can't extract the DSL from `_raw`, leaving Codex with an empty input.

Fix: when `_raw` is present in parsed input, fall back to regex-extracting the
embedded DSL string via `_extract_dsl_from_raw`.
"""
import asyncio
import json
import sys

sys.path.insert(0, '/Users/tingung/Projects/github/llm-proxy')

import pytest

from llm_proxy.protocol.ir import REGISTRY, _resolve
from llm_proxy.protocol.ir.responses import _extract_dsl_from_raw


class MockResp:
    def __init__(self, lines):
        self.lines = lines
        self.status_code = 200

    async def aiter_lines(self):
        for line in self.lines:
            yield line


def _make_args_chunk(args_str: str) -> str:
    data = {
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args_str}}]}
        }]
    }
    return f"data: {json.dumps(data)}"


# ── 单元测试：_extract_dsl_from_raw helper ──


def test_extract_dsl_literal_newlines():
    """MiniMax M3 实际发送的字面 \\n JSON。"""
    raw = '{"input": "*** Begin Patch\n*** Add File: /tmp/x\n+hi\n*** End Patch"}'
    got = _extract_dsl_from_raw(raw)
    assert "*** Begin Patch" in got
    assert "*** End Patch" in got


def test_extract_dsl_escaped_newlines():
    """正常 JSON（\\\\n）。"""
    raw = '{"input": "*** Begin Patch\\n*** Add File: /tmp/x\\n+hi\\n*** End Patch"}'
    got = _extract_dsl_from_raw(raw)
    assert "*** Begin Patch" in got
    assert "*** End Patch" in got


def test_extract_dsl_empty():
    assert _extract_dsl_from_raw("") == ""


def test_extract_dsl_no_input_field():
    assert _extract_dsl_from_raw('{"foo": "bar"}') == ""


# ── 集成测试：完整 stream 流程 ──


async def test_minimax_apply_patch_with_literal_newlines_in_stream():
    """完整重现 Codex→minimax-m3 stream_error 的根因场景。"""
    raw_args = '{"input": "*** Begin Patch\n*** Add File: /tmp/test.txt\n+hello\n*** End Patch"}'

    upstream_lines = [
        'data: {"id":"chatcmpl-x","model":"MiniMax-M3","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}',
        '',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"apply_patch","arguments":""}}]}}]}',
        '',
        _make_args_chunk(raw_args),
        '',
        'data: {"choices":[{"index":0,"finish_reason":"tool_calls"}]}',
        '',
        'data: [DONE]',
    ]

    resp = MockResp(upstream_lines)
    chat_proto = _resolve("openai/chat-completions")
    raw_events = REGISTRY[chat_proto].parse_stream_to_ir(resp, model="MiniMax-M3")

    reverse_tool_map = {"apply_patch": "apply_patch"}
    tool_spec_map = {"apply_patch": {"kind": "custom", "name": "apply_patch"}}

    responses_proto = _resolve("openai/responses")
    chunks = []
    async for chunk in REGISTRY[responses_proto].format_ir_as_sse(
        raw_events, model="minimax-m3",
        reverse_tool_map=reverse_tool_map, tool_spec_map=tool_spec_map,
    ):
        chunks.append(chunk.decode())

    # Find the output_item.done chunk for apply_patch
    apply_patch_done = None
    for c in chunks:
        if 'custom_tool_call' in c and 'output_item.done' in c:
            apply_patch_done = c
            break

    assert apply_patch_done is not None, "No custom_tool_call output_item.done found"

    # Extract the input value
    data_line = [l for l in apply_patch_done.split('\n') if l.startswith('data: ')][0]
    payload = json.loads(data_line[6:])
    input_value = payload['item']['input']

    # DSL 应该完整出现在 input 中（修复前是空字符串）
    assert '*** Begin Patch' in input_value, f"DSL marker missing: {input_value!r}"
    assert '*** Add File: /tmp/test.txt' in input_value
    assert '+hello' in input_value
    assert '*** End Patch' in input_value


if __name__ == "__main__":
    asyncio.run(test_minimax_apply_patch_with_literal_newlines_in_stream())
    test_extract_dsl_literal_newlines()
    test_extract_dsl_escaped_newlines()
    test_extract_dsl_empty()
    test_extract_dsl_no_input_field()
    print("All tests passed")