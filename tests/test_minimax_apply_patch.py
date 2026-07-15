"""Test for MiniMax M3 apply_patch with structured parameters in stream.

Reproduces: Codex→minimax-m3 stream_error after first tool call.

Root cause (original): MiniMax upstream may send tool_call arguments as a JSON
string with literal newlines (e.g. `{"input": "*** Begin Patch\n*** Add File: ..."`),
which is technically invalid JSON. The IR Chat parser's IncrementalJSONParser fails
to parse this, returning an empty dict. The Responses IR custom_tool_call handler
then can't extract the DSL, leaving Codex with an empty input.

With the new structured parameters approach, the upstream should send
`{"action": "add_file", "filePath": "...", "content": "..."}` format.
The reverse conversion (reverse_tool_args_to_apply_patch) converts this back to DSL.
"""
import asyncio
import json
import sys

sys.path.insert(0, '/Users/tingung/Projects/github/llm-proxy')

import pytest

from llm_proxy.protocol.ir import REGISTRY, _resolve


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


async def test_minimax_apply_patch_structured_params_in_stream():
    """apply_patch 调用使用结构化参数，反向转换为 DSL。"""
    args = json.dumps({
        "action": "add_file",
        "filePath": "/tmp/test.txt",
        "content": "hello",
    })

    upstream_lines = [
        'data: {"id":"chatcmpl-x","model":"MiniMax-M3","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}',
        '',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"apply_patch","arguments":""}}]}}]}',
        '',
        _make_args_chunk(args),
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

    apply_patch_done = None
    for c in chunks:
        if 'custom_tool_call' in c and 'output_item.done' in c:
            apply_patch_done = c
            break

    assert apply_patch_done is not None, "No custom_tool_call output_item.done found"

    data_line = [l for l in apply_patch_done.split('\n') if l.startswith('data: ')][0]
    payload = json.loads(data_line[6:])
    input_value = payload['item']['input']

    assert '*** Begin Patch' in input_value, f"DSL marker missing: {input_value!r}"
    assert '*** Add File: /tmp/test.txt' in input_value
    assert '+hello' in input_value
    assert '*** End Patch' in input_value


if __name__ == "__main__":
    asyncio.run(test_minimax_apply_patch_structured_params_in_stream())
    print("All tests passed")
