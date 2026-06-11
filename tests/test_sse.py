"""SSE 流协议单元测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from llm_proxy.protocol.sse import stream_response


class FakeResponse:
    def __init__(self, chunks: list[bytes], fail_after: int = -1):
        self._chunks = chunks
        self._fail_after = fail_after
        self._count = 0

    async def aiter_bytes(self):
        for chunk in self._chunks:
            if 0 <= self._fail_after <= self._count:
                raise ConnectionError("simulated connection drop")
            self._count += 1
            yield chunk

    async def aiter_lines(self):
        for chunk in self._chunks:
            if 0 <= self._fail_after <= self._count:
                raise ConnectionError("simulated connection drop")
            self._count += 1
            for line in chunk.decode().split("\n"):
                yield line


@pytest.mark.asyncio
async def test_stream_forward():
    chunks = [
        b'event: message_start\ndata: {"type":"message_start"}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    resp = FakeResponse(chunks)
    result = []
    async for chunk in stream_response(resp):
        result.append(chunk)
    assert len(result) == 12
    assert result[0] == b'event: message_start\n'
    assert result[-2] == b'\n'
    assert b'message_stop' in result[-3]


@pytest.mark.asyncio
async def test_stream_interrupted():
    chunks = [
        b'event: message_start\ndata: {"type":"message_start"}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
    ]
    resp = FakeResponse(chunks, fail_after=1)
    result = []
    async for chunk in stream_response(resp):
        result.append(chunk)
    assert len(result) >= 2
    assert b'event: error' in result[-1]


@pytest.mark.asyncio
async def test_on_chunk_callback():
    chunks = [b'event: message_start\ndata: {}\n\n']
    resp = FakeResponse(chunks)
    captured = []
    async for _ in stream_response(resp, on_chunk=lambda c: captured.append(c)):
        pass
    assert len(captured) == 4
    assert captured[0] == b'event: message_start\n'
    assert captured[2] == b'\n'


@pytest.mark.asyncio
async def test_on_event_callback():
    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":50}}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":30}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    resp = FakeResponse(chunks)
    events = []
    async for _ in stream_response(resp, on_event=lambda t, d: events.append((t, d))):
        pass
    assert len(events) == 4
    assert events[0][0] == "message_start"
    assert events[0][1]["message"]["usage"]["input_tokens"] == 50
    assert events[2][0] == "message_delta"
    assert events[2][1]["usage"]["output_tokens"] == 30
    assert events[3][0] == "message_stop"
