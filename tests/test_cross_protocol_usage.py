"""跨协议流式代理 usage 追踪测试

验证 _cross_protocol_stream_gen 在流式响应完成后调用 db.record_usage()。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from llm_proxy.handlers.shared.proxy import ProxyStep


class FakeStreamResponse:
    """模拟 httpx 流式响应（OpenAI Chat SSE 格式）"""

    def __init__(self, status_code: int, lines: list[str]):
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


def _make_chat_sse_lines(
    input_tokens: int = 100,
    output_tokens: int = 50,
    content: str = "Hello",
    finish_reason: str = "stop",
) -> list[str]:
    """构造 OpenAI Chat Completions SSE 流的行序列"""
    lines = []
    # chunk 1: role + content start
    lines.append(f'data: {json.dumps({"id":"chatcmpl-1","model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":None}]})}\n')
    # chunk 2: content delta
    lines.append(f'data: {json.dumps({"id":"chatcmpl-1","model":"test","choices":[{"index":0,"delta":{"content":content},"finish_reason":None}]})}\n')
    # chunk 3: finish with usage
    lines.append(f'data: {json.dumps({"id":"chatcmpl-1","model":"test","choices":[{"index":0,"delta":{},"finish_reason":finish_reason}],"usage":{"prompt_tokens":input_tokens,"completion_tokens":output_tokens}})}\n')
    # DONE
    lines.append("data: [DONE]\n")
    return lines


@pytest.mark.asyncio
async def test_cross_protocol_stream_records_usage():
    """跨协议流式请求完成后应调用 db.record_usage()"""
    step = ProxyStep()
    sse_lines = _make_chat_sse_lines(input_tokens=200, output_tokens=80)
    fake_resp = FakeStreamResponse(status_code=200, lines=sse_lines)

    # 模拟 httpx client.stream 返回的 async context manager
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=fake_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    with patch("llm_proxy.handlers.shared.proxy.get_client", return_value=mock_client), \
         patch("llm_proxy.handlers.shared.proxy.db") as mock_db:

        chunks = []
        async for chunk in step._cross_protocol_stream_gen(
            target_url="https://api.example.com/v1/chat/completions",
            req_headers={"Authorization": "Bearer sk-test"},
            chat_body={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            original_body={"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            model="claude-opus-4-8",
            endpoint_id="ep123",
            model_id="glm-5",
        ):
            chunks.append(chunk)

        # 断言 db.record_usage 被调用
        mock_db.record_usage.assert_called_once()
        call_args = mock_db.record_usage.call_args
        assert call_args[0][0] == "ep123"   # endpoint_id
        assert call_args[0][1] == "glm-5"   # model_id
        assert call_args[0][2] == 200        # input_tokens
        assert call_args[0][3] == 80         # output_tokens
        assert call_args[0][4] == "success"  # status


@pytest.mark.asyncio
async def test_cross_protocol_stream_records_estimated_usage_on_missing_tokens():
    """当上游未返回 usage 时，应使用估算值记录"""
    step = ProxyStep()
    # 构造不含 usage 的 SSE 流
    lines = []
    lines.append(f'data: {json.dumps({"id":"chatcmpl-1","model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":None}]})}\n')
    lines.append(f'data: {json.dumps({"id":"chatcmpl-1","model":"test","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":None}]})}\n')
    lines.append(f'data: {json.dumps({"id":"chatcmpl-1","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})}\n')
    lines.append("data: [DONE]\n")
    fake_resp = FakeStreamResponse(status_code=200, lines=lines)

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=fake_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    with patch("llm_proxy.handlers.shared.proxy.get_client", return_value=mock_client), \
         patch("llm_proxy.handlers.shared.proxy.db") as mock_db:

        async for _ in step._cross_protocol_stream_gen(
            target_url="https://api.example.com/v1/chat/completions",
            req_headers={"Authorization": "Bearer sk-test"},
            chat_body={"model": "test", "messages": [{"role": "user", "content": "hello world"}], "stream": True},
            original_body={"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hello world"}], "stream": True},
            model="claude-opus-4-8",
            endpoint_id="ep456",
            model_id="glm-5",
        ):
            pass

        mock_db.record_usage.assert_called_once()
        call_args = mock_db.record_usage.call_args
        assert call_args[0][0] == "ep456"
        assert call_args[0][1] == "glm-5"
        # 没有真实 usage 时应该有估算的 input_tokens (>0)
        assert call_args[0][2] > 0


@pytest.mark.asyncio
async def test_cross_protocol_stream_no_usage_on_upstream_error():
    """上游返回错误时应记录失败状态"""
    step = ProxyStep()
    fake_resp = FakeStreamResponse(status_code=500, lines=[])

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=fake_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    with patch("llm_proxy.handlers.shared.proxy.get_client", return_value=mock_client), \
         patch("llm_proxy.handlers.shared.proxy.db") as mock_db:

        async for _ in step._cross_protocol_stream_gen(
            target_url="https://api.example.com/v1/chat/completions",
            req_headers={"Authorization": "Bearer sk-test"},
            chat_body={"model": "test", "messages": [], "stream": True},
            original_body={"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            model="claude-opus-4-8",
            endpoint_id="ep789",
            model_id="glm-5",
        ):
            pass

        # 上游错误时也应记录（带 error 标记）
        mock_db.record_usage.assert_called_once()
        call_args = mock_db.record_usage.call_args
        assert call_args[0][4] == "error"
