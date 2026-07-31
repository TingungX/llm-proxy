"""IRProxyStep 同协议透传：跳过 IR 转换，body 原样转发（仅 model 映射）。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.handlers.base import PipelineContext
from llm_proxy.handlers.shared.ir_proxy import IRProxyStep

_RESOLVED = (
    "https://api.example.com",
    "upstream-key",
    "upstream-model-name",
    "my-model",
    "openai/responses",
    None,
)


def _make_ctx(body: dict) -> PipelineContext:
    endpoint = {"endpoint_id": "ep-test", "models": [], "model_map": {}}
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test"}
    return PipelineContext(
        request=request,
        body=body,
        headers={},
        error_protocol="openai",
        resolved=_RESOLVED,
        endpoint=endpoint,
        response_model="client-model-name",
        extra={},
    )


@pytest.mark.asyncio
async def test_same_protocol_responses_skips_ir_and_passthrough():
    """upstream 也是 openai/responses 时走裸透传，不调用 to_ir。"""
    apply_patch_body = {
        "model": "my-model",
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** Add File: a.py\n+x\n*** End Patch",
            },
        ],
        "tools": [{"type": "custom", "name": "apply_patch", "description": "patch"}],
        "stream": False,
    }
    ctx = _make_ctx(apply_patch_body)
    mock_state = MagicMock()
    mock_state.protocols_map = {"my-model": {"openai/responses"}}
    mock_state.paths_map = {"my-model": {}}

    with patch("llm_proxy.handlers.shared.ir_proxy.get_state", return_value=mock_state), \
         patch("llm_proxy.handlers.shared.ir_proxy.select_upstream", return_value="openai/responses"), \
         patch("llm_proxy.handlers.shared.ir_proxy.REGISTRY") as mock_registry, \
         patch.object(IRProxyStep, "_proxy_same_protocol", new_callable=AsyncMock) as mock_passthrough:
        await IRProxyStep("openai/responses").execute(ctx)

        mock_passthrough.assert_awaited_once()
        mock_registry.__getitem__.return_value.to_ir.assert_not_called()


@pytest.mark.asyncio
async def test_cross_protocol_still_uses_ir():
    """upstream 为 chat 时仍走 IR 转换。"""
    ctx = _make_ctx({"model": "my-model", "input": "hi", "stream": False})
    mock_state = MagicMock()
    mock_state.protocols_map = {"my-model": {"openai/chat-completions"}}
    mock_state.paths_map = {"my-model": {}}
    mock_state.get_model_effort_mapping.return_value = None

    mock_ir_module = MagicMock()
    mock_ir_request = MagicMock()
    mock_ir_request.stream = False
    mock_ir_request.reasoning_effort = None
    mock_ir_request.extensions = {}
    mock_ir_module.to_ir.return_value = mock_ir_request
    mock_ir_module.to_upstream.return_value = {"model": "upstream-model-name", "messages": []}
    mock_ir_module.response_to_ir.return_value = MagicMock(usage={})
    mock_ir_module.response_from_ir.return_value = {"id": "resp_1"}

    with patch("llm_proxy.handlers.shared.ir_proxy.get_state", return_value=mock_state), \
         patch("llm_proxy.handlers.shared.ir_proxy.select_upstream", return_value="openai/chat-completions"), \
         patch("llm_proxy.handlers.shared.ir_proxy.REGISTRY", {
             "openai/responses": mock_ir_module,
             "openai/chat-completions": mock_ir_module,
         }), \
         patch.object(IRProxyStep, "_proxy_same_protocol", new_callable=AsyncMock) as mock_passthrough, \
         patch.object(IRProxyStep, "_proxy_non_stream", new_callable=AsyncMock) as mock_non_stream:
        await IRProxyStep("openai/responses").execute(ctx)

        mock_passthrough.assert_not_awaited()
        mock_ir_module.to_ir.assert_called_once()
        mock_non_stream.assert_awaited_once()


def test_proxy_responses_direct_maps_model_on_request():
    """裸透传请求侧应替换为 upstream actual_model。"""
    from llm_proxy.handlers.shared.proxy import ProxyStep

    ctx = _make_ctx({
        "model": "client-model-name",
        "input": "hello",
        "stream": False,
    })
    captured: dict = {}

    async def fake_post(url, json, headers, timeout):
        captured["json"] = json
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "r1", "usage": {"input_tokens": 1, "output_tokens": 2}}
        return resp

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=fake_post)

    with patch("llm_proxy.handlers.shared.proxy.get_client", return_value=mock_client), \
         patch("llm_proxy.handlers.shared.proxy.get_state") as mock_state:
        mock_state.return_value.paths_map = {"my-model": {}}
        asyncio.run(ProxyStep()._proxy_responses_direct(
            ctx, "https://api.example.com", "key", "upstream-model-name", "my-model", "ep-test",
        ))

    assert captured["json"]["model"] == "upstream-model-name"
    assert captured["json"]["input"] == "hello"
    assert ctx.response is not None


@pytest.mark.asyncio
async def test_responses_direct_stream_preserves_sse_framing():
    """流式透传应 relay 原始字节（含 \\n\\n 分隔），不能用 aiter_lines 拼接。"""
    from llm_proxy.handlers.shared.proxy import ProxyStep

    raw_sse = (
        b'event: response.created\n'
        b'data: {"type":"response.created","model":"m"}\n'
        b'\n'
        b'event: response.completed\n'
        b'data: {"type":"response.completed","usage":{"input_tokens":3,"output_tokens":7}}\n'
        b'\n'
    )

    class FakeStreamResponse:
        status_code = 200

        async def aiter_bytes(self):
            yield raw_sse

        async def aread(self):
            return b""

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=FakeStreamResponse())
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    chunks: list[bytes] = []
    with patch("llm_proxy.handlers.shared.proxy.get_client", return_value=mock_client), \
         patch("llm_proxy.handlers.shared.proxy.db") as mock_db:
        async for chunk in ProxyStep()._responses_direct_stream(
            "https://api.example.com/v1/responses",
            {"Authorization": "Bearer k"},
            {"model": "m", "input": "hi", "stream": True},
            "m", "ep-test", "my-model",
        ):
            chunks.append(chunk)

    relayed = b"".join(chunks)
    assert b"data: {\"type\":\"response.created\"" in relayed
    assert b"\n\nevent: response.completed" in relayed
    mock_db.record_usage.assert_called_once()
    assert mock_db.record_usage.call_args[0][3] == 7  # output_tokens

