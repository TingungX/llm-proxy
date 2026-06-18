"""Regression tests for ResponsesConvertStep — bug fix for `tool_spec_map` UnboundLocalError.

背景：
  Codex 在 compact 完成后的下一轮请求会省略 `tools` 字段。
  `llm_proxy/handlers/shared/responses_convert.py` 的 `ResponsesConvertStep.execute()`
  在 `if body.get("tools"):` 块里才给 `tool_spec_map` 赋值，但函数末尾无条件
  访问 `tool_spec_map`（line ~117 `ctx.tool_spec_map = tool_spec_map or None`），
  导致 Codex 每次 compact 后下一轮 100% 抛 `UnboundLocalError`，HTTP 500。

证据：
  - /private/tmp/llm-proxy.log @ 16:42:20-16:42:45 (30+ 6-item requests, all 500)
  - /private/tmp/llm-proxy.log @ 17:00:07-17:00:31 (30 6-item requests, all 500)

本测试文件锁定该 bug 的回归行为。
"""

import asyncio
from unittest.mock import MagicMock

from llm_proxy.handlers.base import PipelineContext
from llm_proxy.handlers.shared.responses_convert import ResponsesConvertStep
from llm_proxy.protocol.responses_chat.request import CodexToolSpec


# 来自 Codex 16:42 / 17:00 真实 payload 模式的 6 元组
# (api_base, api_key, actual_model, config_key, upstream_protocol, failover_family)
_RESOLVED = (
    "https://api.example.com",
    "test-api-key",
    "minimax-m3",
    "minimax-m3",
    "openai",
    None,
)


def _make_ctx(
    body: dict,
    *,
    converter: str = "responses_to_chat",
) -> PipelineContext:
    """Build a minimal PipelineContext for ResponsesConvertStep tests.

    Request 字段在 ResponsesConvertStep 中并未被访问，使用 MagicMock 占位即可。
    """
    return PipelineContext(
        request=MagicMock(),
        body=body,
        headers={},
        error_protocol="openai",
        converter=converter,
        resolved=_RESOLVED,
    )


def _execute(ctx: PipelineContext) -> None:
    """Run an async step synchronously for testing."""
    asyncio.run(ResponsesConvertStep().execute(ctx))


class TestResponsesConvertStepNoTools:
    """回归：Codex compact 完成后下一轮（无 `tools` 字段）的请求必须不抛 UnboundLocalError。"""

    def test_no_tools_field_at_all_does_not_raise(self):
        """17:00-17:01 真实场景：body 中完全没有 `tools` 键。

        修复前：抛 UnboundLocalError: cannot access local variable 'tool_spec_map'...
        修复后：干净返回，ctx.tool_spec_map 被设为 None（or {} 的结果）。
        """
        # 来自 17:00:07 [7c9c5d83] 的真实 input 结构
        ctx = _make_ctx({
            "model": "gpt-5.3-codex",
            "input": [
                {"type": "message", "role": "developer", "content": "x" * 30211},
                {"type": "message", "role": "user", "content": "x" * 1809},
                {"type": "message", "role": "user", "content": "x" * 3},
            ],
            "stream": True,
            "reasoning": {"effort": "high"},
        })
        # 关键：body 中没有 "tools" 键
        assert "tools" not in ctx.body

        # 修复前：会抛 UnboundLocalError
        _execute(ctx)

        # 修复后：ctx.tool_spec_map 应该是 None（empty dict 走 `or None` 分支）
        assert ctx.tool_spec_map is None
        assert ctx.reverse_tool_map is None
        # ctx.body 应该是转换后的 chat_body
        assert "messages" in ctx.body
        assert ctx.body["model"] == "minimax-m3"
        assert ctx.body["stream"] is True
        assert ctx.body["stream_options"] == {"include_usage": True}

    def test_empty_tools_list_does_not_raise(self):
        """body 显式带 `tools=[]`（falsy）也必须不抛。"""
        ctx = _make_ctx({
            "model": "gpt-5.3-codex",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [],
        })

        _execute(ctx)

        assert ctx.tool_spec_map is None
        assert ctx.reverse_tool_map is None

    def test_with_tools_still_works(self):
        """Sanity 检查：带 tools 的请求（原有 happy path）必须不受影响。

        修复不能破坏现有行为：function tool 应该被转换，tool_spec_map
        在没有 namespace 工具时仍为 None。
        """
        ctx = _make_ctx({
            "model": "gpt-5.3-codex",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "description": "Execute a shell command",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        })

        _execute(ctx)

        # 转换后 chat_body 包含 tools（来自 function tool 的转换）
        assert "tools" in ctx.body
        assert len(ctx.body["tools"]) == 1
        assert ctx.body["tools"][0]["function"]["name"] == "exec_command"
        # 没有 custom tool，reverse_tool_map 为 None
        assert ctx.reverse_tool_map is None
        # 没有 namespace tool，tool_spec_map 为 None
        assert ctx.tool_spec_map is None

    def test_with_namespace_tools_populates_tool_spec_map(self):
        """带 namespace 工具的请求必须正确填充 tool_spec_map（这是原行为不能破）。"""
        ctx = _make_ctx({
            "model": "gpt-5.3-codex",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__web_search",
                    "tools": [
                        {
                            "type": "function",
                            "name": "search",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    ],
                },
            ],
        })

        _execute(ctx)

        # tool_spec_map 应该包含 upstream 名 → CodexToolSpec
        expected = CodexToolSpec(kind="namespace", name="search", namespace="mcp__web_search")
        assert ctx.tool_spec_map == {"mcp__web_search__search": expected}


class TestResponsesConvertStepEarlyReturn:
    def test_skips_when_converter_not_set(self):
        """converter 不是 'responses_to_chat' 时，步骤应该是 no-op。

        这保护了 Chat Completions 同协议透传路径不会被 Responses 转换污染。
        """
        ctx = _make_ctx(
            body={"model": "gpt-4", "input": [{"type": "message", "role": "user", "content": "hi"}]},
            converter="chat_to_responses",
        )

        _execute(ctx)

        # body 不应该被改
        assert ctx.body == {"model": "gpt-4", "input": [{"type": "message", "role": "user", "content": "hi"}]}
        # 也没有 tool_spec_map 被设置
        assert ctx.tool_spec_map is None
        assert ctx.reverse_tool_map is None
