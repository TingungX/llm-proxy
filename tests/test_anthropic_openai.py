"""Tests for llm_proxy.protocol.anthropic_openai — Anthropic↔Chat 转换"""

import json
import pytest

from llm_proxy.protocol.anthropic_openai.request import (
    anthropic_to_chat,
    strip_leading_anthropic_billing_header,
    is_openai_o_series,
    supports_reasoning_effort,
    resolve_reasoning_effort,
    clean_schema,
    map_tool_choice_to_chat,
)
from llm_proxy.protocol.anthropic_openai.response import chat_to_anthropic
from llm_proxy.protocol.anthropic_openai.rectifier import should_rectify, rectify_request


# ═══════════════════════════════════════════════════════════════════
# Billing Header
# ═══════════════════════════════════════════════════════════════════

class TestBillingHeader:
    def test_strip_leading(self):
        text = "x-anthropic-billing-header: cch=abc\nYou are helpful."
        assert strip_leading_anthropic_billing_header(text) == "You are helpful."

    def test_no_header(self):
        text = "You are helpful."
        assert strip_leading_anthropic_billing_header(text) == text

    def test_only_header(self):
        text = "x-anthropic-billing-header: cch=abc"
        assert strip_leading_anthropic_billing_header(text) == ""

    def test_header_with_crlf(self):
        text = "x-anthropic-billing-header: cch=abc\r\nYou are helpful."
        assert strip_leading_anthropic_billing_header(text) == "You are helpful."

    def test_later_header_preserved(self):
        text = "System prompt\nx-anthropic-billing-header: cch=abc\nMore text"
        assert strip_leading_anthropic_billing_header(text) == text


# ═══════════════════════════════════════════════════════════════════
# O-series & Reasoning Effort
# ═══════════════════════════════════════════════════════════════════

class TestOSeries:
    @pytest.mark.parametrize("model,expected", [
        ("o1", True), ("o1-preview", True), ("o3-mini", True), ("o4-mini", True),
        ("gpt-4o", False), ("openai-gpt", False), ("o", False), ("", False),
    ])
    def test_is_openai_o_series(self, model, expected):
        assert is_openai_o_series(model) == expected

    @pytest.mark.parametrize("model,expected", [
        ("o1", True), ("o3-mini", True), ("gpt-5", True), ("gpt-5.4", True),
        ("gpt-5-codex", True), ("gpt-4o", False), ("claude-sonnet-4-6", False),
    ])
    def test_supports_reasoning_effort(self, model, expected):
        assert supports_reasoning_effort(model) == expected


class TestReasoningEffort:
    def test_output_config_low(self):
        assert resolve_reasoning_effort({"output_config": {"effort": "low"}}) == "low"

    def test_output_config_max(self):
        assert resolve_reasoning_effort({"output_config": {"effort": "max"}}) == "xhigh"

    def test_output_config_priority_over_thinking(self):
        body = {"output_config": {"effort": "low"}, "thinking": {"type": "adaptive"}}
        assert resolve_reasoning_effort(body) == "low"

    def test_thinking_adaptive(self):
        assert resolve_reasoning_effort({"thinking": {"type": "adaptive"}}) == "xhigh"

    def test_thinking_enabled_small_budget(self):
        assert resolve_reasoning_effort({"thinking": {"type": "enabled", "budget_tokens": 2048}}) == "low"

    def test_thinking_enabled_medium_budget(self):
        assert resolve_reasoning_effort({"thinking": {"type": "enabled", "budget_tokens": 8000}}) == "medium"

    def test_thinking_enabled_large_budget(self):
        assert resolve_reasoning_effort({"thinking": {"type": "enabled", "budget_tokens": 32000}}) == "high"

    def test_thinking_enabled_no_budget(self):
        assert resolve_reasoning_effort({"thinking": {"type": "enabled"}}) == "high"

    def test_thinking_disabled(self):
        assert resolve_reasoning_effort({"thinking": {"type": "disabled"}}) is None

    def test_no_thinking(self):
        assert resolve_reasoning_effort({"messages": []}) is None

    def test_unknown_effort(self):
        assert resolve_reasoning_effort({"output_config": {"effort": "turbo"}}) is None


# ═══════════════════════════════════════════════════════════════════
# Schema & Tool Choice
# ═══════════════════════════════════════════════════════════════════

class TestCleanSchema:
    def test_remove_format_uri(self):
        schema = {"type": "string", "format": "uri"}
        result = clean_schema(schema)
        assert "format" not in result

    def test_keep_other_format(self):
        schema = {"type": "string", "format": "date-time"}
        result = clean_schema(schema)
        assert result.get("format") == "date-time"

    def test_recursive(self):
        schema = {"type": "object", "properties": {"url": {"type": "string", "format": "uri"}}}
        result = clean_schema(schema)
        assert "format" not in result["properties"]["url"]


class TestToolChoice:
    def test_string_any(self):
        assert map_tool_choice_to_chat("any") == "required"

    def test_string_auto(self):
        assert map_tool_choice_to_chat("auto") == "auto"

    def test_string_none(self):
        assert map_tool_choice_to_chat("none") == "none"

    def test_object_any(self):
        assert map_tool_choice_to_chat({"type": "any"}) == "required"

    def test_object_tool(self):
        result = map_tool_choice_to_chat({"type": "tool", "name": "bash"})
        assert result == {"type": "function", "function": {"name": "bash"}}


# ═══════════════════════════════════════════════════════════════════
# Request Conversion
# ═══════════════════════════════════════════════════════════════════

class TestAnthropicToChat:
    def test_basic(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_chat(body)
        assert result["model"] == "gpt-4o"
        assert result["max_tokens"] == 1024
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    def test_o_series_max_completion_tokens(self):
        body = {
            "model": "o3-mini",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_chat(body)
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 4096

    def test_system_string(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_chat(body)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."

    def test_system_array(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "Be helpful"}],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_chat(body)
        assert result["messages"][0]["role"] == "system"

    def test_thinking_to_reasoning_effort(self):
        body = {
            "model": "gpt-5.4",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_chat(body)
        assert result["reasoning_effort"] == "low"

    def test_non_reasoning_model_no_effort(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_chat(body)
        assert "reasoning_effort" not in result

    def test_tool_use(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}},
                ],
            }],
        }
        result = anthropic_to_chat(body)
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert any(tc["function"]["name"] == "bash" for tc in msg["tool_calls"])

    def test_tool_result(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file.txt"},
                ],
            }],
        }
        result = anthropic_to_chat(body)
        msg = result["messages"][0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "toolu_1"

    def test_thinking_block(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "The answer is 42."},
                ],
            }],
        }
        result = anthropic_to_chat(body)
        msg = result["messages"][0]
        assert msg["reasoning_content"] == "Let me think..."
        assert msg["content"] == "The answer is 42."

    def test_tools_and_batch_filter(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {"name": "bash", "description": "Run command", "input_schema": {"type": "object"}},
                {"type": "BatchTool", "name": "batch"},
            ],
        }
        result = anthropic_to_chat(body)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "bash"

    def test_stop_sequences(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "stop_sequences": ["\n\n"],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_chat(body)
        assert result["stop"] == ["\n\n"]

    def test_image(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
                ],
            }],
        }
        result = anthropic_to_chat(body)
        part = result["messages"][0]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_billing_header_stripped(self):
        body = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": "x-anthropic-billing-header: cch=abc\nYou are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_chat(body)
        sys_msg = result["messages"][0]
        assert not sys_msg["content"].startswith("x-anthropic-billing-header")
        assert "helpful" in sys_msg["content"]


# ═══════════════════════════════════════════════════════════════════
# Response Conversion
# ═══════════════════════════════════════════════════════════════════

class TestChatToAnthropic:
    def test_basic_text(self):
        resp = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = chat_to_anthropic(resp)
        assert result["type"] == "message"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert any(b["type"] == "text" and b["text"] == "Hello!" for b in result["content"])

    def test_reasoning_content(self):
        resp = {
            "id": "chatcmpl-123",
            "model": "deepseek-v4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "reasoning_content": "I thought...", "content": "Answer: 42"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = chat_to_anthropic(resp)
        blocks = result["content"]
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "I thought..."
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"] == "Answer: 42"

    def test_tool_calls(self):
        resp = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = chat_to_anthropic(resp)
        assert result["stop_reason"] == "tool_use"
        tool_block = [b for b in result["content"] if b["type"] == "tool_use"][0]
        assert tool_block["id"] == "call_123"
        assert tool_block["name"] == "bash"
        assert tool_block["input"]["cmd"] == "ls"

    def test_finish_reason_mapping(self):
        for fr, sr in [("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use"), ("content_filter", "end_turn")]:
            resp = {"id": "x", "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": fr}], "usage": {}}
            result = chat_to_anthropic(resp)
            assert result["stop_reason"] == sr, f"{fr} → {result['stop_reason']}, expected {sr}"

    def test_cache_tokens(self):
        resp = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 80}},
        }
        result = chat_to_anthropic(resp)
        assert result["usage"]["cache_read_input_tokens"] == 80

    def test_empty_choices(self):
        resp = {"id": "x", "model": "m", "choices": [], "usage": {}}
        result = chat_to_anthropic(resp)
        assert result["type"] == "message"
        assert result["content"] == [{"type": "text", "text": ""}]

    def test_refusal_to_text(self):
        resp = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}, {"type": "refusal", "refusal": "I can't do that"}]},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        result = chat_to_anthropic(resp)
        texts = [b["text"] for b in result["content"] if b["type"] == "text"]
        assert "I can't do that" in texts


# ═══════════════════════════════════════════════════════════════════
# Rectifier
# ═══════════════════════════════════════════════════════════════════

class TestRectifier:
    def test_invalid_signature(self):
        assert should_rectify("Invalid `signature` in `thinking` block")

    def test_thought_signature_not_valid(self):
        assert should_rectify("Unable to submit request because Thought signature is not valid")

    def test_must_start_with_thinking(self):
        assert should_rectify("a final assistant message must start with a thinking block")

    def test_expected_thinking_found_tool_use(self):
        assert should_rectify("Expected `thinking` or `redacted_thinking`, but found `tool_use`")

    def test_signature_field_required(self):
        assert should_rectify("***.***.signature: Field required")

    def test_signature_extra_inputs(self):
        assert should_rectify("xxx.signature: Extra inputs are not permitted")

    def test_cannot_be_modified(self):
        assert should_rectify("thinking or redacted_thinking blocks in the response cannot be modified")

    def test_unrelated_error(self):
        assert not should_rectify("Request timeout")
        assert not should_rectify("Connection refused")
        assert not should_rectify(None)

    def test_rectify_removes_thinking_blocks(self):
        body = {
            "model": "claude-test",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "t", "signature": "sig"},
                    {"type": "text", "text": "hello", "signature": "sig_text"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}, "signature": "sig_tool"},
                    {"type": "redacted_thinking", "data": "r"},
                ],
            }],
        }
        result = rectify_request(body)
        assert result.applied
        assert result.removed_thinking_blocks == 1
        assert result.removed_redacted_thinking_blocks == 1
        assert result.removed_signature_fields == 2
        content = body["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert "signature" not in content[0]
        assert content[1]["type"] == "tool_use"
        assert "signature" not in content[1]

    def test_rectify_removes_top_level_thinking(self):
        body = {
            "model": "claude-test",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}],
            }, {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            }],
        }
        result = rectify_request(body)
        assert result.applied
        assert "thinking" not in body

    def test_rectify_adaptive_not_removed(self):
        body = {
            "model": "claude-test",
            "thinking": {"type": "adaptive"},
            "messages": [{
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}],
            }],
        }
        result = rectify_request(body)
        # adaptive not removed from top-level
        assert body.get("thinking", {}).get("type") == "adaptive"

    def test_rectify_no_change(self):
        body = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = rectify_request(body)
        assert not result.applied

