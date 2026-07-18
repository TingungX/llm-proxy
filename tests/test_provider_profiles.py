"""Tests for provider profile registry and thinking format application."""

from __future__ import annotations

import pytest

from llm_proxy.protocol.provider_profiles import (
    ProviderProfile,
    apply_thinking_to_anthropic_body,
    apply_thinking_to_chat_body,
    get_provider_profile,
    load_provider_profiles,
)


def test_load_builtin_profiles():
    """内置 profile 至少包含 deepseek / minimax / moonshot-k2 / moonshot-k3 / zhipu。"""
    registry = load_provider_profiles()
    for key in ("deepseek", "minimax", "moonshot-k2", "moonshot-k3", "zhipu"):
        assert key in registry, f"missing builtin profile {key}"
        p = registry[key]
        assert p.display_name
        assert p.key == key


def test_deepseek_chat_thinking():
    registry = load_provider_profiles()
    profile = registry["deepseek"]

    body: dict = {}
    apply_thinking_to_chat_body(body, "medium", profile)
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"

    body = {}
    apply_thinking_to_chat_body(body, "xhigh", profile)
    assert body["reasoning_effort"] == "max"

    body = {}
    apply_thinking_to_chat_body(body, "none", profile)
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body


def test_deepseek_anthropic_thinking():
    registry = load_provider_profiles()
    profile = registry["deepseek"]

    body: dict = {}
    apply_thinking_to_anthropic_body(body, "medium", profile)
    assert body["output_config"] == {"effort": "high"}

    body = {}
    apply_thinking_to_anthropic_body(body, "none", profile)
    assert body["thinking"] == {"type": "disabled"}


def test_minimax_chat_thinking():
    registry = load_provider_profiles()
    profile = registry["minimax"]

    body: dict = {}
    apply_thinking_to_chat_body(body, "auto", profile)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["reasoning_split"] is True

    body = {}
    apply_thinking_to_chat_body(body, "none", profile)
    assert body["thinking"] == {"type": "disabled"}


def test_moonshot_k3_fixed_effort():
    registry = load_provider_profiles()
    profile = registry["moonshot-k3"]

    body: dict = {}
    apply_thinking_to_chat_body(body, "low", profile)
    assert body["reasoning_effort"] == "max"


def test_kimi_k2_chat_thinking():
    registry = load_provider_profiles()
    profile = registry["moonshot-k2"]

    body: dict = {}
    apply_thinking_to_chat_body(body, "high", profile)
    assert body["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in body

    body = {}
    apply_thinking_to_chat_body(body, "none", profile)
    assert body["thinking"] == {"type": "disabled"}


def test_alibabacloud_glm_boolean():
    registry = load_provider_profiles()
    profile = registry["alibabacloud-glm"]

    body: dict = {}
    apply_thinking_to_chat_body(body, "high", profile)
    assert body["enable_thinking"] is True

    body = {}
    apply_thinking_to_chat_body(body, "none", profile)
    assert body["enable_thinking"] is False


def test_vllm_glm_chat_template_kwargs():
    registry = load_provider_profiles()
    profile = registry["vllm-glm"]

    body: dict = {}
    apply_thinking_to_chat_body(body, "high", profile)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}

    body = {}
    apply_thinking_to_chat_body(body, "none", profile)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_glm_anthropic_reasoning_effort():
    registry = load_provider_profiles()
    profile = registry["glm-anthropic"]

    body: dict = {}
    apply_thinking_to_anthropic_body(body, "low", profile)
    assert body["reasoning_effort"] == "low"

    body = {}
    apply_thinking_to_anthropic_body(body, "none", profile)
    assert "reasoning_effort" not in body


def test_no_profile_is_no_op():
    body: dict = {"reasoning_effort": "high"}
    apply_thinking_to_chat_body(body, "high", None)
    assert body == {"reasoning_effort": "high"}

    body = {"thinking": {"type": "enabled"}}
    apply_thinking_to_anthropic_body(body, "high", None)
    assert body == {"thinking": {"type": "enabled"}}


def test_get_provider_profile():
    registry = load_provider_profiles()
    assert get_provider_profile("deepseek", registry) is not None
    assert get_provider_profile("unknown", registry) is None
    assert get_provider_profile(None, registry) is None


def test_provider_profile_from_dict():
    p = ProviderProfile.from_dict("custom", {
        "display_name": "Custom",
        "default_api_base": "https://example.com/",
        "thinking": {
            "format": "reasoning_effort_only",
            "effort_aliases": {"auto": "medium"},
        },
        "response": {"preserve_reasoning_content": True},
    })
    assert p.key == "custom"
    assert p.display_name == "Custom"
    assert p.default_api_base == "https://example.com/"
    assert p.thinking_format == "reasoning_effort_only"
    assert p.effort_aliases == {"auto": "medium"}
    assert p.preserve_reasoning_content is True


def test_deepseek_preserves_reasoning_content_in_messages():
    """DeepSeek 等声明保留 reasoning_content 的厂商，assistant 消息字段应透传。"""
    registry = load_provider_profiles()
    profile = registry["deepseek"]
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "42", "reasoning_content": "step by step"},
        ],
    }
    apply_thinking_to_chat_body(body, "medium", profile)
    assistant_msg = body["messages"][1]
    assert assistant_msg.get("reasoning_content") == "step by step"


def test_minimax_strips_reasoning_content_in_messages():
    """未声明保留 reasoning_content 的厂商，assistant 消息字段应剥离。"""
    registry = load_provider_profiles()
    profile = registry["minimax"]
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "42", "reasoning_content": "step by step"},
        ],
    }
    apply_thinking_to_chat_body(body, "auto", profile)
    assistant_msg = body["messages"][1]
    assert "reasoning_content" not in assistant_msg
