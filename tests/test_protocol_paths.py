# tests/test_protocol_paths.py
import pytest
from llm_proxy.handlers.shared.paths import resolve_path, DEFAULT_PATHS


def test_default_anthropic_path_is_third_party_compatible():
    assert DEFAULT_PATHS["anthropic/messages"] == "/anthropic/v1/messages"


def test_default_openai_chat_path():
    assert DEFAULT_PATHS["openai/chat-completions"] == "/v1/chat/completions"


def test_default_openai_responses_path():
    assert DEFAULT_PATHS["openai/responses"] == "/v1/responses"


def test_resolve_path_returns_default_when_no_user_config():
    assert resolve_path({}, "anthropic/messages") == "/anthropic/v1/messages"


def test_resolve_path_returns_user_override():
    paths = {"anthropic/messages": "/v1/messages"}
    assert resolve_path(paths, "anthropic/messages") == "/v1/messages"


def test_resolve_path_returns_empty_for_unknown_key():
    assert resolve_path({}, "unknown/protocol") == ""


def test_existing_anthropic_model_gets_legacy_path_injected():
    from llm_proxy.state import State
    cfg = {"models": {"claude-opus-4-7": {"api_base": "https://api.anthropic.com", "api_key": "sk-test", "upstream_protocol": "anthropic"}}}
    paths_map = State._build_paths_map(cfg)
    assert paths_map["claude-opus-4-7"]["anthropic/messages"] == "/v1/messages"


def test_model_with_explicit_paths_not_overridden():
    from llm_proxy.state import State
    cfg = {"models": {"my-model": {"api_base": "https://example.com", "api_key": "sk-test", "upstream_protocol": "anthropic", "upstream_paths": {"anthropic/messages": "custom/v1/messages"}}}}
    paths_map = State._build_paths_map(cfg)
    assert paths_map["my-model"]["anthropic/messages"] == "custom/v1/messages"


def test_non_anthropic_model_not_injected():
    from llm_proxy.state import State
    cfg = {"models": {"gpt-4": {"api_base": "https://api.openai.com", "api_key": "sk-test", "upstream_protocol": "openai"}}}
    paths_map = State._build_paths_map(cfg)
    assert "gpt-4" not in paths_map
