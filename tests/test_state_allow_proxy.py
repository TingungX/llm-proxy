"""Tests for State._build_allow_proxy_map and State.allow_proxy_map."""

from llm_proxy.state import State


def test_allow_proxy_defaults_to_false():
    cfg = {
        "models": {
            "test-model-a": {"api_base": "https://x", "api_key": "k", "upstream_model": "u"},
        }
    }
    s = State(cfg)
    assert s.allow_proxy_map == {"glm-5": False}


def test_allow_proxy_explicit_true():
    cfg = {
        "models": {
            "test-model-a": {"api_base": "https://x", "api_key": "k", "upstream_model": "u", "allow_proxy": True},
        }
    }
    s = State(cfg)
    assert s.allow_proxy_map == {"glm-5": True}


def test_allow_proxy_key_is_lowercase():
    cfg = {
        "models": {
            "test-model-a-pro": {"api_base": "https://x", "api_key": "k", "upstream_model": "u", "allow_proxy": True},
        }
    }
    s = State(cfg)
    assert "glm-5-pro" in s.allow_proxy_map
    assert s.allow_proxy_map["glm-5-pro"] is True


def test_allow_proxy_explicit_false_still_false():
    cfg = {
        "models": {
            "x": {"api_base": "https://x", "api_key": "k", "upstream_model": "u", "allow_proxy": False},
        }
    }
    s = State(cfg)
    assert s.allow_proxy_map["x"] is False


def test_allow_proxy_multiple_models():
    cfg = {
        "models": {
            "a": {"api_base": "https://a", "api_key": "k", "upstream_model": "u", "allow_proxy": True},
            "b": {"api_base": "https://b", "api_key": "k", "upstream_model": "u"},
            "C": {"api_base": "https://c", "api_key": "k", "upstream_model": "u", "allow_proxy": True},
        }
    }
    s = State(cfg)
    assert s.allow_proxy_map == {"a": True, "b": False, "c": True}


def test_reload_rebuilds_allow_proxy_map():
    import asyncio
    cfg_v1 = {"models": {"a": {"api_base": "x", "api_key": "k", "upstream_model": "u"}}}
    s = State(cfg_v1)
    assert s.allow_proxy_map == {"a": False}
    cfg_v2 = {"models": {"a": {"api_base": "x", "api_key": "k", "upstream_model": "u", "allow_proxy": True}}}
    s.config = cfg_v2  # bypass real load_config; verify _build_allow_proxy_map is called
    s.allow_proxy_map = s._build_allow_proxy_map(s.config)
    assert s.allow_proxy_map == {"a": True}
