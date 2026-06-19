"""resolver 模块单元测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_proxy.state import (
    normalize_claude_model,
    build_model_map,
    resolve_model,
    resolve_model_for_endpoint,
    _strip_date_suffix,
)

TEST_CONFIG = {
    "models": {
        "deepseek-v4-pro": {
            "api_base": "https://api.deepseek.com/anthropic",
            "api_key": "sk-deepseek-key-12345",
            "display_name": "DeepSeek V4 Pro",
        },
        "claude-opus-4-6": {
            "api_base": "https://api.anthropic.com",
            "api_key": "sk-anthropic-key-99999",
        },
        "test-model-b": {
            "api_base": "https://api.minimax.io/anthropic",
            "api_key": "sk-minimax-key-67890",
            "upstream_model": "test-model-b",
            "display_name": "Minimax M2.7",
        },
        "test-model-a": {
            "api_base": "https://maas-coding-api.cn-huabei-1.xf-yun.com/anthropic",
            "api_key": "sk-glm-key-abcde",
            "upstream_model": "astron-code-latest",
            "display_name": "Test Model A (Provider)",
        },
    },
}

TEST_ENDPOINT_ROUTING = {
    "haiku": {
        "target": "test-model-b",
    },
    "sonnet": {
        "target": "test-model-b",
        "failover": "haiku",
    },
    "opus-4-7": {
        "target": "deepseek-v4-pro",
        "failover": "opus-4-6",
    },
    "opus-4-6": {
        "target": "test-model-a",
        "failover": "sonnet",
    },
    "opus": {
        "target": "test-model-a",
    },
    "gpt-5": "deepseek-v4-pro",
}


def test_normalize_opus():
    assert normalize_claude_model("claude-opus-4-7") == "opus-4-7"
    assert normalize_claude_model("Claude-Opus-4-6") == "opus-4-6"


def test_normalize_haiku():
    assert normalize_claude_model("claude-haiku-4-5") == "haiku"


def test_normalize_sonnet():
    assert normalize_claude_model("claude-sonnet-4-6") == "sonnet"


def test_normalize_unknown():
    assert normalize_claude_model("gpt-4") is None


def test_build_model_map():
    mm = build_model_map(TEST_CONFIG)
    assert "deepseek-v4-pro" in mm
    assert "test-model-b" in mm
    assert mm["test-model-a"][2] == "astron-code-latest"
    assert mm["test-model-a"][3] is None


def test_resolve_opus_47():
    """无全局映射时，opus-4-7 应返回 None"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model("claude-opus-4-7", TEST_CONFIG, mm)
    assert result is None


def test_resolve_opus_46_with_upstream_model():
    """直接匹配 test-model-a"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model("test-model-a", TEST_CONFIG, mm)
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol = result
    assert upstream == "astron-code-latest"
    assert cfg_key == "test-model-a"
    assert protocol is None


def test_resolve_direct_match():
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model("deepseek-v4-pro", TEST_CONFIG, mm)
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol = result
    assert cfg_key == "deepseek-v4-pro"
    assert upstream == "deepseek-v4-pro"
    assert protocol is None


def test_resolve_claude_model_direct_match():
    """config key 为 claude-opus-4-6 时直接匹配，upstream 用原模型名（用户应设 upstream_model）"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model("claude-opus-4-6", TEST_CONFIG, mm)
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol = result
    assert cfg_key == "claude-opus-4-6"
    assert upstream == "claude-opus-4-6"


def test_resolve_claude_via_routing_full_key():
    """routing key 用完整 claude-opus-4-6 时直接匹配"""
    mm = build_model_map(TEST_CONFIG)
    routing = {"claude-opus-4-6": "deepseek-v4-pro"}
    result = resolve_model_for_endpoint("claude-opus-4-6", TEST_CONFIG, mm, routing)
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol, failover = result
    assert cfg_key == "deepseek-v4-pro"
    assert upstream == "deepseek-v4-pro"

def test_resolve_claude_via_routing_family_key():
    """routing key 用 opus-4-6 时（兼容旧数据），经 normalize 后匹配"""
    mm = build_model_map(TEST_CONFIG)
    routing = {"opus-4-6": "claude-opus-4-6"}
    result = resolve_model_for_endpoint("claude-opus-4-6", TEST_CONFIG, mm, routing)
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol, failover = result
    assert cfg_key == "claude-opus-4-6"
    assert upstream == "claude-opus-4-6"


def test_resolve_unknown():
    mm = build_model_map(TEST_CONFIG)
    assert resolve_model("nonexistent-model", TEST_CONFIG, mm) is None


def test_resolve_with_endpoint_routing_opus_47():
    """端点级路由：opus-4-7 → deepseek-v4-pro，failover=opus-4-6"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model_for_endpoint(
        "claude-opus-4-7", TEST_CONFIG, mm, TEST_ENDPOINT_ROUTING
    )
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol, failover = result
    assert cfg_key == "deepseek-v4-pro"
    assert upstream == "deepseek-v4-pro"
    assert failover == "opus-4-6"


def test_resolve_with_endpoint_routing_sonnet_failover():
    """端点级路由：sonnet → test-model-b，failover=haiku"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model_for_endpoint(
        "claude-sonnet-4-6", TEST_CONFIG, mm, TEST_ENDPOINT_ROUTING
    )
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol, failover = result
    assert cfg_key == "test-model-b"
    assert upstream == "test-model-b"
    assert failover == "haiku"


def test_resolve_with_endpoint_routing_simplified():
    """端点级路由：简化格式，非 Claude 模型直接匹配"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model_for_endpoint(
        "deepseek-v4-pro", TEST_CONFIG, mm, TEST_ENDPOINT_ROUTING
    )
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol, failover = result
    assert cfg_key == "deepseek-v4-pro"
    assert failover is None


def test_resolve_without_endpoint_routing():
    """无端点映射时，直接匹配模型名"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model_for_endpoint(
        "deepseek-v4-pro", TEST_CONFIG, mm, None
    )
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol, failover = result
    assert cfg_key == "deepseek-v4-pro"
    assert failover is None


def test_resolve_endpoint_routing_fallback():
    """端点级路由回退：opus-4-5 无精确匹配，回退到 opus"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model_for_endpoint(
        "claude-opus-4-5", TEST_CONFIG, mm, TEST_ENDPOINT_ROUTING
    )
    assert result is not None
    api_base, api_key, upstream, cfg_key, protocol, failover = result
    assert cfg_key == "test-model-a"
    assert failover is None


def test_resolve_endpoint_routing_unknown():
    """端点级路由：未知模型返回 None"""
    mm = build_model_map(TEST_CONFIG)
    result = resolve_model_for_endpoint(
        "unknown-model", TEST_CONFIG, mm, TEST_ENDPOINT_ROUTING
    )
    assert result is None
