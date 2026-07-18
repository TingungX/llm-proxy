"""State 模型级 thinking_effort_preset 单元测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_proxy.state import State


def _base_config() -> dict:
    return {
        "models": {
            "model-a": {
                "api_base": "https://api.example.com",
                "api_key": "sk-key",
                "thinking_effort_preset": "aggressive",
            },
            "model-b": {
                "api_base": "https://api.example.com",
                "api_key": "sk-key",
            },
        },
        "thinking_effort_mapping": {
            "presets": [
                {
                    "name": "default",
                    "type": "any_to_any",
                    "rules": {"low": "low", "medium": "medium", "high": "high"},
                },
                {
                    "name": "aggressive",
                    "type": "thinking_on",
                    "rules": {"*": "xhigh"},
                },
            ],
            "default_preset": "default",
        },
    }


def test_get_model_effort_mapping_returns_resolved_preset():
    state = State(_base_config())
    mapping = state.get_model_effort_mapping("model-a")
    assert mapping["default_preset"] == "aggressive"
    assert len(mapping["presets"]) == 1
    assert mapping["presets"][0]["rules"]["*"] == "xhigh"


def test_get_model_effort_mapping_falls_back_to_global():
    state = State(_base_config())
    mapping = state.get_model_effort_mapping("model-b")
    assert mapping["default_preset"] == "default"
    assert len(mapping["presets"]) == 2


def test_get_model_effort_mapping_falls_back_for_unknown_preset():
    cfg = _base_config()
    cfg["models"]["model-a"]["thinking_effort_preset"] = "nonexistent"
    state = State(cfg)
    mapping = state.get_model_effort_mapping("model-a")
    # 未知 preset 回退到全局 mapping
    assert mapping["default_preset"] == "default"
    assert len(mapping["presets"]) == 2


def test_get_model_effort_mapping_uses_provider_default_preset():
    """模型未显式配置 preset 时，优先使用厂商 profile 的 default_thinking_effort_preset。"""
    cfg = _base_config()
    cfg["models"]["model-b"]["provider"] = "deepseek"
    state = State(cfg)
    mapping = state.get_model_effort_mapping("model-b")
    assert mapping["default_preset"] == "default"
    assert len(mapping["presets"]) == 1


def test_model_override_beats_provider_default():
    """模型显式 preset 优先级高于厂商默认。"""
    cfg = _base_config()
    cfg["models"]["model-a"]["provider"] = "deepseek"
    state = State(cfg)
    mapping = state.get_model_effort_mapping("model-a")
    assert mapping["default_preset"] == "aggressive"
    assert len(mapping["presets"]) == 1
    assert mapping["presets"][0]["rules"]["*"] == "xhigh"
