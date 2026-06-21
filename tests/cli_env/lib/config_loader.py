"""场景配置管理 — 生成 llm-proxy 配置 + 端点设置"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def create_scenario_config(
    models: dict[str, dict],
    error_handling: dict | None = None,
    compression: dict | None = None,
) -> dict:
    """生成 llm-proxy 的完整 config.json 内容

    Args:
        models: 模型配置字典，格式同 config.example.json 的 "models" 字段
        error_handling: 错误处理配置（可选）
        compression: 压缩配置（可选）

    Returns:
        完整的 config.json dict
    """
    config: dict[str, Any] = {"models": models}

    if error_handling:
        config["error_handling"] = error_handling
    if compression:
        config["compression"] = compression

    return config


def write_temp_config(config: dict, prefix: str = "llm-proxy-scenario") -> str:
    """将 config dict 写入临时文件，返回路径

    使用后调用方负责清理临时文件（server.py 的 cleanup 会处理）
    """
    fd, path = tempfile.mkstemp(suffix=".json", prefix=prefix)
    with os.fdopen(fd, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path


def scenario_db_path(scenario_name: str, run_id: str) -> str:
    """生成场景专用的 usage.db 路径"""
    tmp_dir = Path(tempfile.gettempdir()) / "llm-proxy-cli-env"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return str(tmp_dir / f"{scenario_name}_{run_id}.db")


def build_endpoint_payload(
    name: str,
    api_key: str = "test-key",
    models: list[str] | None = None,
    settings: dict | None = None,
    enabled: bool = True,
    accept_protocols: list[str] | None = None,
    family_routing: dict | None = None,
) -> dict:
    """构建 /api/endpoints POST body"""
    return {
        "name": name,
        "api_key": api_key,
        "models": models or [],
        "settings": settings or {},
        "enabled": enabled,
        "accept_protocols": accept_protocols or ["anthropic", "openai"],
        "family_routing": family_routing,
    }


def make_openai_model_config(
    api_base: str,
    upstream_model: str = "mock-model",
    api_key: str = "mock-key",
    context_window: int = 1000000,
    display_name: str = "Mock Model",
    vision_support: bool = False,
) -> dict:
    """构建一个 OpenAI 格式模型的配置项"""
    return {
        "api_base": api_base,
        "api_key": api_key,
        "upstream_model": upstream_model,
        "context_window": context_window,
        "display_name": display_name,
        "vision_support": vision_support,
        "upstream_paths": {
            "openai/chat-completions": "/v1/chat/completions",
        },
        "upstream_protocols": ["openai"],
    }


def make_anthropic_model_config(
    api_base: str,
    upstream_model: str = "mock-model",
    api_key: str = "mock-key",
    context_window: int = 1000000,
    display_name: str = "Mock Model",
    vision_support: bool = True,
) -> dict:
    """构建一个 Anthropic 格式模型的配置项"""
    return {
        "api_base": api_base,
        "api_key": api_key,
        "upstream_model": upstream_model,
        "context_window": context_window,
        "display_name": display_name,
        "vision_support": vision_support,
        "upstream_paths": {
            "anthropic/messages": "anthropic/v1/messages",
            "openai/chat-completions": "/v1/chat/completions",
        },
        "upstream_protocols": ["openai/chat-completions", "anthropic"],
    }


def make_dual_model_config(
    api_base: str,
    upstream_model: str = "mock-model",
    api_key: str = "mock-key",
    context_window: int = 1000000,
    display_name: str = "Mock Model",
) -> dict:
    """构建双协议模型的配置项"""
    return {
        "api_base": api_base,
        "api_key": api_key,
        "upstream_model": upstream_model,
        "context_window": context_window,
        "display_name": display_name,
        "vision_support": True,
        "upstream_paths": {
            "anthropic/messages": "anthropic/v1/messages",
            "openai/chat-completions": "/v1/chat/completions",
        },
        "upstream_protocols": ["anthropic", "openai"],
    }
