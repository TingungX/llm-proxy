"""厂商 profile 注册表。

把各家模型对 thinking / reasoning 的特殊请求格式集中维护，避免在协议层硬编码
厂商判断。模型配置通过 `provider` 字段声明所属厂商，协议层再按 profile 构造请求。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_PROFILES_PATH = BASE_DIR / "provider_profiles.json"
EXAMPLE_PROFILES_PATH = BASE_DIR / "provider_profiles.example.json"
PROFILE_PATH = Path(os.environ.get("LLM_PROXY_PROVIDER_PROFILES_PATH", str(DEFAULT_PROFILES_PATH)))


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """单个厂商的 profile。"""

    key: str
    display_name: str
    default_api_base: str
    thinking_format: str | None
    default_thinking_type: str | None
    disable_thinking_value: str | None
    effort_field: str | None
    fixed_effort: str | None
    effort_aliases: dict[str, str | None]
    supports_reasoning_split: bool
    preserve_reasoning_content: bool
    default_thinking_effort_preset: str | None

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> "ProviderProfile":
        thinking = data.get("thinking") or {}
        response = data.get("response") or {}
        aliases = thinking.get("effort_aliases") or {}
        return cls(
            key=key,
            display_name=data.get("display_name") or key,
            default_api_base=data.get("default_api_base", ""),
            thinking_format=thinking.get("format") or None,
            default_thinking_type=thinking.get("default_type") or None,
            disable_thinking_value=thinking.get("disable_value") or None,
            effort_field=thinking.get("field") or None,
            fixed_effort=thinking.get("fixed_value") or None,
            effort_aliases={str(k): v for k, v in aliases.items()},
            supports_reasoning_split=bool(response.get("supports_reasoning_split", False)),
            preserve_reasoning_content=bool(response.get("preserve_reasoning_content", False)),
            default_thinking_effort_preset=data.get("default_thinking_effort_preset") or None,
        )


def _builtin_profiles() -> dict[str, dict[str, Any]]:
    """内置默认 profile，当外部 JSON 不存在时使用。"""
    return {
        "deepseek": {
            "display_name": "DeepSeek",
            "default_api_base": "https://api.deepseek.com/",
            "default_thinking_effort_preset": "default",
            "thinking": {
                "format": "thinking_type_plus_reasoning_effort",
                "default_type": "enabled",
                "disable_value": "disabled",
                "effort_aliases": {
                    "none": "disabled",
                    "auto": "high",
                    "minimal": "high",
                    "low": "high",
                    "medium": "high",
                    "high": "high",
                    "xhigh": "max",
                    "max": "max",
                },
            },
            "response": {"preserve_reasoning_content": True},
        },
        "minimax": {
            "display_name": "MiniMax",
            "default_api_base": "https://api.minimax.io/",
            "default_thinking_effort_preset": "default",
            "thinking": {
                "format": "thinking_adaptive_disabled",
                "default_type": "adaptive",
                "disable_value": "disabled",
            },
            "response": {"supports_reasoning_split": True},
        },
        "moonshot-k2": {
            "display_name": "Kimi K2.x",
            "default_api_base": "https://api.moonshot.cn/",
            "default_thinking_effort_preset": "default",
            "thinking": {
                "format": "thinking_enabled_disabled",
                "default_type": "enabled",
                "disable_value": "disabled",
            },
        },
        "moonshot-k3": {
            "display_name": "Kimi K3",
            "default_api_base": "https://api.moonshot.cn/",
            "default_thinking_effort_preset": "default",
            "thinking": {
                "format": "reasoning_effort_fixed",
                "fixed_value": "max",
            },
        },
        "zhipu": {
            "display_name": "智谱 GLM",
            "default_api_base": "https://open.bigmodel.cn/api/paas/v4/",
            "default_thinking_effort_preset": "default",
            "thinking": {
                "format": "thinking_enabled_disabled",
                "default_type": "enabled",
                "disable_value": "disabled",
            },
        },
        "alibabacloud-glm": {
            "display_name": "阿里云 GLM",
            "default_api_base": "",
            "default_thinking_effort_preset": "default",
            "thinking": {
                "format": "enable_thinking_boolean",
                "field": "enable_thinking",
            },
        },
        "vllm-glm": {
            "display_name": "vLLM GLM",
            "default_api_base": "",
            "default_thinking_effort_preset": "default",
            "thinking": {
                "format": "chat_template_kwargs_enable_thinking",
            },
        },
        "glm-anthropic": {
            "display_name": "GLM (Anthropic 入口)",
            "default_api_base": "",
            "default_thinking_effort_preset": "default",
            "thinking": {
                "format": "reasoning_effort_only",
                "effort_aliases": {
                    "none": None,
                    "auto": "medium",
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "high",
                    "max": "high",
                },
            },
        },
    }


def load_provider_profiles(path: Path | str | None = None) -> dict[str, ProviderProfile]:
    """加载厂商 profile 注册表。

    优先级：
    1. 指定 path
    2. 环境变量 LLM_PROXY_PROVIDER_PROFILES_PATH
    3. 项目根目录 provider_profiles.json
    4. provider_profiles.example.json
    5. 内置默认
    """
    if path is None:
        path = PROFILE_PATH
    else:
        path = Path(path)

    raw: dict[str, Any] | None = None
    tried: list[str] = []

    for candidate in (path, EXAMPLE_PROFILES_PATH):
        tried.append(str(candidate))
        try:
            with open(candidate, encoding="utf-8") as f:
                raw = json.load(f)
            logger.info("Loaded provider profiles from %s", candidate)
            break
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as e:
            logger.error(
                "Provider profiles JSON parse error at %s line %s col %s: %s",
                candidate, e.lineno, e.colno, e.msg,
            )
            raise

    if raw is None:
        logger.info("No external provider_profiles.json found; using built-in defaults")
        raw = _builtin_profiles()

    if not isinstance(raw, dict):
        raise ValueError(f"Provider profiles must be a JSON object, got {type(raw).__name__}")

    return {k: ProviderProfile.from_dict(k, v) for k, v in raw.items() if isinstance(v, dict)}


def get_provider_profile(provider: str | None, registry: dict[str, ProviderProfile] | None = None) -> ProviderProfile | None:
    """按 provider key 取 profile；未找到返回 None。"""
    if not provider or not registry:
        return None
    return registry.get(provider)


def _resolve_effort(effort: str | None, aliases: dict[str, str | None]) -> str | None:
    """把规范 effort 映射为厂商接受的值。"""
    if effort is None:
        return None
    if effort in aliases:
        return aliases[effort]
    return effort


def _clear_thinking_fields(body: dict[str, Any]) -> None:
    """清除 body 中已有的 thinking 相关字段，避免与 profile 规则冲突。"""
    for key in (
        "reasoning_effort",
        "thinking",
        "reasoning_split",
        "enable_thinking",
        "chat_template_kwargs",
        "output_config",
    ):
        body.pop(key, None)


def _strip_message_reasoning_content(body: dict[str, Any]) -> None:
    """从 messages 中 assistant 消息的 reasoning_content 字段剥离。"""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)


def _alias_or_default(effort: str | None, aliases: dict[str, str | None], default: str) -> str:
    resolved = _resolve_effort(effort, aliases)
    return resolved if resolved else default


def apply_thinking_to_chat_body(
    body: dict[str, Any],
    effort: str | None,
    profile: ProviderProfile | None,
) -> None:
    """按厂商 profile 把规范 effort 编码为 OpenAI Chat Completions 请求字段。

    该函数会原地修改 body。无 profile 时不做任何事。
    """
    if not profile or not profile.thinking_format:
        return

    fmt = profile.thinking_format
    _clear_thinking_fields(body)

    if fmt == "thinking_type_plus_reasoning_effort":
        # DeepSeek：thinking.type + reasoning_effort（仅接受 high/max）
        if effort == "none":
            body["thinking"] = {"type": profile.disable_thinking_value or "disabled"}
        else:
            body["thinking"] = {"type": profile.default_thinking_type or "enabled"}
            resolved = _alias_or_default(effort, profile.effort_aliases, "high")
            if resolved != profile.disable_thinking_value:
                body["reasoning_effort"] = resolved

    elif fmt == "thinking_enabled_disabled":
        # Kimi K2.x / 智谱：仅 thinking.type
        if effort == "none":
            body["thinking"] = {"type": profile.disable_thinking_value or "disabled"}
        else:
            body["thinking"] = {"type": profile.default_thinking_type or "enabled"}

    elif fmt == "thinking_adaptive_disabled":
        # MiniMax：adaptive 开思考，disabled 关；可配 reasoning_split
        if effort == "none":
            body["thinking"] = {"type": profile.disable_thinking_value or "disabled"}
        else:
            body["thinking"] = {"type": profile.default_thinking_type or "adaptive"}
        if profile.supports_reasoning_split:
            body["reasoning_split"] = True

    elif fmt == "reasoning_effort_only":
        # GLM Anthropic 入口：顶层 reasoning_effort
        if effort and effort != "none":
            resolved = _resolve_effort(effort, profile.effort_aliases)
            if resolved:
                body["reasoning_effort"] = resolved

    elif fmt == "reasoning_effort_fixed":
        # Kimi K3：固定 max
        if profile.fixed_effort:
            body["reasoning_effort"] = profile.fixed_effort

    elif fmt == "enable_thinking_boolean":
        # 阿里云 GLM
        field = profile.effort_field or "enable_thinking"
        body[field] = effort != "none"

    elif fmt == "chat_template_kwargs_enable_thinking":
        # vLLM GLM
        body["chat_template_kwargs"] = {"enable_thinking": effort != "none"}

    else:
        logger.debug("Unknown thinking_format '%s' for provider %s", fmt, profile.key)

    # reasoning_content 维护：仅声明保留的厂商才向上游透传 assistant 消息里的
    # reasoning_content，否则剥离，避免不支持的厂商触发校验或解析错误。
    if not profile.preserve_reasoning_content:
        _strip_message_reasoning_content(body)


def apply_thinking_to_anthropic_body(
    body: dict[str, Any],
    effort: str | None,
    profile: ProviderProfile | None,
) -> None:
    """按厂商 profile 把规范 effort 编码为 Anthropic Messages 请求字段。

    该函数会原地修改 body。无 profile 时不做任何事。
    """
    if not profile or not profile.thinking_format:
        return

    fmt = profile.thinking_format
    # 只清除本会覆盖的字段，保留用户传入的 output_config 其他子字段
    body.pop("thinking", None)
    body.pop("reasoning_effort", None)

    if fmt == "thinking_enabled_disabled":
        if effort == "none":
            body["thinking"] = {"type": profile.disable_thinking_value or "disabled"}
        else:
            body["thinking"] = {"type": profile.default_thinking_type or "enabled"}

    elif fmt == "thinking_adaptive_disabled":
        if effort == "none":
            body["thinking"] = {"type": profile.disable_thinking_value or "disabled"}
        else:
            body["thinking"] = {"type": profile.default_thinking_type or "adaptive"}

    elif fmt == "thinking_type_plus_reasoning_effort":
        # DeepSeek Anthropic 入口：output_config.effort 控制深度
        if effort == "none":
            body["thinking"] = {"type": profile.disable_thinking_value or "disabled"}
        else:
            resolved = _alias_or_default(effort, profile.effort_aliases, "high")
            output_config = body.get("output_config")
            if isinstance(output_config, dict):
                body["output_config"] = {**output_config, "effort": resolved}
            else:
                body["output_config"] = {"effort": resolved}

    elif fmt == "reasoning_effort_only":
        if effort and effort != "none":
            resolved = _resolve_effort(effort, profile.effort_aliases)
            if resolved:
                body["reasoning_effort"] = resolved

    elif fmt == "reasoning_effort_fixed":
        if profile.fixed_effort:
            body["reasoning_effort"] = profile.fixed_effort

    # enable_thinking_boolean / chat_template_kwargs_* 不适用于 Anthropic 协议，忽略
