"""Thinking effort 全局映射配置的纯函数实现。

从 config.thinking_effort_mapping 读取多 preset 规则，提供统一的 effort 映射。
支持 "*" 兜底键，未匹配的 effort 会走兜底或被过滤。

mapping_config 结构示例：
{
    "presets": [
        {
            "name": "default",
            "type": "any_to_any",
            "rules": {"low": "low", "medium": "medium", "high": "high", "*": "medium"}
        },
        {
            "name": "Thinking:ON",
            "type": "thinking_on",
            "rules": {"*": "high"}
        }
    ],
    "default_preset": "default"
}

type 含义：
- any_to_any：标准 effort 到 effort 的映射（low/medium/high/...）
- thinking_on：变体，把所有输入统一映射为某个 effort（用 "*" 实现）
- 其他自定义 type 同样按 rules 字典解析
"""

from __future__ import annotations


def get_default_mapping_config() -> dict:
    """返回内置的默认 thinking_effort_mapping 配置。

    config 缺失时兜底使用；规则集合等于原 4 处硬编码 effort_map 的并集，
    保证未配置时行为与旧版本完全一致。
    """
    return {
        "presets": [
            {
                "name": "default",
                "type": "any_to_any",
                "rules": {
                    "none": "none",
                    "auto": "auto",
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "xhigh",
                    "max": "xhigh",
                },
            },
        ],
        "default_preset": "default",
    }


def _resolve_preset(mapping_config: dict) -> dict | None:
    """根据 default_preset 选取 preset，缺失时回退到第一个。"""
    presets = mapping_config.get("presets", [])
    if not isinstance(presets, list) or not presets:
        return None

    default_name = mapping_config.get("default_preset")
    if default_name:
        for p in presets:
            if isinstance(p, dict) and p.get("name") == default_name:
                return p

    p0 = presets[0]
    return p0 if isinstance(p0, dict) else None


def apply_effort_mapping(effort: str | None, mapping_config: dict | None) -> str | None:
    """应用 thinking_effort_mapping，返回映射后的 effort。

    Args:
        effort: 输入的 reasoning effort（如 low/medium/high/xhigh/none/auto/minimal/max）
        mapping_config: 全局 thinking_effort_mapping 配置 dict
            None 或空 dict 时使用 get_default_mapping_config() 兜底

    Returns:
        映射后的 effort 字符串；
        effort 为 None 时返回 None；
        规则命中时返回 mapped value；
        未命中且有 "*" 兜底时返回 "*" 的值；
        未命中且无 "*" 兜底时返回 None（调用方据此跳过注入）。
    """
    if effort is None:
        return None

    cfg = mapping_config if mapping_config else get_default_mapping_config()
    preset = _resolve_preset(cfg)
    if not preset:
        return effort

    rules = preset.get("rules", {})
    if not isinstance(rules, dict) or not rules:
        return effort

    if effort in rules:
        mapped = rules[effort]
        return mapped if isinstance(mapped, str) and mapped else effort
    if "*" in rules:
        mapped = rules["*"]
        return mapped if isinstance(mapped, str) and mapped else effort
    return None
