"""URL 工具 — 规范化 api_base"""

import re


def normalize_api_base(api_base: str) -> str:
    """去掉 /v1/{path} 等后缀，得到根 URL

    "https://api.example.com/v1/chat/completions" → "https://api.example.com"
    "https://api.example.com/anthropic/v1/messages" → "https://api.example.com/anthropic"
    "https://api.example.com/v1" → "https://api.example.com"
    "https://api.example.com/anthropic" → "https://api.example.com/anthropic"
    """
    base = api_base.rstrip("/")
    base = re.sub(r"/v1(/(?:messages|chat/completions|responses))?$", "", base)
    return base


def normalize_model_name(name: str) -> str:
    """Normalize a model name: strip whitespace, lowercase, replace _ with -,
    strip common provider prefixes (anthropic/, openai/, google/, meta/, mistral/).

    Examples:
        >>> normalize_model_name("Claude-Sonnet-4 ")
        'claude-sonnet-4'
        >>> normalize_model_name("anthropic/Claude-Opus-4-7")
        'claude-opus-4-7'
        >>> normalize_model_name("openai/Codex_opus_4_7")
        'codex-opus-4-7'
    """
    if not isinstance(name, str):
        return ""
    s = name.strip().lower().replace("_", "-")
    for prefix in ("anthropic/", "openai/", "google/", "meta/", "mistral/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s
