"""Multi-format usage extraction for OpenAI / Claude / Gemini upstream responses.

Auto-detects format and normalizes to a unified dict with:
- input_tokens (net of cached tokens for OpenAI/Gemini)
- output_tokens
- total_tokens (recalculated for Claude with cache)
- cached_tokens, reasoning_tokens, cache details where applicable
"""


def _get_int(usage_map: dict, key: str) -> int:
    v = usage_map.get(key)
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    return 0


def _get_nested_int(usage_map: dict, dotted_key: str) -> int:
    parts = dotted_key.split(".")
    current = usage_map
    for part in parts:
        if not isinstance(current, dict):
            return 0
        current = current.get(part, {})
    if isinstance(current, (int, float)):
        return int(current)
    return 0


def _effective_cache_creation(cache_creation: int, cache_5m: int, cache_1h: int) -> int:
    if cache_creation > 0:
        return cache_creation
    return cache_5m + cache_1h


def _calculate_claude_total(
    input_tokens: int, output_tokens: int,
    cache_read: int, cache_creation: int, cache_5m: int, cache_1h: int,
) -> int:
    return input_tokens + output_tokens + cache_read + _effective_cache_creation(cache_creation, cache_5m, cache_1h)


def _normalize_openai_input_tokens(
    input_tokens: int, cache_total: int,
) -> int:
    if cache_total <= 0:
        return input_tokens
    return max(input_tokens - cache_total, 0)


def _parse_openai_usage(usage_map: dict) -> dict:
    result = {}
    input_tokens = _get_int(usage_map, "prompt_tokens")
    output_tokens = _get_int(usage_map, "completion_tokens")
    total_tokens = _get_int(usage_map, "total_tokens")
    cached_tokens = _get_nested_int(usage_map, "prompt_tokens_details.cached_tokens")
    if cached_tokens == 0:
        cached_tokens = _get_nested_int(usage_map, "input_tokens_details.cached_tokens")
    reasoning_tokens = _get_nested_int(usage_map, "completion_tokens_details.reasoning_tokens")
    if reasoning_tokens == 0:
        reasoning_tokens = _get_nested_int(usage_map, "output_tokens_details.reasoning_tokens")

    if cached_tokens > 0:
        input_tokens = _normalize_openai_input_tokens(input_tokens, cached_tokens)

    if total_tokens == 0 or cached_tokens > 0:
        total_tokens = input_tokens + output_tokens + cached_tokens

    result["input_tokens"] = input_tokens
    result["output_tokens"] = output_tokens
    result["total_tokens"] = total_tokens
    if cached_tokens > 0:
        result["cached_tokens"] = cached_tokens
    if reasoning_tokens > 0:
        result["reasoning_tokens"] = reasoning_tokens
    return result


def _parse_claude_usage(usage_map: dict) -> dict:
    result = {}
    input_tokens = _get_int(usage_map, "input_tokens")
    output_tokens = _get_int(usage_map, "output_tokens")
    cache_read = _get_int(usage_map, "cache_read_input_tokens")
    cache_creation = _get_int(usage_map, "cache_creation_input_tokens")
    cache_5m = _get_int(usage_map, "cache_creation_5m_input_tokens")
    cache_1h = _get_int(usage_map, "cache_creation_1h_input_tokens")

    has_5m = cache_5m > 0
    has_1h = cache_1h > 0
    cache_ttl = ""
    if has_5m and has_1h:
        cache_ttl = "mixed"
    elif has_1h:
        cache_ttl = "1h"
    elif has_5m:
        cache_ttl = "5m"

    total_tokens = _calculate_claude_total(
        input_tokens, output_tokens, cache_read, cache_creation, cache_5m, cache_1h
    )

    result["input_tokens"] = input_tokens
    result["output_tokens"] = output_tokens
    result["total_tokens"] = total_tokens
    if cache_read > 0:
        result["cached_tokens"] = cache_read
    if cache_creation > 0:
        result["cache_creation_input_tokens"] = cache_creation
    if cache_5m > 0:
        result["cache_creation_5m_input_tokens"] = cache_5m
    if cache_1h > 0:
        result["cache_creation_1h_input_tokens"] = cache_1h
    if cache_ttl:
        result["cache_ttl"] = cache_ttl
    return result


def _parse_gemini_usage(usage_map: dict) -> dict:
    result = {}
    prompt_tokens = _get_int(usage_map, "promptTokenCount")
    cached_tokens = _get_int(usage_map, "cachedContentTokenCount")
    output_tokens = _get_int(usage_map, "candidatesTokenCount")

    actual_input = max(prompt_tokens - cached_tokens, 0)

    result["input_tokens"] = actual_input
    result["output_tokens"] = output_tokens
    result["total_tokens"] = actual_input + output_tokens
    if cached_tokens > 0:
        result["cached_tokens"] = cached_tokens
    return result


def extract_usage_metrics(usage_raw) -> dict:
    if usage_raw is None:
        return {}
    if not isinstance(usage_raw, dict):
        return {}
    if not usage_raw:
        return {}

    has_cache_creation = "cache_creation_input_tokens" in usage_raw
    has_cache_read = "cache_read_input_tokens" in usage_raw
    has_cache_5m = "cache_creation_5m_input_tokens" in usage_raw
    has_cache_1h = "cache_creation_1h_input_tokens" in usage_raw
    has_input_details = "input_tokens_details" in usage_raw or "prompt_tokens_details" in usage_raw
    has_openai_keys = "prompt_tokens" in usage_raw or "completion_tokens" in usage_raw
    is_gemini = "promptTokenCount" in usage_raw

    if has_cache_creation or has_cache_5m or has_cache_1h:
        return _parse_claude_usage(usage_raw)
    if has_cache_read and not has_input_details:
        return _parse_claude_usage(usage_raw)
    if is_gemini:
        return _parse_gemini_usage(usage_raw)
    if not has_openai_keys and ("input_tokens" in usage_raw or "output_tokens" in usage_raw):
        return _parse_claude_usage(usage_raw)
    return _parse_openai_usage(usage_raw)
