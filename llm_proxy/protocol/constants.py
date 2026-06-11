# ── Stop Reason 映射 ──────────────────────────────────────────────────

STOP_REASON_MAP = {
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "end_turn",
}
