DEFAULT_PATHS = {
    "anthropic/messages":       "anthropic/v1/messages",
    "openai/chat-completions":  "/v1/chat/completions",
    "openai/responses":         "/v1/responses",
}


def resolve_path(model_paths: dict, protocol_key: str) -> str:
    if protocol_key in model_paths:
        return model_paths[protocol_key]
    return DEFAULT_PATHS.get(protocol_key, "")
