from llm_proxy.infra.url_utils import normalize_api_base, normalize_model_name


def test_lowercase():
    assert normalize_model_name("Claude-Sonnet-4") == "claude-sonnet-4"


def test_strip_whitespace():
    assert normalize_model_name("  hello  ") == "hello"


def test_underscore_to_dash():
    assert normalize_model_name("Codex_opus_4_7") == "codex-opus-4-7"


def test_strip_anthropic_prefix():
    assert normalize_model_name("anthropic/Claude-Opus-4-7") == "claude-opus-4-7"


def test_strip_openai_prefix():
    assert normalize_model_name("openai/Codex_opus_4_7") == "codex-opus-4-7"


def test_empty_string():
    assert normalize_model_name("") == ""


def test_non_string_input():
    assert normalize_model_name(None) == ""
    assert normalize_model_name(123) == ""


def test_strip_v1_chat_completions():
    assert normalize_api_base("https://api.example.com/v1/chat/completions") == "https://api.example.com"


def test_strip_v1_messages():
    assert normalize_api_base("https://api.example.com/anthropic/v1/messages") == "https://api.example.com/anthropic"


def test_strip_trailing_v1():
    assert normalize_api_base("https://api.example.com/v1") == "https://api.example.com"


def test_no_v1_path():
    assert normalize_api_base("https://api.example.com/anthropic") == "https://api.example.com/anthropic"


def test_trailing_slash():
    assert normalize_api_base("https://api.example.com/v1/") == "https://api.example.com"


def test_root_url():
    assert normalize_api_base("https://api.example.com") == "https://api.example.com"
