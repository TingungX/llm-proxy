"""Tests for multi-format usage extraction"""
import pytest
from llm_proxy.protocol.responses_chat.usage import extract_usage_metrics


class TestOpenAIUsageFormat:
    def test_basic_openai_usage(self):
        usage_raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_openai_with_cached_tokens(self):
        usage_raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 30},
        }
        result = extract_usage_metrics(usage_raw)
        assert result["input_tokens"] == 70
        assert result["output_tokens"] == 50
        assert result["cached_tokens"] == 30

    def test_openai_with_reasoning_tokens(self):
        usage_raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 20},
        }
        result = extract_usage_metrics(usage_raw)
        assert result["reasoning_tokens"] == 20


class TestClaudeUsageFormat:
    def test_basic_claude_usage(self):
        usage_raw = {
            "input_tokens": 80,
            "output_tokens": 40,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["input_tokens"] == 80
        assert result["output_tokens"] == 40

    def test_claude_with_cache_fields(self):
        usage_raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 10,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cached_tokens"] == 30
        assert result["cache_creation_input_tokens"] == 10
        assert result["total_tokens"] > 0

    def test_claude_with_ttl_breakdown(self):
        usage_raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "cache_creation_5m_input_tokens": 5,
            "cache_creation_1h_input_tokens": 10,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["cache_creation_5m_input_tokens"] == 5
        assert result["cache_creation_1h_input_tokens"] == 10
        assert result["cache_ttl"] == "mixed"

    def test_claude_1h_only_ttl(self):
        usage_raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_1h_input_tokens": 10,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["cache_ttl"] == "1h"


class TestGeminiUsageFormat:
    def test_gemini_dedup_cached_tokens(self):
        usage_raw = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "cachedContentTokenCount": 30,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["input_tokens"] == 70
        assert result["output_tokens"] == 50
        assert result["cached_tokens"] == 30

    def test_gemini_no_cached(self):
        usage_raw = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50


class TestUsageFormatDetection:
    def test_none_input_returns_empty(self):
        result = extract_usage_metrics(None)
        assert result == {}

    def test_empty_dict_returns_empty(self):
        result = extract_usage_metrics({})
        assert result == {}

    def test_claude_detected_by_cache_creation(self):
        usage_raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 10,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["input_tokens"] == 100

    def test_openai_without_details_format(self):
        usage_raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        result = extract_usage_metrics(usage_raw)
        assert result["input_tokens"] == 100
