"""Tests for vision_service"""

import pytest
from llm_proxy.services.vision_service import (
    resolve_image,
    replace_images_in_chat_messages,
    replace_images_in_responses_input,
    replace_images_in_anthropic_messages,
    _is_data_uri,
    _parse_data_uri,
)


def test_is_data_uri_true():
    assert _is_data_uri("data:image/png;base64,abc123")


def test_is_data_uri_false():
    assert not _is_data_uri("https://example.com/image.jpg")


def test_parse_data_uri():
    import base64
    raw = b"fake_image_bytes"
    b64 = base64.b64encode(raw).decode()
    uri = f"data:image/png;base64,{b64}"
    data, mime = _parse_data_uri(uri)
    assert data == raw
    assert mime == "image/png"


@pytest.mark.asyncio
async def test_replace_images_in_chat_messages_no_images():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    result = await replace_images_in_chat_messages(msgs)
    assert result == msgs


@pytest.mark.asyncio
async def test_replace_images_in_chat_messages_with_image(monkeypatch):
    async def mock_resolve(url):
        return "[描述：一只猫]"
    monkeypatch.setattr(
        "llm_proxy.services.vision_service.resolve_image", mock_resolve
    )

    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]}]
    result = await replace_images_in_chat_messages(msgs)
    assert result[0]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "text", "text": "[描述：一只猫]"},
    ]


@pytest.mark.asyncio
async def test_replace_images_in_responses_input(monkeypatch):
    async def mock_resolve(url):
        return "[描述：一只狗]"
    monkeypatch.setattr(
        "llm_proxy.services.vision_service.resolve_image", mock_resolve
    )

    inp = [
        {"type": "input_text", "text": "hello"},
        {"type": "input_image", "image_url": "https://example.com/dog.jpg"},
    ]
    result = await replace_images_in_responses_input(inp)
    assert result == [
        {"type": "input_text", "text": "hello"},
        {"type": "input_text", "text": "[描述：一只狗]"},
    ]


@pytest.mark.asyncio
async def test_replace_images_in_anthropic_messages(monkeypatch):
    import base64

    async def mock_image_to_text(image_data, mime):
        return "[描述：风景]"
    monkeypatch.setattr(
        "llm_proxy.services.vision_service._image_to_text", mock_image_to_text
    )

    raw = b"fake_image"
    b64 = base64.b64encode(raw).decode()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image", "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": b64,
        }},
    ]}]
    result = await replace_images_in_anthropic_messages(msgs)
    assert result[0]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "text", "text": "[描述：风景]"},
    ]


@pytest.mark.asyncio
async def test_resolve_image_data_uri(monkeypatch):
    async def mock_image_to_text(image_data, mime):
        return "data uri result"
    monkeypatch.setattr(
        "llm_proxy.services.vision_service._image_to_text", mock_image_to_text
    )

    import base64
    raw = b"test_image_data"
    b64 = base64.b64encode(raw).decode()
    uri = f"data:image/png;base64,{b64}"
    result = await resolve_image(uri)
    assert result == "data uri result"
