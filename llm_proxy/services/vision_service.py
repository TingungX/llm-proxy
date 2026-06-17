"""Vision Fallback: 将图片转为文字描述，供不支持多模态的模型使用。"""

import base64
import json
import logging
from urllib.parse import urlparse

from llm_proxy.infra.http_client import get_client

logger = logging.getLogger(__name__)

OMLX_URL = "http://localhost:8000/v1/chat/completions"
OMLX_MODEL = "gemma-4-e2b-it-4bit"
IMAGE_PROMPT = "请详细描述这张图片的内容。"
DOWNLOAD_TIMEOUT = 30.0
HTTP_TIMEOUT = 120.0


def _is_data_uri(url: str) -> bool:
    return url.startswith("data:")


def _parse_data_uri(data_uri: str) -> tuple[bytes, str]:
    """解析 data URI，返回 (bytes, mime_type)"""
    # data:[<mime>][;base64],<data>
    header, _, encoded = data_uri.partition(",")
    mime = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
    padding = 4 - len(encoded) % 4
    if padding != 4:
        encoded += "=" * padding
    return base64.b64decode(encoded), mime


async def _download_image(url: str) -> tuple[bytes, str]:
    """下载远程图片，返回 (bytes, mime_type)"""
    client = get_client()
    resp = await client.get(url, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "image/jpeg")
    return resp.content, ctype


async def _image_to_text(image_data: bytes, mime: str) -> str:
    """调用 Omlx 视觉模型将图片转为文字描述"""
    b64 = base64.b64encode(image_data).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    payload = {
        "model": OMLX_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": 1024,
    }
    try:
        client = get_client()
        resp = await client.post(OMLX_URL, json=payload, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        logger.debug(
            "Omlx vision returned %d chars for image (mime=%s)", len(text), mime
        )
        return text
    except Exception as e:
        logger.error("Omlx vision call failed: %s", e)
        return ""


async def resolve_image(url: str) -> str:
    """下载/解码图片 → Omlx 识别 → 返回文字描述"""
    try:
        if _is_data_uri(url):
            image_data, mime = _parse_data_uri(url)
        else:
            image_data, mime = await _download_image(url)
        text = await _image_to_text(image_data, mime)
        if text:
            return text
    except Exception as e:
        logger.error("resolve_image failed for %s: %s", url[:80], e)
    return f"[Image: {url[:120]}]"


async def replace_images_in_chat_messages(messages: list[dict]) -> list[dict]:
    """替换 Chat Completions 格式消息中的 image_url 为文字描述"""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url:
                    text = await resolve_image(url)
                    new_content.append({"type": "text", "text": text})
                    continue
            new_content.append(part)
        msg["content"] = new_content
    return messages


async def replace_images_in_responses_input(input_data: list) -> list:
    """替换 Responses API input 中的 input_image 为 input_text"""
    result = []
    for item in input_data:
        if not isinstance(item, dict):
            result.append(item)
            continue
        if item.get("type") != "input_image":
            result.append(item)
            continue
        url = item.get("image_url") or item.get("image", {}).get("url", "")
        if url:
            text = await resolve_image(url)
            result.append({"type": "input_text", "text": text})
        else:
            result.append(item)
    return result


async def replace_images_in_anthropic_messages(messages: list[dict]) -> list[dict]:
    """替换 Anthropic 格式消息中的 image 为 text"""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                source = part.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    image_data = base64.b64decode(source.get("data", ""))
                    text = await _image_to_text(image_data, media_type)
                    if text:
                        new_content.append({"type": "text", "text": text})
                        continue
                    new_content.append({"type": "text", "text": "[Image (Anthropic format)]"})
                    continue
            new_content.append(part)
        msg["content"] = new_content
    return messages
