"""Mock upstream LLM API 服务器 — 模拟 OpenAI Chat / Anthropic Messages 响应

支持：非流式、流式（SSE）、自定义状态码
"""

import json
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MockUpstream:
    """轻量 mock LLM 上游服务器"""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self._server: asyncio.AbstractServer | None = None
        self._default_status: int = 200
        self._default_body: Any = {}
        self._call_log: list[dict] = []

    def set_default_response(self, status: int = 200, body: Any = None):
        self._default_status = status
        self._default_body = body or {}

    @property
    def call_log(self) -> list[dict]:
        return list(self._call_log)

    def clear(self):
        self._call_log.clear()

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle_connection, host=self.host, port=self.port,
        )
        sock = self._server.sockets[0]
        self.port = sock.getsockname()[1]
        logger.info("Mock upstream started on %s:%s", self.host, self.port)
        return self.port

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Mock upstream stopped")

    def get_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ── handlers ────────────────────────────────────────────────

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            req = await self._read_request(reader)
            if not req:
                writer.close()
                return
            method, path, headers, body = req

            self._call_log.append({
                "method": method, "path": path,
                "headers": dict(headers), "body": body,
            })

            status = self._default_status
            resp_body = self._default_body

            if status >= 400:
                await self._send_json(writer, status, resp_body)
            elif body and isinstance(body, dict) and body.get("stream"):
                await self._send_stream(writer, body, path)
            else:
                await self._send_json(writer, status, resp_body)
        except Exception as e:
            logger.error("Mock upstream handler error: %s", e)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    # ── SSE streaming ───────────────────────────────────────────

    async def _send_stream(self, writer: asyncio.StreamWriter, body: dict, path: str = ""):
        """根据请求路径发送 SSE 流式响应"""
        if "/chat/completions" in path:
            await self._stream_openai(writer, body)
        else:
            await self._stream_anthropic(writer, body)

    async def _stream_openai(self, writer: asyncio.StreamWriter, body: dict):
        """发送 OpenAI Chat Completions SSE 格式"""
        CRLF = "\r\n"
        # HTTP 响应头
        writer.write(b"HTTP/1.1 200 OK\r\n")
        writer.write(b"Content-Type: text/event-stream\r\n")
        writer.write(b"Cache-Control: no-cache\r\n")
        writer.write(b"Connection: close\r\n")
        writer.write(b"\r\n")
        await writer.drain()

        model = body.get("model", "mock-model")
        words = "Streaming response from mock! ".split() * 3

        # role chunk
        await self._sse_write(writer, {
            "id": "chatcmpl-mock-stream", "object": "chat.completion.chunk",
            "created": 1700000000, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        })
        # content chunks
        for word in words:
            await self._sse_write(writer, {
                "id": "chatcmpl-mock-stream", "object": "chat.completion.chunk",
                "created": 1700000000, "model": model,
                "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
            })
        # finish chunk
        await self._sse_write(writer, {
            "id": "chatcmpl-mock-stream", "object": "chat.completion.chunk",
            "created": 1700000000, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        # [DONE]
        writer.write(b"data: [DONE]\r\n\r\n")
        await writer.drain()

    async def _stream_anthropic(self, writer: asyncio.StreamWriter, body: dict):
        """发送 Anthropic Messages SSE 格式"""
        # HTTP 响应头
        writer.write(b"HTTP/1.1 200 OK\r\n")
        writer.write(b"Content-Type: text/event-stream\r\n")
        writer.write(b"Cache-Control: no-cache\r\n")
        writer.write(b"Connection: close\r\n")
        writer.write(b"\r\n")
        await writer.drain()

        model = body.get("model", "mock-model")
        words = "Streaming response from mock! ".split() * 3

        await self._sse_write_event(writer, "message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_mock_stream", "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        })
        await self._sse_write_event(writer, "content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        for word in words:
            await self._sse_write_event(writer, "content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": word + " "},
            })
        await self._sse_write_event(writer, "content_block_stop", {
            "type": "content_block_stop", "index": 0,
        })
        await self._sse_write_event(writer, "message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        })
        await self._sse_write_event(writer, "message_stop", {
            "type": "message_stop",
        })

    async def _sse_write(self, writer: asyncio.StreamWriter, data: dict):
        line = f"data: {json.dumps(data)}\r\n\r\n"
        writer.write(line.encode())
        await writer.drain()

    async def _sse_write_event(self, writer: asyncio.StreamWriter, event: str, data: dict):
        line = f"event: {event}\r\ndata: {json.dumps(data)}\r\n\r\n"
        writer.write(line.encode())
        await writer.drain()

    # ── helpers ─────────────────────────────────────────────────

    async def _send_json(self, writer: asyncio.StreamWriter, status: int, body: Any):
        body_bytes = json.dumps(body).encode()
        status_text = {200: "OK", 400: "Bad Request", 401: "Unauthorized",
                       403: "Forbidden", 404: "Not Found", 429: "Too Many Requests",
                       500: "Internal Server Error", 502: "Bad Gateway",
                       503: "Service Unavailable"}.get(status, "Unknown")
        writer.write(f"HTTP/1.1 {status} {status_text}\r\n".encode())
        writer.write(b"Content-Type: application/json\r\n")
        writer.write(f"Content-Length: {len(body_bytes)}\r\n".encode())
        writer.write(b"Connection: close\r\n\r\n")
        writer.write(body_bytes)
        await writer.drain()

    async def _read_request(self, reader: asyncio.StreamReader) -> tuple | None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not request_line:
                return None
            parts = request_line.decode().strip().split(" ")
            if len(parts) < 2:
                return None
            method, path = parts[0], parts[1]

            headers = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                line = line.decode().strip()
                if not line:
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            content_length = int(headers.get("content-length", "0"))
            body_bytes = b""
            if content_length > 0:
                body_bytes = await asyncio.wait_for(reader.readexactly(content_length), timeout=10)

            body = None
            if "json" in headers.get("content-type", "") and body_bytes:
                try:
                    body = json.loads(body_bytes)
                except json.JSONDecodeError:
                    body = body_bytes.decode()
            elif body_bytes:
                body = body_bytes.decode()

            return method, path, headers, body
        except Exception as e:
            logger.error("Error reading request: %s", e)
            return None

    def make_chat_response(self, text: str = "Mock response", model: str = "mock-model") -> dict:
        return {
            "id": "chatcmpl-mock-001", "object": "chat.completion",
            "created": 1700000000, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def make_anthropic_response(self, text: str = "Mock response", model: str = "mock-model") -> dict:
        return {
            "id": "msg_mock_001", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": text}], "model": model,
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
