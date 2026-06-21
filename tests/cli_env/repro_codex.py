"""复现 Codex 报错：stream closed before response.completed

模拟 Codex CLI 发送带工具的 Responses 流式请求，
mock 上游返回带 tool_calls 的 Chat Completions SSE。
检查响应流是否包含 response.completed 事件。
"""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.cli_env.lib import config_loader as cfg
from tests.cli_env.lib.client import send_responses, configure_endpoint
from tests.cli_env.lib.mock_upstream import MockUpstream
from tests.cli_env.lib.server import ServerManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("repro")


async def tool_stream_handler(reader, writer):
    """Mock 上游：返回带 tool_calls 的 Chat SSE 流（apply_patch 风格）。"""
    # 读请求
    req_line = await asyncio.wait_for(reader.readline(), timeout=10)
    headers = {}
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=10)
        line = line.decode().strip()
        if not line:
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    cl = int(headers.get("content-length", "0"))
    if cl:
        body = await asyncio.wait_for(reader.readexactly(cl), timeout=10)
        body = json.loads(body)
    else:
        body = {}

    logger.info("mock upstream received request: stream=%s", body.get("stream"))

    # 响应头
    writer.write(b"HTTP/1.1 200 OK\r\n")
    writer.write(b"Content-Type: text/event-stream\r\n")
    writer.write(b"Connection: close\r\n\r\n")
    await writer.drain()

    model = body.get("model", "test-model")

    def chunk(delta, finish=None):
        return json.dumps({
            "id": "chatcmpl-tool", "object": "chat.completion.chunk",
            "created": 1700000000, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    # 1) tool_calls begin
    writer.write(f"data: {json.dumps({'id':'chatcmpl-tool','object':'chat.completion.chunk','created':1700000000,'model':model,'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'call_abc123','type':'function','function':{'name':'apply_patch','arguments':''}}]},'finish_reason':None}]})}\n\n".encode())
    await writer.drain()
    # 2) tool_calls argument delta
    dsl = "*** Begin Patch\n*** Add File: foo.txt\n+hello\n*** End Patch"
    payload = json.dumps({"input": dsl})
    writer.write(f"data: {json.dumps({'id':'chatcmpl-tool','object':'chat.completion.chunk','created':1700000000,'model':model,'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':payload}}]},'finish_reason':None}]})}\n\n".encode())
    await writer.drain()
    # 3) finish
    writer.write(f"data: {json.dumps({'id':'chatcmpl-tool','object':'chat.completion.chunk','created':1700000000,'model':model,'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]})}\n\n".encode())
    await writer.drain()
    writer.write(b"data: [DONE]\n\n")
    await writer.drain()
    writer.close()


async def main():
    # 启动 mock upstream（自定义 handler 返回 tool_calls 流）
    server = await asyncio.start_server(tool_stream_handler, "127.0.0.1", 0)
    sock = server.sockets[0]
    mock_port = sock.getsockname()[1]
    mock_url = f"http://127.0.0.1:{mock_port}"
    logger.info("mock upstream on %s", mock_url)

    # 配置
    model = cfg.make_openai_model_config(
        api_base=mock_url, upstream_model="test-model",
        api_key="mock-key", display_name="Test",
    )
    config = cfg.create_scenario_config(models={"test-model": model})
    config_path = cfg.write_temp_config(config, prefix="repro-")
    db_path = cfg.scenario_db_path("repro", "001")

    ps = ServerManager()
    proxy_port = await ps.start(config_path=config_path, db_path=db_path)
    logger.info("proxy on %s", ps.url)

    ep = cfg.build_endpoint_payload(name="test-ep", api_key="test-key", models=[])
    await configure_endpoint(ps.url, ep)

    # 发送 Codex 风格的 Responses 请求（带 tools）
    body = {
        "model": "test-model",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "create foo.txt"}]}
        ],
        "tools": [
            {"type": "function", "name": "apply_patch",
             "description": "Apply a patch", "parameters": {"type": "object", "properties": {}}},
        ],
        "tool_choice": "auto",
    }
    status, hdrs, events, elapsed = await send_responses(ps.url, body, "test-key", stream=True)
    logger.info("status=%s, %d events, %.0fms", status, len(events), elapsed)

    types = []
    has_completed = False
    has_done = False
    for ev in events:
        s = ev if isinstance(ev, str) else (ev.get("data", "") if isinstance(ev, dict) else str(ev))
        if s.strip() == "[DONE]":
            has_done = True
            types.append("[DONE]")
            continue
        try:
            d = json.loads(s)
        except Exception:
            types.append(f"<raw:{s[:80]}>")
            continue
        if isinstance(d, dict):
            t = d.get("type", f"<no-type>")
            types.append(t)
            if t == "response.completed":
                has_completed = True

    print("=== Raw events ===")
    for i, ev in enumerate(events):
        s = ev if isinstance(ev, str) else (ev.get("data", "") if isinstance(ev, dict) else str(ev))
        print(f"  [{i}] {s[:160]}")
    print("=== Event types in order ===")
    for i, t in enumerate(types):
        print(f"  [{i}] {t}")
    print("=== Summary ===")
    print(f"  response.completed present: {has_completed}")
    print(f"  [DONE] present: {has_done}")

    # 捕获 server 日志（先停再读）
    server_logs = await ps.read_stderr()
    await ps.stop()
    print("\n=== Server stderr (last 30 lines) ===")
    for line in server_logs.splitlines()[-30:]:
        print("  ", line)
    server.close()
    await server.wait_closed()

    if not has_completed:
        print("\n>>> REPRODUCED: response.completed MISSING — Codex would report 'stream closed before response.completed'")
        sys.exit(1)
    else:
        print("\n>>> OK: response.completed present")


if __name__ == "__main__":
    asyncio.run(main())
