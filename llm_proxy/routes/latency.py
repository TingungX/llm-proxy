"""POST /api/latency — 延迟测试"""

import asyncio
import time
import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.main import app
from llm_proxy.state import get_state, resolve_model
from llm_proxy.infra.http_client import get_client

logger = logging.getLogger(__name__)


@app.post("/api/latency")
async def api_latency(request: Request):
    body = await request.json()
    model = body.get("model", "")
    rounds = body.get("rounds", 3)

    s = get_state()
    resolved = resolve_model(model, s.config, s.model_map)
    if not resolved:
        return JSONResponse({"error": f"Unknown model: {model}"}, status_code=400)

    api_base, api_key, actual_model, endpoint_id, model_id = resolved
    target_url = f"{api_base}/v1/messages"
    test_body = {
        "model": actual_model,
        "messages": [{"role": "user", "content": "OK"}],
        "max_tokens": 2,
    }
    req_headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    async def do_one():
        t0 = time.perf_counter()
        try:
            client = get_client()
            await client.post(target_url, json=test_body, headers=req_headers, timeout=15.0)
            return time.perf_counter() - t0
        except Exception:
            return -1

    times = await asyncio.gather(*[do_one() for _ in range(rounds)])
    valid = [t for t in times if t > 0]
    return {
        "model": model,
        "target": target_url,
        "times": times,
        "avg": sum(valid) / max(1, len(valid)),
        "rounds": rounds,
    }
