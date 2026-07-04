"""GET/PUT /api/config, PUT /api/models/{model_id}, DELETE /api/models/{model_id}, POST /api/providers/{model_id}/detect, POST /api/detect-protocol"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.main import app
from llm_proxy.config_loader import save_config
from llm_proxy.protocol.detector import detect_upstream_protocols
from llm_proxy.state import get_state

logger = logging.getLogger(__name__)


def _entries_from_protocols(protocols: list[str]) -> list[dict]:
    """将探测到的协议列表转换为新版 entry 格式。"""
    path_defaults = {
        "anthropic": "/v1/messages",
        "openai/chat-completions": "/v1/chat/completions",
        "openai/responses": "/v1/responses",
    }
    return [
        {"protocol": p, "enabled": True, "path": path_defaults.get(p, "/v1/" + p.split("/")[-1])}
        for p in protocols
    ]


@app.get("/api/config")
async def api_get_config():
    return get_state().config


@app.put("/api/config")
async def api_update_config(request: Request):
    new_config = await request.json()
    new_config.pop("family_routing", None)
    new_config.pop("model_map", None)
    save_config(new_config)
    await get_state().reload()
    logger.info("Config reloaded")
    return {"status": "ok"}


@app.put("/api/models/{model_id}")
async def api_update_model(model_id: str, request: Request):
    """更新单个模型配置（增量合并，不覆盖其他模型）"""
    body = await request.json()
    s = get_state()
    if model_id not in s.config["models"]:
        s.config["models"][model_id] = {}
    model_cfg = s.config["models"][model_id]
    for key, value in body.items():
        if value is None:
            model_cfg.pop(key, None)
        else:
            model_cfg[key] = value
    if body.get("upstream_protocols") is not None:
        model_cfg.pop("upstream_protocol", None)
        model_cfg.pop("upstream_paths", None)
    save_config(s.config)
    await s.reload()
    logger.info(f"Model {model_id} updated")
    return {"status": "ok"}


@app.delete("/api/models/{model_id}")
async def api_delete_model(model_id: str):
    """删除单个模型配置"""
    s = get_state()
    if model_id not in s.config["models"]:
        return JSONResponse({"error": f"Model {model_id} not found"}, status_code=404)
    del s.config["models"][model_id]
    save_config(s.config)
    await s.reload()
    logger.info(f"Model {model_id} deleted")
    return {"status": "ok"}


@app.post("/api/detect-protocol")
async def api_detect_protocol(request: Request):
    """探测给定 api_base 的协议，不依赖已保存的模型"""
    body = await request.json()
    api_base = body.get("api_base", "").strip()
    api_key = body.get("api_key", "").strip()

    if not api_base or not api_key:
        return JSONResponse(
            {"error": "api_base and api_key are required"},
            status_code=400
        )

    protocols = await detect_upstream_protocols(api_base, api_key)

    if not protocols:
        return JSONResponse(
            {"error": "Failed to detect protocol. Please specify upstream_protocol manually."},
            status_code=400
        )

    return {
        "status": "ok",
        "upstream_protocol": protocols[0],
        "upstream_protocols": _entries_from_protocols(protocols),
    }


@app.post("/api/providers/{model_id}/detect")
async def api_detect_provider_protocol(model_id: str):
    """手动触发协议探测，记录所有支持的协议和路径"""
    model_id_lower = model_id.lower()
    
    if model_id_lower not in get_state().model_map:
        return JSONResponse(
            {"error": f"Model {model_id} not found in config"},
            status_code=404
        )
    
    api_base, api_key, _, _ = get_state().model_map[model_id_lower]
    
    protocols = await detect_upstream_protocols(api_base, api_key)
    
    if not protocols:
        return JSONResponse(
            {"error": "Failed to detect protocol. Please specify upstream_protocol manually."},
            status_code=400
        )
    
    # 更新 config.json：探测结果写入新版 entry 格式
    s = get_state()
    if model_id in s.config["models"]:
        entries = _entries_from_protocols(protocols)
        s.config["models"][model_id]["upstream_protocols"] = entries
        s.config["models"][model_id].pop("upstream_protocol", None)
        s.config["models"][model_id].pop("upstream_paths", None)
        save_config(s.config)
    await s.reload()

    return {
        "status": "ok",
        "model_id": model_id,
        "upstream_protocols": _entries_from_protocols(protocols),
    }
