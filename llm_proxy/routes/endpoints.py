"""/api/endpoints CRUD"""

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.main import app
from llm_proxy.infra import db


@app.get("/api/endpoints")
async def api_get_endpoints():
    return db.get_all_endpoints()


@app.get("/api/endpoints/{endpoint_id}")
async def api_get_endpoint(endpoint_id: str):
    ep = db.get_endpoint(endpoint_id)
    if not ep:
        return JSONResponse({"error": "Endpoint not found"}, status_code=404)
    return ep


@app.post("/api/endpoints")
async def api_create_endpoint(request: Request):
    body = await request.json()
    name = body.get("name")
    api_key = body.get("api_key")
    models = body.get("models", [])
    settings = body.get("settings", {})
    enabled = body.get("enabled", True)
    accept_protocols = body.get("accept_protocols", ["anthropic", "openai"])
    family_routing = body.get("family_routing")

    if not name or not api_key:
        return JSONResponse({"error": "name and api_key are required"}, status_code=400)

    endpoint_id = db.get_endpoint_id(api_key)
    if db.get_endpoint(endpoint_id):
        return JSONResponse({"error": "Endpoint with this API Key already exists"}, status_code=400)

    db.create_endpoint(endpoint_id, name, api_key, models, settings, enabled, accept_protocols, family_routing)
    return {"status": "ok", "endpoint_id": endpoint_id}


@app.put("/api/endpoints/{endpoint_id}")
async def api_update_endpoint(endpoint_id: str, request: Request):
    body = await request.json()
    ep = db.get_endpoint(endpoint_id)
    if not ep:
        return JSONResponse({"error": "Endpoint not found"}, status_code=404)

    if ep.get("is_default") and body.get("api_key"):
        return JSONResponse({"error": "Cannot change default endpoint's API key"}, status_code=400)

    db.update_endpoint(
        endpoint_id,
        name=body.get("name"),
        api_key=body.get("api_key"),
        models=body.get("models"),
        settings=body.get("settings"),
        enabled=body.get("enabled"),
        accept_protocols=body.get("accept_protocols"),
        family_routing=body.get("family_routing"),
    )
    return {"status": "ok"}


@app.delete("/api/endpoints/{endpoint_id}")
async def api_delete_endpoint(endpoint_id: str):
    ep = db.get_endpoint(endpoint_id)
    if not ep:
        return JSONResponse({"error": "Endpoint not found"}, status_code=404)
    
    if ep.get("is_default"):
        return JSONResponse({"error": "Cannot delete default endpoint"}, status_code=400)
    
    db.delete_endpoint(endpoint_id)
    return {"status": "ok"}
