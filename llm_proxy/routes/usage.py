"""GET /api/usage + /api/usage/summary"""

from datetime import datetime, timedelta, timezone

from llm_proxy.main import app
from llm_proxy.infra import db

BEIJING_TZ = timezone(timedelta(hours=8))


@app.get("/api/usage")
async def api_get_usage(
    days: int = 30,
    group_by: str = "model",
    granularity: str = "day",
    endpoint_id: str = None,
    view: str = None,
):
    end = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")
    start = (datetime.now(BEIJING_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")

    if view == "heatmap":
        data = db.get_usage_heatmap(start, end, group_by, endpoint_id)
        return {"data": data, "start": start, "end": end}

    data = db.get_usage(start, end, group_by, granularity, endpoint_id)
    return {"data": data, "start": start, "end": end, "group_by": group_by, "granularity": granularity}


@app.get("/api/usage/summary")
async def api_get_usage_summary():
    return db.get_usage_summary()
