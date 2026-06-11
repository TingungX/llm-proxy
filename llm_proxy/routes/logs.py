"""日志查询 API — /api/logs/list, /api/logs/summary, /api/logs/filter-options"""

from datetime import datetime

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.main import app
from llm_proxy.infra import db

# Hard caps to prevent unbounded queries from DOSing the DB
# (see docs/stability-improvement-directions.md §5)
MAX_LOG_LIMIT = 1000
MAX_LOG_TIME_SPAN_DAYS = 90


def _parse_date_param(value: str | None, field: str) -> str | None:
    """Validate ISO-8601 date or datetime. Returns the value, or raises via JSONResponse."""
    if value is None:
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise _BadDate(field, value)
    return value


class _BadDate(Exception):
    def __init__(self, field: str, value: str):
        self.field = field
        self.value = value


@app.get("/api/logs/list")
async def get_logs_list(request: Request):
    qp = request.query_params
    try:
        since = _parse_date_param(qp.get("since"), "since")
        until = _parse_date_param(qp.get("until"), "until")
    except _BadDate as e:
        return JSONResponse({"error": f"invalid {e.field} format: {e.value}"}, status_code=400)

    if since and until:
        span_days = (datetime.fromisoformat(until) - datetime.fromisoformat(since)).days
        if span_days > MAX_LOG_TIME_SPAN_DAYS:
            return JSONResponse(
                {"error": f"time span {span_days} days exceeds max {MAX_LOG_TIME_SPAN_DAYS}"},
                status_code=400,
            )
    endpoint_id = qp.get("endpoint_id")
    model_id = qp.get("model_id")
    status = qp.get("status")
    try:
        limit = int(qp.get("limit", "50"))
        offset = int(qp.get("offset", "0"))
    except ValueError:
        return JSONResponse({"error": "limit/offset must be int"}, status_code=400)

    capped = False
    if limit > MAX_LOG_LIMIT:
        limit = MAX_LOG_LIMIT
        capped = True

    records, total = db.get_logs_list(
        since=since, until=until, endpoint_id=endpoint_id, model_id=model_id,
        status=status, limit=limit, offset=offset,
    )
    response = JSONResponse({"records": records, "total": total, "limit": limit, "offset": offset})
    if capped:
        response.headers["x-limit-capped"] = "true"
    return response


@app.get("/api/logs/summary")
async def get_logs_summary(request: Request):
    qp = request.query_params
    summary = db.get_logs_summary(
        since=qp.get("since"),
        until=qp.get("until"),
        endpoint_id=qp.get("endpoint_id"),
        model_id=qp.get("model_id"),
    )
    return JSONResponse(summary)


@app.get("/api/logs/filter-options")
async def get_filter_options():
    return JSONResponse(db.get_filter_options())
