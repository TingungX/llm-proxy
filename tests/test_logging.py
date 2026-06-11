import logging
from llm_proxy.logging_config import RequestIdFilter, setup_logging


def test_request_id_filter_injects_request_id():
    filt = RequestIdFilter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    filt.filter(record)
    assert hasattr(record, "request_id")


def test_request_id_filter_defaults_to_no_request():
    filt = RequestIdFilter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    filt.filter(record)
    assert record.request_id == "-"


def test_setup_logging_format_includes_name():
    setup_logging()
    root = logging.getLogger()
    handler = root.handlers[0]
    fmt = handler.formatter._fmt
    assert "%(name)s" in fmt
    assert "%(request_id)s" in fmt


from fastapi import FastAPI
from fastapi.testclient import TestClient
from llm_proxy.logging_config import REQUEST_ID_CTX
from llm_proxy.middleware.request_id import request_id_middleware


def test_request_id_middleware_sets_context():
    app = FastAPI()
    app.middleware("http")(request_id_middleware)

    @app.get("/test")
    async def test_route():
        return {"request_id": REQUEST_ID_CTX.get("-")}

    client = TestClient(app)
    resp = client.get("/test")
    assert resp.status_code == 200
    assert resp.json()["request_id"] != "-"
    assert len(resp.json()["request_id"]) == 8


def test_request_id_middleware_returns_header():
    app = FastAPI()
    app.middleware("http")(request_id_middleware)

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/test")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) == 8


def test_request_id_different_per_request():
    app = FastAPI()
    app.middleware("http")(request_id_middleware)

    @app.get("/test")
    async def test_route():
        return {"request_id": REQUEST_ID_CTX.get("-")}

    client = TestClient(app)
    ids = set()
    for _ in range(5):
        resp = client.get("/test")
        ids.add(resp.json()["request_id"])
    assert len(ids) == 5


import logging
from llm_proxy.middleware.access_log import access_log_middleware


def test_access_log_middleware_logs_request(caplog):
    app = FastAPI()
    app.middleware("http")(access_log_middleware)

    @app.post("/v1/messages")
    async def test_route():
        return {"ok": True}

    client = TestClient(app)
    with caplog.at_level(logging.INFO, logger="llm_proxy.access"):
        resp = client.post("/v1/messages")
    assert resp.status_code == 200
    assert any("POST /v1/messages 200" in r.message for r in caplog.records)


def test_access_log_middleware_includes_duration(caplog):
    app = FastAPI()
    app.middleware("http")(access_log_middleware)

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    client = TestClient(app)
    with caplog.at_level(logging.INFO, logger="llm_proxy.access"):
        resp = client.get("/test")
    assert resp.status_code == 200
    access_logs = [r for r in caplog.records if r.name == "llm_proxy.access"]
    assert len(access_logs) == 1
    assert "ms" in access_logs[0].message


def test_access_log_middleware_uses_pipeline_stop_status(caplog):
    from fastapi.responses import JSONResponse
    from llm_proxy.handlers.base import PipelineStop

    app = FastAPI()
    app.middleware("http")(access_log_middleware)

    @app.post("/v1/responses")
    async def test_route():
        raise PipelineStop(JSONResponse({"error": {"message": "Unknown model"}}, status_code=400))

    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.INFO, logger="llm_proxy.access"):
        client.post("/v1/responses")
    access_logs = [r for r in caplog.records if r.name == "llm_proxy.access"]
    assert len(access_logs) == 1
    assert "POST /v1/responses 400" in access_logs[0].message
    assert " 500 " not in access_logs[0].message
