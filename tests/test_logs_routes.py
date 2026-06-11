import os
import sqlite3
import tempfile
import pytest
from datetime import datetime
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def client(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from llm_proxy.infra import db
    monkeypatch.setattr(db, "DB_PATH", db.Path(path))
    db.init_db()

    # Insert sample data
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO endpoints (endpoint_id) VALUES ('ep-A')")
    c.execute("""INSERT INTO usage_records
        (timestamp, endpoint_id, model_id, input_tokens, output_tokens, request_status, latency_ms, error_type)
        VALUES ('2026-06-01 10:00:00', 'ep-A', 'm-1', 100, 50, 'success', 200, NULL)""")
    c.execute("""INSERT INTO usage_records
        (timestamp, endpoint_id, model_id, input_tokens, output_tokens, request_status, latency_ms, error_type)
        VALUES ('2026-06-01 11:00:00', 'ep-A', 'm-2', 200, 80, 'error', 300, '5xx')""")
    conn.commit()
    conn.close()

    from llm_proxy.main import app
    yield TestClient(app)


def test_logs_list(client):
    r = client.get("/api/logs/list")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert len(data["records"]) == 2
    assert data["records"][0]["endpoint_id"] == "ep-A"


def test_logs_list_with_filters(client):
    r = client.get("/api/logs/list?status=error")
    data = r.json()
    assert data["total"] == 1
    assert data["records"][0]["error_type"] == "5xx"


def test_logs_list_pagination(client):
    r = client.get("/api/logs/list?limit=1&offset=0")
    data = r.json()
    assert data["total"] == 2
    assert len(data["records"]) == 1
    assert data["limit"] == 1
    assert data["offset"] == 0


def test_logs_list_invalid_pagination(client):
    r = client.get("/api/logs/list?limit=abc")
    assert r.status_code == 400


def test_logs_list_caps_excessive_limit(client):
    """Requesting limit > MAX should be clamped, not allowed to run an unbounded query."""
    r = client.get("/api/logs/list?limit=10000")
    assert r.status_code == 200
    data = r.json()
    assert data["limit"] == 1000  # clamped, not 10000
    assert r.headers.get("x-limit-capped") == "true"


@pytest.mark.parametrize("field", ["since", "until"])
def test_logs_list_rejects_invalid_date_format(client, field):
    """Malformed since/until should be 400, not silently produce wrong results."""
    r = client.get(f"/api/logs/list?{field}=not-a-date")
    assert r.status_code == 400
    assert field in r.json()["error"].lower()


def test_logs_list_rejects_excessive_time_span(client):
    """A 6-year window should be 400 — long ranges must be paginated, not loaded at once."""
    r = client.get("/api/logs/list?since=2020-01-01&until=2026-01-01")
    assert r.status_code == 400
    assert "span" in r.json()["error"].lower()


def test_logs_summary(client):
    r = client.get("/api/logs/summary")
    data = r.json()
    assert data["total_requests"] == 2
    assert data["error_count"] == 1
    assert data["total_input_tokens"] == 300
    assert data["total_output_tokens"] == 130
    assert data["avg_latency_ms"] == 250


def test_filter_options(client):
    r = client.get("/api/logs/filter-options")
    data = r.json()
    assert any(e["id"] == "ep-A" for e in data["endpoints"])
    assert set(data["models"]) == {"m-1", "m-2"}
    assert set(data["statuses"]) == {"success", "error"}
    assert set(data["error_types"]) == {"5xx"}


def test_logs_list_order_by_datetime_not_string(client, monkeypatch):
    """所有 timestamp 都是北京时间字符串，'14:xx' < '21:xx' 是字符串序也是时间序。"""
    import os, sqlite3, tempfile
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from llm_proxy.infra import db
    monkeypatch.setattr(db, "DB_PATH", db.Path(path))
    db.init_db()
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO endpoints (endpoint_id) VALUES ('ep-A')")
    c.execute("""INSERT INTO usage_records
        (timestamp, endpoint_id, model_id, input_tokens, output_tokens, request_status)
        VALUES ('2026-06-06 14:00:00', 'ep-A', 'm-older', 1, 1, 'success')""")
    c.execute("""INSERT INTO usage_records
        (timestamp, endpoint_id, model_id, input_tokens, output_tokens, request_status)
        VALUES ('2026-06-06 22:59:00', 'ep-A', 'm-newer', 1, 1, 'success')""")
    conn.commit()
    conn.close()

    from fastapi.testclient import TestClient
    from llm_proxy.main import app
    tc = TestClient(app)
    r = tc.get("/api/logs/list")
    assert r.status_code == 200
    records = r.json()["records"]
    assert records[0]["model_id"] == "m-newer", f"got {records[0]} (sort bug)"
    assert records[1]["model_id"] == "m-older"
