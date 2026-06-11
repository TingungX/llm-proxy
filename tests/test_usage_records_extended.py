# tests/test_usage_records_extended.py
import os
import sqlite3
import pytest
from datetime import datetime, timezone


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_usage.db"
    monkeypatch.setenv("LLM_PROXY_DB_PATH", str(db_path))
    from llm_proxy.infra import db
    db.DB_PATH = db_path
    db.init_db()
    yield db
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_record_usage_with_new_columns(tmp_db):
    tmp_db.record_usage("ep1", "opus-4-7", 100, 50, "success",
                        request_id="ab12cd34", latency_ms=250, error_type=None)
    conn = sqlite3.connect(tmp_db.DB_PATH)
    row = conn.execute("SELECT request_id, latency_ms, error_type FROM usage_records WHERE endpoint_id='ep1'").fetchone()
    conn.close()
    assert row == ("ab12cd34", 250, None)


def test_record_usage_backward_compat(tmp_db):
    tmp_db.record_usage("ep1", "opus-4-7", 100, 50)
    conn = sqlite3.connect(tmp_db.DB_PATH)
    row = conn.execute("SELECT request_id, latency_ms, error_type FROM usage_records WHERE endpoint_id='ep1'").fetchone()
    conn.close()
    assert row == ("", None, None)


def test_record_usage_error_path(tmp_db):
    tmp_db.record_usage("ep1", "opus-4-7", 0, 0, "error",
                        request_id="ef56gh78", error_type="timeout")
    conn = sqlite3.connect(tmp_db.DB_PATH)
    row = conn.execute("SELECT request_status, error_type FROM usage_records WHERE request_id='ef56gh78'").fetchone()
    conn.close()
    assert row == ("error", "timeout")
