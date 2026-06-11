# tests/test_usage_timezone.py
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


def test_record_usage_timestamp_is_utc(tmp_db):
    tmp_db.record_usage("ep1", "opus-4-7", 100, 50, "success")
    conn = sqlite3.connect(tmp_db.DB_PATH)
    ts = conn.execute("SELECT timestamp FROM usage_records").fetchone()[0]
    conn.close()
    # Should end with Z or be clearly UTC (no local timezone offset)
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is None or parsed.tzinfo == timezone.utc
