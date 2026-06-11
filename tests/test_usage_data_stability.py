# tests/test_usage_data_stability.py
import os
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_usage.db"
    monkeypatch.setenv("LLM_PROXY_DB_PATH", str(db_path))
    from llm_proxy.infra import db
    db.DB_PATH = db_path
    db.RECORD_RETENTION_DAYS = 7
    db.init_db()
    yield db
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_get_usage_uses_stable_window_not_today(tmp_db):
    """Records segment should cover RECORD_RETENTION_DAYS, not just 'today'"""
    # Insert a record 2 days ago
    conn = sqlite3.connect(tmp_db.DB_PATH)
    ts = (datetime.now(timezone.utc) - timedelta(days=2)).strftime('%Y-%m-%d %H:00:00')
    conn.execute("INSERT INTO usage_records (timestamp, endpoint_id, model_id, input_tokens, output_tokens, request_status) VALUES (?, 'ep1', 'opus', 100, 50, 'success')", (ts,))
    conn.commit()
    conn.close()
    # get_usage should return data for that day even without hourly aggregation
    result = tmp_db.get_usage(
        (datetime.now(timezone.utc) - timedelta(days=3)).strftime('%Y-%m-%d'),
        datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'model', 'hour'
    )
    total = sum(r['total_tokens'] for r in result)
    assert total == 150  # 100 + 50


def test_aggregate_daily_idempotent(tmp_db):
    """aggregate_daily should produce same result when run twice"""
    # Insert hourly data older than HOURLY_RETENTION_DAYS (90)
    conn = sqlite3.connect(tmp_db.DB_PATH)
    conn.execute("INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count) VALUES ('2025-01-01 10:00:00', 'ep1', 'opus', 100, 50, 1)")
    conn.commit()
    conn.close()
    tmp_db.aggregate_daily()
    # Run again
    tmp_db.aggregate_daily()
    conn = sqlite3.connect(tmp_db.DB_PATH)
    row = conn.execute("SELECT total_input_tokens, total_output_tokens, request_count FROM usage_daily WHERE date='2025-01-01'").fetchone()
    conn.close()
    assert row == (100, 50, 1)
