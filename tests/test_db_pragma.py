"""Tests that init_db configures SQLite reliability PRAGMAs.

The doc stability-improvement-directions.md §5 calls out:
- WAL mode to reduce writer/reader contention
- busy_timeout so concurrent writers wait rather than fail immediately
- synchronous=NORMAL for durability/perf tradeoff
"""
import os
import sqlite3
import tempfile
import pytest

from llm_proxy.infra import db


@pytest.fixture
def tmp_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setattr(db, "DB_PATH", db.Path(path))
    db.init_db()
    yield db
    os.unlink(path)


def test_init_db_enables_wal_journal_mode(tmp_db):
    """journal_mode should be 'wal' after init_db (not the default 'delete')."""
    conn = sqlite3.connect(tmp_db.DB_PATH)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode == "wal"


def test_init_db_sets_busy_timeout_30s(tmp_db):
    """busy_timeout should be 30s (not Python's 5s default) so writers wait longer under contention."""
    conn = tmp_db._connect()
    try:
        busy_timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        conn.close()
    assert busy_timeout_ms == 30000


def test_connect_helper_sets_synchronous_normal(tmp_db):
    """Connections opened via _connect() should have synchronous=NORMAL (1)."""
    conn = tmp_db._connect()
    try:
        # 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
        sync = conn.execute("PRAGMA synchronous").fetchone()[0]
    finally:
        conn.close()
    assert sync == 1


def test_init_db_logs_pragma_values_for_verification(tmp_db, caplog):
    """Startup log should report the actual journal_mode, busy_timeout, synchronous values
    so an operator can confirm the reliability PRAGMAs actually took effect."""
    import logging
    with caplog.at_level(logging.INFO, logger="llm_proxy.infra.db"):
        # Force re-run of init_db to capture its log output
        tmp_db.init_db()
    text = caplog.text
    assert "journal_mode=wal" in text
    assert "busy_timeout=30000" in text
    assert "synchronous=1" in text  # NORMAL is integer 1 in PRAGMA output
