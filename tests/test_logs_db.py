import os
import sqlite3
import tempfile
import pytest
from datetime import datetime, timedelta
from llm_proxy.infra import db


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setattr(db, "DB_PATH", db.Path(path))
    db.init_db()
    yield path
    os.unlink(path)


def _insert(c, ts, ep='ep-test', m='m-a', inp=100, out=50, st='success', err=None, lat=None):
    c.execute("""
        INSERT INTO usage_records
            (timestamp, endpoint_id, model_id, input_tokens, output_tokens,
             request_status, request_id, latency_ms, error_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, ep, m, inp, out, st, None, lat, err))


def test_get_logs_list_empty(tmp_db):
    records, total = db.get_logs_list()
    assert records == []
    assert total == 0


def test_get_logs_list_basic(tmp_db):
    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()
    for i in range(3):
        _insert(c, (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'),
                ep='ep-1', m='m-1', st='success' if i % 2 == 0 else 'error',
                err=None if i % 2 == 0 else '5xx', lat=100 + i * 10)
    conn.commit(); conn.close()
    records, total = db.get_logs_list()
    assert total == 3
    assert len(records) == 3
    assert records[0]['endpoint_id'] == 'ep-1'


def test_get_logs_list_filters(tmp_db):
    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()
    _insert(c, '2026-06-01 10:00:00', ep='ep-A', m='m-1', st='success')
    _insert(c, '2026-06-01 11:00:00', ep='ep-B', m='m-2', st='error', err='5xx')
    conn.commit(); conn.close()

    records, total = db.get_logs_list(endpoint_id='ep-B')
    assert total == 1
    assert records[0]['model_id'] == 'm-2'

    records, total = db.get_logs_list(status='success')
    assert total == 1
    assert records[0]['endpoint_id'] == 'ep-A'


def test_get_logs_list_pagination(tmp_db):
    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()
    for i in range(5):
        _insert(c, f'2026-06-01 1{i}:00:00', ep='ep-1', m='m-1')
    conn.commit(); conn.close()
    records, total = db.get_logs_list(limit=2, offset=0)
    assert total == 5
    assert len(records) == 2


def test_get_logs_summary(tmp_db):
    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()
    _insert(c, '2026-06-01 10:00:00', inp=100, out=50, st='success', lat=200)
    _insert(c, '2026-06-01 11:00:00', inp=200, out=80, st='error', err='5xx', lat=300)
    conn.commit(); conn.close()

    summary = db.get_logs_summary()
    assert summary['total_requests'] == 2
    assert summary['error_count'] == 1
    assert summary['total_input_tokens'] == 300
    assert summary['total_output_tokens'] == 130
    assert summary['avg_latency_ms'] == 250


def test_get_filter_options(tmp_db):
    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()
    _insert(c, '2026-06-01 10:00:00', ep='ep-A', m='m-1', st='success')
    _insert(c, '2026-06-01 11:00:00', ep='ep-B', m='m-2', st='error', err='5xx')
    conn.commit(); conn.close()

    opts = db.get_filter_options()
    assert set(e['id'] for e in opts['endpoints']) == {'ep-A', 'ep-B'}
    assert set(opts['models']) == {'m-1', 'm-2'}
    assert set(opts['statuses']) == {'success', 'error'}
    assert set(opts['error_types']) == {'5xx'}
