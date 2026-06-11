# tests/test_archive.py
import json
import os
import time
from pathlib import Path
import pytest


@pytest.fixture
def archive_dir(tmp_path, monkeypatch):
    d = tmp_path / "archive"
    d.mkdir()
    monkeypatch.setattr("llm_proxy.infra.archive.ARCHIVE_DIR", d)
    # Reset module state
    import llm_proxy.infra.archive as arch
    arch._archive_q.queue.clear()
    arch._worker_started = False
    yield d


def test_archive_record_creates_jsonl(archive_dir):
    from llm_proxy.infra.archive import archive_record
    archive_record({"ts": "2026-06-01T12:00:00Z", "endpoint_id": "ep1", "model_id": "opus", "status": "success"})
    archive_record({"ts": "2026-06-01T12:01:00Z", "endpoint_id": "ep1", "model_id": "opus", "status": "success"})
    time.sleep(0.5)  # wait for worker
    files = list(archive_dir.glob("*.jsonl"))
    assert len(files) >= 1
    lines = files[0].read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["endpoint_id"] == "ep1"


def test_archive_write_failure_does_not_raise(archive_dir):
    from llm_proxy.infra.archive import archive_record
    # Make dir read-only to force write failure
    os.chmod(archive_dir, 0o444)
    archive_record({"ts": "2026-06-01T12:00:00Z", "endpoint_id": "ep1", "status": "success"})
    time.sleep(0.5)
    os.chmod(archive_dir, 0o755)
    # No exception raised = pass
