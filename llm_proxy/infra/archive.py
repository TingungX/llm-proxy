import atexit
import json
import logging
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "data" / "usage-archive"

_archive_q: queue.Queue = queue.Queue()
_worker_started = False
_worker_thread: threading.Thread | None = None


def _worker():
    while True:
        rec = _archive_q.get()
        if rec is None:
            return
        try:
            _write_record(rec)
        except Exception as e:
            logger.warning("archive write failed: %s", e)


def _write_record(rec: dict):
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    date = rec.pop("_archive_date", None) or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = ARCHIVE_DIR / f"usage-{date}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _flush():
    _archive_q.put(None)
    if _worker_thread is not None:
        _worker_thread.join(timeout=5)


def archive_record(rec: dict):
    global _worker_started, _worker_thread
    if not _worker_started:
        _worker_thread = threading.Thread(target=_worker, daemon=True, name="archive-writer")
        _worker_thread.start()
        _worker_started = True
        atexit.register(_flush)
    _archive_q.put_nowait(rec)
