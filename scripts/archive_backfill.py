#!/usr/bin/env python3
"""One-time backfill: write all usage_records to JSONL archive files."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "usage.db"
ARCHIVE_DIR = Path(__file__).parent.parent / "data" / "usage-archive"


def main():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT timestamp, endpoint_id, model_id, input_tokens, output_tokens, request_status, request_id
        FROM usage_records ORDER BY timestamp
    """).fetchall()
    conn.close()

    files_written = {}
    for row in rows:
        ts, endpoint_id, model_id, inp, out, status, request_id = row
        dt = datetime.fromisoformat(ts)
        date_str = dt.strftime("%Y-%m-%d")
        path = ARCHIVE_DIR / f"usage-{date_str}.jsonl"
        rec = {"ts": ts, "endpoint_id": endpoint_id, "model_id": model_id,
               "input_tokens": inp, "output_tokens": out, "status": status, "request_id": request_id or ""}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        files_written[date_str] = files_written.get(date_str, 0) + 1

    print(f"Backfilled {len(rows)} records into {len(files_written)} files")
    for d, count in sorted(files_written.items()):
        print(f"  {d}: {count} records")


if __name__ == "__main__":
    main()
