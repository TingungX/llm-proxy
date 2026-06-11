#!/usr/bin/env python3
"""CLI wrapper for common usage_records queries."""
import argparse
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "usage.db"

SINCE_MAP = {"1h": "-1 hour", "24h": "-1 day", "7d": "-7 days", "30d": "-30 days"}


def _since_to_sql(since: str) -> str:
    return SINCE_MAP.get(since, f"-{since}")


def cmd_errors(args):
    since_sql = _since_to_sql(args.since)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"""
        SELECT timestamp, endpoint_id, model_id, request_status, error_type, latency_ms, request_id
        FROM usage_records
        WHERE timestamp > datetime('now', '{since_sql}') AND request_status = 'error'
        ORDER BY timestamp DESC LIMIT ?
    """, (args.limit,)).fetchall()
    conn.close()
    for r in rows:
        print(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4] or '-'} {r[5] or '-'}ms req={r[6] or '-'}")


def cmd_latency(args):
    since_sql = _since_to_sql(args.since)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"""
        SELECT model_id, COUNT(*), ROUND(AVG(latency_ms)), MAX(latency_ms)
        FROM usage_records
        WHERE timestamp > datetime('now', '{since_sql}') AND latency_ms IS NOT NULL
        GROUP BY model_id ORDER BY AVG(latency_ms) DESC
    """).fetchall()
    conn.close()
    for r in rows:
        print(f"{r[0]} count={r[1]} avg={r[2]}ms max={r[3]}ms")


def cmd_top_endpoints(args):
    since_sql = _since_to_sql(args.since)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"""
        SELECT endpoint_id, SUM(input_tokens + output_tokens) AS total
        FROM usage_records WHERE timestamp > datetime('now', '{since_sql}')
        GROUP BY endpoint_id ORDER BY total DESC LIMIT ?
    """, (args.limit,)).fetchall()
    conn.close()
    for r in rows:
        print(f"{r[0]} tokens={r[1]}")


def main():
    parser = argparse.ArgumentParser(description="Query usage_records")
    sub = parser.add_subparsers(required=True)

    p_errors = sub.add_parser("errors")
    p_errors.add_argument("--since", default="1h")
    p_errors.add_argument("--endpoint", default=None)
    p_errors.add_argument("--limit", type=int, default=20)
    p_errors.set_defaults(func=cmd_errors)

    p_latency = sub.add_parser("latency")
    p_latency.add_argument("--since", default="24h")
    p_latency.add_argument("--model", default=None)
    p_latency.set_defaults(func=cmd_latency)

    p_top = sub.add_parser("top-endpoints")
    p_top.add_argument("--since", default="7d")
    p_top.add_argument("--limit", type=int, default=10)
    p_top.set_defaults(func=cmd_top_endpoints)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
