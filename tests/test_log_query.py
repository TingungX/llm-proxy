# tests/test_log_query.py
import subprocess
import sys


def test_log_query_errors_subcommand():
    result = subprocess.run([sys.executable, "scripts/log_query.py", "errors", "--since", "1h"], capture_output=True, text=True)
    assert result.returncode == 0


def test_log_query_latency_subcommand():
    result = subprocess.run([sys.executable, "scripts/log_query.py", "latency", "--since", "24h"], capture_output=True, text=True)
    assert result.returncode == 0


def test_log_query_top_endpoints_subcommand():
    result = subprocess.run([sys.executable, "scripts/log_query.py", "top-endpoints", "--since", "7d"], capture_output=True, text=True)
    assert result.returncode == 0
