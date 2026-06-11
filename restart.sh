#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-4000}"
HOST="${HOST:-0.0.0.0}"
LOG_FILE="${LOG_FILE:-proxy.log}"
PID_FILE="${PID_FILE:-.llm-proxy.pid}"
START_SCRIPT="$ROOT_DIR/start.sh"

pids_on_port() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser "$PORT/tcp" 2>/dev/null || true
  fi
}

stop_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  kill "$pid" 2>/dev/null || true
}

wait_for_exit() {
  local pid="$1"
  local i
  for i in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.2
  done
  kill -9 "$pid" 2>/dev/null || true
}

echo "Stopping llm-proxy on port $PORT..."
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]]; then
    stop_pid "$old_pid"
    wait_for_exit "$old_pid"
  fi
  rm -f "$PID_FILE"
fi

mapfile -t port_pids < <(pids_on_port)
if (( ${#port_pids[@]} > 0 )); then
  for pid in "${port_pids[@]}"; do
    stop_pid "$pid"
  done
  for pid in "${port_pids[@]}"; do
    wait_for_exit "$pid"
  done
  echo "Stopped process(es): ${port_pids[*]}"
else
  echo "No listener found on port $PORT"
fi

echo "Starting llm-proxy on port $PORT..."
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
PORT="$PORT" HOST="$HOST" LOG_FILE="$LOG_FILE" nohup "$START_SCRIPT" >> "$LOG_FILE" 2>&1 &
new_pid=$!
echo "$new_pid" > "$PID_FILE"

sleep 2
if ! kill -0 "$new_pid" 2>/dev/null; then
  echo "ERROR: llm-proxy failed to start. Last log lines:" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE"
  exit 1
fi

echo "Started pid $new_pid — http://localhost:$PORT"
echo "Log: $LOG_FILE"
