#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

APP_MODULE="${APP_MODULE:-llm_proxy.main:app}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-4000}"
# 日志由 logging_config.py 统一写到 logs/llm-proxy.log（唯一真相源）。
# 这里不再重定向 stdout 到独立的 proxy.log，避免两套文件名。
# 若仍需 stdout 落盘（如 systemd 重定向），可显式设 LOG_FILE。
LOG_FILE="${LOG_FILE:-logs/llm-proxy.log}"
RELOAD="${RELOAD:-1}"
FRONTEND_DIR="$ROOT_DIR/static"
DIST_INDEX="$FRONTEND_DIR/dist/index.html"

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "ERROR: no Python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
  return 1
}

PYTHON="$(find_python)"

if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import fastapi, httpx, uvicorn
PY
then
  cat >&2 <<EOF2
ERROR: Python dependencies are missing for $PYTHON.
Install them in your environment (for example: pip install fastapi uvicorn httpx),
or set PYTHON_BIN to an interpreter that already has them.
EOF2
  exit 1
fi

build_frontend_if_needed() {
  [[ -f "$FRONTEND_DIR/package.json" ]] || return 0
  if [[ -f "$DIST_INDEX" && "${FORCE_FRONTEND_BUILD:-0}" != "1" ]]; then
    return 0
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "WARNING: npm not found; /static/ UI may be unavailable until static/dist is built." >&2
    return 0
  fi

  echo "Building frontend assets..."
  (
    cd "$FRONTEND_DIR"
    if [[ ! -d node_modules ]]; then
      npm ci
    fi
    npm run build
  )
}

build_frontend_if_needed

# 日志目录由 logging_config.py 创建并写入；这里只做存在性预创建，
# 以便权限问题在启动前就暴露，而不是在 logging_config 的 except 里静默吞掉。
mkdir -p "$(dirname "$LOG_FILE")"

# 代理：Steam++ (Watt Toolkit) MITM proxy，用于访问被墙上游（opencode.ai 等）
# allow_proxy=true 的模型走此代理，其余直连
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:1082}"

echo "Starting $APP_MODULE on http://$HOST:$PORT"
echo "Logging to $LOG_FILE (managed by llm_proxy.logging_config)"

UVICORN_ARGS=("$APP_MODULE" --host "$HOST" --port "$PORT")
if [[ "$RELOAD" != "0" ]]; then
  UVICORN_ARGS+=(--reload --reload-include '*.html' --reload-include '*.py' --reload-include '*.css' --reload-include '*.js' --reload-include '*.ts' --reload-include '*.tsx')
fi

# 不再重定向 stdout——logging_config.py 的 RotatingFileHandler 统一落盘到 $LOG_FILE。
# stdout 直接走调用者终端（systemd / 手动运行），便于实时观察。
exec "$PYTHON" -m uvicorn "${UVICORN_ARGS[@]}"
