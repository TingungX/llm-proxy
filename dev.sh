#!/bin/bash
# llm-proxy dev server — 端口 4010，代码热更新
# 用法: ./dev.sh [start|stop|restart|status|log]
#
# 环境变量:
#   LLM_PROXY_DEV_PORT  — 端口，默认 4010
#   LLM_PROXY_LOG_LEVEL — 日志级别，默认 DEBUG（dev 环境保留全量日志）

set -euo pipefail
cd "$(dirname "$0")"

PORT="${LLM_PROXY_DEV_PORT:-4010}"
LOG_LEVEL="${LLM_PROXY_LOG_LEVEL:-DEBUG}"
SCREEN_NAME="llm-proxy-dev"

# 加载 .dev-env 环境变量
if [ -f ".dev-env" ]; then
    set -a
    source .dev-env
    set +a
fi

LOG_FILE="dev-server.log"

stop_dev() {
    # Kill screen session
    if screen -list | grep -q "$SCREEN_NAME"; then
        echo "Stopping dev server (screen session: $SCREEN_NAME)..."
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null
        sleep 1
        # Fallback: kill by port
        PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            for PID in $PIDS; do
                kill "$PID" 2>/dev/null
            done
            sleep 1
            PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
            if [ -n "$PIDS" ]; then
                for PID in $PIDS; do
                    kill -9 "$PID" 2>/dev/null
                done
            fi
        fi
        echo "Stopped."
    else
        # Fallback: find by port
        PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            echo "Found process(es) on port $PORT, killing..."
            for PID in $PIDS; do
                kill "$PID" 2>/dev/null
            done
            sleep 1
            echo "Stopped."
        else
            echo "No dev server running on port $PORT."
        fi
    fi
}

start_dev() {
    # Check if already running
    EXISTING=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -n "$EXISTING" ]; then
        echo "Port $PORT already in use. Run './dev.sh stop' first."
        exit 1
    fi

    if screen -list | grep -q "$SCREEN_NAME"; then
        echo "Screen session '$SCREEN_NAME' already exists. Run './dev.sh stop' first."
        exit 1
    fi

    echo "Starting dev server on port $PORT (log level=$LOG_LEVEL)..."

    # Create screen session running the dev server
    screen -dmS "$SCREEN_NAME" \
        env LLM_PROXY_DEV=true LLM_PROXY_LOG_LEVEL="$LOG_LEVEL" \
        .venv/bin/uvicorn llm_proxy.main:app \
            --host 0.0.0.0 \
            --port "$PORT" \
            --reload \
            --reload-dir llm_proxy \
            --reload-dir static \
            --reload-include '*.py' \
            --reload-include '*.html' \
            --reload-include '*.js' \
            --reload-include '*.css' \
            --reload-include '*.json' \
            2>&1 | tee "$LOG_FILE"

    # Wait for server to be ready
    for i in $(seq 1 15); do
        sleep 1
        if curl -sf http://127.0.0.1:$PORT/v1/models -H "x-api-key: default" -o /dev/null 2>/dev/null; then
            echo "Dev server started (port=$PORT, screen=$SCREEN_NAME)"
            echo "Logs: tail -f $LOG_FILE"
            echo "Attach: screen -r $SCREEN_NAME"
            return 0
        fi
        if ! screen -list | grep -q "$SCREEN_NAME"; then
            echo "Failed to start. Check $LOG_FILE for errors."
            exit 1
        fi
    done

    echo "Server process running but not responding on port $PORT yet. Check $LOG_FILE."
}

case "${1:-start}" in
    start)   start_dev ;;
    stop)    stop_dev ;;
    restart) stop_dev; sleep 1; start_dev ;;
    status)
        PID=$(lsof -ti:$PORT 2>/dev/null || true)
        if [ -n "$PID" ]; then
            echo "Dev server running on port $PORT (PID=$PID, screen=$SCREEN_NAME)"
        else
            echo "Dev server not running."
        fi
        ;;
    log)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status|log]"
        exit 1
        ;;
esac

