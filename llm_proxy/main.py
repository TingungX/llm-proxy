"""FastAPI 应用入口：app 创建、lifespan、日志、静态文件"""

import asyncio
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from llm_proxy.infra import db
from llm_proxy.middleware import catch_all_exceptions
from llm_proxy.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
lifecycle = logging.getLogger("llm_proxy.lifecycle")


async def hourly_aggregator():
    """每小时聚合任务"""
    while True:
        await asyncio.sleep(3600)
        try:
            db.aggregate_hourly()
            lifecycle.info("Hourly aggregation completed")
        except Exception as e:
            logger.error(f"Hourly aggregation error: {e}")


async def daily_cleanup():
    """每日清理任务"""
    while True:
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_midnight = (midnight - now).total_seconds()
        await asyncio.sleep(seconds_until_midnight)
        try:
            db.aggregate_daily()
            db.cleanup_old_records()
            lifecycle.info("Daily cleanup completed")
        except Exception as e:
            logger.error(f"Daily cleanup error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Step 1: Database ──────────────────────────────────────
    try:
        db.init_db()
        lifecycle.info("Database initialized at %s", db.DB_PATH)
    except Exception as e:
        lifecycle.error(
            "Database initialization FAILED at %s: %s",
            getattr(db, "DB_PATH", "unknown"),
            e,
            exc_info=True,
        )
        raise

    # ── Step 2: Config / State ────────────────────────────────
    from llm_proxy.state import init_state, get_state
    try:
        init_state()
        lifecycle.info("State initialized")
    except FileNotFoundError:
        lifecycle.error(
            "Config file not found at %s. Create config.json from config.example.json",
            "config.json (check LLM_PROXY_CONFIG_PATH)",
        )
        raise
    except Exception as e:
        lifecycle.error("State initialization FAILED: %s", e, exc_info=True)
        raise

    # ── Step 3: Background tasks ──────────────────────────────
    aggregator_task = asyncio.create_task(hourly_aggregator())
    cleanup_task = asyncio.create_task(daily_cleanup())
    lifecycle.info("Background tasks started")

    state = get_state()
    model_count = len(state.config.get("models", {}))
    lifecycle.info(
        "Server ready — %d models loaded, port 4000", model_count,
    )

    yield

    aggregator_task.cancel()
    cleanup_task.cancel()

    from llm_proxy.infra.http_client import close_client
    await close_client()
    lifecycle.info("HTTP client closed")


app = FastAPI(lifespan=lifespan)
from llm_proxy.middleware.request_id import request_id_middleware
from llm_proxy.middleware.access_log import access_log_middleware
app.middleware("http")(request_id_middleware)
app.middleware("http")(access_log_middleware)
app.middleware("http")(catch_all_exceptions)

# 在 app 定义后导入路由，避免循环依赖
import llm_proxy.routes.misc       # noqa: F402
import llm_proxy.routes.messages   # noqa: F402
import llm_proxy.routes.config     # noqa: F402
import llm_proxy.routes.usage      # noqa: F402
import llm_proxy.routes.endpoints  # noqa: F402
import llm_proxy.routes.latency    # noqa: F402
import llm_proxy.routes.openai     # noqa: F402
import llm_proxy.routes.responses  # noqa: F402
import llm_proxy.routes.logs      # noqa: F402

# 挂载静态文件：Vite 构建产物在 static/dist/
# dist/index.html 引用 /static/assets/*，所以挂载 dist/ 到 /static/
BASE_DIR = Path(__file__).parent.parent
DIST_DIR = BASE_DIR / "static" / "dist"
app.mount("/static", StaticFiles(directory=str(DIST_DIR), html=True, check_dir=False), name="static")
