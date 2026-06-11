import os
import sqlite3
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))

_DEFAULT_DB_DIR = Path(__file__).parent.parent.parent  # llm-proxy project root


def _get_db_path() -> Path:
    env_path = os.environ.get("LLM_PROXY_DB_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_DB_DIR / "usage.db"


DB_PATH = _get_db_path()

# 数据保留策略
RECORD_RETENTION_DAYS = 7  # 明细数据保留天数
HOURLY_RETENTION_DAYS = 90  # 小时聚合数据保留天数

# SQLite reliability PRAGMAs (see docs/stability-improvement-directions.md §5)
# busy_timeout is per-connection and not persisted, so it lives in _connect().
# journal_mode (WAL) and synchronous are persisted to the DB file by init_db().
_BUSY_TIMEOUT_MS = 30000  # writers wait 30s for the lock; 5s Python default is too short under load


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_endpoint_id(api_key: str) -> str:
    """生成端点 ID（API Key 的 SHA256 前 16 位）"""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def ensure_default_endpoint():
    """确保默认端点存在"""
    default_api_key = "default"
    default_endpoint_id = get_endpoint_id(default_api_key)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT endpoint_id FROM endpoints WHERE is_default = 1")
    if c.fetchone():
        conn.close()
        return

    c.execute("""
        INSERT INTO endpoints (endpoint_id, name, api_key, api_key_hash, models, settings, enabled, is_default)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (default_endpoint_id, "Default", default_api_key, get_endpoint_id(default_api_key), "[]", "{}", 1, 1))

    conn.commit()
    conn.close()


def migrate_global_family_routing():
    """将全局 family_routing 迁移到默认端点"""
    config_path = _DEFAULT_DB_DIR / "config.json"
    if not config_path.exists():
        return

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return

    global_family_routing = config.get("family_routing")
    if not global_family_routing:
        return

    default_endpoint_id = get_endpoint_id("default")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT family_routing FROM endpoints WHERE endpoint_id = ? AND is_default = 1", (default_endpoint_id,))
    row = c.fetchone()

    if row and row[0] is None:
        c.execute(
            "UPDATE endpoints SET family_routing = ? WHERE endpoint_id = ? AND is_default = 1",
            (json.dumps(global_family_routing), default_endpoint_id)
        )
        conn.commit()

    conn.close()


def init_db():
    """初始化数据库和表结构"""
    conn = _connect()
    c = conn.cursor()

    # Persistent reliability PRAGMA (written to the DB file, applies to all future connections)
    c.execute("PRAGMA journal_mode=WAL")
    journal_mode = c.execute("PRAGMA journal_mode").fetchone()[0]
    busy_timeout = c.execute("PRAGMA busy_timeout").fetchone()[0]
    synchronous = c.execute("PRAGMA synchronous").fetchone()[0]
    logger.info(
        "sqlite pragmas applied: journal_mode=%s busy_timeout=%s synchronous=%s db_path=%s",
        journal_mode, busy_timeout, synchronous, DB_PATH,
    )

    # 明细表
    c.execute("""
        CREATE TABLE IF NOT EXISTS usage_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            endpoint_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            request_status TEXT DEFAULT 'success'
        )
    """)

    # 小时聚合表
    c.execute("""
        CREATE TABLE IF NOT EXISTS usage_hourly (
            hour_start DATETIME NOT NULL,
            endpoint_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            total_input_tokens INTEGER DEFAULT 0,
            total_output_tokens INTEGER DEFAULT 0,
            request_count INTEGER DEFAULT 0,
            PRIMARY KEY (hour_start, endpoint_id, model_id)
        )
    """)

    # 日级聚合表（永久保留）
    c.execute("""
        CREATE TABLE IF NOT EXISTS usage_daily (
            date TEXT NOT NULL,
            model_id TEXT NOT NULL,
            endpoint_id TEXT NOT NULL,
            total_input_tokens INTEGER DEFAULT 0,
            total_output_tokens INTEGER DEFAULT 0,
            request_count INTEGER DEFAULT 0,
            PRIMARY KEY (date, model_id, endpoint_id)
        )
    """)

    # 端点配置表（扩展）
    c.execute("""
        CREATE TABLE IF NOT EXISTS endpoints (
            endpoint_id TEXT PRIMARY KEY,
            name TEXT,
            api_key TEXT,
            api_key_hash TEXT,
            models TEXT,
            settings TEXT,
            enabled INTEGER DEFAULT 1,
            alias TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        c.execute("ALTER TABLE endpoints ADD COLUMN api_key TEXT")
    except sqlite3.OperationalError:
        pass  # 列已存在

    # 新增字段
    for col, col_type in [
        ("is_default", "INTEGER DEFAULT 0"),
        ("accept_protocols", "TEXT DEFAULT '[\"anthropic\", \"openai\"]'"),
        ("family_routing", "TEXT DEFAULT NULL"),
    ]:
        try:
            c.execute(f"ALTER TABLE endpoints ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # 列已存在

    # 创建索引加速查询
    c.execute("CREATE INDEX IF NOT EXISTS idx_records_timestamp ON usage_records(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_records_endpoint ON usage_records(endpoint_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_records_model ON usage_records(model_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_hourly_hour ON usage_hourly(hour_start)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON usage_daily(date)")

    # usage_records 扩展列
    for col, col_type in [
        ("request_id",   "TEXT"),
        ("latency_ms",   "INTEGER"),
        ("error_type",   "TEXT"),
        ("completed_at", "DATETIME"),
        ("client_ip",    "TEXT"),
        ("user_agent",   "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE usage_records ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    # 新增索引
    c.execute("CREATE INDEX IF NOT EXISTS idx_records_request_id ON usage_records(request_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_records_status_ts ON usage_records(request_status, timestamp)")

    conn.commit()
    conn.close()

    ensure_default_endpoint()
    migrate_global_family_routing()
    backfill_daily()


def record_usage(
    endpoint_id: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    status: str = "success",
    request_id: str = "",
    latency_ms: int | None = None,
    error_type: str | None = None,
    client_ip: str = "",
    user_agent: str = "",
):
    """记录单次请求用量"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
    c.execute("""
        INSERT INTO usage_records (timestamp, endpoint_id, model_id, input_tokens, output_tokens, request_status,
                                   request_id, latency_ms, error_type, client_ip, user_agent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now, endpoint_id, model_id, input_tokens, output_tokens, status,
          request_id, latency_ms, error_type, client_ip, user_agent))

    # 确保端点存在
    c.execute("""
        INSERT OR IGNORE INTO endpoints (endpoint_id) VALUES (?)
    """, (endpoint_id,))

    conn.commit()
    conn.close()

    # DB 写入成功后再归档（DB 是 source of truth）
    from llm_proxy.infra.archive import archive_record
    archive_record({
        "ts": now,
        "endpoint_id": endpoint_id,
        "model_id": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": status,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "error_type": error_type,
    })


def aggregate_hourly(catch_up: bool = False):
    """将上一小时的明细聚合到 hourly 表

    Args:
        catch_up: 若为 True，补齐所有缺失的小时聚合（用于清理前确保数据完整）
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if not catch_up:
        now = datetime.now(BEIJING_TZ)
        last_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        last_hour_str = last_hour.strftime('%Y-%m-%d %H:%M:%S')
        c.execute("SELECT COUNT(*) FROM usage_hourly WHERE hour_start = ?", (last_hour_str,))
        if c.fetchone()[0] > 0:
            conn.close()
            return
        next_hour_str = (last_hour + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        c.execute("""
            INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count)
            SELECT ? as hour_start, endpoint_id, model_id,
                   SUM(input_tokens), SUM(output_tokens), COUNT(*)
            FROM usage_records WHERE timestamp >= ? AND timestamp < ?
            GROUP BY endpoint_id, model_id
        """, (last_hour_str, last_hour_str, next_hour_str))
    else:
        c.execute("""
            SELECT DISTINCT strftime('%Y-%m-%d %H:00:00', timestamp) as hour
            FROM usage_records r
            WHERE NOT EXISTS (
                SELECT 1 FROM usage_hourly h
                WHERE h.hour_start = strftime('%Y-%m-%d %H:00:00', r.timestamp)
            )
            ORDER BY hour
        """)
        missing_hours = [row[0] for row in c.fetchall()]
        for hour_start in missing_hours:
            hour_dt = datetime.strptime(hour_start, '%Y-%m-%d %H:%M:%S')
            next_hour = (hour_dt + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            c.execute("""
                INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count)
                SELECT ? as hour_start, endpoint_id, model_id,
                       SUM(input_tokens), SUM(output_tokens), COUNT(*)
                FROM usage_records WHERE timestamp >= ? AND timestamp < ?
                GROUP BY endpoint_id, model_id
            """, (hour_start, hour_start, next_hour))

    conn.commit()
    conn.close()


def backfill_hourly():
    """补齐历史聚合数据（一次性修复）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 找出所有有记录的小时
    c.execute("""
        SELECT DISTINCT strftime('%Y-%m-%d %H:00:00', timestamp) as hour
        FROM usage_records
        ORDER BY hour
    """)
    hours = [row[0] for row in c.fetchall()]

    for hour_start in hours:
        # 检查是否已聚合
        c.execute("SELECT COUNT(*) FROM usage_hourly WHERE hour_start = ?", (hour_start,))
        if c.fetchone()[0] > 0:
            continue

        # 计算时间范围
        hour_dt = datetime.strptime(hour_start, '%Y-%m-%d %H:%M:%S')
        next_hour = (hour_dt + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

        # 聚合该小时的数据
        c.execute("""
            INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count)
            SELECT
                ? as hour_start,
                endpoint_id,
                model_id,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                COUNT(*) as request_count
            FROM usage_records
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY endpoint_id, model_id
        """, (hour_start, hour_start, next_hour))

    conn.commit()
    conn.close()


def backfill_daily():
    """一次性将所有 usage_hourly 数据灌入 usage_daily（幂等）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT OR REPLACE INTO usage_daily (date, model_id, endpoint_id, total_input_tokens, total_output_tokens, request_count)
        SELECT
            strftime('%Y-%m-%d', hour_start) as date,
            model_id,
            endpoint_id,
            SUM(total_input_tokens),
            SUM(total_output_tokens),
            SUM(request_count)
        FROM usage_hourly
        GROUP BY date, model_id, endpoint_id
    """)

    conn.commit()
    conn.close()


def aggregate_daily():
    """将过期的小时聚合数据滚动合并到日级表，然后删除对应小时数据"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    cutoff = (datetime.now(BEIJING_TZ) - timedelta(days=HOURLY_RETENTION_DAYS)).strftime('%Y-%m-%d %H:%M:%S')

    # 先 DELETE 再 INSERT，确保幂等
    c.execute("SELECT MIN(strftime('%Y-%m-%d', hour_start)) FROM usage_hourly WHERE hour_start < ?", (cutoff,))
    row = c.fetchone()
    min_date = row[0] if row else None
    if min_date:
        c.execute("DELETE FROM usage_daily WHERE date >= ? AND date < ?", (min_date, cutoff[:10]))

    c.execute("""
        INSERT INTO usage_daily (date, model_id, endpoint_id, total_input_tokens, total_output_tokens, request_count)
        SELECT
            strftime('%Y-%m-%d', hour_start) as date,
            model_id,
            endpoint_id,
            SUM(total_input_tokens),
            SUM(total_output_tokens),
            SUM(request_count)
        FROM usage_hourly
        WHERE hour_start < ?
        GROUP BY date, model_id, endpoint_id
    """, (cutoff,))

    c.execute("DELETE FROM usage_hourly WHERE hour_start < ?", (cutoff,))

    conn.commit()
    conn.close()


def cleanup_old_records():
    """清理过期数据"""
    aggregate_hourly(catch_up=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now(BEIJING_TZ)

    cutoff_records = (now - timedelta(days=RECORD_RETENTION_DAYS)).strftime('%Y-%m-%d %H:%M:%S')
    c.execute("DELETE FROM usage_records WHERE timestamp < ?", (cutoff_records,))

    cutoff_hourly = (now - timedelta(days=HOURLY_RETENTION_DAYS)).strftime('%Y-%m-%d %H:%M:%S')
    c.execute("DELETE FROM usage_hourly WHERE hour_start < ?", (cutoff_hourly,))

    # 清理孤立端点（无任何记录的端点）
    c.execute("""
        DELETE FROM endpoints WHERE endpoint_id NOT IN (
            SELECT DISTINCT endpoint_id FROM usage_records
            UNION
            SELECT DISTINCT endpoint_id FROM usage_hourly
        )
    """)

    conn.commit()
    conn.close()

    aggregate_daily()


def get_usage(
    start: str,
    end: str,
    group_by: str = "model",
    granularity: str = "hour",
    endpoint_id: str = None
) -> list[dict]:
    """查询用量数据

    Args:
        start: 开始日期 (YYYY-MM-DD)
        end: 结束日期 (YYYY-MM-DD)
        group_by: 分组维度 (model/endpoint)
        granularity: 时间粒度 (hour/day)
        endpoint_id: 按端点筛选（可选）

    Returns:
        [{time, group_key, input_tokens, output_tokens, count}]
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 确定分组字段
    group_expr = "model_id" if group_by == "model" else "endpoint_id"

    # 扩展结束日期到当天结束
    end_dt = datetime.fromisoformat(end) + timedelta(days=1)
    end_extended = end_dt.strftime("%Y-%m-%d")

    # 构建端点筛选条件
    ep_where = "AND endpoint_id = ?" if endpoint_id else ""
    ep_params = [endpoint_id] if endpoint_id else []

    record_window_start = (datetime.now(BEIJING_TZ) - timedelta(days=RECORD_RETENTION_DAYS)).strftime("%Y-%m-%d %H:00:00")
    today_start = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")

    if granularity == "hour":
        time_expr = "strftime('%Y-%m-%d %H:00', timestamp)"
        hourly_time_expr = "strftime('%Y-%m-%d %H:00', hour_start)"

        c.execute(f"""
            SELECT
                {hourly_time_expr} as time,
                {group_expr} as group_key,
                SUM(total_input_tokens) as input_tokens,
                SUM(total_output_tokens) as output_tokens,
                SUM(request_count) as count
            FROM usage_hourly
            WHERE hour_start >= ? AND hour_start < ? {ep_where}
            GROUP BY time, group_key
        """, [start, end_extended] + ep_params)
        hourly_rows = c.fetchall()

        c.execute(f"""
            SELECT
                {time_expr} as time,
                {group_expr} as group_key,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                COUNT(*) as count
            FROM usage_records
            WHERE timestamp >= ? AND timestamp < ? {ep_where}
            GROUP BY time, group_key
        """, [record_window_start, end_extended] + ep_params)
        record_rows = c.fetchall()

        all_rows = hourly_rows + record_rows
        slot_delta = timedelta(hours=1)
        slot_fmt = "%Y-%m-%d %H:00"
    else:
        # granularity=day：usage_daily（历史）+ usage_records（近7天）
        c.execute(f"""
            SELECT
                date as time,
                {group_expr} as group_key,
                SUM(total_input_tokens) as input_tokens,
                SUM(total_output_tokens) as output_tokens,
                SUM(request_count) as count
            FROM usage_daily
            WHERE date >= ? AND date < ? {ep_where}
            GROUP BY time, group_key
        """, [start, end_extended] + ep_params)
        daily_rows = c.fetchall()

        c.execute(f"""
            SELECT
                strftime('%Y-%m-%d', timestamp) as time,
                {group_expr} as group_key,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                COUNT(*) as count
            FROM usage_records
            WHERE timestamp >= ? AND timestamp < ? {ep_where}
            GROUP BY time, group_key
        """, [record_window_start, end_extended] + ep_params)
        record_rows = c.fetchall()

        all_rows = daily_rows + record_rows
        slot_delta = timedelta(days=1)
        slot_fmt = "%Y-%m-%d"

    conn.close()

    # 转换为字典，合并同 (time, group_key) 的数据
    merged = {}
    if granularity == "hour":
        # hour 模式：hourly_rows（历史）+ record_rows（今天）不重叠，直接累加
        for row in hourly_rows + record_rows:
            key = (row[0], row[1])
            if key in merged:
                merged[key]["input_tokens"] += row[2]
                merged[key]["output_tokens"] += row[3]
                merged[key]["count"] += row[4]
            else:
                merged[key] = {
                    "time": row[0],
                    "group_key": row[1],
                    "input_tokens": row[2],
                    "output_tokens": row[3],
                    "total_tokens": row[2] + row[3],
                    "count": row[4]
                }
    else:
        # day 模式：record_rows 优先（更新更完整），daily_rows 补充 records 未覆盖的日期
        for row in record_rows:
            key = (row[0], row[1])
            merged[key] = {
                "time": row[0],
                "group_key": row[1],
                "input_tokens": row[2],
                "output_tokens": row[3],
                "total_tokens": row[2] + row[3],
                "count": row[4]
            }
        for row in daily_rows:
            key = (row[0], row[1])
            if key not in merged:
                merged[key] = {
                    "time": row[0],
                    "group_key": row[1],
                    "input_tokens": row[2],
                    "output_tokens": row[3],
                    "total_tokens": row[2] + row[3],
                    "count": row[4]
                }

    result = list(merged.values())

    # 补零：生成范围内所有时间点的完整序列
    all_groups = list(set(r["group_key"] for r in result))
    existing = set((r["time"], r["group_key"]) for r in result)
    t = datetime.fromisoformat(start)
    end_dt_obj = datetime.fromisoformat(end_extended)
    while t < end_dt_obj:
        ts = t.strftime(slot_fmt)
        for g in all_groups:
            if (ts, g) not in existing:
                result.append({
                    "time": ts,
                    "group_key": g,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "count": 0
                })
        t += slot_delta

    result.sort(key=lambda r: (r["time"], r["group_key"]))
    return result


def get_usage_heatmap(
    start: str,
    end: str,
    group_by: str = "model",
    endpoint_id: str = None
) -> list[dict]:
    """查询热力图数据（按天聚合的扁平列表）

    Args:
        start: 开始日期 (YYYY-MM-DD)
        end: 结束日期 (YYYY-MM-DD)
        group_by: 分组维度（仅用于筛选，结果不拆分 group_key）
        endpoint_id: 按端点筛选（可选）

    Returns:
        [{date, total_tokens}]
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    end_dt = datetime.fromisoformat(end) + timedelta(days=1)
    end_extended = end_dt.strftime("%Y-%m-%d")

    ep_where = "AND endpoint_id = ?" if endpoint_id else ""
    ep_params = [endpoint_id] if endpoint_id else []

    # usage_daily（历史）
    c.execute(f"""
        SELECT
            date,
            SUM(total_input_tokens) as input_tokens,
            SUM(total_output_tokens) as output_tokens
        FROM usage_daily
        WHERE date >= ? AND date < ? {ep_where}
        GROUP BY date
    """, [start, end_extended] + ep_params)
    daily_rows = c.fetchall()

    # usage_records（近7天，补充 usage_daily 未覆盖的近期数据）
    recent_start = (datetime.now(BEIJING_TZ) - timedelta(days=RECORD_RETENTION_DAYS)).strftime("%Y-%m-%d")
    c.execute(f"""
        SELECT
            strftime('%Y-%m-%d', timestamp) as date,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens
        FROM usage_records
        WHERE timestamp >= ? AND timestamp < ? {ep_where}
        GROUP BY date
    """, [recent_start, end_extended] + ep_params)
    record_rows = c.fetchall()

    conn.close()

    # 合并同一天的数据，records 优先（更新更完整），daily 补充 records 未覆盖的日期
    merged = {}
    for row in record_rows:
        merged[row[0]] = row[1] + row[2]
    for row in daily_rows:
        if row[0] not in merged:
            merged[row[0]] = row[1] + row[2]

    return [{"date": d, "total_tokens": v} for d, v in sorted(merged.items())]


def get_endpoints() -> list[dict]:
    """获取所有端点列表"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT
            e.endpoint_id,
            e.alias,
            e.created_at,
            MAX(r.timestamp) as last_used
        FROM endpoints e
        LEFT JOIN usage_records r ON e.endpoint_id = r.endpoint_id
        GROUP BY e.endpoint_id
        ORDER BY last_used DESC
    """)

    rows = c.fetchall()
    conn.close()

    return [
        {
            "endpoint_id": row[0],
            "alias": row[1],
            "created_at": row[2],
            "last_used": row[3]
        }
        for row in rows
    ]


def set_endpoint_alias(endpoint_id: str, alias: str):
    """设置端点别名"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT INTO endpoints (endpoint_id, alias) VALUES (?, ?)
        ON CONFLICT(endpoint_id) DO UPDATE SET alias = ?
    """, (endpoint_id, alias, alias))

    conn.commit()
    conn.close()


def get_usage_summary() -> dict:
    """获取用量汇总统计"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 优先从聚合表获取，若无数据则从明细表获取
    c.execute("SELECT COUNT(*) FROM usage_hourly")
    use_hourly = c.fetchone()[0] > 0

    if use_hourly:
        # 总用量
        c.execute("""
            SELECT
                COALESCE(SUM(total_input_tokens), 0),
                COALESCE(SUM(total_output_tokens), 0),
                COALESCE(SUM(request_count), 0)
            FROM usage_hourly
        """)
        total = c.fetchone()

        # 今日用量
        today = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")
        c.execute("""
            SELECT
                COALESCE(SUM(total_input_tokens), 0),
                COALESCE(SUM(total_output_tokens), 0),
                COALESCE(SUM(request_count), 0)
            FROM usage_hourly
            WHERE hour_start >= ?
        """, (today,))
        today_stats = c.fetchone()

        # 活跃端点数（仅已配置的）
        c.execute("""
            SELECT COUNT(DISTINCT h.endpoint_id)
            FROM usage_hourly h
            JOIN endpoints e ON h.endpoint_id = e.endpoint_id
            WHERE e.api_key_hash IS NOT NULL
        """)
        active_endpoints = c.fetchone()[0]

        # 活跃模型数
        c.execute("SELECT COUNT(DISTINCT model_id) FROM usage_hourly")
        active_models = c.fetchone()[0]
    else:
        # 从明细表获取
        c.execute("""
            SELECT
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COUNT(*)
            FROM usage_records
        """)
        total = c.fetchone()

        today = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")
        c.execute("""
            SELECT
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COUNT(*)
            FROM usage_records
            WHERE timestamp >= ?
        """, (today,))
        today_stats = c.fetchone()

        c.execute("""
            SELECT COUNT(DISTINCT r.endpoint_id)
            FROM usage_records r
            JOIN endpoints e ON r.endpoint_id = e.endpoint_id
            WHERE e.api_key_hash IS NOT NULL
        """)
        active_endpoints = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT model_id) FROM usage_records")
        active_models = c.fetchone()[0]

    conn.close()

    return {
        "total_input_tokens": total[0],
        "total_output_tokens": total[1],
        "total_tokens": total[0] + total[1],
        "total_requests": total[2],
        "today_input_tokens": today_stats[0],
        "today_output_tokens": today_stats[1],
        "today_tokens": today_stats[0] + today_stats[1],
        "today_requests": today_stats[2],
        "active_endpoints": active_endpoints,
        "active_models": active_models
    }


def create_endpoint(
    endpoint_id: str,
    name: str,
    api_key: str,
    models: list[str],
    settings: dict,
    enabled: bool = True,
    accept_protocols: list[str] = None,
    is_default: bool = False,
    family_routing: dict = None
):
    """创建端点"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    api_key_hash = get_endpoint_id(api_key)

    if accept_protocols is None:
        accept_protocols = ["anthropic", "openai"]

    c.execute("""
        INSERT INTO endpoints (endpoint_id, name, api_key, api_key_hash, models, settings, enabled, is_default, accept_protocols, family_routing)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        endpoint_id, name, api_key, api_key_hash,
        json.dumps(models), json.dumps(settings),
        1 if enabled else 0, 1 if is_default else 0,
        json.dumps(accept_protocols),
        json.dumps(family_routing) if family_routing else None
    ))
    conn.commit()
    conn.close()


def update_endpoint(
    endpoint_id: str,
    name: str = None,
    api_key: str = None,
    models: list[str] = None,
    settings: dict = None,
    enabled: bool = None,
    accept_protocols: list[str] = None,
    family_routing: dict = None
):
    """更新端点"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    updates = []
    params = []
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if api_key is not None:
        updates.append("api_key = ?")
        params.append(api_key)
        updates.append("api_key_hash = ?")
        params.append(get_endpoint_id(api_key))
    if models is not None:
        updates.append("models = ?")
        params.append(json.dumps(models))
    if settings is not None:
        updates.append("settings = ?")
        params.append(json.dumps(settings))
    if enabled is not None:
        updates.append("enabled = ?")
        params.append(1 if enabled else 0)
    if accept_protocols is not None:
        updates.append("accept_protocols = ?")
        params.append(json.dumps(accept_protocols))
    if family_routing is not None:
        updates.append("family_routing = ?")
        params.append(json.dumps(family_routing))

    if updates:
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(endpoint_id)
        c.execute(f"UPDATE endpoints SET {', '.join(updates)} WHERE endpoint_id = ?", params)
        conn.commit()
    conn.close()


def delete_endpoint(endpoint_id: str):
    """删除端点（默认端点不可删除）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT is_default FROM endpoints WHERE endpoint_id = ?", (endpoint_id,))
    row = c.fetchone()
    if row and row[0]:
        conn.close()
        raise ValueError("Cannot delete default endpoint")

    c.execute("DELETE FROM endpoints WHERE endpoint_id = ?", (endpoint_id,))
    conn.commit()
    conn.close()


def get_endpoint(endpoint_id: str) -> dict | None:
    """获取单个端点"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT endpoint_id, name, api_key, api_key_hash, models, settings, enabled, alias, created_at, updated_at, is_default, accept_protocols, family_routing
        FROM endpoints WHERE endpoint_id = ?
    """, (endpoint_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "endpoint_id": row[0],
        "name": row[1],
        "api_key": row[2],
        "api_key_hash": row[3],
        "models": json.loads(row[4]) if row[4] else [],
        "settings": json.loads(row[5]) if row[5] else {},
        "enabled": bool(row[6]),
        "alias": row[7],
        "created_at": row[8],
        "updated_at": row[9],
        "is_default": bool(row[10]),
        "accept_protocols": json.loads(row[11]) if row[11] else ["anthropic", "openai"],
        "family_routing": json.loads(row[12]) if row[12] else None
    }


def get_endpoint_by_api_key(api_key: str) -> dict | None:
    """根据 API Key 获取端点"""
    api_key_hash = get_endpoint_id(api_key)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT endpoint_id, name, api_key_hash, models, settings, enabled, alias, is_default, accept_protocols, family_routing
        FROM endpoints WHERE api_key_hash = ?
    """, (api_key_hash,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "endpoint_id": row[0],
        "name": row[1],
        "api_key_hash": row[2],
        "models": json.loads(row[3]) if row[3] else [],
        "settings": json.loads(row[4]) if row[4] else {},
        "enabled": bool(row[5]),
        "alias": row[6],
        "is_default": bool(row[7]),
        "accept_protocols": json.loads(row[8]) if row[8] else ["anthropic", "openai"],
        "family_routing": json.loads(row[9]) if row[9] else None
    }


def get_all_endpoints() -> list[dict]:
    """获取所有端点（含用量统计）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT
            e.endpoint_id, e.name, e.api_key_hash, e.models, e.settings, e.enabled, e.alias, e.is_default, e.accept_protocols, e.family_routing,
            MAX(r.timestamp) as last_used
        FROM endpoints e
        LEFT JOIN usage_records r ON e.endpoint_id = r.endpoint_id
        WHERE e.api_key_hash IS NOT NULL
        GROUP BY e.endpoint_id
        ORDER BY e.created_at DESC
    """)
    rows = c.fetchall()
    conn.close()
    return [
        {
            "endpoint_id": row[0],
            "name": row[1],
            "api_key_hash": row[2],
            "models": json.loads(row[3]) if row[3] else [],
            "settings": json.loads(row[4]) if row[4] else {},
            "enabled": bool(row[5]),
            "alias": row[6],
            "is_default": bool(row[7]),
            "accept_protocols": json.loads(row[8]) if row[8] else ["anthropic", "openai"],
            "family_routing": json.loads(row[9]) if row[9] else None,
            "last_used": row[10]
        }
        for row in rows
    ]


# ─── Logs queries ─────────────────────────────────────────────────────

def ensure_log_columns():
    """确保 usage_records 包含 logs 所需的列（P1 已加，这里幂等保护）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for col, col_type in [
        ("request_id",   "TEXT"),
        ("latency_ms",   "INTEGER"),
        ("error_type",   "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE usage_records ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def get_logs_list(
    since: str | None = None,
    until: str | None = None,
    endpoint_id: str | None = None,
    model_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """获取日志列表 + 总数。since/until 是北京时间 'YYYY-MM-DD HH:MM:SS'。"""
    ensure_log_columns()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    where: list[str] = []
    params: list = []
    if since:
        where.append("datetime(timestamp) >= datetime(?)"); params.append(since)
    if until:
        where.append("datetime(timestamp) < datetime(?)"); params.append(until)
    if endpoint_id:
        where.append("endpoint_id = ?"); params.append(endpoint_id)
    if model_id:
        where.append("model_id = ?"); params.append(model_id)
    if status:
        where.append("request_status = ?"); params.append(status)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    c.execute(f"SELECT COUNT(*) FROM usage_records {where_sql}", params)
    total = c.fetchone()[0]

    c.execute(f"""
        SELECT id, timestamp, endpoint_id, model_id, input_tokens, output_tokens,
               request_status, request_id, latency_ms, error_type, client_ip, user_agent
        FROM usage_records {where_sql}
        ORDER BY datetime(timestamp) DESC
        LIMIT ? OFFSET ?
    """, params + [limit, offset])
    rows = c.fetchall()
    conn.close()

    records = [
        {
            "id": r[0], "timestamp": r[1], "endpoint_id": r[2], "model_id": r[3],
            "input_tokens": r[4], "output_tokens": r[5], "request_status": r[6],
            "request_id": r[7], "latency_ms": r[8], "error_type": r[9],
            "client_ip": r[10], "user_agent": r[11],
        }
        for r in rows
    ]
    return records, total


def get_logs_summary(
    since: str | None = None,
    until: str | None = None,
    endpoint_id: str | None = None,
    model_id: str | None = None,
) -> dict:
    ensure_log_columns()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    where: list[str] = []
    params: list = []
    if since:
        where.append("datetime(timestamp) >= datetime(?)"); params.append(since)
    if until:
        where.append("datetime(timestamp) < datetime(?)"); params.append(until)
    if endpoint_id:
        where.append("endpoint_id = ?"); params.append(endpoint_id)
    if model_id:
        where.append("model_id = ?"); params.append(model_id)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    c.execute(f"""
        SELECT COUNT(*),
               SUM(CASE WHEN request_status='error' THEN 1 ELSE 0 END),
               COALESCE(AVG(latency_ms), 0),
               COALESCE(SUM(input_tokens), 0),
               COALESCE(SUM(output_tokens), 0)
        FROM usage_records {where_sql}
    """, params)
    row = c.fetchone()
    total_requests, error_count, avg_latency_ms, total_in, total_out = row

    c.execute(f"""
        SELECT latency_ms FROM usage_records
        {where_sql} {'AND' if where_sql else 'WHERE'} latency_ms IS NOT NULL
        ORDER BY latency_ms
    """, params)
    latencies = [r[0] for r in c.fetchall()]
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else None

    conn.close()
    return {
        "total_requests": total_requests or 0,
        "error_count": error_count or 0,
        "avg_latency_ms": round(avg_latency_ms) if avg_latency_ms else None,
        "p95_latency_ms": p95,
        "total_input_tokens": total_in or 0,
        "total_output_tokens": total_out or 0,
    }


def get_filter_options() -> dict:
    ensure_log_columns()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT endpoint_id, model_id, request_status, error_type FROM usage_records")
    rows = c.fetchall()
    conn.close()

    endpoint_ids = sorted({r[0] for r in rows if r[0]})
    models = sorted({r[1] for r in rows if r[1]})
    statuses = sorted({r[2] for r in rows if r[2]})
    error_types = sorted({r[3] for r in rows if r[3]})

    return {
        "endpoints": [{"id": eid, "name": eid} for eid in endpoint_ids],
        "models": models,
        "statuses": statuses,
        "error_types": error_types,
    }
