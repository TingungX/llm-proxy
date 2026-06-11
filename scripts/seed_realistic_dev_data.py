"""生成 dev DB 真实场景的 usage 数据（不调用上游，只模拟真实 proxy 写入）。

写入到与 dev server 共用的 usage.db，让前端能看到真实分布的图表/热力图/日志。
"""
import random
import sqlite3
from datetime import datetime, timedelta, timezone

from llm_proxy.infra import db

random.seed(123)  # 可复现

DB_PATH = db.DB_PATH
print(f"DB path: {DB_PATH}")

# 示例场景配置（替换为你的实际模型和端点）
MODELS = [
    ('model-a', 'openai'),
    ('model-b', 'anthropic'),
    ('model-c', 'openai'),
    ('model-d', 'openai'),
    ('model-e', 'anthropic'),
]
ENDPOINTS = ['ep-default', 'ep-custom-1']  # 替换为你的实际端点 ID

# 不同时间段的请求量分布
HOUR_WEIGHTS = [1, 1, 1, 1, 1, 2, 4, 8, 12, 10, 8, 7, 8, 9, 10, 8, 6, 5, 4, 3, 2, 2, 1, 1]

# 错误率（5xx/4xx 偶尔）
ERROR_RATE = 0.04
TIMEOUT_RATE = 0.01

now = datetime.now(timezone.utc)
records = []

for day_offset in range(60):  # 60 天
    date = now - timedelta(days=day_offset)
    weekday = date.weekday()
    # 周末减半
    base = 60 if weekday < 5 else 25
    # 5% 突发高峰
    if random.random() < 0.05:
        base *= 4
    # 8% 完全空日
    if random.random() < 0.08:
        continue

    # 当天请求数
    n = random.randint(max(8, base // 2), base * 2)
    for _ in range(n):
        hour = random.choices(range(24), weights=HOUR_WEIGHTS)[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = date.replace(hour=hour, minute=minute, second=second, microsecond=0)

        endpoint = random.choice(ENDPOINTS)
        model, protocol = random.choice(MODELS)

        # token 数：长尾分布（小请求多，大请求少）
        base_tokens = random.choices(
            [200, 800, 2500, 8000, 25000, 80000],
            weights=[40, 30, 15, 8, 5, 2],
        )[0]
        input_tokens = int(base_tokens * random.uniform(0.6, 1.4))
        output_tokens = int(input_tokens * random.uniform(0.2, 0.9))
        # 偶尔 input=0（错配请求）
        if random.random() < 0.01:
            input_tokens = 0
        # 偶尔 output=0（超时截断）
        if random.random() < 0.02:
            output_tokens = 0

        # 状态 + 错误类型
        r = random.random()
        if r < ERROR_RATE:
            status = 'error'
            error_type = random.choice(['5xx', '4xx', '5xx', '5xx', '4xx', 'parse_error'])
            latency = random.randint(100, 5000)
        elif r < ERROR_RATE + TIMEOUT_RATE:
            status = 'error'
            error_type = 'timeout'
            latency = random.randint(30000, 60000)
        else:
            status = 'success'
            error_type = None
            # 成功请求 latency：与 token 数正相关
            latency = int(200 + (input_tokens + output_tokens) * 0.05 + random.uniform(-100, 500))

        records.append({
            'timestamp': ts,
            'endpoint_id': endpoint,
            'model_id': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'status': status,
            'latency_ms': latency,
            'error_type': error_type,
        })

# 写库
print(f"准备写入 {len(records)} 条 records")
conn = sqlite3.connect(str(DB_PATH))
c = conn.cursor()
c.execute('DELETE FROM usage_records')
c.execute('DELETE FROM usage_hourly')
c.execute('DELETE FROM usage_daily')
conn.commit()
conn.close()

# 先 INSERT 用 stable request_id + 正确 timestamp
conn = sqlite3.connect(str(DB_PATH))
c = conn.cursor()
c.execute('DELETE FROM usage_records')
c.execute('DELETE FROM usage_hourly')
c.execute('DELETE FROM usage_daily')

for i, r in enumerate(records):
    req_id = f"req-{i:06d}"
    c.execute("""
        INSERT INTO usage_records (timestamp, endpoint_id, model_id, input_tokens, output_tokens, request_status,
                                   request_id, latency_ms, error_type, client_ip, user_agent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (r['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
          r['endpoint_id'], r['model_id'], r['input_tokens'], r['output_tokens'],
          r['status'], req_id, r['latency_ms'], r['error_type'], '', ''))
conn.commit()
conn.close()

# 重新聚合到 hourly / daily
db.aggregate_hourly(catch_up=True)
db.aggregate_daily()
print("聚合完成")

# 验证
conn = sqlite3.connect(str(DB_PATH))
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM usage_records')
print(f"usage_records: {c.fetchone()[0]}")
c.execute('SELECT COUNT(DISTINCT date) FROM usage_daily')
print(f"有数据的日期: {c.fetchone()[0]}")
c.execute('SELECT COUNT(DISTINCT model_id) FROM usage_records')
print(f"模型数: {c.fetchone()[0]}")
c.execute('SELECT COUNT(DISTINCT endpoint_id) FROM usage_records')
print(f"端点数: {c.fetchone()[0]}")
print(f"时间范围: ", end='')
c.execute('SELECT MIN(timestamp), MAX(timestamp) FROM usage_records')
print(c.fetchone())
