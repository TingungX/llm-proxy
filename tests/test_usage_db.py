"""Tests for usage_daily table and heatmap query"""
import os
import tempfile
import pytest
from llm_proxy.infra import db


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    """用临时数据库替代生产数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setattr(db, "DB_PATH", db.Path(path))
    db.init_db()
    yield path
    os.unlink(path)


class TestUsageDailyTable:
    def test_usage_daily_table_created(self):
        """init_db 应创建 usage_daily 表"""
        import sqlite3
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_daily'")
        assert c.fetchone() is not None
        conn.close()

    def test_usage_daily_has_correct_columns(self):
        import sqlite3
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("PRAGMA table_info(usage_daily)")
        columns = {row[1] for row in c.fetchall()}
        assert "date" in columns
        assert "model_id" in columns
        assert "endpoint_id" in columns
        assert "total_input_tokens" in columns
        assert "total_output_tokens" in columns
        assert "request_count" in columns
        conn.close()


class TestAggregateDaily:
    def test_aggregate_daily_merges_hourly_to_daily(self):
        """aggregate_daily 应将过期小时数据聚合到日级表"""
        import sqlite3
        from datetime import datetime, timedelta

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()

        # 插入 91 天前的小时数据（应被聚合）
        old_date = (datetime.now() - timedelta(days=91)).strftime('%Y-%m-%d')
        old_hour = f"{old_date} 10:00:00"
        c.execute("""
            INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_hour, "ep1", "model-a", 100, 50, 5))
        old_hour2 = f"{old_date} 11:00:00"
        c.execute("""
            INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_hour2, "ep1", "model-a", 200, 100, 10))
        conn.commit()
        conn.close()

        db.aggregate_daily()

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT total_input_tokens, total_output_tokens, request_count FROM usage_daily WHERE date = ? AND model_id = ?", (old_date, "model-a"))
        row = c.fetchone()
        assert row is not None
        assert row[0] == 300
        assert row[1] == 150
        assert row[2] == 15
        conn.close()

    def test_aggregate_daily_idempotent(self):
        """重复调用 aggregate_daily 不应产生重复数据"""
        import sqlite3
        from datetime import datetime, timedelta

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()

        old_date = (datetime.now() - timedelta(days=91)).strftime('%Y-%m-%d')
        old_hour = f"{old_date} 10:00:00"
        c.execute("""
            INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_hour, "ep1", "model-b", 100, 50, 5))
        conn.commit()
        conn.close()

        db.aggregate_daily()
        db.aggregate_daily()

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM usage_daily WHERE date = ? AND model_id = ?", (old_date, "model-b"))
        count = c.fetchone()[0]
        assert count == 1
        conn.close()

    def test_aggregate_daily_not_touch_recent_hourly(self):
        """aggregate_daily 不应聚合近期小时数据"""
        import sqlite3
        from datetime import datetime, timedelta

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()

        recent_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        recent_hour = f"{recent_date} 10:00:00"
        c.execute("""
            INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (recent_hour, "ep1", "model-c", 100, 50, 5))
        conn.commit()
        conn.close()

        db.aggregate_daily()

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM usage_daily WHERE date = ? AND model_id = ?", (recent_date, "model-c"))
        assert c.fetchone()[0] == 0
        conn.close()


class TestGetUsageWithDaily:
    def test_granularity_day_uses_daily_table(self):
        """granularity=day 应从 usage_daily 表获取历史数据"""
        import sqlite3
        from datetime import datetime, timedelta

        old_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO usage_daily (date, model_id, endpoint_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_date, "model-x", "ep1", 500, 300, 20))
        conn.commit()
        conn.close()

        start = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        result = db.get_usage(start, end, group_by="model", granularity="day")
        matching = [r for r in result if r["time"] == old_date and r["group_key"] == "model-x"]
        assert len(matching) == 1
        assert matching[0]["total_tokens"] == 800

    def test_get_usage_heatmap_returns_flat_list(self):
        """get_usage_heatmap 应返回 [{date, total_tokens}] 扁平列表"""
        import sqlite3
        from datetime import datetime, timedelta

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        old_date = (datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d')
        c.execute("""
            INSERT INTO usage_daily (date, model_id, endpoint_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_date, "model-y", "ep1", 1000, 500, 30))
        c.execute("""
            INSERT INTO usage_daily (date, model_id, endpoint_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_date, "model-z", "ep1", 200, 100, 10))
        conn.commit()
        conn.close()

        start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        result = db.get_usage_heatmap(start, end)
        matching = [r for r in result if r["date"] == old_date]
        assert len(matching) == 1
        assert matching[0]["total_tokens"] == 1800

    def test_get_usage_heatmap_with_endpoint_filter(self):
        """get_usage_heatmap 支持端点筛选"""
        import sqlite3
        from datetime import datetime, timedelta

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        old_date = (datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d')
        c.execute("""
            INSERT INTO usage_daily (date, model_id, endpoint_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_date, "model-y", "ep1", 1000, 500, 30))
        c.execute("""
            INSERT INTO usage_daily (date, model_id, endpoint_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_date, "model-y", "ep2", 200, 100, 10))
        conn.commit()
        conn.close()

        start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        result = db.get_usage_heatmap(start, end, endpoint_id="ep1")
        matching = [r for r in result if r["date"] == old_date]
        assert len(matching) == 1
        assert matching[0]["total_tokens"] == 1500


class TestCleanupIntegration:
    def test_cleanup_calls_aggregate_first(self):
        """cleanup_old_records 应先聚合再删除，数据不丢"""
        import sqlite3
        from datetime import datetime, timedelta

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()

        # 插入 91 天前的小时数据
        old_date = (datetime.now() - timedelta(days=91)).strftime('%Y-%m-%d')
        old_hour = f"{old_date} 10:00:00"
        c.execute("""
            INSERT INTO usage_hourly (hour_start, endpoint_id, model_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_hour, "ep1", "model-int", 100, 50, 5))
        conn.commit()
        conn.close()

        # 先聚合再清理
        db.aggregate_daily()
        db.cleanup_old_records()

        # 小时数据应被删除
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM usage_hourly WHERE model_id = ?", ("model-int",))
        assert c.fetchone()[0] == 0

        # 日级数据应存在
        c.execute("SELECT total_input_tokens FROM usage_daily WHERE date = ? AND model_id = ?", (old_date, "model-int"))
        assert c.fetchone()[0] == 100
        conn.close()

    def test_heatmap_returns_empty_for_no_data(self):
        """无数据时热力图返回空列表"""
        from datetime import datetime, timedelta
        start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        result = db.get_usage_heatmap(start, end)
        assert isinstance(result, list)
