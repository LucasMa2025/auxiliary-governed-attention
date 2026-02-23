"""
tests/test_observability/test_audit_storage.py — AuditStorage 测试
"""
import time
import json
import pytest
from aga.instrumentation import EventBus
from aga_observability.audit_storage import (
    FileAuditStorage,
    SQLiteAuditStorage,
)


class TestFileAuditStorage:
    """FileAuditStorage 测试"""

    def test_store_and_query(self, tmp_path):
        """存储和查询"""
        path = str(tmp_path / "audit.jsonl")
        storage = FileAuditStorage(path=path, flush_interval=999)

        storage.store({
            "timestamp": time.time(),
            "operation": "register",
            "details": {"id": "test_001"},
            "success": 1,
        })
        storage._flush()

        results = storage.query()
        assert len(results) == 1
        assert results[0]["operation"] == "register"
        storage.shutdown()

    def test_store_batch(self, tmp_path):
        """批量存储"""
        path = str(tmp_path / "audit.jsonl")
        storage = FileAuditStorage(path=path, flush_interval=999)

        entries = [
            {
                "timestamp": time.time(),
                "operation": f"op_{i}",
                "details": {},
                "success": 1,
            }
            for i in range(10)
        ]
        storage.store_batch(entries)

        results = storage.query()
        assert len(results) == 10
        storage.shutdown()

    def test_query_filter_operation(self, tmp_path):
        """按操作类型过滤"""
        path = str(tmp_path / "audit.jsonl")
        storage = FileAuditStorage(path=path, flush_interval=999)

        storage.store_batch([
            {"timestamp": time.time(), "operation": "register", "details": {}, "success": 1},
            {"timestamp": time.time(), "operation": "unregister", "details": {}, "success": 1},
            {"timestamp": time.time(), "operation": "register", "details": {}, "success": 1},
        ])

        results = storage.query(operation="register")
        assert len(results) == 2
        storage.shutdown()

    def test_subscribe_event_bus(self, tmp_path):
        """订阅 EventBus"""
        path = str(tmp_path / "audit.jsonl")
        storage = FileAuditStorage(path=path, flush_interval=999, batch_size=999)
        bus = EventBus()

        storage.subscribe(bus)

        bus.emit("audit", {
            "timestamp": time.time(),
            "operation": "register",
            "details": {"id": "test"},
            "success": True,
        })

        storage._flush()
        results = storage.query()
        assert len(results) == 1
        storage.shutdown()

    def test_cleanup(self, tmp_path):
        """清理过期数据"""
        path = str(tmp_path / "audit.jsonl")
        storage = FileAuditStorage(path=path, flush_interval=999)

        # 写入旧数据
        old_time = time.time() - 100 * 86400  # 100 天前
        storage.store_batch([
            {"timestamp": old_time, "operation": "old", "details": {}, "success": 1},
            {"timestamp": time.time(), "operation": "new", "details": {}, "success": 1},
        ])

        removed = storage.cleanup(retention_days=90)
        assert removed == 1

        results = storage.query()
        assert len(results) == 1
        assert results[0]["operation"] == "new"
        storage.shutdown()

    def test_get_stats(self, tmp_path):
        """统计信息"""
        path = str(tmp_path / "audit.jsonl")
        storage = FileAuditStorage(path=path, flush_interval=999)
        stats = storage.get_stats()
        assert "path" in stats
        assert "total_written" in stats
        storage.shutdown()


class TestSQLiteAuditStorage:
    """SQLiteAuditStorage 测试"""

    def test_store_and_query(self, tmp_path):
        """存储和查询"""
        path = str(tmp_path / "audit.db")
        storage = SQLiteAuditStorage(path=path, flush_interval=999)

        storage.store({
            "timestamp": time.time(),
            "operation": "register",
            "details": json.dumps({"id": "test_001"}),
            "success": 1,
            "error": None,
        })
        storage._flush()

        results = storage.query()
        assert len(results) == 1
        assert results[0]["operation"] == "register"
        storage.shutdown()

    def test_store_batch(self, tmp_path):
        """批量存储"""
        path = str(tmp_path / "audit.db")
        storage = SQLiteAuditStorage(path=path, flush_interval=999)

        entries = [
            {
                "timestamp": time.time(),
                "operation": f"op_{i}",
                "details": json.dumps({}),
                "success": 1,
                "error": None,
            }
            for i in range(10)
        ]
        storage.store_batch(entries)

        results = storage.query()
        assert len(results) == 10
        storage.shutdown()

    def test_query_filter(self, tmp_path):
        """过滤查询"""
        path = str(tmp_path / "audit.db")
        storage = SQLiteAuditStorage(path=path, flush_interval=999)

        storage.store_batch([
            {"timestamp": time.time(), "operation": "register", "details": "{}", "success": 1, "error": None},
            {"timestamp": time.time(), "operation": "unregister", "details": "{}", "success": 1, "error": None},
            {"timestamp": time.time(), "operation": "register", "details": "{}", "success": 1, "error": None},
        ])

        results = storage.query(operation="register")
        assert len(results) == 2
        storage.shutdown()

    def test_cleanup(self, tmp_path):
        """清理过期数据"""
        path = str(tmp_path / "audit.db")
        storage = SQLiteAuditStorage(path=path, flush_interval=999)

        old_time = time.time() - 100 * 86400
        storage.store_batch([
            {"timestamp": old_time, "operation": "old", "details": "{}", "success": 1, "error": None},
            {"timestamp": time.time(), "operation": "new", "details": "{}", "success": 1, "error": None},
        ])

        removed = storage.cleanup(retention_days=90)
        assert removed == 1

        results = storage.query()
        assert len(results) == 1
        assert results[0]["operation"] == "new"
        storage.shutdown()

    def test_subscribe_event_bus(self, tmp_path):
        """订阅 EventBus"""
        path = str(tmp_path / "audit.db")
        storage = SQLiteAuditStorage(path=path, flush_interval=999, batch_size=999)
        bus = EventBus()

        storage.subscribe(bus)

        bus.emit("audit", {
            "timestamp": time.time(),
            "operation": "register",
            "details": {"id": "test"},
            "success": True,
        })

        storage._flush()
        results = storage.query()
        assert len(results) == 1
        storage.shutdown()

    def test_get_stats(self, tmp_path):
        """统计信息"""
        path = str(tmp_path / "audit.db")
        storage = SQLiteAuditStorage(path=path, flush_interval=999)
        stats = storage.get_stats()
        assert "path" in stats
        assert "total_written" in stats
        assert "total_records" in stats
        storage.shutdown()
