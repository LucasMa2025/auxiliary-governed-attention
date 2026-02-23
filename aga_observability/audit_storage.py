"""
aga_observability/audit_storage.py — 审计日志持久化

将 EventBus 的 audit 事件持久化到外部存储。

支持的后端:
  - memory: 仅内存（aga-core 内置，此处不需要）
  - file: JSONL 文件（支持 rotation）
  - sqlite: SQLite 数据库

设计要点:
  - 批量写入（减少 I/O）
  - 异步刷新（不阻塞推理）
  - 自动清理过期数据
"""
import json
import time
import logging
import threading
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditStorageBackend(ABC):
    """审计存储后端抽象基类"""

    @abstractmethod
    def store(self, entry: Dict[str, Any]) -> None:
        """存储单条审计记录"""
        ...

    @abstractmethod
    def store_batch(self, entries: List[Dict[str, Any]]) -> None:
        """批量存储审计记录"""
        ...

    @abstractmethod
    def query(
        self,
        operation: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """查询审计记录"""
        ...

    @abstractmethod
    def cleanup(self, retention_days: int) -> int:
        """清理过期数据，返回删除数量"""
        ...

    def shutdown(self) -> None:
        """关闭"""
        pass


class FileAuditStorage(AuditStorageBackend):
    """
    JSONL 文件审计存储

    每行一个 JSON 对象，支持文件 rotation。

    使用方式:
        storage = FileAuditStorage("audit.jsonl")
        storage.subscribe(event_bus)
        # ... 推理 ...
        storage.shutdown()
    """

    SUBSCRIBER_ID = "aga-observability-audit-file"

    def __init__(
        self,
        path: str = "aga_audit.jsonl",
        flush_interval: int = 10,
        batch_size: int = 100,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        max_files: int = 10,
    ):
        self._path = Path(path)
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._max_file_size = max_file_size
        self._max_files = max_files

        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._flush_timer: Optional[threading.Timer] = None
        self._running = False
        self._total_written = 0

        # 确保目录存在
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, event_bus) -> None:
        """订阅 audit 事件"""
        event_bus.subscribe(
            "audit",
            self._on_audit,
            subscriber_id=self.SUBSCRIBER_ID,
        )
        self._running = True
        self._schedule_flush()
        logger.info(f"FileAuditStorage 已启动: {self._path}")

    def unsubscribe(self, event_bus) -> None:
        """取消订阅"""
        event_bus.unsubscribe(
            "audit",
            subscriber_id=self.SUBSCRIBER_ID,
        )
        self._running = False

    def _on_audit(self, event) -> None:
        """处理 audit 事件"""
        entry = {
            "timestamp": event.data.get("timestamp", event.timestamp),
            "operation": event.data.get("operation", "unknown"),
            "details": event.data.get("details", {}),
            "success": event.data.get("success", True),
            "error": event.data.get("error"),
        }

        with self._lock:
            self._buffer.append(entry)

        # 缓冲区满时立即刷新
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def store(self, entry: Dict[str, Any]) -> None:
        """存储单条记录"""
        with self._lock:
            self._buffer.append(entry)

    def store_batch(self, entries: List[Dict[str, Any]]) -> None:
        """批量存储"""
        with self._lock:
            self._buffer.extend(entries)
        self._flush()

    def query(
        self,
        operation: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        查询审计记录（从文件读取）

        注意: 大文件查询可能较慢，建议使用 SQLite 后端。
        """
        results = []
        if not self._path.exists():
            return results

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if operation and entry.get("operation") != operation:
                            continue
                        if since and entry.get("timestamp", 0) < since:
                            continue
                        results.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"审计文件读取失败: {e}")

        return results[-limit:]

    def cleanup(self, retention_days: int) -> int:
        """清理过期数据（重写文件）"""
        if not self._path.exists():
            return 0

        cutoff = time.time() - retention_days * 86400
        kept = []
        removed = 0

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("timestamp", 0) >= cutoff:
                            kept.append(line)
                        else:
                            removed += 1
                    except json.JSONDecodeError:
                        kept.append(line)

            with open(self._path, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")

        except Exception as e:
            logger.warning(f"审计文件清理失败: {e}")

        return removed

    def _flush(self) -> None:
        """刷新缓冲区到文件"""
        with self._lock:
            if not self._buffer:
                return
            entries = list(self._buffer)
            self._buffer.clear()

        try:
            # 检查文件大小，必要时 rotate
            if self._path.exists() and self._path.stat().st_size > self._max_file_size:
                self._rotate()

            with open(self._path, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

            self._total_written += len(entries)

        except Exception as e:
            logger.warning(f"审计文件写入失败: {e}")

    def _rotate(self) -> None:
        """文件 rotation"""
        for i in range(self._max_files - 1, 0, -1):
            src = self._path.with_suffix(f".{i}.jsonl")
            dst = self._path.with_suffix(f".{i + 1}.jsonl")
            if src.exists():
                if i + 1 >= self._max_files:
                    src.unlink()
                else:
                    src.rename(dst)

        if self._path.exists():
            self._path.rename(self._path.with_suffix(".1.jsonl"))

    def _schedule_flush(self) -> None:
        """定时刷新"""
        if not self._running:
            return

        self._flush()

        self._flush_timer = threading.Timer(
            self._flush_interval,
            self._schedule_flush,
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def shutdown(self) -> None:
        """关闭并刷新"""
        self._running = False
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush()
        logger.info(f"FileAuditStorage 已关闭 (共写入 {self._total_written} 条)")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "path": str(self._path),
            "total_written": self._total_written,
            "buffer_size": len(self._buffer),
            "file_exists": self._path.exists(),
            "file_size": self._path.stat().st_size if self._path.exists() else 0,
        }


class SQLiteAuditStorage(AuditStorageBackend):
    """
    SQLite 审计存储

    使用方式:
        storage = SQLiteAuditStorage("audit.db")
        storage.subscribe(event_bus)
        # ... 推理 ...
        storage.shutdown()
    """

    SUBSCRIBER_ID = "aga-observability-audit-sqlite"

    def __init__(
        self,
        path: str = "aga_audit.db",
        flush_interval: int = 10,
        batch_size: int = 100,
    ):
        self._path = path
        self._flush_interval = flush_interval
        self._batch_size = batch_size

        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._flush_timer: Optional[threading.Timer] = None
        self._running = False
        self._total_written = 0

        # 初始化数据库
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表"""
        conn = sqlite3.connect(self._path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    operation TEXT NOT NULL,
                    details TEXT,
                    success INTEGER DEFAULT 1,
                    error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_log(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_operation
                ON audit_log(operation)
            """)
            conn.commit()
        finally:
            conn.close()

    def subscribe(self, event_bus) -> None:
        """订阅 audit 事件"""
        event_bus.subscribe(
            "audit",
            self._on_audit,
            subscriber_id=self.SUBSCRIBER_ID,
        )
        self._running = True
        self._schedule_flush()
        logger.info(f"SQLiteAuditStorage 已启动: {self._path}")

    def unsubscribe(self, event_bus) -> None:
        """取消订阅"""
        event_bus.unsubscribe(
            "audit",
            subscriber_id=self.SUBSCRIBER_ID,
        )
        self._running = False

    def _on_audit(self, event) -> None:
        """处理 audit 事件"""
        entry = {
            "timestamp": event.data.get("timestamp", event.timestamp),
            "operation": event.data.get("operation", "unknown"),
            "details": json.dumps(
                event.data.get("details", {}), ensure_ascii=False, default=str
            ),
            "success": 1 if event.data.get("success", True) else 0,
            "error": event.data.get("error"),
        }

        with self._lock:
            self._buffer.append(entry)

        if len(self._buffer) >= self._batch_size:
            self._flush()

    def store(self, entry: Dict[str, Any]) -> None:
        """存储单条记录"""
        with self._lock:
            self._buffer.append(entry)

    def store_batch(self, entries: List[Dict[str, Any]]) -> None:
        """批量存储"""
        with self._lock:
            self._buffer.extend(entries)
        self._flush()

    def query(
        self,
        operation: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """查询审计记录"""
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            sql = "SELECT * FROM audit_log WHERE 1=1"
            params = []

            if operation:
                sql += " AND operation = ?"
                params.append(operation)
            if since:
                sql += " AND timestamp >= ?"
                params.append(since)

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                entry = dict(row)
                # 解析 details JSON
                if entry.get("details"):
                    try:
                        entry["details"] = json.loads(entry["details"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                entry["success"] = bool(entry.get("success", 1))
                results.append(entry)

            return list(reversed(results))  # 按时间正序

        finally:
            conn.close()

    def cleanup(self, retention_days: int) -> int:
        """清理过期数据"""
        cutoff = time.time() - retention_days * 86400
        conn = sqlite3.connect(self._path)
        try:
            cursor = conn.execute(
                "DELETE FROM audit_log WHERE timestamp < ?",
                (cutoff,),
            )
            conn.commit()
            deleted = cursor.rowcount
            if deleted > 0:
                conn.execute("VACUUM")
                logger.info(f"清理了 {deleted} 条过期审计记录")
            return deleted
        finally:
            conn.close()

    def _flush(self) -> None:
        """刷新缓冲区到数据库"""
        with self._lock:
            if not self._buffer:
                return
            entries = list(self._buffer)
            self._buffer.clear()

        conn = sqlite3.connect(self._path)
        try:
            conn.executemany(
                """
                INSERT INTO audit_log (timestamp, operation, details, success, error)
                VALUES (:timestamp, :operation, :details, :success, :error)
                """,
                entries,
            )
            conn.commit()
            self._total_written += len(entries)
        except Exception as e:
            logger.warning(f"审计数据库写入失败: {e}")
        finally:
            conn.close()

    def _schedule_flush(self) -> None:
        """定时刷新"""
        if not self._running:
            return

        self._flush()

        self._flush_timer = threading.Timer(
            self._flush_interval,
            self._schedule_flush,
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def shutdown(self) -> None:
        """关闭并刷新"""
        self._running = False
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush()
        logger.info(f"SQLiteAuditStorage 已关闭 (共写入 {self._total_written} 条)")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        count = 0
        try:
            conn = sqlite3.connect(self._path)
            cursor = conn.execute("SELECT COUNT(*) FROM audit_log")
            count = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass

        return {
            "path": self._path,
            "total_written": self._total_written,
            "buffer_size": len(self._buffer),
            "total_records": count,
        }
