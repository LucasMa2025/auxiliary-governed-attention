"""
aga-knowledge SQLite 持久化适配器

明文 KV 版本：不存储向量数据，只存储 condition/decision 文本对。
支持完整的 CRUD、审计日志、WAL 模式。
"""

import json
import sqlite3
import threading
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any

from .base import PersistenceAdapter
from ..types import LifecycleState

logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_BASE = 0.1


class SQLiteAdapter(PersistenceAdapter):
    """
    SQLite 持久化适配器（明文 KV 版本）

    特性：
    - 完整的 CRUD 操作
    - 审计日志
    - 索引优化
    - 事务支持
    - WAL 模式提升并发性能
    - 不存储向量数据
    """

    def __init__(
        self,
        db_path: str = "aga_knowledge.db",
        enable_audit: bool = True,
        enable_wal: bool = True,
        busy_timeout_ms: int = 5000,
    ):
        self.db_path = db_path
        self.enable_audit = enable_audit
        self.enable_wal = enable_wal
        self.busy_timeout_ms = busy_timeout_ms
        self._lock = threading.RLock()
        self._connected = False

    @contextmanager
    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        if self.enable_wal:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def _retry_on_busy(self, func, *args, **kwargs):
        """在数据库忙时重试"""
        last_error = None
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) or "busy" in str(e).lower():
                    last_error = e
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Database busy, retrying in {delay:.2f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                else:
                    raise
        raise last_error

    def _init_tables(self):
        """初始化数据库表（明文 KV 版本）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 知识表（无向量字段）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    lu_id TEXT NOT NULL,
                    condition_text TEXT,
                    decision_text TEXT,
                    lifecycle_state TEXT NOT NULL DEFAULT 'probationary',
                    trust_tier TEXT NOT NULL DEFAULT 'standard',
                    hit_count INTEGER DEFAULT 0,
                    consecutive_misses INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(namespace, lu_id)
                )
            ''')

            # 审计日志表
            if self.enable_audit:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        namespace TEXT NOT NULL,
                        lu_id TEXT,
                        action TEXT NOT NULL,
                        old_state TEXT,
                        new_state TEXT,
                        details TEXT,
                        source TEXT DEFAULT 'portal',
                        reason TEXT,
                        timestamp TEXT NOT NULL
                    )
                ''')

            # 索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_namespace ON knowledge(namespace)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_lu_id ON knowledge(lu_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_state ON knowledge(namespace, lifecycle_state)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_tier ON knowledge(namespace, trust_tier)')

            if self.enable_audit:
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_namespace ON audit_log(namespace)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_lu_id ON audit_log(lu_id)')

            conn.commit()
        logger.info(f"SQLite database initialized: {self.db_path}")

    def _log_audit(self, namespace: str, lu_id: Optional[str], action: str,
                   old_state: Optional[str] = None, new_state: Optional[str] = None,
                   details: Optional[str] = None, reason: Optional[str] = None):
        """记录审计日志"""
        if not self.enable_audit:
            return
        def _do_log():
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_log
                    (namespace, lu_id, action, old_state, new_state, details, reason, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (namespace, lu_id, action, old_state, new_state, details, reason,
                      datetime.utcnow().isoformat()))
                conn.commit()
        try:
            self._retry_on_busy(_do_log)
        except Exception as e:
            logger.warning(f"Failed to log audit: {e}")

    # ==================== 连接管理 ====================

    async def connect(self) -> bool:
        with self._lock:
            try:
                self._init_tables()
                self._connected = True
                return True
            except Exception as e:
                logger.error(f"Failed to connect to SQLite: {e}")
                return False

    async def disconnect(self):
        self._connected = False

    async def is_connected(self) -> bool:
        return self._connected

    async def health_check(self) -> Dict[str, Any]:
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM knowledge")
                count = cursor.fetchone()['count']
            return {
                "status": "healthy",
                "adapter": "sqlite",
                "db_path": self.db_path,
                "total_records": count,
            }
        except Exception as e:
            return {"status": "unhealthy", "adapter": "sqlite", "error": str(e)}

    # ==================== 知识 CRUD ====================

    async def save_knowledge(self, namespace: str, lu_id: str, data: Dict[str, Any]) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    now = datetime.utcnow().isoformat()
                    metadata_json = json.dumps(data.get("metadata")) if data.get("metadata") else None

                    cursor.execute('''
                        INSERT OR REPLACE INTO knowledge
                        (namespace, lu_id, condition_text, decision_text,
                         lifecycle_state, trust_tier, hit_count,
                         consecutive_misses, version, metadata,
                         created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?,
                                ?, COALESCE((SELECT version FROM knowledge
                                WHERE namespace = ? AND lu_id = ?), 0) + 1,
                                ?,
                                COALESCE((SELECT created_at FROM knowledge
                                WHERE namespace = ? AND lu_id = ?), ?), ?)
                    ''', (
                        namespace, lu_id,
                        data.get("condition", ""),
                        data.get("decision", ""),
                        data.get("lifecycle_state", "probationary"),
                        data.get("trust_tier", "standard"),
                        data.get("hit_count", 0),
                        data.get("consecutive_misses", 0),
                        namespace, lu_id,
                        metadata_json,
                        namespace, lu_id, now, now,
                    ))
                    conn.commit()

                self._log_audit(namespace, lu_id, "SAVE",
                                new_state=data.get("lifecycle_state", "probationary"))
                return True
            except Exception as e:
                logger.error(f"Failed to save knowledge: {e}")
                return False

    async def load_knowledge(self, namespace: str, lu_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT * FROM knowledge WHERE namespace = ? AND lu_id = ?',
                        (namespace, lu_id),
                    )
                    row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_dict(row)
            except Exception as e:
                logger.error(f"Failed to load knowledge: {e}")
                return None

    async def delete_knowledge(self, namespace: str, lu_id: str) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'DELETE FROM knowledge WHERE namespace = ? AND lu_id = ?',
                        (namespace, lu_id),
                    )
                    conn.commit()
                self._log_audit(namespace, lu_id, "DELETE")
                return True
            except Exception as e:
                logger.error(f"Failed to delete knowledge: {e}")
                return False

    async def knowledge_exists(self, namespace: str, lu_id: str) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT 1 FROM knowledge WHERE namespace = ? AND lu_id = ?',
                        (namespace, lu_id),
                    )
                    return cursor.fetchone() is not None
            except Exception as e:
                logger.error(f"Failed to check knowledge existence: {e}")
                return False

    # ==================== 批量操作 ====================

    async def save_batch(self, namespace: str, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        with self._lock:
            count = 0
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    now = datetime.utcnow().isoformat()
                    cursor.execute("BEGIN TRANSACTION")
                    try:
                        for data in records:
                            lu_id = data.get("lu_id", "")
                            if not lu_id:
                                continue
                            metadata_json = json.dumps(data.get("metadata")) if data.get("metadata") else None
                            cursor.execute('''
                                INSERT OR REPLACE INTO knowledge
                                (namespace, lu_id, condition_text, decision_text,
                                 lifecycle_state, trust_tier, hit_count,
                                 consecutive_misses, version, metadata,
                                 created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?,
                                        ?, COALESCE((SELECT version FROM knowledge
                                        WHERE namespace = ? AND lu_id = ?), 0) + 1,
                                        ?,
                                        COALESCE((SELECT created_at FROM knowledge
                                        WHERE namespace = ? AND lu_id = ?), ?), ?)
                            ''', (
                                namespace, lu_id,
                                data.get("condition", ""),
                                data.get("decision", ""),
                                data.get("lifecycle_state", "probationary"),
                                data.get("trust_tier", "standard"),
                                data.get("hit_count", 0),
                                data.get("consecutive_misses", 0),
                                namespace, lu_id,
                                metadata_json,
                                namespace, lu_id, now, now,
                            ))
                            count += 1
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                self._log_audit(namespace, None, "BATCH_SAVE", details=f"count={count}")
                return count
            except Exception as e:
                logger.error(f"Failed to save batch: {e}")
                return 0

    async def load_active_knowledge(self, namespace: str) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM knowledge
                        WHERE namespace = ? AND lifecycle_state != ?
                        ORDER BY lu_id
                    ''', (namespace, LifecycleState.QUARANTINED.value))
                    rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to load active knowledge: {e}")
                return []

    async def load_all_knowledge(self, namespace: str) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT * FROM knowledge WHERE namespace = ? ORDER BY lu_id',
                        (namespace,),
                    )
                    rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to load all knowledge: {e}")
                return []

    # ==================== 查询 ====================

    async def query_knowledge(
        self,
        namespace: str,
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    query = "SELECT * FROM knowledge WHERE namespace = ?"
                    params: list = [namespace]

                    if lifecycle_states:
                        placeholders = ','.join(['?'] * len(lifecycle_states))
                        query += f" AND lifecycle_state IN ({placeholders})"
                        params.extend(lifecycle_states)

                    if trust_tiers:
                        placeholders = ','.join(['?'] * len(trust_tiers))
                        query += f" AND trust_tier IN ({placeholders})"
                        params.extend(trust_tiers)

                    query += " ORDER BY lu_id LIMIT ? OFFSET ?"
                    params.extend([limit, offset])

                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to query knowledge: {e}")
                return []

    # ==================== 生命周期管理 ====================

    async def update_lifecycle(self, namespace: str, lu_id: str, new_state: str) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT lifecycle_state FROM knowledge WHERE namespace = ? AND lu_id = ?',
                        (namespace, lu_id),
                    )
                    row = cursor.fetchone()
                    old_state = row['lifecycle_state'] if row else None

                    now = datetime.utcnow().isoformat()
                    cursor.execute('''
                        UPDATE knowledge SET lifecycle_state = ?, updated_at = ?
                        WHERE namespace = ? AND lu_id = ?
                    ''', (new_state, now, namespace, lu_id))
                    conn.commit()

                self._log_audit(namespace, lu_id, "UPDATE_LIFECYCLE", old_state, new_state)
                return True
            except Exception as e:
                logger.error(f"Failed to update lifecycle: {e}")
                return False

    async def update_trust_tier(self, namespace: str, lu_id: str, new_tier: str) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    now = datetime.utcnow().isoformat()
                    cursor.execute('''
                        UPDATE knowledge SET trust_tier = ?, updated_at = ?
                        WHERE namespace = ? AND lu_id = ?
                    ''', (new_tier, now, namespace, lu_id))
                    conn.commit()
                self._log_audit(namespace, lu_id, "UPDATE_TRUST_TIER", new_state=new_tier)
                return True
            except Exception as e:
                logger.error(f"Failed to update trust tier: {e}")
                return False

    # ==================== 统计 ====================

    async def get_knowledge_count(self, namespace: str, state: Optional[str] = None) -> int:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    if state:
                        cursor.execute(
                            'SELECT COUNT(*) as count FROM knowledge WHERE namespace = ? AND lifecycle_state = ?',
                            (namespace, state),
                        )
                    else:
                        cursor.execute(
                            'SELECT COUNT(*) as count FROM knowledge WHERE namespace = ?',
                            (namespace,),
                        )
                    return cursor.fetchone()['count']
            except Exception as e:
                logger.error(f"Failed to get knowledge count: {e}")
                return 0

    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT COUNT(*) as count FROM knowledge WHERE namespace = ?',
                        (namespace,),
                    )
                    total = cursor.fetchone()['count']

                    cursor.execute('''
                        SELECT lifecycle_state, COUNT(*) as count
                        FROM knowledge WHERE namespace = ?
                        GROUP BY lifecycle_state
                    ''', (namespace,))
                    state_dist = {row['lifecycle_state']: row['count'] for row in cursor.fetchall()}

                    cursor.execute('''
                        SELECT SUM(hit_count) as total_hits,
                               AVG(hit_count) as avg_hits,
                               MAX(hit_count) as max_hits
                        FROM knowledge WHERE namespace = ?
                    ''', (namespace,))
                    hit_stats = cursor.fetchone()

                return {
                    "namespace": namespace,
                    "total_knowledge": total,
                    "state_distribution": state_dist,
                    "total_hits": hit_stats['total_hits'] or 0,
                    "avg_hits": hit_stats['avg_hits'] or 0,
                    "max_hits": hit_stats['max_hits'] or 0,
                    "db_path": self.db_path,
                }
            except Exception as e:
                logger.error(f"Failed to get statistics: {e}")
                return {"namespace": namespace, "error": str(e)}

    async def increment_hit_count(self, namespace: str, lu_ids: List[str]) -> bool:
        if not lu_ids:
            return True
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    placeholders = ','.join(['?'] * len(lu_ids))
                    cursor.execute(f'''
                        UPDATE knowledge SET hit_count = hit_count + 1
                        WHERE namespace = ? AND lu_id IN ({placeholders})
                    ''', [namespace] + lu_ids)
                    conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to increment hit count: {e}")
                return False

    # ==================== 命名空间 ====================

    async def get_namespaces(self) -> List[str]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT DISTINCT namespace FROM knowledge')
                    rows = cursor.fetchall()
                return [row['namespace'] for row in rows]
            except Exception as e:
                logger.error(f"Failed to get namespaces: {e}")
                return []

    # ==================== 审计日志 ====================

    async def save_audit_log(self, entry: Dict[str, Any]) -> bool:
        if not self.enable_audit:
            return True
        self._log_audit(
            namespace=entry.get("namespace", "default"),
            lu_id=entry.get("lu_id"),
            action=entry.get("action", "unknown"),
            old_state=entry.get("old_state"),
            new_state=entry.get("new_state"),
            details=json.dumps(entry.get("details")) if entry.get("details") else None,
            reason=entry.get("reason"),
        )
        return True

    async def query_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if not self.enable_audit:
            return []
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    query = "SELECT * FROM audit_log WHERE 1=1"
                    params: list = []
                    if namespace:
                        query += " AND namespace = ?"
                        params.append(namespace)
                    if lu_id:
                        query += " AND lu_id = ?"
                        params.append(lu_id)
                    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to query audit log: {e}")
                return []

    # ==================== 工具方法 ====================

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """转换数据库行到字典"""
        metadata = json.loads(row['metadata']) if row['metadata'] else None
        return {
            "lu_id": row['lu_id'],
            "namespace": row['namespace'],
            "condition": row['condition_text'],
            "decision": row['decision_text'],
            "lifecycle_state": row['lifecycle_state'],
            "trust_tier": row['trust_tier'],
            "hit_count": row['hit_count'],
            "consecutive_misses": row['consecutive_misses'],
            "version": row['version'],
            "created_at": row['created_at'],
            "updated_at": row['updated_at'],
            "metadata": metadata,
        }
