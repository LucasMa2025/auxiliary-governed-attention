"""
AGA SQLite 持久化适配器

用于开发/测试环境，支持完整的持久化功能。

版本: v3.1

v3.1 更新:
- 增强事务支持和错误处理
- 添加 WAL 模式支持提升并发性能
- 改进审计日志的重试机制
- 优化批量操作的原子性
"""
import asyncio
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import time

from .base import PersistenceAdapter, KnowledgeRecord, PersistenceError, SerializationError
from ..types import LifecycleState

logger = logging.getLogger(__name__)

# 重试配置
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_BASE = 0.1  # 秒


class SQLiteAdapter(PersistenceAdapter):
    """
    SQLite 持久化适配器
    
    特性：
    - 完整的 CRUD 操作
    - 审计日志
    - 索引优化
    - 事务支持
    - WAL 模式提升并发性能
    """
    
    def __init__(
        self,
        db_path: str = "aga_data.db",
        enable_audit: bool = True,
        enable_wal: bool = True,
        busy_timeout_ms: int = 5000,
    ):
        """
        初始化 SQLite 适配器
        
        Args:
            db_path: 数据库文件路径
            enable_audit: 是否启用审计日志
            enable_wal: 是否启用 WAL 模式（提升并发性能）
            busy_timeout_ms: 忙等待超时时间（毫秒）
        """
        self.db_path = db_path
        self.enable_audit = enable_audit
        self.enable_wal = enable_wal
        self.busy_timeout_ms = busy_timeout_ms
        self._lock = threading.RLock()
        self._connected = False
        self._schema_version = 1  # 用于未来 schema 迁移
    
    @contextmanager
    def _get_conn(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        
        # 启用 WAL 模式
        if self.enable_wal:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        
        try:
            yield conn
        finally:
            conn.close()
    
    def _retry_on_busy(self, func, *args, **kwargs):
        """在数据库忙时重试操作"""
        last_error = None
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) or "busy" in str(e).lower():
                    last_error = e
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Database busy, retrying in {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})")
                    time.sleep(delay)
                else:
                    raise
        raise last_error
    
    def _init_tables(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 知识表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    slot_idx INTEGER NOT NULL,
                    lu_id TEXT NOT NULL,
                    condition TEXT,
                    decision TEXT,
                    key_vector TEXT NOT NULL,
                    value_vector TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
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
                        timestamp TEXT NOT NULL
                    )
                ''')
            
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_namespace 
                ON knowledge(namespace)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_lu_id 
                ON knowledge(lu_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_state 
                ON knowledge(namespace, lifecycle_state)
            ''')
            
            if self.enable_audit:
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_audit_namespace 
                    ON audit_log(namespace)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                    ON audit_log(timestamp)
                ''')
            
            conn.commit()
        
        logger.info(f"SQLite database initialized: {self.db_path}")
    
    def _log_audit(
        self,
        namespace: str,
        lu_id: Optional[str],
        action: str,
        old_state: Optional[str],
        new_state: Optional[str],
        details: Optional[str],
    ):
        """记录审计日志（带重试机制）"""
        if not self.enable_audit:
            return
        
        def _do_log():
            with self._get_conn() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute('''
                    INSERT INTO audit_log 
                    (namespace, lu_id, action, old_state, new_state, details, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (namespace, lu_id, action, old_state, new_state, details, now))
                conn.commit()
        
        try:
            self._retry_on_busy(_do_log)
        except Exception as e:
            # 审计日志失败不应影响主流程，但需要记录
            logger.warning(f"Failed to log audit after retries: {e}")
    
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
            return {
                "status": "unhealthy",
                "adapter": "sqlite",
                "error": str(e),
            }
    
    # ==================== 槽位操作 ====================
    
    async def save_slot(self, namespace: str, record: KnowledgeRecord) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    
                    key_json = json.dumps(record.key_vector)
                    value_json = json.dumps(record.value_vector)
                    metadata_json = json.dumps(record.metadata) if record.metadata else None
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO knowledge 
                        (namespace, slot_idx, lu_id, condition, decision, 
                         key_vector, value_vector, lifecycle_state, hit_count,
                         consecutive_misses, version, metadata, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, COALESCE((SELECT created_at FROM knowledge 
                                WHERE namespace = ? AND lu_id = ?), ?), ?)
                    ''', (
                        namespace, record.slot_idx, record.lu_id, 
                        record.condition, record.decision,
                        key_json, value_json, record.lifecycle_state,
                        record.hit_count, record.consecutive_misses, record.version,
                        metadata_json, namespace, record.lu_id, now, now
                    ))
                    
                    conn.commit()
                
                self._log_audit(
                    namespace, record.lu_id, "save_slot", 
                    None, record.lifecycle_state, f"slot={record.slot_idx}"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to save slot: {e}")
                return False
    
    async def load_slot(self, namespace: str, lu_id: str) -> Optional[KnowledgeRecord]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM knowledge WHERE namespace = ? AND lu_id = ?
                    ''', (namespace, lu_id))
                    row = cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_record(row)
            except Exception as e:
                logger.error(f"Failed to load slot: {e}")
                return None
    
    async def delete_slot(self, namespace: str, lu_id: str) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        DELETE FROM knowledge WHERE namespace = ? AND lu_id = ?
                    ''', (namespace, lu_id))
                    conn.commit()
                
                self._log_audit(namespace, lu_id, "delete_slot", None, None, None)
                return True
            except Exception as e:
                logger.error(f"Failed to delete slot: {e}")
                return False
    
    async def slot_exists(self, namespace: str, lu_id: str) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT 1 FROM knowledge WHERE namespace = ? AND lu_id = ?
                    ''', (namespace, lu_id))
                    return cursor.fetchone() is not None
            except Exception as e:
                logger.error(f"Failed to check slot existence: {e}")
                return False
    
    # ==================== 批量操作 ====================
    
    async def save_batch(self, namespace: str, records: List[KnowledgeRecord]) -> int:
        """
        批量保存（事务原子性）
        
        使用单一事务保存所有记录，失败时回滚。
        """
        if not records:
            return 0
        
        with self._lock:
            success_count = 0
            failed_records = []
            
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    
                    # 开始事务
                    cursor.execute("BEGIN TRANSACTION")
                    
                    try:
                        for record in records:
                            try:
                                key_json = json.dumps(record.key_vector)
                                value_json = json.dumps(record.value_vector)
                            except (TypeError, ValueError) as e:
                                failed_records.append((record.lu_id, f"Serialization error: {e}"))
                                continue
                            
                            metadata_json = json.dumps(record.metadata) if record.metadata else None
                            
                            cursor.execute('''
                                INSERT OR REPLACE INTO knowledge 
                                (namespace, slot_idx, lu_id, condition, decision, 
                                 key_vector, value_vector, lifecycle_state, hit_count,
                                 consecutive_misses, version, metadata, created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                        ?, COALESCE((SELECT created_at FROM knowledge 
                                        WHERE namespace = ? AND lu_id = ?), ?), ?)
                            ''', (
                                namespace, record.slot_idx, record.lu_id,
                                record.condition, record.decision,
                                key_json, value_json, record.lifecycle_state,
                                record.hit_count, record.consecutive_misses, record.version,
                                metadata_json, namespace, record.lu_id, now, now
                            ))
                            success_count += 1
                        
                        # 提交事务
                        conn.commit()
                        
                    except Exception as e:
                        # 回滚事务
                        conn.rollback()
                        logger.error(f"Batch save transaction failed, rolling back: {e}")
                        raise
                
                # 记录失败的记录
                if failed_records:
                    for lu_id, reason in failed_records:
                        logger.warning(f"Failed to save record {lu_id}: {reason}")
                
                self._log_audit(
                    namespace, None, "save_batch", 
                    None, None, f"count={success_count}, failed={len(failed_records)}"
                )
                return success_count
                
            except Exception as e:
                logger.error(f"Failed to save batch: {e}")
                return 0
    
    async def load_active_slots(self, namespace: str) -> List[KnowledgeRecord]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM knowledge 
                        WHERE namespace = ? AND lifecycle_state != ?
                        ORDER BY slot_idx
                    ''', (namespace, LifecycleState.QUARANTINED.value))
                    rows = cursor.fetchall()
                
                return [self._row_to_record(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to load active slots: {e}")
                return []
    
    async def load_all_slots(self, namespace: str) -> List[KnowledgeRecord]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM knowledge WHERE namespace = ?
                        ORDER BY slot_idx
                    ''', (namespace,))
                    rows = cursor.fetchall()
                
                return [self._row_to_record(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to load all slots: {e}")
                return []
    
    def _row_to_record(self, row) -> KnowledgeRecord:
        """转换数据库行到记录"""
        metadata = json.loads(row['metadata']) if row['metadata'] else None
        return KnowledgeRecord(
            slot_idx=row['slot_idx'],
            lu_id=row['lu_id'],
            condition=row['condition'] or '',
            decision=row['decision'] or '',
            key_vector=json.loads(row['key_vector']),
            value_vector=json.loads(row['value_vector']),
            lifecycle_state=row['lifecycle_state'],
            namespace=row['namespace'],
            hit_count=row['hit_count'],
            consecutive_misses=row['consecutive_misses'],
            version=row['version'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            metadata=metadata,
        )
    
    # ==================== 生命周期管理 ====================
    
    async def update_lifecycle(
        self, 
        namespace: str, 
        lu_id: str, 
        new_state: LifecycleState
    ) -> bool:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    
                    # 获取旧状态
                    cursor.execute('''
                        SELECT lifecycle_state FROM knowledge 
                        WHERE namespace = ? AND lu_id = ?
                    ''', (namespace, lu_id))
                    row = cursor.fetchone()
                    old_state = row['lifecycle_state'] if row else None
                    
                    # 更新
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        UPDATE knowledge 
                        SET lifecycle_state = ?, updated_at = ?
                        WHERE namespace = ? AND lu_id = ?
                    ''', (new_state.value, now, namespace, lu_id))
                    
                    conn.commit()
                
                self._log_audit(
                    namespace, lu_id, "update_lifecycle", 
                    old_state, new_state.value, None
                )
                return True
            except Exception as e:
                logger.error(f"Failed to update lifecycle: {e}")
                return False
    
    async def update_lifecycle_batch(
        self,
        namespace: str,
        updates: List[tuple]
    ) -> int:
        if not updates:
            return 0
        
        with self._lock:
            success_count = 0
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    
                    for lu_id, new_state in updates:
                        cursor.execute('''
                            UPDATE knowledge 
                            SET lifecycle_state = ?, updated_at = ?
                            WHERE namespace = ? AND lu_id = ?
                        ''', (new_state.value, now, namespace, lu_id))
                        success_count += cursor.rowcount
                    
                    conn.commit()
                
                self._log_audit(
                    namespace, None, "update_lifecycle_batch",
                    None, None, f"count={success_count}"
                )
                return success_count
            except Exception as e:
                logger.error(f"Failed to update lifecycle batch: {e}")
                return success_count
    
    # ==================== 统计查询 ====================
    
    async def get_slot_count(
        self, 
        namespace: str, 
        state: Optional[LifecycleState] = None
    ) -> int:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    
                    if state:
                        cursor.execute('''
                            SELECT COUNT(*) as count FROM knowledge 
                            WHERE namespace = ? AND lifecycle_state = ?
                        ''', (namespace, state.value))
                    else:
                        cursor.execute('''
                            SELECT COUNT(*) as count FROM knowledge 
                            WHERE namespace = ?
                        ''', (namespace,))
                    
                    return cursor.fetchone()['count']
            except Exception as e:
                logger.error(f"Failed to get slot count: {e}")
                return 0
    
    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    
                    # 总数
                    cursor.execute('''
                        SELECT COUNT(*) as count FROM knowledge WHERE namespace = ?
                    ''', (namespace,))
                    total = cursor.fetchone()['count']
                    
                    # 状态分布
                    cursor.execute('''
                        SELECT lifecycle_state, COUNT(*) as count 
                        FROM knowledge WHERE namespace = ?
                        GROUP BY lifecycle_state
                    ''', (namespace,))
                    state_dist = {row['lifecycle_state']: row['count'] for row in cursor.fetchall()}
                    
                    # 命中统计
                    cursor.execute('''
                        SELECT SUM(hit_count) as total_hits,
                               AVG(hit_count) as avg_hits,
                               MAX(hit_count) as max_hits
                        FROM knowledge WHERE namespace = ?
                    ''', (namespace,))
                    hit_stats = cursor.fetchone()
                
                return {
                    "namespace": namespace,
                    "total_slots": total,
                    "state_distribution": state_dist,
                    "total_hits": hit_stats['total_hits'] or 0,
                    "avg_hits": hit_stats['avg_hits'] or 0,
                    "max_hits": hit_stats['max_hits'] or 0,
                    "db_path": self.db_path,
                }
            except Exception as e:
                logger.error(f"Failed to get statistics: {e}")
                return {"namespace": namespace, "error": str(e)}
    
    # ==================== 命中计数 ====================
    
    async def increment_hit_count(
        self, 
        namespace: str, 
        lu_ids: List[str]
    ) -> bool:
        if not lu_ids:
            return True
        
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    
                    placeholders = ','.join(['?'] * len(lu_ids))
                    cursor.execute(f'''
                        UPDATE knowledge 
                        SET hit_count = hit_count + 1
                        WHERE namespace = ? AND lu_id IN ({placeholders})
                    ''', [namespace] + lu_ids)
                    
                    conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to increment hit count: {e}")
                return False
    
    # ==================== 审计日志 ====================
    
    async def get_audit_log(
        self, 
        namespace: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取审计日志"""
        if not self.enable_audit:
            return []
        
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM audit_log 
                        WHERE namespace = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (namespace, limit))
                    rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to get audit log: {e}")
                return []
    
    async def save_audit_log(self, entry: Dict[str, Any]) -> bool:
        """保存审计日志（Portal 扩展接口）"""
        if not self.enable_audit:
            return True
        
        self._log_audit(
            namespace=entry.get("namespace", "default"),
            lu_id=entry.get("lu_id"),
            action=entry.get("action", "unknown"),
            old_state=entry.get("old_state"),
            new_state=entry.get("new_state"),
            details=json.dumps(entry.get("details")) if entry.get("details") else entry.get("reason"),
        )
        return True
    
    async def query_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """查询审计日志（Portal 扩展接口）"""
        if not self.enable_audit:
            return []
        
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    
                    # 构建查询
                    query = "SELECT * FROM audit_log WHERE 1=1"
                    params = []
                    
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
    
    # ==================== 命名空间管理 ====================
    
    async def get_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT DISTINCT namespace FROM knowledge
                    ''')
                    rows = cursor.fetchall()
                
                return [row['namespace'] for row in rows]
            except Exception as e:
                logger.error(f"Failed to get namespaces: {e}")
                return []
