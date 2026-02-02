"""
AGA SQLite 持久化适配器

用于开发/测试环境，支持完整的持久化功能。

版本: v3.0
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

from .base import PersistenceAdapter, KnowledgeRecord, PersistenceError
from ..types import LifecycleState

logger = logging.getLogger(__name__)


class SQLiteAdapter(PersistenceAdapter):
    """
    SQLite 持久化适配器
    
    特性：
    - 完整的 CRUD 操作
    - 审计日志
    - 索引优化
    - 事务支持
    """
    
    def __init__(
        self,
        db_path: str = "aga_data.db",
        enable_audit: bool = True,
    ):
        """
        初始化 SQLite 适配器
        
        Args:
            db_path: 数据库文件路径
            enable_audit: 是否启用审计日志
        """
        self.db_path = db_path
        self.enable_audit = enable_audit
        self._lock = threading.RLock()
        self._connected = False
    
    @contextmanager
    def _get_conn(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
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
        """记录审计日志"""
        if not self.enable_audit:
            return
        
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute('''
                    INSERT INTO audit_log 
                    (namespace, lu_id, action, old_state, new_state, details, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (namespace, lu_id, action, old_state, new_state, details, now))
                conn.commit()
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
        if not records:
            return 0
        
        with self._lock:
            success_count = 0
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    
                    for record in records:
                        try:
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
                            success_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to save record {record.lu_id}: {e}")
                    
                    conn.commit()
                
                self._log_audit(
                    namespace, None, "save_batch", 
                    None, None, f"count={success_count}"
                )
                return success_count
            except Exception as e:
                logger.error(f"Failed to save batch: {e}")
                return success_count
    
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
