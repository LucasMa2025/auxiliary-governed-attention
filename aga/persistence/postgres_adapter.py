"""
AGA PostgreSQL 持久化适配器

用于冷存储层 (L2)，支持完整的数据持久化和审计。

版本: v3.0
"""
import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from .base import PersistenceAdapter, KnowledgeRecord, ConnectionError
from ..types import LifecycleState

logger = logging.getLogger(__name__)

# 尝试导入 asyncpg
try:
    import asyncpg
    _HAS_ASYNCPG = True
except ImportError:
    _HAS_ASYNCPG = False


class PostgresAdapter(PersistenceAdapter):
    """
    PostgreSQL 持久化适配器
    
    特性：
    - 完整的 ACID 事务
    - 审计日志
    - 向量存储（可选 pgvector）
    - 连接池
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "aga",
        user: str = "aga",
        password: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        enable_audit: bool = True,
    ):
        """
        初始化 PostgreSQL 适配器
        
        Args:
            host: 数据库主机
            port: 数据库端口
            database: 数据库名
            user: 用户名
            password: 密码
            pool_size: 连接池大小
            max_overflow: 最大溢出连接数
            enable_audit: 是否启用审计日志
        """
        if not _HAS_ASYNCPG:
            raise ImportError("PostgreSQL adapter requires 'asyncpg' package")
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.enable_audit = enable_audit
        
        self._pool: Optional[asyncpg.Pool] = None
        self._connected = False
    
    async def _init_tables(self):
        """初始化数据库表"""
        async with self._pool.acquire() as conn:
            # 知识表
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id SERIAL PRIMARY KEY,
                    namespace VARCHAR(255) NOT NULL,
                    slot_idx INTEGER NOT NULL,
                    lu_id VARCHAR(255) NOT NULL,
                    condition TEXT,
                    decision TEXT,
                    key_vector JSONB NOT NULL,
                    value_vector JSONB NOT NULL,
                    lifecycle_state VARCHAR(50) NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    consecutive_misses INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    UNIQUE(namespace, lu_id)
                )
            ''')
            
            # 审计日志表
            if self.enable_audit:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id SERIAL PRIMARY KEY,
                        namespace VARCHAR(255) NOT NULL,
                        lu_id VARCHAR(255),
                        action VARCHAR(100) NOT NULL,
                        old_state VARCHAR(50),
                        new_state VARCHAR(50),
                        details TEXT,
                        timestamp TIMESTAMP NOT NULL
                    )
                ''')
            
            # 创建索引
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_namespace 
                ON knowledge(namespace)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_lu_id 
                ON knowledge(lu_id)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_state 
                ON knowledge(namespace, lifecycle_state)
            ''')
            
            if self.enable_audit:
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_audit_namespace 
                    ON audit_log(namespace)
                ''')
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                    ON audit_log(timestamp)
                ''')
        
        logger.info("PostgreSQL tables initialized")
    
    async def _log_audit(
        self,
        conn,
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
            await conn.execute('''
                INSERT INTO audit_log 
                (namespace, lu_id, action, old_state, new_state, details, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', namespace, lu_id, action, old_state, new_state, details, datetime.now())
        except Exception as e:
            logger.warning(f"Failed to log audit: {e}")
    
    # ==================== 连接管理 ====================
    
    async def connect(self) -> bool:
        try:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.pool_size,
                max_size=self.pool_size + self.max_overflow,
            )
            
            await self._init_tables()
            self._connected = True
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._connected = False
    
    async def is_connected(self) -> bool:
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            if not self._pool:
                return {"status": "disconnected", "adapter": "postgres"}
            
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow("SELECT version()")
            
            return {
                "status": "healthy",
                "adapter": "postgres",
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "version": result[0] if result else None,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "adapter": "postgres",
                "error": str(e),
            }
    
    # ==================== 槽位操作 ====================
    
    async def save_slot(self, namespace: str, record: KnowledgeRecord) -> bool:
        if not self._pool:
            return False
        
        try:
            async with self._pool.acquire() as conn:
                now = datetime.now()
                
                await conn.execute('''
                    INSERT INTO knowledge 
                    (namespace, slot_idx, lu_id, condition, decision, 
                     key_vector, value_vector, lifecycle_state, hit_count,
                     consecutive_misses, version, metadata, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (namespace, lu_id) DO UPDATE SET
                        slot_idx = EXCLUDED.slot_idx,
                        condition = EXCLUDED.condition,
                        decision = EXCLUDED.decision,
                        key_vector = EXCLUDED.key_vector,
                        value_vector = EXCLUDED.value_vector,
                        lifecycle_state = EXCLUDED.lifecycle_state,
                        hit_count = EXCLUDED.hit_count,
                        consecutive_misses = EXCLUDED.consecutive_misses,
                        version = knowledge.version + 1,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                ''', 
                    namespace, record.slot_idx, record.lu_id,
                    record.condition, record.decision,
                    json.dumps(record.key_vector), json.dumps(record.value_vector),
                    record.lifecycle_state, record.hit_count,
                    record.consecutive_misses, record.version,
                    json.dumps(record.metadata) if record.metadata else None,
                    now, now
                )
                
                await self._log_audit(
                    conn, namespace, record.lu_id, "save_slot",
                    None, record.lifecycle_state, f"slot={record.slot_idx}"
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to save slot to PostgreSQL: {e}")
            return False
    
    async def load_slot(self, namespace: str, lu_id: str) -> Optional[KnowledgeRecord]:
        if not self._pool:
            return None
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow('''
                    SELECT * FROM knowledge WHERE namespace = $1 AND lu_id = $2
                ''', namespace, lu_id)
            
            if not row:
                return None
            
            return self._row_to_record(row)
        except Exception as e:
            logger.error(f"Failed to load slot from PostgreSQL: {e}")
            return None
    
    async def delete_slot(self, namespace: str, lu_id: str) -> bool:
        if not self._pool:
            return False
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute('''
                    DELETE FROM knowledge WHERE namespace = $1 AND lu_id = $2
                ''', namespace, lu_id)
                
                await self._log_audit(
                    conn, namespace, lu_id, "delete_slot", None, None, None
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete slot from PostgreSQL: {e}")
            return False
    
    async def slot_exists(self, namespace: str, lu_id: str) -> bool:
        if not self._pool:
            return False
        
        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow('''
                    SELECT 1 FROM knowledge WHERE namespace = $1 AND lu_id = $2
                ''', namespace, lu_id)
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check slot existence: {e}")
            return False
    
    # ==================== 批量操作 ====================
    
    async def save_batch(self, namespace: str, records: List[KnowledgeRecord]) -> int:
        if not self._pool or not records:
            return 0
        
        try:
            async with self._pool.acquire() as conn:
                now = datetime.now()
                
                # 使用事务
                async with conn.transaction():
                    for record in records:
                        await conn.execute('''
                            INSERT INTO knowledge 
                            (namespace, slot_idx, lu_id, condition, decision, 
                             key_vector, value_vector, lifecycle_state, hit_count,
                             consecutive_misses, version, metadata, created_at, updated_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                            ON CONFLICT (namespace, lu_id) DO UPDATE SET
                                slot_idx = EXCLUDED.slot_idx,
                                condition = EXCLUDED.condition,
                                decision = EXCLUDED.decision,
                                key_vector = EXCLUDED.key_vector,
                                value_vector = EXCLUDED.value_vector,
                                lifecycle_state = EXCLUDED.lifecycle_state,
                                hit_count = EXCLUDED.hit_count,
                                consecutive_misses = EXCLUDED.consecutive_misses,
                                version = knowledge.version + 1,
                                metadata = EXCLUDED.metadata,
                                updated_at = EXCLUDED.updated_at
                        ''',
                            namespace, record.slot_idx, record.lu_id,
                            record.condition, record.decision,
                            json.dumps(record.key_vector), json.dumps(record.value_vector),
                            record.lifecycle_state, record.hit_count,
                            record.consecutive_misses, record.version,
                            json.dumps(record.metadata) if record.metadata else None,
                            now, now
                        )
                    
                    await self._log_audit(
                        conn, namespace, None, "save_batch",
                        None, None, f"count={len(records)}"
                    )
            
            return len(records)
        except Exception as e:
            logger.error(f"Failed to save batch to PostgreSQL: {e}")
            return 0
    
    async def load_active_slots(self, namespace: str) -> List[KnowledgeRecord]:
        if not self._pool:
            return []
        
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM knowledge 
                    WHERE namespace = $1 AND lifecycle_state != $2
                    ORDER BY slot_idx
                ''', namespace, LifecycleState.QUARANTINED.value)
            
            return [self._row_to_record(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to load active slots from PostgreSQL: {e}")
            return []
    
    async def load_all_slots(self, namespace: str) -> List[KnowledgeRecord]:
        if not self._pool:
            return []
        
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM knowledge WHERE namespace = $1
                    ORDER BY slot_idx
                ''', namespace)
            
            return [self._row_to_record(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to load all slots from PostgreSQL: {e}")
            return []
    
    def _row_to_record(self, row) -> KnowledgeRecord:
        """转换数据库行到记录"""
        key_vector = row['key_vector']
        value_vector = row['value_vector']
        metadata = row['metadata']
        
        # 处理 JSONB
        if isinstance(key_vector, str):
            key_vector = json.loads(key_vector)
        if isinstance(value_vector, str):
            value_vector = json.loads(value_vector)
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return KnowledgeRecord(
            slot_idx=row['slot_idx'],
            lu_id=row['lu_id'],
            condition=row['condition'] or '',
            decision=row['decision'] or '',
            key_vector=key_vector,
            value_vector=value_vector,
            lifecycle_state=row['lifecycle_state'],
            namespace=row['namespace'],
            hit_count=row['hit_count'],
            consecutive_misses=row['consecutive_misses'],
            version=row['version'],
            created_at=row['created_at'].isoformat() if row['created_at'] else None,
            updated_at=row['updated_at'].isoformat() if row['updated_at'] else None,
            metadata=metadata,
        )
    
    # ==================== 生命周期管理 ====================
    
    async def update_lifecycle(
        self, 
        namespace: str, 
        lu_id: str, 
        new_state: LifecycleState
    ) -> bool:
        if not self._pool:
            return False
        
        try:
            async with self._pool.acquire() as conn:
                # 获取旧状态
                row = await conn.fetchrow('''
                    SELECT lifecycle_state FROM knowledge 
                    WHERE namespace = $1 AND lu_id = $2
                ''', namespace, lu_id)
                old_state = row['lifecycle_state'] if row else None
                
                # 更新
                await conn.execute('''
                    UPDATE knowledge 
                    SET lifecycle_state = $1, updated_at = $2
                    WHERE namespace = $3 AND lu_id = $4
                ''', new_state.value, datetime.now(), namespace, lu_id)
                
                await self._log_audit(
                    conn, namespace, lu_id, "update_lifecycle",
                    old_state, new_state.value, None
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to update lifecycle in PostgreSQL: {e}")
            return False
    
    async def update_lifecycle_batch(
        self,
        namespace: str,
        updates: List[tuple]
    ) -> int:
        if not self._pool or not updates:
            return 0
        
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    count = 0
                    now = datetime.now()
                    
                    for lu_id, new_state in updates:
                        result = await conn.execute('''
                            UPDATE knowledge 
                            SET lifecycle_state = $1, updated_at = $2
                            WHERE namespace = $3 AND lu_id = $4
                        ''', new_state.value, now, namespace, lu_id)
                        
                        if "UPDATE 1" in result:
                            count += 1
                    
                    await self._log_audit(
                        conn, namespace, None, "update_lifecycle_batch",
                        None, None, f"count={count}"
                    )
            
            return count
        except Exception as e:
            logger.error(f"Failed to update lifecycle batch in PostgreSQL: {e}")
            return 0
    
    # ==================== 统计查询 ====================
    
    async def get_slot_count(
        self, 
        namespace: str, 
        state: Optional[LifecycleState] = None
    ) -> int:
        if not self._pool:
            return 0
        
        try:
            async with self._pool.acquire() as conn:
                if state:
                    result = await conn.fetchrow('''
                        SELECT COUNT(*) as count FROM knowledge 
                        WHERE namespace = $1 AND lifecycle_state = $2
                    ''', namespace, state.value)
                else:
                    result = await conn.fetchrow('''
                        SELECT COUNT(*) as count FROM knowledge 
                        WHERE namespace = $1
                    ''', namespace)
                
                return result['count']
        except Exception as e:
            logger.error(f"Failed to get slot count from PostgreSQL: {e}")
            return 0
    
    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        if not self._pool:
            return {"namespace": namespace, "error": "Not connected"}
        
        try:
            async with self._pool.acquire() as conn:
                # 总数
                total_result = await conn.fetchrow('''
                    SELECT COUNT(*) as count FROM knowledge WHERE namespace = $1
                ''', namespace)
                
                # 状态分布
                state_rows = await conn.fetch('''
                    SELECT lifecycle_state, COUNT(*) as count 
                    FROM knowledge WHERE namespace = $1
                    GROUP BY lifecycle_state
                ''', namespace)
                state_dist = {row['lifecycle_state']: row['count'] for row in state_rows}
                
                # 命中统计
                hit_result = await conn.fetchrow('''
                    SELECT SUM(hit_count) as total_hits,
                           AVG(hit_count) as avg_hits,
                           MAX(hit_count) as max_hits
                    FROM knowledge WHERE namespace = $1
                ''', namespace)
            
            return {
                "namespace": namespace,
                "total_slots": total_result['count'],
                "state_distribution": state_dist,
                "total_hits": hit_result['total_hits'] or 0,
                "avg_hits": float(hit_result['avg_hits'] or 0),
                "max_hits": hit_result['max_hits'] or 0,
                "adapter": "postgres",
                "host": self.host,
                "database": self.database,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics from PostgreSQL: {e}")
            return {"namespace": namespace, "error": str(e)}
    
    # ==================== 命中计数 ====================
    
    async def increment_hit_count(
        self, 
        namespace: str, 
        lu_ids: List[str]
    ) -> bool:
        if not self._pool or not lu_ids:
            return True
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute('''
                    UPDATE knowledge 
                    SET hit_count = hit_count + 1
                    WHERE namespace = $1 AND lu_id = ANY($2)
                ''', namespace, lu_ids)
            
            return True
        except Exception as e:
            logger.error(f"Failed to increment hit count in PostgreSQL: {e}")
            return False
    
    # ==================== 审计日志 ====================
    
    async def get_audit_log(
        self, 
        namespace: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取审计日志"""
        if not self._pool or not self.enable_audit:
            return []
        
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM audit_log 
                    WHERE namespace = $1 
                    ORDER BY timestamp DESC 
                    LIMIT $2
                ''', namespace, limit)
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get audit log from PostgreSQL: {e}")
            return []
    
    async def save_audit_log(self, entry: Dict[str, Any]) -> bool:
        """保存审计日志（Portal 扩展接口）"""
        if not self._pool or not self.enable_audit:
            return True
        
        try:
            async with self._pool.acquire() as conn:
                await self._log_audit(
                    conn,
                    namespace=entry.get("namespace", "default"),
                    lu_id=entry.get("lu_id"),
                    action=entry.get("action", "unknown"),
                    old_state=entry.get("old_state"),
                    new_state=entry.get("new_state"),
                    details=json.dumps(entry.get("details")) if entry.get("details") else entry.get("reason"),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save audit log to PostgreSQL: {e}")
            return False
    
    async def query_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """查询审计日志（Portal 扩展接口）"""
        if not self._pool or not self.enable_audit:
            return []
        
        try:
            async with self._pool.acquire() as conn:
                # 构建查询
                conditions = []
                params = []
                param_idx = 1
                
                if namespace:
                    conditions.append(f"namespace = ${param_idx}")
                    params.append(namespace)
                    param_idx += 1
                if lu_id:
                    conditions.append(f"lu_id = ${param_idx}")
                    params.append(lu_id)
                    param_idx += 1
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                query = f'''
                    SELECT * FROM audit_log 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC 
                    LIMIT ${param_idx} OFFSET ${param_idx + 1}
                '''
                params.extend([limit, offset])
                
                rows = await conn.fetch(query, *params)
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query audit log from PostgreSQL: {e}")
            return []
    
    # ==================== 命名空间管理 ====================
    
    async def get_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        if not self._pool:
            return []
        
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT DISTINCT namespace FROM knowledge
                ''')
            
            return [row['namespace'] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get namespaces from PostgreSQL: {e}")
            return []
    
    # ==================== DSN 连接支持 ====================
    
    @classmethod
    async def from_dsn(cls, dsn: str, **kwargs) -> "PostgresAdapter":
        """
        从 DSN 创建适配器
        
        Args:
            dsn: PostgreSQL 连接字符串 (postgresql://user:pass@host:port/db)
            **kwargs: 其他参数
        
        Returns:
            PostgresAdapter 实例
        """
        # 解析 DSN
        import urllib.parse
        parsed = urllib.parse.urlparse(dsn)
        
        adapter = cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "aga",
            user=parsed.username or "aga",
            password=parsed.password,
            **kwargs
        )
        
        await adapter.connect()
        return adapter
