"""
aga-knowledge PostgreSQL 持久化适配器

明文 KV 版本：存储 condition/decision 文本对，不含向量数据。
支持完整的 ACID 事务、审计日志、连接池。

依赖:
    pip install asyncpg

配置示例:
    persistence:
      type: "postgres"
      postgres_url: "postgresql://user:pass@localhost:5432/aga_knowledge"
      postgres_pool_size: 10
      postgres_max_overflow: 20
      enable_audit: true
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from .base import PersistenceAdapter
from ..types import LifecycleState

logger = logging.getLogger(__name__)

# 延迟导入
try:
    import asyncpg
    _HAS_ASYNCPG = True
except ImportError:
    _HAS_ASYNCPG = False


class PostgresAdapter(PersistenceAdapter):
    """
    PostgreSQL 持久化适配器（明文 KV 版本）

    特性:
    - 完整的 ACID 事务
    - 审计日志
    - 连接池（asyncpg）
    - 索引优化
    - 命名空间隔离
    - 不存储向量数据
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "aga_knowledge",
        user: str = "aga",
        password: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        enable_audit: bool = True,
        dsn: Optional[str] = None,
    ):
        """
        初始化 PostgreSQL 适配器

        Args:
            host: 数据库主机
            port: 数据库端口
            database: 数据库名
            user: 用户名
            password: 密码
            pool_size: 连接池最小大小
            max_overflow: 连接池最大溢出
            enable_audit: 是否启用审计日志
            dsn: 完整的 DSN 连接字符串（优先于单独参数）
        """
        if not _HAS_ASYNCPG:
            raise ImportError(
                "PostgreSQL adapter 需要 asyncpg 包。\n"
                "请运行: pip install asyncpg"
            )

        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.enable_audit = enable_audit
        self.dsn = dsn

        self._pool: Optional[asyncpg.Pool] = None
        self._connected = False

    # ==================== 连接管理 ====================

    async def connect(self) -> bool:
        try:
            if self.dsn:
                self._pool = await asyncpg.create_pool(
                    dsn=self.dsn,
                    min_size=self.pool_size,
                    max_size=self.pool_size + self.max_overflow,
                )
            else:
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
            logger.info(
                f"Connected to PostgreSQL: "
                f"{self.dsn or f'{self.host}:{self.port}/{self.database}'}"
            )
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
        logger.info("Disconnected from PostgreSQL")

    async def is_connected(self) -> bool:
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
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
                "pool_size": self._pool.get_size() if self._pool else 0,
            }
        except Exception as e:
            return {"status": "unhealthy", "adapter": "postgres", "error": str(e)}

    async def _init_tables(self):
        """初始化数据库表"""
        async with self._pool.acquire() as conn:
            # 知识表（明文 KV，无向量字段）
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id SERIAL PRIMARY KEY,
                    namespace VARCHAR(255) NOT NULL,
                    lu_id VARCHAR(255) NOT NULL,
                    condition_text TEXT,
                    decision_text TEXT,
                    lifecycle_state VARCHAR(50) NOT NULL DEFAULT 'probationary',
                    trust_tier VARCHAR(50) NOT NULL DEFAULT 'standard',
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
                        source VARCHAR(100) DEFAULT 'portal',
                        reason TEXT,
                        timestamp TIMESTAMP NOT NULL
                    )
                ''')

            # 索引
            await conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_knowledge_namespace '
                'ON knowledge(namespace)'
            )
            await conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_knowledge_lu_id '
                'ON knowledge(lu_id)'
            )
            await conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_knowledge_state '
                'ON knowledge(namespace, lifecycle_state)'
            )
            await conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_knowledge_tier '
                'ON knowledge(namespace, trust_tier)'
            )

            if self.enable_audit:
                await conn.execute(
                    'CREATE INDEX IF NOT EXISTS idx_audit_namespace '
                    'ON audit_log(namespace)'
                )
                await conn.execute(
                    'CREATE INDEX IF NOT EXISTS idx_audit_timestamp '
                    'ON audit_log(timestamp)'
                )
                await conn.execute(
                    'CREATE INDEX IF NOT EXISTS idx_audit_lu_id '
                    'ON audit_log(lu_id)'
                )

        logger.info("PostgreSQL tables initialized")

    async def _log_audit(
        self,
        conn,
        namespace: str,
        lu_id: Optional[str],
        action: str,
        old_state: Optional[str] = None,
        new_state: Optional[str] = None,
        details: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """记录审计日志"""
        if not self.enable_audit:
            return
        try:
            await conn.execute('''
                INSERT INTO audit_log
                (namespace, lu_id, action, old_state, new_state, details, reason, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', namespace, lu_id, action, old_state, new_state,
                details, reason, datetime.utcnow())
        except Exception as e:
            logger.warning(f"Failed to log audit: {e}")

    # ==================== 知识 CRUD ====================

    async def save_knowledge(
        self, namespace: str, lu_id: str, data: Dict[str, Any]
    ) -> bool:
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                now = datetime.utcnow()
                metadata_json = json.dumps(data.get("metadata")) if data.get("metadata") else None

                await conn.execute('''
                    INSERT INTO knowledge
                    (namespace, lu_id, condition_text, decision_text,
                     lifecycle_state, trust_tier, hit_count,
                     consecutive_misses, version, metadata,
                     created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1, $9, $10, $11)
                    ON CONFLICT (namespace, lu_id) DO UPDATE SET
                        condition_text = EXCLUDED.condition_text,
                        decision_text = EXCLUDED.decision_text,
                        lifecycle_state = EXCLUDED.lifecycle_state,
                        trust_tier = EXCLUDED.trust_tier,
                        hit_count = EXCLUDED.hit_count,
                        consecutive_misses = EXCLUDED.consecutive_misses,
                        version = knowledge.version + 1,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                ''',
                    namespace, lu_id,
                    data.get("condition", ""),
                    data.get("decision", ""),
                    data.get("lifecycle_state", "probationary"),
                    data.get("trust_tier", "standard"),
                    data.get("hit_count", 0),
                    data.get("consecutive_misses", 0),
                    metadata_json,
                    now, now,
                )

                await self._log_audit(
                    conn, namespace, lu_id, "SAVE",
                    new_state=data.get("lifecycle_state", "probationary"),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to save knowledge to PostgreSQL: {e}")
            return False

    async def load_knowledge(
        self, namespace: str, lu_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow('''
                    SELECT * FROM knowledge
                    WHERE namespace = $1 AND lu_id = $2
                ''', namespace, lu_id)

            if not row:
                return None
            return self._row_to_dict(row)
        except Exception as e:
            logger.error(f"Failed to load knowledge from PostgreSQL: {e}")
            return None

    async def delete_knowledge(self, namespace: str, lu_id: str) -> bool:
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute('''
                    DELETE FROM knowledge
                    WHERE namespace = $1 AND lu_id = $2
                ''', namespace, lu_id)

                await self._log_audit(conn, namespace, lu_id, "DELETE")

            return True
        except Exception as e:
            logger.error(f"Failed to delete knowledge from PostgreSQL: {e}")
            return False

    async def knowledge_exists(self, namespace: str, lu_id: str) -> bool:
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow('''
                    SELECT 1 FROM knowledge
                    WHERE namespace = $1 AND lu_id = $2
                ''', namespace, lu_id)
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check knowledge existence: {e}")
            return False

    # ==================== 批量操作 ====================

    async def save_batch(
        self, namespace: str, records: List[Dict[str, Any]]
    ) -> int:
        if not self._pool or not records:
            return 0

        try:
            async with self._pool.acquire() as conn:
                now = datetime.utcnow()
                count = 0

                async with conn.transaction():
                    for data in records:
                        lu_id = data.get("lu_id", "")
                        if not lu_id:
                            continue

                        metadata_json = (
                            json.dumps(data.get("metadata"))
                            if data.get("metadata") else None
                        )

                        await conn.execute('''
                            INSERT INTO knowledge
                            (namespace, lu_id, condition_text, decision_text,
                             lifecycle_state, trust_tier, hit_count,
                             consecutive_misses, version, metadata,
                             created_at, updated_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1, $9, $10, $11)
                            ON CONFLICT (namespace, lu_id) DO UPDATE SET
                                condition_text = EXCLUDED.condition_text,
                                decision_text = EXCLUDED.decision_text,
                                lifecycle_state = EXCLUDED.lifecycle_state,
                                trust_tier = EXCLUDED.trust_tier,
                                version = knowledge.version + 1,
                                metadata = EXCLUDED.metadata,
                                updated_at = EXCLUDED.updated_at
                        ''',
                            namespace, lu_id,
                            data.get("condition", ""),
                            data.get("decision", ""),
                            data.get("lifecycle_state", "probationary"),
                            data.get("trust_tier", "standard"),
                            data.get("hit_count", 0),
                            data.get("consecutive_misses", 0),
                            metadata_json,
                            now, now,
                        )
                        count += 1

                    await self._log_audit(
                        conn, namespace, None, "BATCH_SAVE",
                        details=f"count={count}",
                    )

            return count
        except Exception as e:
            logger.error(f"Failed to save batch to PostgreSQL: {e}")
            return 0

    async def load_active_knowledge(self, namespace: str) -> List[Dict[str, Any]]:
        if not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM knowledge
                    WHERE namespace = $1 AND lifecycle_state != $2
                    ORDER BY lu_id
                ''', namespace, LifecycleState.QUARANTINED.value)

            return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to load active knowledge: {e}")
            return []

    async def load_all_knowledge(self, namespace: str) -> List[Dict[str, Any]]:
        if not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM knowledge
                    WHERE namespace = $1
                    ORDER BY lu_id
                ''', namespace)

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
        if not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                query = "SELECT * FROM knowledge WHERE namespace = $1"
                params: list = [namespace]
                param_idx = 2

                if lifecycle_states:
                    placeholders = ", ".join(
                        f"${param_idx + i}" for i in range(len(lifecycle_states))
                    )
                    query += f" AND lifecycle_state IN ({placeholders})"
                    params.extend(lifecycle_states)
                    param_idx += len(lifecycle_states)

                if trust_tiers:
                    placeholders = ", ".join(
                        f"${param_idx + i}" for i in range(len(trust_tiers))
                    )
                    query += f" AND trust_tier IN ({placeholders})"
                    params.extend(trust_tiers)
                    param_idx += len(trust_tiers)

                query += f" ORDER BY lu_id LIMIT ${param_idx} OFFSET ${param_idx + 1}"
                params.extend([limit, offset])

                rows = await conn.fetch(query, *params)

            return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}")
            return []

    # ==================== 生命周期管理 ====================

    async def update_lifecycle(
        self, namespace: str, lu_id: str, new_state: str
    ) -> bool:
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow('''
                    SELECT lifecycle_state FROM knowledge
                    WHERE namespace = $1 AND lu_id = $2
                ''', namespace, lu_id)
                old_state = row['lifecycle_state'] if row else None

                now = datetime.utcnow()
                await conn.execute('''
                    UPDATE knowledge
                    SET lifecycle_state = $1, updated_at = $2
                    WHERE namespace = $3 AND lu_id = $4
                ''', new_state, now, namespace, lu_id)

                await self._log_audit(
                    conn, namespace, lu_id, "UPDATE_LIFECYCLE",
                    old_state, new_state,
                )

            return True
        except Exception as e:
            logger.error(f"Failed to update lifecycle: {e}")
            return False

    async def update_trust_tier(
        self, namespace: str, lu_id: str, new_tier: str
    ) -> bool:
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                now = datetime.utcnow()
                await conn.execute('''
                    UPDATE knowledge
                    SET trust_tier = $1, updated_at = $2
                    WHERE namespace = $3 AND lu_id = $4
                ''', new_tier, now, namespace, lu_id)

                await self._log_audit(
                    conn, namespace, lu_id, "UPDATE_TRUST_TIER",
                    new_state=new_tier,
                )

            return True
        except Exception as e:
            logger.error(f"Failed to update trust tier: {e}")
            return False

    # ==================== 统计 ====================

    async def get_knowledge_count(
        self, namespace: str, state: Optional[str] = None
    ) -> int:
        if not self._pool:
            return 0

        try:
            async with self._pool.acquire() as conn:
                if state:
                    result = await conn.fetchrow('''
                        SELECT COUNT(*) as count FROM knowledge
                        WHERE namespace = $1 AND lifecycle_state = $2
                    ''', namespace, state)
                else:
                    result = await conn.fetchrow('''
                        SELECT COUNT(*) as count FROM knowledge
                        WHERE namespace = $1
                    ''', namespace)

                return result['count']
        except Exception as e:
            logger.error(f"Failed to get knowledge count: {e}")
            return 0

    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        if not self._pool:
            return {"namespace": namespace, "error": "Not connected"}

        try:
            async with self._pool.acquire() as conn:
                total_result = await conn.fetchrow('''
                    SELECT COUNT(*) as count FROM knowledge
                    WHERE namespace = $1
                ''', namespace)

                state_rows = await conn.fetch('''
                    SELECT lifecycle_state, COUNT(*) as count
                    FROM knowledge WHERE namespace = $1
                    GROUP BY lifecycle_state
                ''', namespace)
                state_dist = {
                    row['lifecycle_state']: row['count'] for row in state_rows
                }

                hit_result = await conn.fetchrow('''
                    SELECT SUM(hit_count) as total_hits,
                           AVG(hit_count) as avg_hits,
                           MAX(hit_count) as max_hits
                    FROM knowledge WHERE namespace = $1
                ''', namespace)

            return {
                "namespace": namespace,
                "total_knowledge": total_result['count'],
                "state_distribution": state_dist,
                "total_hits": hit_result['total_hits'] or 0,
                "avg_hits": float(hit_result['avg_hits'] or 0),
                "max_hits": hit_result['max_hits'] or 0,
                "adapter": "postgres",
                "host": self.host,
                "database": self.database,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"namespace": namespace, "error": str(e)}

    async def increment_hit_count(
        self, namespace: str, lu_ids: List[str]
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
            logger.error(f"Failed to increment hit count: {e}")
            return False

    # ==================== 命名空间 ====================

    async def get_namespaces(self) -> List[str]:
        if not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    'SELECT DISTINCT namespace FROM knowledge'
                )
            return [row['namespace'] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get namespaces: {e}")
            return []

    # ==================== 审计日志 ====================

    async def save_audit_log(self, entry: Dict[str, Any]) -> bool:
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
                    details=(
                        json.dumps(entry.get("details"))
                        if entry.get("details") else None
                    ),
                    reason=entry.get("reason"),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save audit log: {e}")
            return False

    async def query_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if not self._pool or not self.enable_audit:
            return []

        try:
            async with self._pool.acquire() as conn:
                query = "SELECT * FROM audit_log WHERE 1=1"
                params: list = []
                param_idx = 1

                if namespace:
                    query += f" AND namespace = ${param_idx}"
                    params.append(namespace)
                    param_idx += 1
                if lu_id:
                    query += f" AND lu_id = ${param_idx}"
                    params.append(lu_id)
                    param_idx += 1

                query += (
                    f" ORDER BY timestamp DESC"
                    f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
                )
                params.extend([limit, offset])

                rows = await conn.fetch(query, *params)

            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query audit log: {e}")
            return []

    # ==================== 工具方法 ====================

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """转换数据库行到字典"""
        metadata = row['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = None

        return {
            "lu_id": row['lu_id'],
            "namespace": row['namespace'],
            "condition": row['condition_text'] or "",
            "decision": row['decision_text'] or "",
            "lifecycle_state": row['lifecycle_state'],
            "trust_tier": row['trust_tier'],
            "hit_count": row['hit_count'],
            "consecutive_misses": row['consecutive_misses'],
            "version": row['version'],
            "created_at": (
                row['created_at'].isoformat() if row['created_at'] else None
            ),
            "updated_at": (
                row['updated_at'].isoformat() if row['updated_at'] else None
            ),
            "metadata": metadata,
        }

    @classmethod
    async def from_dsn(cls, dsn: str, **kwargs) -> "PostgresAdapter":
        """
        从 DSN 创建适配器

        Args:
            dsn: PostgreSQL 连接字符串
            **kwargs: 其他参数

        Returns:
            已连接的 PostgresAdapter 实例
        """
        adapter = cls(dsn=dsn, **kwargs)
        await adapter.connect()
        return adapter
