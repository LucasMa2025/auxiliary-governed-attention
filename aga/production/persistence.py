"""
AGA 生产级持久化层 (单机部署模式)

⚠️ 重要：本模块专为单机部署设计
=========================================

本模块使用 **同步** 客户端（redis、SQLAlchemy），设计目标是：
- 与 AGA 推理进程同机部署
- 最小化持久化延迟
- 简化运维复杂度

如需分离部署（API 服务与推理服务分开），请使用：
- aga.persistence: 异步持久化适配器
- aga.portal: 独立的 API 服务
- aga.sync: Portal ↔ Runtime 同步协议

架构对比：
---------
单机部署 (本模块):
    LLM + AGA + persistence.py (同进程)
    └─── 直接写入本地 Redis/PostgreSQL

分离部署 (aga.portal + aga.runtime):
    Portal (无 GPU) ────┬──── PostgreSQL
                        │
                        │ Redis Pub/Sub
                        ▼
    Runtime (GPU) ←── 订阅同步消息

支持的存储：
- Redis: 热槽位缓存 (同步客户端)
- PostgreSQL: 冷存储 + 审计日志 (SQLAlchemy)
- 混合模式: Redis + PostgreSQL

新增功能：
- Redis→PostgreSQL 增量同步
- Prometheus 指标导出
"""
import json
import time
import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

from .config import PersistenceConfig
from .slot_pool import Slot, LifecycleState

logger = logging.getLogger(__name__)


# ============== Prometheus 指标 ==============

class PrometheusMetrics:
    """
    Prometheus 指标导出器
    
    导出 AGA 持久化层的关键指标：
    - 槽位数量 (按命名空间和状态)
    - 命中计数
    - 持久化操作延迟
    - 同步状态
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._metrics = {}
        self._collectors = []
        self._prometheus_available = False
        
        try:
            from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY
            self._prometheus_available = True
            self._registry = REGISTRY
            
            # 定义指标
            self._init_metrics()
            
        except ImportError:
            logger.info("prometheus_client not installed, metrics disabled")
        
        self._initialized = True
    
    def _init_metrics(self):
        """初始化 Prometheus 指标"""
        from prometheus_client import Counter, Gauge, Histogram
        
        # 槽位数量 (gauge)
        self._metrics["slots_total"] = Gauge(
            "aga_slots_total",
            "Total number of slots",
            ["namespace", "lifecycle_state"]
        )
        
        # 活跃槽位数量
        self._metrics["slots_active"] = Gauge(
            "aga_slots_active",
            "Number of active (non-quarantined) slots",
            ["namespace"]
        )
        
        # 操作计数 (counter)
        self._metrics["operations_total"] = Counter(
            "aga_persistence_operations_total",
            "Total number of persistence operations",
            ["namespace", "operation", "status"]
        )
        
        # 操作延迟 (histogram)
        self._metrics["operation_duration_seconds"] = Histogram(
            "aga_persistence_operation_duration_seconds",
            "Persistence operation duration in seconds",
            ["namespace", "operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        # 命中计数 (counter)
        self._metrics["hits_total"] = Counter(
            "aga_slot_hits_total",
            "Total slot hits",
            ["namespace"]
        )
        
        # 同步状态
        self._metrics["sync_last_success"] = Gauge(
            "aga_sync_last_success_timestamp",
            "Timestamp of last successful sync"
        )
        
        self._metrics["sync_records_synced"] = Counter(
            "aga_sync_records_synced_total",
            "Total records synced from Redis to PostgreSQL"
        )
        
        self._metrics["sync_errors"] = Counter(
            "aga_sync_errors_total",
            "Total sync errors"
        )
    
    @property
    def enabled(self) -> bool:
        """是否启用指标"""
        return self._prometheus_available
    
    def record_slot_count(self, namespace: str, state: str, count: int):
        """记录槽位数量"""
        if not self.enabled:
            return
        self._metrics["slots_total"].labels(
            namespace=namespace, 
            lifecycle_state=state
        ).set(count)
    
    def record_active_slots(self, namespace: str, count: int):
        """记录活跃槽位数量"""
        if not self.enabled:
            return
        self._metrics["slots_active"].labels(namespace=namespace).set(count)
    
    def record_operation(
        self, 
        namespace: str, 
        operation: str, 
        status: str, 
        duration: float
    ):
        """记录持久化操作"""
        if not self.enabled:
            return
        self._metrics["operations_total"].labels(
            namespace=namespace,
            operation=operation,
            status=status
        ).inc()
        self._metrics["operation_duration_seconds"].labels(
            namespace=namespace,
            operation=operation
        ).observe(duration)
    
    def record_hits(self, namespace: str, count: int = 1):
        """记录命中"""
        if not self.enabled:
            return
        self._metrics["hits_total"].labels(namespace=namespace).inc(count)
    
    def record_sync_success(self, records_synced: int):
        """记录同步成功"""
        if not self.enabled:
            return
        self._metrics["sync_last_success"].set(time.time())
        self._metrics["sync_records_synced"].inc(records_synced)
    
    def record_sync_error(self):
        """记录同步错误"""
        if not self.enabled:
            return
        self._metrics["sync_errors"].inc()
    
    def get_text_metrics(self) -> str:
        """获取文本格式的指标（用于 HTTP 端点）"""
        if not self.enabled:
            return "# Prometheus metrics not available\n"
        
        from prometheus_client import generate_latest
        return generate_latest(self._registry).decode("utf-8")


# 全局指标实例
metrics = PrometheusMetrics()


# ============== 抽象基类 ==============

class BasePersistence(ABC):
    """持久化抽象基类"""
    
    @abstractmethod
    def save_slot(self, namespace: str, slot: Slot) -> bool:
        """保存槽位"""
        pass
    
    @abstractmethod
    def save_slots_batch(self, namespace: str, slots: List[Slot]) -> int:
        """批量保存槽位"""
        pass
    
    @abstractmethod
    def load_slot(self, namespace: str, lu_id: str) -> Optional[Dict[str, Any]]:
        """加载槽位"""
        pass
    
    @abstractmethod
    def load_active_slots(self, namespace: str) -> List[Dict[str, Any]]:
        """加载活跃槽位"""
        pass
    
    @abstractmethod
    def delete_slot(self, namespace: str, lu_id: str) -> bool:
        """删除槽位"""
        pass
    
    @abstractmethod
    def update_lifecycle(self, namespace: str, lu_id: str, new_state: str) -> bool:
        """更新生命周期"""
        pass
    
    @abstractmethod
    def increment_hit_count(self, namespace: str, lu_ids: List[str]) -> bool:
        """批量增加命中计数"""
        pass
    
    @abstractmethod
    def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """获取统计信息"""
        pass


# ============== Redis 实现 ==============

class RedisPersistence(BasePersistence):
    """
    Redis 持久化实现
    
    数据结构：
    - aga:{namespace}:slots (Hash): lu_id -> slot_data_json
    - aga:{namespace}:meta (Hash): 元数据
    - aga:changes (Pub/Sub): 变更通知
    """
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self._client = None
        self._pool = None
        self._lock = threading.Lock()
    
    @property
    def client(self):
        """懒加载 Redis 客户端"""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._connect()
        return self._client
    
    def _connect(self):
        """建立 Redis 连接"""
        try:
            import redis
            
            self._pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                max_connections=self.config.redis_pool_size,
                decode_responses=True,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            self._client.ping()
            logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
        except ImportError:
            logger.error("redis package not installed. Run: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _slot_key(self, namespace: str) -> str:
        """生成槽位 Hash key"""
        return f"{self.config.redis_key_prefix}:{namespace}:slots"
    
    def _meta_key(self, namespace: str) -> str:
        """生成元数据 Hash key"""
        return f"{self.config.redis_key_prefix}:{namespace}:meta"
    
    def save_slot(self, namespace: str, slot: Slot) -> bool:
        """保存槽位到 Redis"""
        start_time = time.perf_counter()
        try:
            key = self._slot_key(namespace)
            data = json.dumps(slot.to_dict())
            
            # 使用 pipeline 提高效率
            pipe = self.client.pipeline()
            pipe.hset(key, slot.lu_id, data)
            
            # 设置 TTL（如果配置了）
            ttl_seconds = self.config.redis_slot_ttl_days * 86400
            pipe.expire(key, ttl_seconds)
            
            pipe.execute()
            
            # 发布变更通知
            self._publish_change(namespace, "save", slot.lu_id)
            
            # 记录指标
            duration = time.perf_counter() - start_time
            metrics.record_operation(namespace, "save", "success", duration)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save slot to Redis: {e}")
            duration = time.perf_counter() - start_time
            metrics.record_operation(namespace, "save", "error", duration)
            return False
    
    def save_slots_batch(self, namespace: str, slots: List[Slot]) -> int:
        """批量保存槽位"""
        if not slots:
            return 0
        
        try:
            key = self._slot_key(namespace)
            
            pipe = self.client.pipeline()
            for slot in slots:
                data = json.dumps(slot.to_dict())
                pipe.hset(key, slot.lu_id, data)
            
            # 设置 TTL
            ttl_seconds = self.config.redis_slot_ttl_days * 86400
            pipe.expire(key, ttl_seconds)
            
            pipe.execute()
            
            # 发布变更通知
            self._publish_change(namespace, "batch_save", None)
            
            return len(slots)
        except Exception as e:
            logger.error(f"Failed to batch save slots to Redis: {e}")
            return 0
    
    def load_slot(self, namespace: str, lu_id: str) -> Optional[Dict[str, Any]]:
        """从 Redis 加载槽位"""
        try:
            key = self._slot_key(namespace)
            data = self.client.hget(key, lu_id)
            
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to load slot from Redis: {e}")
            return None
    
    def load_active_slots(self, namespace: str) -> List[Dict[str, Any]]:
        """加载所有活跃槽位"""
        try:
            key = self._slot_key(namespace)
            all_data = self.client.hgetall(key)
            
            slots = []
            for lu_id, data in all_data.items():
                slot_data = json.loads(data)
                # 过滤掉已隔离的
                if slot_data.get("lifecycle_state") != LifecycleState.QUARANTINED.value:
                    slots.append(slot_data)
            
            return slots
        except Exception as e:
            logger.error(f"Failed to load active slots from Redis: {e}")
            return []
    
    def delete_slot(self, namespace: str, lu_id: str) -> bool:
        """删除槽位"""
        try:
            key = self._slot_key(namespace)
            self.client.hdel(key, lu_id)
            self._publish_change(namespace, "delete", lu_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete slot from Redis: {e}")
            return False
    
    def update_lifecycle(self, namespace: str, lu_id: str, new_state: str) -> bool:
        """更新生命周期状态"""
        try:
            slot_data = self.load_slot(namespace, lu_id)
            if not slot_data:
                return False
            
            slot_data["lifecycle_state"] = new_state
            slot_data["updated_at"] = time.time()
            
            key = self._slot_key(namespace)
            self.client.hset(key, lu_id, json.dumps(slot_data))
            
            self._publish_change(namespace, "lifecycle", lu_id)
            return True
        except Exception as e:
            logger.error(f"Failed to update lifecycle in Redis: {e}")
            return False
    
    def increment_hit_count(self, namespace: str, lu_ids: List[str]) -> bool:
        """批量增加命中计数（使用 Lua 脚本保证原子性）"""
        if not lu_ids:
            return True
        
        try:
            key = self._slot_key(namespace)
            
            # 使用 Lua 脚本原子更新，避免 read-modify-write 竞态条件
            # KEYS[1] = hash key
            # ARGV[1] = 当前时间戳
            # ARGV[2..N] = lu_id 列表
            lua_script = """
            local key = KEYS[1]
            local timestamp = ARGV[1]
            local updated = 0
            for i = 2, #ARGV do
                local lu_id = ARGV[i]
                local data = redis.call('HGET', key, lu_id)
                if data then
                    local slot = cjson.decode(data)
                    slot['hit_count'] = (slot['hit_count'] or 0) + 1
                    slot['consecutive_misses'] = 0
                    slot['last_hit_ts'] = tonumber(timestamp)
                    redis.call('HSET', key, lu_id, cjson.encode(slot))
                    updated = updated + 1
                end
            end
            return updated
            """
            
            current_ts = str(time.time())
            self.client.eval(lua_script, 1, key, current_ts, *lu_ids)
            return True
        except Exception as e:
            logger.error(f"Failed to increment hit count in Redis: {e}")
            return False
    
    def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            key = self._slot_key(namespace)
            count = self.client.hlen(key)
            
            # 更新 Prometheus 指标
            metrics.record_active_slots(namespace, count)
            
            return {
                "backend": "redis",
                "namespace": namespace,
                "slot_count": count,
                "host": self.config.redis_host,
                "port": self.config.redis_port,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics from Redis: {e}")
            return {"backend": "redis", "error": str(e)}
    
    def _publish_change(self, namespace: str, action: str, lu_id: Optional[str]):
        """发布变更通知"""
        try:
            channel = f"{self.config.redis_key_prefix}:changes"
            message = json.dumps({
                "namespace": namespace,
                "action": action,
                "lu_id": lu_id,
                "timestamp": time.time(),
            })
            self.client.publish(channel, message)
        except Exception as e:
            logger.warning(f"Failed to publish change notification: {e}")
    
    def subscribe_changes(self):
        """订阅变更通知"""
        try:
            pubsub = self.client.pubsub()
            channel = f"{self.config.redis_key_prefix}:changes"
            pubsub.subscribe(channel)
            return pubsub
        except Exception as e:
            logger.error(f"Failed to subscribe to changes: {e}")
            return None


# ============== PostgreSQL 实现 ==============

class PostgreSQLPersistence(BasePersistence):
    """
    PostgreSQL 持久化实现
    
    表结构：
    - aga_slots: 槽位数据
    - aga_audit_log: 审计日志
    """
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self._engine = None
        self._session_factory = None
        self._lock = threading.Lock()
    
    def _connect(self):
        """建立 PostgreSQL 连接"""
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy.pool import QueuePool
            
            url = (f"postgresql://{self.config.postgres_user}:{self.config.postgres_password}"
                   f"@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_db}")
            
            self._engine = create_engine(
                url,
                poolclass=QueuePool,
                pool_size=self.config.postgres_pool_size,
                max_overflow=self.config.postgres_max_overflow,
            )
            
            self._session_factory = sessionmaker(bind=self._engine)
            
            # 初始化表
            self._init_tables()
            
            logger.info(f"Connected to PostgreSQL at {self.config.postgres_host}:{self.config.postgres_port}")
        except ImportError:
            logger.error("sqlalchemy or psycopg2 not installed. Run: pip install sqlalchemy psycopg2-binary")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _init_tables(self):
        """初始化数据库表"""
        from sqlalchemy import text
        
        create_slots_table = """
        CREATE TABLE IF NOT EXISTS aga_slots (
            id SERIAL PRIMARY KEY,
            namespace VARCHAR(100) NOT NULL,
            lu_id VARCHAR(100) NOT NULL,
            slot_idx INTEGER NOT NULL,
            key_vector JSONB NOT NULL,
            value_vector JSONB NOT NULL,
            lifecycle_state VARCHAR(50) NOT NULL,
            condition TEXT,
            decision TEXT,
            hit_count INTEGER DEFAULT 0,
            consecutive_misses INTEGER DEFAULT 0,
            last_hit_ts TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            version INTEGER DEFAULT 1,
            UNIQUE(namespace, lu_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_aga_slots_namespace ON aga_slots(namespace);
        CREATE INDEX IF NOT EXISTS idx_aga_slots_lifecycle ON aga_slots(namespace, lifecycle_state);
        """
        
        create_audit_table = """
        CREATE TABLE IF NOT EXISTS aga_audit_log (
            id SERIAL PRIMARY KEY,
            namespace VARCHAR(100) NOT NULL,
            lu_id VARCHAR(100),
            action VARCHAR(50) NOT NULL,
            old_state VARCHAR(50),
            new_state VARCHAR(50),
            details JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_aga_audit_namespace ON aga_audit_log(namespace);
        CREATE INDEX IF NOT EXISTS idx_aga_audit_created ON aga_audit_log(created_at);
        """
        
        with self._engine.connect() as conn:
            conn.execute(text(create_slots_table))
            conn.execute(text(create_audit_table))
            conn.commit()
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._connect()
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def save_slot(self, namespace: str, slot: Slot) -> bool:
        """保存槽位到 PostgreSQL"""
        try:
            from sqlalchemy import text
            
            with self._get_session() as session:
                # UPSERT
                sql = text("""
                INSERT INTO aga_slots 
                    (namespace, lu_id, slot_idx, key_vector, value_vector, lifecycle_state,
                     condition, decision, hit_count, consecutive_misses, last_hit_ts, version)
                VALUES 
                    (:namespace, :lu_id, :slot_idx, :key_vector, :value_vector, :lifecycle_state,
                     :condition, :decision, :hit_count, :consecutive_misses, :last_hit_ts, :version)
                ON CONFLICT (namespace, lu_id) DO UPDATE SET
                    slot_idx = EXCLUDED.slot_idx,
                    key_vector = EXCLUDED.key_vector,
                    value_vector = EXCLUDED.value_vector,
                    lifecycle_state = EXCLUDED.lifecycle_state,
                    condition = EXCLUDED.condition,
                    decision = EXCLUDED.decision,
                    hit_count = EXCLUDED.hit_count,
                    consecutive_misses = EXCLUDED.consecutive_misses,
                    last_hit_ts = EXCLUDED.last_hit_ts,
                    version = aga_slots.version + 1,
                    updated_at = NOW()
                """)
                
                session.execute(sql, {
                    "namespace": namespace,
                    "lu_id": slot.lu_id,
                    "slot_idx": slot.slot_idx,
                    "key_vector": json.dumps(slot.key_vector.cpu().tolist()),
                    "value_vector": json.dumps(slot.value_vector.cpu().tolist()),
                    "lifecycle_state": slot.lifecycle_state.value,
                    "condition": slot.condition,
                    "decision": slot.decision,
                    "hit_count": slot.hit_count,
                    "consecutive_misses": slot.consecutive_misses,
                    "last_hit_ts": datetime.fromtimestamp(slot.last_hit_ts),
                    "version": slot.version,
                })
                
                # 记录审计日志
                self._log_audit(session, namespace, slot.lu_id, "save", None, slot.lifecycle_state.value)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save slot to PostgreSQL: {e}")
            return False
    
    def save_slots_batch(self, namespace: str, slots: List[Slot]) -> int:
        """批量保存槽位"""
        success_count = 0
        for slot in slots:
            if self.save_slot(namespace, slot):
                success_count += 1
        return success_count
    
    def load_slot(self, namespace: str, lu_id: str) -> Optional[Dict[str, Any]]:
        """从 PostgreSQL 加载槽位"""
        try:
            from sqlalchemy import text
            
            with self._get_session() as session:
                sql = text("""
                SELECT * FROM aga_slots WHERE namespace = :namespace AND lu_id = :lu_id
                """)
                result = session.execute(sql, {"namespace": namespace, "lu_id": lu_id})
                row = result.fetchone()
                
                if row:
                    return self._row_to_dict(row)
                return None
        except Exception as e:
            logger.error(f"Failed to load slot from PostgreSQL: {e}")
            return None
    
    def load_active_slots(self, namespace: str) -> List[Dict[str, Any]]:
        """加载所有活跃槽位"""
        try:
            from sqlalchemy import text
            
            with self._get_session() as session:
                sql = text("""
                SELECT * FROM aga_slots 
                WHERE namespace = :namespace AND lifecycle_state != :quarantined
                ORDER BY slot_idx
                """)
                result = session.execute(sql, {
                    "namespace": namespace,
                    "quarantined": LifecycleState.QUARANTINED.value,
                })
                
                return [self._row_to_dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to load active slots from PostgreSQL: {e}")
            return []
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """转换数据库行到字典"""
        return {
            "slot_idx": row.slot_idx,
            "lu_id": row.lu_id,
            "key_vector": json.loads(row.key_vector) if isinstance(row.key_vector, str) else row.key_vector,
            "value_vector": json.loads(row.value_vector) if isinstance(row.value_vector, str) else row.value_vector,
            "lifecycle_state": row.lifecycle_state,
            "condition": row.condition,
            "decision": row.decision,
            "namespace": row.namespace,
            "hit_count": row.hit_count,
            "consecutive_misses": row.consecutive_misses,
            "last_hit_ts": row.last_hit_ts.timestamp() if row.last_hit_ts else time.time(),
            "created_at": row.created_at.timestamp() if row.created_at else time.time(),
            "updated_at": row.updated_at.timestamp() if row.updated_at else time.time(),
            "version": row.version,
        }
    
    def delete_slot(self, namespace: str, lu_id: str) -> bool:
        """删除槽位"""
        try:
            from sqlalchemy import text
            
            with self._get_session() as session:
                sql = text("""
                DELETE FROM aga_slots WHERE namespace = :namespace AND lu_id = :lu_id
                """)
                session.execute(sql, {"namespace": namespace, "lu_id": lu_id})
                self._log_audit(session, namespace, lu_id, "delete", None, None)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete slot from PostgreSQL: {e}")
            return False
    
    def update_lifecycle(self, namespace: str, lu_id: str, new_state: str) -> bool:
        """更新生命周期状态"""
        try:
            from sqlalchemy import text
            
            with self._get_session() as session:
                # 获取旧状态
                sql = text("""
                SELECT lifecycle_state FROM aga_slots WHERE namespace = :namespace AND lu_id = :lu_id
                """)
                result = session.execute(sql, {"namespace": namespace, "lu_id": lu_id})
                row = result.fetchone()
                old_state = row.lifecycle_state if row else None
                
                # 更新
                sql = text("""
                UPDATE aga_slots SET lifecycle_state = :new_state, updated_at = NOW()
                WHERE namespace = :namespace AND lu_id = :lu_id
                """)
                session.execute(sql, {"namespace": namespace, "lu_id": lu_id, "new_state": new_state})
                
                self._log_audit(session, namespace, lu_id, "lifecycle", old_state, new_state)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update lifecycle in PostgreSQL: {e}")
            return False
    
    def increment_hit_count(self, namespace: str, lu_ids: List[str]) -> bool:
        """批量增加命中计数"""
        if not lu_ids:
            return True
        
        try:
            from sqlalchemy import text
            
            with self._get_session() as session:
                sql = text("""
                UPDATE aga_slots SET 
                    hit_count = hit_count + 1,
                    last_hit_ts = NOW()
                WHERE namespace = :namespace AND lu_id = ANY(:lu_ids)
                """)
                session.execute(sql, {"namespace": namespace, "lu_ids": lu_ids})
            
            return True
        except Exception as e:
            logger.error(f"Failed to increment hit count in PostgreSQL: {e}")
            return False
    
    def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            from sqlalchemy import text
            
            with self._get_session() as session:
                # 总数
                sql = text("SELECT COUNT(*) FROM aga_slots WHERE namespace = :namespace")
                total = session.execute(sql, {"namespace": namespace}).scalar()
                
                # 按状态分组
                sql = text("""
                SELECT lifecycle_state, COUNT(*) as count 
                FROM aga_slots WHERE namespace = :namespace 
                GROUP BY lifecycle_state
                """)
                result = session.execute(sql, {"namespace": namespace})
                state_dist = {row.lifecycle_state: row.count for row in result.fetchall()}
            
            return {
                "backend": "postgresql",
                "namespace": namespace,
                "slot_count": total,
                "state_distribution": state_dist,
                "host": self.config.postgres_host,
                "port": self.config.postgres_port,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics from PostgreSQL: {e}")
            return {"backend": "postgresql", "error": str(e)}
    
    def _log_audit(
        self,
        session,
        namespace: str,
        lu_id: Optional[str],
        action: str,
        old_state: Optional[str],
        new_state: Optional[str],
        details: Optional[Dict] = None,
    ):
        """记录审计日志"""
        from sqlalchemy import text
        
        sql = text("""
        INSERT INTO aga_audit_log (namespace, lu_id, action, old_state, new_state, details)
        VALUES (:namespace, :lu_id, :action, :old_state, :new_state, :details)
        """)
        session.execute(sql, {
            "namespace": namespace,
            "lu_id": lu_id,
            "action": action,
            "old_state": old_state,
            "new_state": new_state,
            "details": json.dumps(details) if details else None,
        })


# ============== 混合持久化 ==============

class HybridPersistence(BasePersistence):
    """
    混合持久化：Redis（热） + PostgreSQL（冷）
    
    策略：
    - 写入：同时写入 Redis 和 PostgreSQL
    - 读取：优先 Redis，miss 时回源 PostgreSQL
    - 同步：后台线程定期同步
    """
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.redis = RedisPersistence(config) if config.redis_enabled else None
        self.postgres = PostgreSQLPersistence(config) if config.postgres_enabled else None
        
        if not self.redis and not self.postgres:
            raise ValueError("At least one persistence backend must be enabled")
    
    def save_slot(self, namespace: str, slot: Slot) -> bool:
        """保存槽位（双写）"""
        success = True
        
        if self.redis:
            if not self.redis.save_slot(namespace, slot):
                success = False
        
        if self.postgres:
            if not self.postgres.save_slot(namespace, slot):
                success = False
        
        return success
    
    def save_slots_batch(self, namespace: str, slots: List[Slot]) -> int:
        """批量保存槽位"""
        redis_count = 0
        pg_count = 0
        
        if self.redis:
            redis_count = self.redis.save_slots_batch(namespace, slots)
        
        if self.postgres:
            pg_count = self.postgres.save_slots_batch(namespace, slots)
        
        return max(redis_count, pg_count)
    
    def load_slot(self, namespace: str, lu_id: str) -> Optional[Dict[str, Any]]:
        """加载槽位（Redis 优先）"""
        if self.redis:
            slot = self.redis.load_slot(namespace, lu_id)
            if slot:
                return slot
        
        if self.postgres:
            slot = self.postgres.load_slot(namespace, lu_id)
            if slot and self.redis:
                # 回填 Redis
                self.redis.save_slot(namespace, Slot.from_dict(slot))
            return slot
        
        return None
    
    def load_active_slots(self, namespace: str) -> List[Dict[str, Any]]:
        """加载活跃槽位"""
        if self.redis:
            slots = self.redis.load_active_slots(namespace)
            if slots:
                return slots
        
        if self.postgres:
            return self.postgres.load_active_slots(namespace)
        
        return []
    
    def delete_slot(self, namespace: str, lu_id: str) -> bool:
        """删除槽位（双删）"""
        success = True
        
        if self.redis:
            if not self.redis.delete_slot(namespace, lu_id):
                success = False
        
        if self.postgres:
            if not self.postgres.delete_slot(namespace, lu_id):
                success = False
        
        return success
    
    def update_lifecycle(self, namespace: str, lu_id: str, new_state: str) -> bool:
        """更新生命周期（双写）"""
        success = True
        
        if self.redis:
            if not self.redis.update_lifecycle(namespace, lu_id, new_state):
                success = False
        
        if self.postgres:
            if not self.postgres.update_lifecycle(namespace, lu_id, new_state):
                success = False
        
        return success
    
    def increment_hit_count(self, namespace: str, lu_ids: List[str]) -> bool:
        """批量增加命中计数"""
        # 只更新 Redis（高频操作），定期同步到 PostgreSQL
        if self.redis:
            return self.redis.increment_hit_count(namespace, lu_ids)
        elif self.postgres:
            return self.postgres.increment_hit_count(namespace, lu_ids)
        return False
    
    def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {"backend": "hybrid"}
        
        if self.redis:
            stats["redis"] = self.redis.get_statistics(namespace)
        
        if self.postgres:
            stats["postgresql"] = self.postgres.get_statistics(namespace)
        
        return stats


# ============== 持久化管理器 ==============

class PersistenceManager:
    """
    持久化管理器
    
    提供统一的持久化接口，管理后台同步。
    
    特性：
    - 带重试的保存操作（指数退避）
    - 带降级的加载操作（Redis → PostgreSQL）
    - Redis 缓存回填
    """
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.persistence = HybridPersistence(config)
        
        # 重试配置
        self._max_retries = 3
        self._retry_delay = 0.5  # 秒
        self._retry_backoff = 2.0  # 指数退避
        
        # 后台同步
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync = threading.Event()
    
    def start_sync(self):
        """启动后台同步"""
        if self._sync_thread is not None:
            return
        
        self._stop_sync.clear()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Started persistence sync thread")
    
    def stop_sync(self):
        """停止后台同步"""
        if self._sync_thread is None:
            return
        
        self._stop_sync.set()
        self._sync_thread.join(timeout=5)
        self._sync_thread = None
        logger.info("Stopped persistence sync thread")
    
    def _sync_loop(self):
        """同步循环"""
        while not self._stop_sync.is_set():
            try:
                # 从 Redis 同步到 PostgreSQL（增量同步）
                self._sync_redis_to_postgres()
            except Exception as e:
                logger.error(f"Sync error: {e}")
            
            self._stop_sync.wait(self.config.sync_interval_seconds)
    
    def _sync_redis_to_postgres(self):
        """
        Redis → PostgreSQL 增量同步
        
        策略：
        1. 获取 Redis 中所有命名空间
        2. 对比 Redis 和 PostgreSQL 的版本号
        3. 只同步有变更的记录（version 不一致）
        4. 同步命中计数增量
        """
        if not (self.persistence.redis and self.persistence.postgres):
            return
        
        redis = self.persistence.redis
        postgres = self.persistence.postgres
        
        try:
            # 获取所有命名空间
            namespaces = self._get_all_namespaces_from_redis()
            
            sync_stats = {
                "total_synced": 0,
                "total_skipped": 0,
                "total_errors": 0,
            }
            
            for namespace in namespaces:
                ns_stats = self._sync_namespace(namespace, redis, postgres)
                sync_stats["total_synced"] += ns_stats["synced"]
                sync_stats["total_skipped"] += ns_stats["skipped"]
                sync_stats["total_errors"] += ns_stats["errors"]
            
            if sync_stats["total_synced"] > 0:
                logger.info(f"Redis→PostgreSQL sync complete: {sync_stats}")
                
        except Exception as e:
            logger.error(f"Redis→PostgreSQL sync failed: {e}")
    
    def _get_all_namespaces_from_redis(self) -> List[str]:
        """从 Redis 获取所有命名空间"""
        try:
            redis = self.persistence.redis
            pattern = f"{redis.config.redis_key_prefix}:*:slots"
            keys = redis.client.keys(pattern)
            
            namespaces = []
            for key in keys:
                # 解析 namespace: aga:{namespace}:slots
                parts = key.split(":")
                if len(parts) >= 3:
                    namespaces.append(parts[1])
            
            return list(set(namespaces))
        except Exception as e:
            logger.warning(f"Failed to get namespaces from Redis: {e}")
            return []
    
    def _sync_namespace(
        self, 
        namespace: str, 
        redis: RedisPersistence, 
        postgres: PostgreSQLPersistence
    ) -> Dict[str, int]:
        """同步单个命名空间"""
        stats = {"synced": 0, "skipped": 0, "errors": 0}
        
        try:
            # 1. 获取 Redis 中的所有槽位
            redis_slots = redis.load_active_slots(namespace)
            # 也加载隔离的，确保状态同步
            redis_key = redis._slot_key(namespace)
            all_redis_data = redis.client.hgetall(redis_key)
            
            redis_slots_map = {}
            for lu_id, data_str in all_redis_data.items():
                slot_data = json.loads(data_str)
                redis_slots_map[lu_id] = slot_data
            
            # 2. 获取 PostgreSQL 中的版本映射
            pg_versions = self._get_pg_versions(namespace, postgres)
            
            # 3. 对比并同步
            for lu_id, redis_data in redis_slots_map.items():
                try:
                    redis_version = redis_data.get("version", 1)
                    redis_updated = redis_data.get("updated_at", 0)
                    
                    pg_info = pg_versions.get(lu_id, {"version": 0, "updated_at": 0})
                    pg_version = pg_info["version"]
                    
                    # 如果 Redis 版本更新，同步到 PostgreSQL
                    if redis_version > pg_version or redis_updated > pg_info["updated_at"]:
                        if self._sync_slot_to_postgres(namespace, redis_data, postgres):
                            stats["synced"] += 1
                        else:
                            stats["errors"] += 1
                    else:
                        stats["skipped"] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to sync slot {lu_id}: {e}")
                    stats["errors"] += 1
            
            # 4. 同步命中计数增量（高频数据）
            self._sync_hit_counts(namespace, redis_slots_map, postgres)
            
        except Exception as e:
            logger.error(f"Failed to sync namespace {namespace}: {e}")
            stats["errors"] += 1
        
        return stats
    
    def _get_pg_versions(
        self, 
        namespace: str, 
        postgres: PostgreSQLPersistence
    ) -> Dict[str, Dict[str, Any]]:
        """获取 PostgreSQL 中的版本信息"""
        try:
            from sqlalchemy import text
            
            with postgres._get_session() as session:
                sql = text("""
                SELECT lu_id, version, EXTRACT(EPOCH FROM updated_at) as updated_at
                FROM aga_slots WHERE namespace = :namespace
                """)
                result = session.execute(sql, {"namespace": namespace})
                
                return {
                    row.lu_id: {
                        "version": row.version,
                        "updated_at": row.updated_at or 0
                    }
                    for row in result.fetchall()
                }
        except Exception as e:
            logger.warning(f"Failed to get PG versions: {e}")
            return {}
    
    def _sync_slot_to_postgres(
        self, 
        namespace: str, 
        redis_data: Dict[str, Any], 
        postgres: PostgreSQLPersistence
    ) -> bool:
        """同步单个槽位到 PostgreSQL"""
        try:
            from sqlalchemy import text
            import torch
            
            # 构建 Slot 对象
            slot = Slot(
                slot_idx=redis_data.get("slot_idx", 0),
                lu_id=redis_data["lu_id"],
                key_vector=torch.tensor(redis_data["key_vector"]),
                value_vector=torch.tensor(redis_data["value_vector"]),
                lifecycle_state=LifecycleState(redis_data["lifecycle_state"]),
                namespace=namespace,
                condition=redis_data.get("condition"),
                decision=redis_data.get("decision"),
                hit_count=redis_data.get("hit_count", 0),
                consecutive_misses=redis_data.get("consecutive_misses", 0),
                last_hit_ts=redis_data.get("last_hit_ts", time.time()),
                created_at=redis_data.get("created_at", time.time()),
                updated_at=redis_data.get("updated_at", time.time()),
                version=redis_data.get("version", 1),
            )
            
            return postgres.save_slot(namespace, slot)
        except Exception as e:
            logger.warning(f"Failed to sync slot to PostgreSQL: {e}")
            return False
    
    def _sync_hit_counts(
        self, 
        namespace: str, 
        redis_slots: Dict[str, Dict[str, Any]], 
        postgres: PostgreSQLPersistence
    ):
        """同步命中计数到 PostgreSQL"""
        try:
            from sqlalchemy import text
            
            with postgres._get_session() as session:
                for lu_id, data in redis_slots.items():
                    redis_hit_count = data.get("hit_count", 0)
                    
                    # 更新 PostgreSQL 中的命中计数（取较大值）
                    sql = text("""
                    UPDATE aga_slots 
                    SET hit_count = GREATEST(hit_count, :hit_count),
                        last_hit_ts = COALESCE(
                            CASE WHEN :hit_count > hit_count 
                                 THEN to_timestamp(:last_hit_ts) 
                                 ELSE last_hit_ts 
                            END, 
                            last_hit_ts
                        )
                    WHERE namespace = :namespace AND lu_id = :lu_id
                    """)
                    session.execute(sql, {
                        "namespace": namespace,
                        "lu_id": lu_id,
                        "hit_count": redis_hit_count,
                        "last_hit_ts": data.get("last_hit_ts", time.time()),
                    })
        except Exception as e:
            logger.warning(f"Failed to sync hit counts: {e}")
    
    def save_slot(self, namespace: str, slot: Slot) -> bool:
        """保存槽位（带重试机制）"""
        for attempt in range(self._max_retries):
            try:
                return self.persistence.save_slot(namespace, slot)
            except Exception as e:
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (self._retry_backoff ** attempt)
                    logger.warning(
                        f"Save slot failed (attempt {attempt + 1}/{self._max_retries}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Save slot failed after {self._max_retries} attempts: {e}")
                    return False
        return False
    
    def load_active_slots(self, namespace: str) -> List[Dict[str, Any]]:
        """加载活跃槽位（带降级保护）"""
        # 优先从 Redis 加载
        if self.persistence.redis:
            try:
                redis_slots = self.persistence.redis.load_active_slots(namespace)
                if redis_slots:
                    logger.debug(f"Loaded {len(redis_slots)} slots from Redis")
                    return redis_slots
            except Exception as e:
                logger.warning(f"Failed to load from Redis: {e}, falling back to PostgreSQL")
        
        # 降级到 PostgreSQL
        if self.persistence.postgres:
            try:
                pg_slots = self.persistence.postgres.load_active_slots(namespace)
                logger.info(f"Loaded {len(pg_slots)} slots from PostgreSQL (fallback)")
                
                # 回填 Redis 缓存
                if self.persistence.redis and pg_slots:
                    self._backfill_redis(namespace, pg_slots)
                
                return pg_slots
            except Exception as e:
                logger.error(f"Failed to load from PostgreSQL: {e}")
        
        return []
    
    def _backfill_redis(self, namespace: str, slots_data: List[Dict[str, Any]]):
        """回填 Redis 缓存"""
        try:
            import torch
            backfilled = 0
            for slot_data in slots_data:
                try:
                    slot = Slot(
                        slot_idx=slot_data.get("slot_idx", 0),
                        lu_id=slot_data["lu_id"],
                        key_vector=torch.tensor(slot_data["key_vector"]),
                        value_vector=torch.tensor(slot_data["value_vector"]),
                        lifecycle_state=LifecycleState(slot_data["lifecycle_state"]),
                        namespace=namespace,
                        condition=slot_data.get("condition"),
                        decision=slot_data.get("decision"),
                        hit_count=slot_data.get("hit_count", 0),
                        consecutive_misses=slot_data.get("consecutive_misses", 0),
                        last_hit_ts=slot_data.get("last_hit_ts", time.time()),
                        created_at=slot_data.get("created_at", time.time()),
                        updated_at=slot_data.get("updated_at", time.time()),
                        version=slot_data.get("version", 1),
                    )
                    self.persistence.redis.save_slot(namespace, slot)
                    backfilled += 1
                except Exception as e:
                    logger.warning(f"Failed to backfill slot {slot_data.get('lu_id')}: {e}")
            
            if backfilled > 0:
                logger.info(f"Backfilled {backfilled} slots to Redis")
        except Exception as e:
            logger.warning(f"Failed to backfill Redis: {e}")
    
    def update_lifecycle(self, namespace: str, lu_id: str, new_state: str) -> bool:
        """更新生命周期"""
        return self.persistence.update_lifecycle(namespace, lu_id, new_state)
    
    def quarantine(self, namespace: str, lu_id: str) -> bool:
        """隔离槽位"""
        return self.persistence.update_lifecycle(namespace, lu_id, LifecycleState.QUARANTINED.value)
    
    def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """获取统计信息"""
        return self.persistence.get_statistics(namespace)

