"""
AGA Redis 持久化适配器

用于热缓存层 (L1)，支持高性能读写。

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

# 尝试导入 redis
try:
    import redis.asyncio as aioredis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class RedisAdapter(PersistenceAdapter):
    """
    Redis 持久化适配器
    
    特性：
    - 高性能读写
    - TTL 自动过期
    - Pub/Sub 支持（用于分布式同步）
    - 管道批量操作
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "aga",
        slot_ttl_days: int = 7,
        pool_size: int = 10,
    ):
        """
        初始化 Redis 适配器
        
        Args:
            host: Redis 主机
            port: Redis 端口
            db: 数据库编号
            password: 密码
            key_prefix: 键前缀
            slot_ttl_days: 槽位 TTL（天）
            pool_size: 连接池大小
        """
        if not _HAS_REDIS:
            raise ImportError("Redis adapter requires 'redis' package")
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.slot_ttl_seconds = slot_ttl_days * 86400
        self.pool_size = pool_size
        
        self._client: Optional[aioredis.Redis] = None
        self._connected = False
    
    def _make_key(self, namespace: str, lu_id: str) -> str:
        """生成 Redis 键"""
        return f"{self.key_prefix}:{namespace}:slot:{lu_id}"
    
    def _make_index_key(self, namespace: str) -> str:
        """生成索引键"""
        return f"{self.key_prefix}:{namespace}:index"
    
    def _make_stats_key(self, namespace: str) -> str:
        """生成统计键"""
        return f"{self.key_prefix}:{namespace}:stats"
    
    # ==================== 连接管理 ====================
    
    async def connect(self) -> bool:
        try:
            self._client = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.pool_size,
                decode_responses=True,
            )
            
            # 测试连接
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        if self._client:
            await self._client.close()
            self._client = None
        self._connected = False
    
    async def is_connected(self) -> bool:
        if not self._client:
            return False
        try:
            await self._client.ping()
            return True
        except:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            if not self._client:
                return {"status": "disconnected", "adapter": "redis"}
            
            info = await self._client.info("server")
            return {
                "status": "healthy",
                "adapter": "redis",
                "host": self.host,
                "port": self.port,
                "redis_version": info.get("redis_version"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "adapter": "redis",
                "error": str(e),
            }
    
    # ==================== 槽位操作 ====================
    
    async def save_slot(self, namespace: str, record: KnowledgeRecord) -> bool:
        if not self._client:
            return False
        
        try:
            key = self._make_key(namespace, record.lu_id)
            index_key = self._make_index_key(namespace)
            
            # 更新时间戳
            now = datetime.now().isoformat()
            if record.created_at is None:
                record.created_at = now
            record.updated_at = now
            
            # 序列化
            data = record.to_json()
            
            # 使用管道
            async with self._client.pipeline() as pipe:
                # 保存数据
                pipe.set(key, data, ex=self.slot_ttl_seconds)
                # 添加到索引
                pipe.sadd(index_key, record.lu_id)
                await pipe.execute()
            
            return True
        except Exception as e:
            logger.error(f"Failed to save slot to Redis: {e}")
            return False
    
    async def load_slot(self, namespace: str, lu_id: str) -> Optional[KnowledgeRecord]:
        if not self._client:
            return None
        
        try:
            key = self._make_key(namespace, lu_id)
            data = await self._client.get(key)
            
            if not data:
                return None
            
            return KnowledgeRecord.from_json(data)
        except Exception as e:
            logger.error(f"Failed to load slot from Redis: {e}")
            return None
    
    async def delete_slot(self, namespace: str, lu_id: str) -> bool:
        if not self._client:
            return False
        
        try:
            key = self._make_key(namespace, lu_id)
            index_key = self._make_index_key(namespace)
            
            async with self._client.pipeline() as pipe:
                pipe.delete(key)
                pipe.srem(index_key, lu_id)
                await pipe.execute()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete slot from Redis: {e}")
            return False
    
    async def slot_exists(self, namespace: str, lu_id: str) -> bool:
        if not self._client:
            return False
        
        try:
            key = self._make_key(namespace, lu_id)
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.error(f"Failed to check slot existence: {e}")
            return False
    
    # ==================== 批量操作 ====================
    
    async def save_batch(self, namespace: str, records: List[KnowledgeRecord]) -> int:
        if not self._client or not records:
            return 0
        
        try:
            index_key = self._make_index_key(namespace)
            now = datetime.now().isoformat()
            
            async with self._client.pipeline() as pipe:
                for record in records:
                    if record.created_at is None:
                        record.created_at = now
                    record.updated_at = now
                    
                    key = self._make_key(namespace, record.lu_id)
                    data = record.to_json()
                    
                    pipe.set(key, data, ex=self.slot_ttl_seconds)
                    pipe.sadd(index_key, record.lu_id)
                
                await pipe.execute()
            
            return len(records)
        except Exception as e:
            logger.error(f"Failed to save batch to Redis: {e}")
            return 0
    
    async def load_active_slots(self, namespace: str) -> List[KnowledgeRecord]:
        if not self._client:
            return []
        
        try:
            index_key = self._make_index_key(namespace)
            lu_ids = await self._client.smembers(index_key)
            
            if not lu_ids:
                return []
            
            # 批量获取
            keys = [self._make_key(namespace, lu_id) for lu_id in lu_ids]
            values = await self._client.mget(keys)
            
            records = []
            for data in values:
                if data:
                    record = KnowledgeRecord.from_json(data)
                    if record.lifecycle_state != LifecycleState.QUARANTINED.value:
                        records.append(record)
            
            return records
        except Exception as e:
            logger.error(f"Failed to load active slots from Redis: {e}")
            return []
    
    async def load_all_slots(self, namespace: str) -> List[KnowledgeRecord]:
        if not self._client:
            return []
        
        try:
            index_key = self._make_index_key(namespace)
            lu_ids = await self._client.smembers(index_key)
            
            if not lu_ids:
                return []
            
            keys = [self._make_key(namespace, lu_id) for lu_id in lu_ids]
            values = await self._client.mget(keys)
            
            records = []
            for data in values:
                if data:
                    records.append(KnowledgeRecord.from_json(data))
            
            return records
        except Exception as e:
            logger.error(f"Failed to load all slots from Redis: {e}")
            return []
    
    # ==================== 生命周期管理 ====================
    
    async def update_lifecycle(
        self, 
        namespace: str, 
        lu_id: str, 
        new_state: LifecycleState
    ) -> bool:
        if not self._client:
            return False
        
        try:
            record = await self.load_slot(namespace, lu_id)
            if not record:
                return False
            
            record.lifecycle_state = new_state.value
            record.updated_at = datetime.now().isoformat()
            
            return await self.save_slot(namespace, record)
        except Exception as e:
            logger.error(f"Failed to update lifecycle in Redis: {e}")
            return False
    
    async def update_lifecycle_batch(
        self,
        namespace: str,
        updates: List[tuple]
    ) -> int:
        count = 0
        for lu_id, new_state in updates:
            if await self.update_lifecycle(namespace, lu_id, new_state):
                count += 1
        return count
    
    # ==================== 统计查询 ====================
    
    async def get_slot_count(
        self, 
        namespace: str, 
        state: Optional[LifecycleState] = None
    ) -> int:
        if not self._client:
            return 0
        
        try:
            if state is None:
                index_key = self._make_index_key(namespace)
                return await self._client.scard(index_key)
            
            # 需要遍历检查状态
            records = await self.load_all_slots(namespace)
            return sum(1 for r in records if r.lifecycle_state == state.value)
        except Exception as e:
            logger.error(f"Failed to get slot count from Redis: {e}")
            return 0
    
    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        if not self._client:
            return {"namespace": namespace, "error": "Not connected"}
        
        try:
            records = await self.load_all_slots(namespace)
            
            state_dist = {}
            total_hits = 0
            for record in records:
                state = record.lifecycle_state
                state_dist[state] = state_dist.get(state, 0) + 1
                total_hits += record.hit_count
            
            return {
                "namespace": namespace,
                "total_slots": len(records),
                "state_distribution": state_dist,
                "total_hits": total_hits,
                "adapter": "redis",
                "host": self.host,
                "port": self.port,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics from Redis: {e}")
            return {"namespace": namespace, "error": str(e)}
    
    # ==================== 命中计数 ====================
    
    async def increment_hit_count(
        self, 
        namespace: str, 
        lu_ids: List[str]
    ) -> bool:
        if not self._client or not lu_ids:
            return True
        
        try:
            for lu_id in lu_ids:
                record = await self.load_slot(namespace, lu_id)
                if record:
                    record.hit_count += 1
                    await self.save_slot(namespace, record)
            return True
        except Exception as e:
            logger.error(f"Failed to increment hit count in Redis: {e}")
            return False
    
    # ==================== Pub/Sub 支持 ====================
    
    async def publish_event(self, channel: str, event: Dict[str, Any]) -> bool:
        """发布事件（用于分布式同步）"""
        if not self._client:
            return False
        
        try:
            await self._client.publish(channel, json.dumps(event))
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    async def subscribe(self, channel: str):
        """订阅频道"""
        if not self._client:
            return None
        
        pubsub = self._client.pubsub()
        await pubsub.subscribe(channel)
        return pubsub
