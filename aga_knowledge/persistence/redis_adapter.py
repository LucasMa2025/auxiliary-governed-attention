"""
aga-knowledge Redis 持久化适配器

明文 KV 版本：用于热缓存层，支持高性能读写。
不存储向量数据，只存储 condition/decision 文本对。

特性:
- 高性能读写（Pipeline 批量操作）
- TTL 自动过期
- Pub/Sub 支持（用于分布式同步）
- Lua 脚本原子操作
- 审计日志（List + TTL）

依赖:
    pip install redis

配置示例:
    persistence:
      type: "redis"
      redis_host: "localhost"
      redis_port: 6379
      redis_db: 0
      redis_password: null
      redis_key_prefix: "aga_knowledge"
      redis_ttl_days: 30
      redis_pool_size: 10
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
    import redis.asyncio as aioredis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class RedisAdapter(PersistenceAdapter):
    """
    Redis 持久化适配器（明文 KV 版本）

    特性:
    - 高性能读写
    - TTL 自动过期
    - Pub/Sub 支持（用于分布式同步）
    - Pipeline 批量操作
    - Lua 脚本原子操作
    - 审计日志
    - 不存储向量数据
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "aga_knowledge",
        ttl_days: int = 30,
        pool_size: int = 10,
        enable_audit: bool = True,
        audit_max_entries: int = 1000,
        audit_ttl_days: int = 30,
        url: Optional[str] = None,
    ):
        """
        初始化 Redis 适配器

        Args:
            host: Redis 主机
            port: Redis 端口
            db: 数据库编号
            password: 密码
            key_prefix: 键前缀
            ttl_days: 知识 TTL（天），0 表示不过期
            pool_size: 连接池大小
            enable_audit: 是否启用审计日志
            audit_max_entries: 每个命名空间最大审计日志条数
            audit_ttl_days: 审计日志 TTL（天）
            url: Redis URL（优先于单独参数）
        """
        if not _HAS_REDIS:
            raise ImportError(
                "Redis adapter 需要 redis 包。\n"
                "请运行: pip install redis"
            )

        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_days * 86400 if ttl_days > 0 else 0
        self.pool_size = pool_size
        self.enable_audit = enable_audit
        self.audit_max_entries = audit_max_entries
        self.audit_ttl_seconds = audit_ttl_days * 86400
        self.url = url

        self._client: Optional[aioredis.Redis] = None
        self._connected = False

    # ==================== 键名生成 ====================

    def _make_key(self, namespace: str, lu_id: str) -> str:
        """生成知识键"""
        return f"{self.key_prefix}:{namespace}:knowledge:{lu_id}"

    def _make_index_key(self, namespace: str) -> str:
        """生成索引键（Set 存储所有 lu_id）"""
        return f"{self.key_prefix}:{namespace}:index"

    def _make_audit_key(self, namespace: str) -> str:
        """生成审计日志键"""
        return f"{self.key_prefix}:{namespace}:audit"

    # ==================== 连接管理 ====================

    async def connect(self) -> bool:
        try:
            if self.url:
                self._client = aioredis.from_url(
                    self.url,
                    max_connections=self.pool_size,
                    decode_responses=True,
                )
            else:
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
            logger.info(
                f"Connected to Redis: "
                f"{self.url or f'{self.host}:{self.port}/{self.db}'}"
            )
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
        logger.info("Disconnected from Redis")

    async def is_connected(self) -> bool:
        if not self._client:
            return False
        try:
            await self._client.ping()
            return True
        except Exception:
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
                "db": self.db,
                "redis_version": info.get("redis_version"),
            }
        except Exception as e:
            return {"status": "unhealthy", "adapter": "redis", "error": str(e)}

    # ==================== 知识 CRUD ====================

    async def save_knowledge(
        self, namespace: str, lu_id: str, data: Dict[str, Any]
    ) -> bool:
        if not self._client:
            return False

        try:
            key = self._make_key(namespace, lu_id)
            index_key = self._make_index_key(namespace)
            now = datetime.utcnow().isoformat()

            # 读取已有记录以保留 created_at 和 version
            existing_raw = await self._client.get(key)
            existing = json.loads(existing_raw) if existing_raw else None

            record = {
                "lu_id": lu_id,
                "namespace": namespace,
                "condition": data.get("condition", ""),
                "decision": data.get("decision", ""),
                "lifecycle_state": data.get("lifecycle_state", "probationary"),
                "trust_tier": data.get("trust_tier", "standard"),
                "hit_count": data.get(
                    "hit_count",
                    existing.get("hit_count", 0) if existing else 0,
                ),
                "consecutive_misses": data.get("consecutive_misses", 0),
                "version": (
                    (existing.get("version", 0) + 1) if existing else 1
                ),
                "created_at": (
                    existing.get("created_at", now) if existing else now
                ),
                "updated_at": now,
                "metadata": data.get("metadata"),
            }

            async with self._client.pipeline() as pipe:
                if self.ttl_seconds > 0:
                    pipe.set(key, json.dumps(record), ex=self.ttl_seconds)
                else:
                    pipe.set(key, json.dumps(record))
                pipe.sadd(index_key, lu_id)
                await pipe.execute()

            if self.enable_audit:
                await self._log_audit(
                    namespace, lu_id, "SAVE",
                    new_state=data.get("lifecycle_state", "probationary"),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to save knowledge to Redis: {e}")
            return False

    async def load_knowledge(
        self, namespace: str, lu_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None

        try:
            key = self._make_key(namespace, lu_id)
            raw = await self._client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Failed to load knowledge from Redis: {e}")
            return None

    async def delete_knowledge(self, namespace: str, lu_id: str) -> bool:
        if not self._client:
            return False

        try:
            key = self._make_key(namespace, lu_id)
            index_key = self._make_index_key(namespace)

            async with self._client.pipeline() as pipe:
                pipe.delete(key)
                pipe.srem(index_key, lu_id)
                await pipe.execute()

            if self.enable_audit:
                await self._log_audit(namespace, lu_id, "DELETE")

            return True
        except Exception as e:
            logger.error(f"Failed to delete knowledge from Redis: {e}")
            return False

    async def knowledge_exists(self, namespace: str, lu_id: str) -> bool:
        if not self._client:
            return False

        try:
            key = self._make_key(namespace, lu_id)
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.error(f"Failed to check knowledge existence: {e}")
            return False

    # ==================== 批量操作 ====================

    async def save_batch(
        self, namespace: str, records: List[Dict[str, Any]]
    ) -> int:
        if not self._client or not records:
            return 0

        try:
            index_key = self._make_index_key(namespace)
            now = datetime.utcnow().isoformat()
            count = 0

            async with self._client.pipeline() as pipe:
                for data in records:
                    lu_id = data.get("lu_id", "")
                    if not lu_id:
                        continue

                    key = self._make_key(namespace, lu_id)

                    record = {
                        "lu_id": lu_id,
                        "namespace": namespace,
                        "condition": data.get("condition", ""),
                        "decision": data.get("decision", ""),
                        "lifecycle_state": data.get(
                            "lifecycle_state", "probationary"
                        ),
                        "trust_tier": data.get("trust_tier", "standard"),
                        "hit_count": data.get("hit_count", 0),
                        "consecutive_misses": data.get(
                            "consecutive_misses", 0
                        ),
                        "version": 1,
                        "created_at": now,
                        "updated_at": now,
                        "metadata": data.get("metadata"),
                    }

                    if self.ttl_seconds > 0:
                        pipe.set(key, json.dumps(record), ex=self.ttl_seconds)
                    else:
                        pipe.set(key, json.dumps(record))
                    pipe.sadd(index_key, lu_id)
                    count += 1

                await pipe.execute()

            if self.enable_audit:
                await self._log_audit(
                    namespace, None, "BATCH_SAVE",
                    details=f"count={count}",
                )

            return count
        except Exception as e:
            logger.error(f"Failed to save batch to Redis: {e}")
            return 0

    async def load_active_knowledge(
        self, namespace: str
    ) -> List[Dict[str, Any]]:
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
            for raw in values:
                if raw:
                    record = json.loads(raw)
                    if record.get("lifecycle_state") != LifecycleState.QUARANTINED.value:
                        records.append(record)

            return records
        except Exception as e:
            logger.error(f"Failed to load active knowledge from Redis: {e}")
            return []

    async def load_all_knowledge(
        self, namespace: str
    ) -> List[Dict[str, Any]]:
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
            for raw in values:
                if raw:
                    records.append(json.loads(raw))

            return records
        except Exception as e:
            logger.error(f"Failed to load all knowledge from Redis: {e}")
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
        if not self._client:
            return []

        try:
            # Redis 不支持复杂查询，需要加载全部后过滤
            all_records = await self.load_all_knowledge(namespace)

            if lifecycle_states:
                all_records = [
                    r for r in all_records
                    if r.get("lifecycle_state") in lifecycle_states
                ]
            if trust_tiers:
                all_records = [
                    r for r in all_records
                    if r.get("trust_tier") in trust_tiers
                ]

            # 排序
            all_records.sort(key=lambda r: r.get("lu_id", ""))

            return all_records[offset:offset + limit]
        except Exception as e:
            logger.error(f"Failed to query knowledge from Redis: {e}")
            return []

    # ==================== 生命周期管理 ====================

    async def update_lifecycle(
        self, namespace: str, lu_id: str, new_state: str
    ) -> bool:
        if not self._client:
            return False

        try:
            key = self._make_key(namespace, lu_id)
            raw = await self._client.get(key)
            if not raw:
                return False

            record = json.loads(raw)
            old_state = record.get("lifecycle_state")
            record["lifecycle_state"] = new_state
            record["updated_at"] = datetime.utcnow().isoformat()

            if self.ttl_seconds > 0:
                await self._client.set(
                    key, json.dumps(record), ex=self.ttl_seconds
                )
            else:
                await self._client.set(key, json.dumps(record))

            if self.enable_audit:
                await self._log_audit(
                    namespace, lu_id, "UPDATE_LIFECYCLE",
                    old_state=old_state, new_state=new_state,
                )

            return True
        except Exception as e:
            logger.error(f"Failed to update lifecycle in Redis: {e}")
            return False

    async def update_trust_tier(
        self, namespace: str, lu_id: str, new_tier: str
    ) -> bool:
        if not self._client:
            return False

        try:
            key = self._make_key(namespace, lu_id)
            raw = await self._client.get(key)
            if not raw:
                return False

            record = json.loads(raw)
            record["trust_tier"] = new_tier
            record["updated_at"] = datetime.utcnow().isoformat()

            if self.ttl_seconds > 0:
                await self._client.set(
                    key, json.dumps(record), ex=self.ttl_seconds
                )
            else:
                await self._client.set(key, json.dumps(record))

            if self.enable_audit:
                await self._log_audit(
                    namespace, lu_id, "UPDATE_TRUST_TIER",
                    new_state=new_tier,
                )

            return True
        except Exception as e:
            logger.error(f"Failed to update trust tier in Redis: {e}")
            return False

    # ==================== 统计 ====================

    async def get_knowledge_count(
        self, namespace: str, state: Optional[str] = None
    ) -> int:
        if not self._client:
            return 0

        try:
            if state is None:
                index_key = self._make_index_key(namespace)
                return await self._client.scard(index_key)

            # 需要遍历检查状态
            records = await self.load_all_knowledge(namespace)
            return sum(
                1 for r in records
                if r.get("lifecycle_state") == state
            )
        except Exception as e:
            logger.error(f"Failed to get knowledge count from Redis: {e}")
            return 0

    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        if not self._client:
            return {"namespace": namespace, "error": "Not connected"}

        try:
            records = await self.load_all_knowledge(namespace)

            state_dist: Dict[str, int] = {}
            total_hits = 0
            for record in records:
                state = record.get("lifecycle_state", "unknown")
                state_dist[state] = state_dist.get(state, 0) + 1
                total_hits += record.get("hit_count", 0)

            return {
                "namespace": namespace,
                "total_knowledge": len(records),
                "state_distribution": state_dist,
                "total_hits": total_hits,
                "adapter": "redis",
                "host": self.host,
                "port": self.port,
                "db": self.db,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics from Redis: {e}")
            return {"namespace": namespace, "error": str(e)}

    # Lua 脚本：原子地对多个知识键批量 +1 hit_count
    _INCREMENT_HIT_LUA = """
    local now = ARGV[1]
    local ttl = tonumber(ARGV[2])
    local updated = 0
    for i = 1, #KEYS do
        local raw = redis.call('GET', KEYS[i])
        if raw then
            local record = cjson.decode(raw)
            record['hit_count'] = (record['hit_count'] or 0) + 1
            record['updated_at'] = now
            if ttl > 0 then
                redis.call('SET', KEYS[i], cjson.encode(record), 'EX', ttl)
            else
                redis.call('SET', KEYS[i], cjson.encode(record))
            end
            updated = updated + 1
        end
    end
    return updated
    """

    async def increment_hit_count(
        self, namespace: str, lu_ids: List[str]
    ) -> bool:
        if not self._client or not lu_ids:
            return True

        try:
            keys = [self._make_key(namespace, lu_id) for lu_id in lu_ids]
            now = datetime.utcnow().isoformat()

            updated = await self._client.eval(
                self._INCREMENT_HIT_LUA,
                len(keys),
                *keys,
                now,
                self.ttl_seconds,
            )
            logger.debug(
                f"increment_hit_count: {updated}/{len(lu_ids)} records updated"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to increment hit count in Redis: {e}")
            return False

    # ==================== 命名空间 ====================

    async def get_namespaces(self) -> List[str]:
        if not self._client:
            return []

        try:
            pattern = f"{self.key_prefix}:*:index"
            namespaces = []

            async for key in self._client.scan_iter(match=pattern):
                # 从键中提取命名空间: prefix:namespace:index
                parts = key.split(":")
                if len(parts) >= 3:
                    namespaces.append(parts[1])

            return list(set(namespaces))
        except Exception as e:
            logger.error(f"Failed to get namespaces from Redis: {e}")
            return []

    # ==================== 审计日志 ====================

    async def _log_audit(
        self,
        namespace: str,
        lu_id: Optional[str],
        action: str,
        old_state: Optional[str] = None,
        new_state: Optional[str] = None,
        details: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """记录审计日志"""
        if not self.enable_audit or not self._client:
            return

        try:
            audit_key = self._make_audit_key(namespace)
            entry = {
                "namespace": namespace,
                "lu_id": lu_id,
                "action": action,
                "old_state": old_state,
                "new_state": new_state,
                "details": details,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }

            async with self._client.pipeline() as pipe:
                pipe.lpush(audit_key, json.dumps(entry))
                pipe.ltrim(audit_key, 0, self.audit_max_entries - 1)
                if self.audit_ttl_seconds > 0:
                    pipe.expire(audit_key, self.audit_ttl_seconds)
                await pipe.execute()
        except Exception as e:
            logger.warning(f"Failed to log audit to Redis: {e}")

    async def save_audit_log(self, entry: Dict[str, Any]) -> bool:
        if not self._client or not self.enable_audit:
            return True

        try:
            namespace = entry.get("namespace", "default")
            await self._log_audit(
                namespace=namespace,
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
            logger.error(f"Failed to save audit log to Redis: {e}")
            return False

    async def query_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if not self._client or not self.enable_audit:
            return []

        try:
            if namespace:
                audit_key = self._make_audit_key(namespace)
                if lu_id:
                    # 需要过滤 lu_id，先获取足够多的条目再过滤
                    entries_raw = await self._client.lrange(
                        audit_key, 0, (offset + limit) * 3
                    )
                else:
                    entries_raw = await self._client.lrange(
                        audit_key, offset, offset + limit - 1
                    )
            else:
                # 获取所有命名空间的审计日志
                namespaces = await self.get_namespaces()
                entries_raw = []
                for ns in namespaces:
                    audit_key = self._make_audit_key(ns)
                    ns_entries = await self._client.lrange(
                        audit_key, 0, (offset + limit) * 2
                    )
                    entries_raw.extend(ns_entries)

            # 解析并过滤
            result = []
            for entry_str in entries_raw:
                try:
                    entry = json.loads(entry_str)
                    if lu_id and entry.get("lu_id") != lu_id:
                        continue
                    result.append(entry)
                except json.JSONDecodeError:
                    continue

            if namespace and not lu_id:
                return result

            # 排序并应用分页
            result.sort(
                key=lambda x: x.get("timestamp", ""), reverse=True
            )
            return result[offset:offset + limit]

        except Exception as e:
            logger.error(f"Failed to query audit log from Redis: {e}")
            return []

    # ==================== Pub/Sub 支持 ====================

    async def publish_event(
        self, channel: str, event: Dict[str, Any]
    ) -> bool:
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

    # ==================== 便捷方法 ====================

    @classmethod
    async def from_url(cls, url: str, **kwargs) -> "RedisAdapter":
        """
        从 URL 创建适配器

        Args:
            url: Redis URL (redis://host:port/db)
            **kwargs: 其他参数

        Returns:
            已连接的 RedisAdapter 实例
        """
        adapter = cls(url=url, **kwargs)
        await adapter.connect()
        return adapter
