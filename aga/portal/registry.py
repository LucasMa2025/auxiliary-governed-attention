"""
AGA Runtime 注册表

管理所有活跃的 Runtime 实例。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class RuntimeInfo:
    """Runtime 实例信息"""
    instance_id: str
    namespaces: List[str]
    host: Optional[str] = None
    port: Optional[int] = None
    
    # 状态
    status: str = "active"  # active, inactive, unhealthy
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    
    # 统计
    messages_received: int = 0
    last_message_at: Optional[float] = None
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "namespaces": self.namespaces,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "registered_at": self.registered_at,
            "messages_received": self.messages_received,
            "last_message_at": self.last_message_at,
            "metadata": self.metadata,
        }


class RuntimeRegistry:
    """
    Runtime 注册表
    
    跟踪所有活跃的 AGA Runtime 实例。
    
    功能：
    - 注册/注销 Runtime
    - 心跳检测
    - 获取活跃 Runtime 列表
    """
    
    def __init__(
        self,
        heartbeat_timeout: int = 90,
        cleanup_interval: int = 30,
    ):
        """
        初始化注册表
        
        Args:
            heartbeat_timeout: 心跳超时（秒）
            cleanup_interval: 清理间隔（秒）
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.cleanup_interval = cleanup_interval
        
        self._runtimes: Dict[str, RuntimeInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """启动注册表"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("RuntimeRegistry started")
    
    async def stop(self):
        """停止注册表"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("RuntimeRegistry stopped")
    
    async def _cleanup_loop(self):
        """定期清理不活跃的 Runtime"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_inactive()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_inactive(self):
        """清理不活跃的 Runtime"""
        now = time.time()
        inactive = []
        
        for instance_id, info in self._runtimes.items():
            if now - info.last_heartbeat > self.heartbeat_timeout:
                inactive.append(instance_id)
                info.status = "inactive"
        
        for instance_id in inactive:
            logger.warning(f"Runtime {instance_id} inactive, removing")
            del self._runtimes[instance_id]
    
    def register(
        self,
        instance_id: str,
        namespaces: List[str],
        host: Optional[str] = None,
        port: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        注册 Runtime
        
        Args:
            instance_id: 实例 ID
            namespaces: 支持的命名空间
            host: 主机地址（可选）
            port: 端口（可选）
            metadata: 元数据
        """
        info = RuntimeInfo(
            instance_id=instance_id,
            namespaces=namespaces,
            host=host,
            port=port,
            metadata=metadata,
        )
        self._runtimes[instance_id] = info
        logger.info(f"Runtime registered: {instance_id}, namespaces={namespaces}")
    
    def deregister(self, instance_id: str):
        """注销 Runtime"""
        if instance_id in self._runtimes:
            del self._runtimes[instance_id]
            logger.info(f"Runtime deregistered: {instance_id}")
    
    def heartbeat(self, instance_id: str):
        """
        更新心跳
        
        Args:
            instance_id: 实例 ID
        """
        if instance_id in self._runtimes:
            self._runtimes[instance_id].last_heartbeat = time.time()
            self._runtimes[instance_id].status = "active"
    
    def record_message(self, instance_id: str):
        """记录消息接收"""
        if instance_id in self._runtimes:
            self._runtimes[instance_id].messages_received += 1
            self._runtimes[instance_id].last_message_at = time.time()
    
    def get_runtime(self, instance_id: str) -> Optional[RuntimeInfo]:
        """获取 Runtime 信息"""
        return self._runtimes.get(instance_id)
    
    def get_all_runtimes(self) -> List[RuntimeInfo]:
        """获取所有 Runtime"""
        return list(self._runtimes.values())
    
    def get_active_runtimes(self) -> List[RuntimeInfo]:
        """获取活跃的 Runtime"""
        return [r for r in self._runtimes.values() if r.status == "active"]
    
    def get_runtimes_for_namespace(self, namespace: str) -> List[RuntimeInfo]:
        """获取支持指定命名空间的 Runtime"""
        return [
            r for r in self._runtimes.values()
            if r.status == "active" and namespace in r.namespaces
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self._runtimes)
        active = len([r for r in self._runtimes.values() if r.status == "active"])
        
        return {
            "total_runtimes": total,
            "active_runtimes": active,
            "inactive_runtimes": total - active,
            "runtimes": [r.to_dict() for r in self._runtimes.values()],
        }


class RedisRuntimeRegistry(RuntimeRegistry):
    """
    基于 Redis 的 Runtime 注册表
    
    支持分布式部署的多 Portal 实例。
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        key_prefix: str = "aga:runtime:",
        heartbeat_timeout: int = 90,
        cleanup_interval: int = 30,
    ):
        super().__init__(heartbeat_timeout, cleanup_interval)
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.key_prefix = key_prefix
        
        self._redis = None
    
    async def start(self):
        """启动注册表"""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError("需要安装 redis: pip install redis")
        
        self._redis = aioredis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True,
        )
        
        await super().start()
        logger.info(f"RedisRuntimeRegistry started: {self.redis_host}:{self.redis_port}")
    
    async def stop(self):
        """停止注册表"""
        await super().stop()
        if self._redis:
            await self._redis.close()
    
    def register(
        self,
        instance_id: str,
        namespaces: List[str],
        host: Optional[str] = None,
        port: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """注册到 Redis"""
        import json
        
        info = RuntimeInfo(
            instance_id=instance_id,
            namespaces=namespaces,
            host=host,
            port=port,
            metadata=metadata,
        )
        
        # 本地缓存
        self._runtimes[instance_id] = info
        
        # 存储到 Redis（异步任务）
        asyncio.create_task(self._redis_register(instance_id, info))
    
    async def _redis_register(self, instance_id: str, info: RuntimeInfo):
        """异步存储到 Redis"""
        import json
        
        key = f"{self.key_prefix}{instance_id}"
        value = json.dumps(info.to_dict())
        
        await self._redis.setex(key, self.heartbeat_timeout * 2, value)
    
    def heartbeat(self, instance_id: str):
        """更新心跳"""
        super().heartbeat(instance_id)
        
        # 更新 Redis TTL
        if instance_id in self._runtimes:
            asyncio.create_task(self._redis_heartbeat(instance_id))
    
    async def _redis_heartbeat(self, instance_id: str):
        """异步更新 Redis TTL"""
        key = f"{self.key_prefix}{instance_id}"
        await self._redis.expire(key, self.heartbeat_timeout * 2)
    
    async def sync_from_redis(self):
        """从 Redis 同步所有 Runtime"""
        import json
        
        pattern = f"{self.key_prefix}*"
        keys = await self._redis.keys(pattern)
        
        for key in keys:
            value = await self._redis.get(key)
            if value:
                data = json.loads(value)
                instance_id = data["instance_id"]
                self._runtimes[instance_id] = RuntimeInfo(
                    instance_id=instance_id,
                    namespaces=data["namespaces"],
                    host=data.get("host"),
                    port=data.get("port"),
                    status=data.get("status", "active"),
                    last_heartbeat=data.get("last_heartbeat", time.time()),
                    registered_at=data.get("registered_at", time.time()),
                    metadata=data.get("metadata"),
                )
