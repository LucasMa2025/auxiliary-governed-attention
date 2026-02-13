"""
AGA 实例协调器

管理多实例的注册、发现和健康检查。

版本: v3.0
"""
import asyncio
import json
import ast
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class InstanceInfo:
    """
    实例信息
    
    记录 AGA 实例的状态和元数据。
    """
    instance_id: str
    namespace: str
    host: str
    port: int
    
    # 状态
    status: str = "unknown"  # unknown, healthy, unhealthy, offline
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    
    # 能力
    capabilities: List[str] = field(default_factory=list)
    
    # 负载
    active_slots: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_healthy(self) -> bool:
        """是否健康"""
        return self.status == "healthy"
    
    @property
    def seconds_since_heartbeat(self) -> float:
        """距离上次心跳的秒数"""
        return time.time() - self.last_heartbeat
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "instance_id": self.instance_id,
            "namespace": self.namespace,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "registered_at": self.registered_at,
            "capabilities": self.capabilities,
            "active_slots": self.active_slots,
            "total_requests": self.total_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstanceInfo":
        """从字典创建"""
        return cls(
            instance_id=data.get("instance_id", ""),
            namespace=data.get("namespace", "default"),
            host=data.get("host", "unknown"),
            port=int(data.get("port", 0)),
            status=data.get("status", "unknown"),
            last_heartbeat=float(data.get("last_heartbeat", time.time())),
            registered_at=float(data.get("registered_at", time.time())),
            capabilities=list(data.get("capabilities") or []),
            active_slots=int(data.get("active_slots", 0)),
            total_requests=int(data.get("total_requests", 0)),
            avg_latency_ms=float(data.get("avg_latency_ms", 0.0)),
            metadata=data.get("metadata"),
        )


class InstanceCoordinator:
    """
    实例协调器
    
    功能：
    - 实例注册和发现
    - 健康检查
    - 负载均衡
    - 领导者选举（可选）
    """
    
    def __init__(
        self,
        instance_id: str,
        namespace: str = "default",
        backend: str = "redis",  # redis, etcd, consul
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化协调器
        
        Args:
            instance_id: 本实例 ID
            namespace: 命名空间
            backend: 后端类型
            config: 后端配置
        """
        self.instance_id = instance_id
        self.namespace = namespace
        self.backend = backend
        self.config = config or {}
        
        # 实例注册表
        self._instances: Dict[str, InstanceInfo] = {}
        
        # 本实例信息
        self._self_info: Optional[InstanceInfo] = None
        
        # 状态
        self._running = False
        self._client = None
        
        # 配置
        self.heartbeat_interval = self.config.get("heartbeat_interval", 10)
        self.instance_timeout = self.config.get("instance_timeout", 30)
        
        # 回调
        self._on_instance_join: Optional[Callable] = None
        self._on_instance_leave: Optional[Callable] = None
        self._on_leader_change: Optional[Callable] = None
    
    async def start(
        self,
        host: str = "localhost",
        port: int = 8000,
        capabilities: Optional[List[str]] = None,
    ) -> bool:
        """
        启动协调器
        
        Args:
            host: 本实例主机
            port: 本实例端口
            capabilities: 本实例能力
        """
        try:
            # 初始化后端
            if self.backend == "redis":
                await self._init_redis()
            
            # 创建本实例信息
            self._self_info = InstanceInfo(
                instance_id=self.instance_id,
                namespace=self.namespace,
                host=host,
                port=port,
                status="healthy",
                capabilities=capabilities or [],
            )
            
            # 注册本实例
            await self._register_self()
            
            self._running = True
            
            # 启动后台任务
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._discovery_loop())
            
            logger.info(f"Coordinator started: instance={self.instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start coordinator: {e}")
            return False
    
    async def stop(self):
        """停止协调器"""
        if not self._running:
            return
        
        self._running = False
        
        # 注销本实例
        await self._deregister_self()
        
        if self._client:
            await self._client.close()
        
        logger.info(f"Coordinator stopped: instance={self.instance_id}")
    
    async def _init_redis(self):
        """初始化 Redis 后端"""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError("Redis backend requires 'redis' package")
        
        self._client = aioredis.Redis(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 6379),
            db=self.config.get("db", 0),
            password=self.config.get("password"),
        )
    
    async def _register_self(self):
        """注册本实例"""
        if self.backend == "redis":
            key = f"aga:{self.namespace}:instances:{self.instance_id}"
            payload = json.dumps(self._self_info.to_dict(), ensure_ascii=True)
            await self._client.set(
                key,
                payload,
                ex=self.instance_timeout * 2,
            )
    
    async def _deregister_self(self):
        """注销本实例"""
        if self.backend == "redis":
            key = f"aga:{self.namespace}:instances:{self.instance_id}"
            await self._client.delete(key)
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            try:
                if self._self_info:
                    self._self_info.last_heartbeat = time.time()
                    await self._register_self()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _discovery_loop(self):
        """发现循环"""
        while self._running:
            try:
                await self._discover_instances()
                await self._check_instance_health()
            except Exception as e:
                logger.error(f"Discovery error: {e}")
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _discover_instances(self):
        """发现其他实例"""
        if self.backend == "redis":
            pattern = f"aga:{self.namespace}:instances:*"
            keys = await self._client.keys(pattern)
            
            for key in keys:
                instance_id = key.decode().split(":")[-1]
                if instance_id == self.instance_id:
                    continue
                
                data = await self._client.get(key)
                if data:
                    info = self._parse_instance_data(data)
                    if info is None:
                        # 无法解析时保底登记
                        info = InstanceInfo(
                            instance_id=instance_id,
                            namespace=self.namespace,
                            host="unknown",
                            port=0,
                            status="healthy",
                        )
                    
                    # 更新/新增实例
                    existing = self._instances.get(instance_id)
                    if existing:
                        existing.host = info.host
                        existing.port = info.port
                        existing.status = info.status
                        existing.capabilities = info.capabilities
                        existing.active_slots = info.active_slots
                        existing.total_requests = info.total_requests
                        existing.avg_latency_ms = info.avg_latency_ms
                        existing.metadata = info.metadata
                        existing.last_heartbeat = info.last_heartbeat
                    else:
                        self._instances[instance_id] = info
                        if self._on_instance_join:
                            await self._on_instance_join(info)

    def _parse_instance_data(self, data: Any) -> Optional[InstanceInfo]:
        """解析实例信息，兼容旧格式"""
        try:
            text = data.decode() if isinstance(data, (bytes, bytearray)) else str(data)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = ast.literal_eval(text)
            if isinstance(payload, dict):
                return InstanceInfo.from_dict(payload)
        except Exception as e:
            logger.debug(f"Failed to parse instance data: {e}")
        return None
    
    async def _check_instance_health(self):
        """检查实例健康状态"""
        now = time.time()
        offline_instances = []
        
        for instance_id, info in self._instances.items():
            if info.seconds_since_heartbeat > self.instance_timeout:
                info.status = "offline"
                offline_instances.append(instance_id)
        
        # 移除离线实例
        for instance_id in offline_instances:
            info = self._instances.pop(instance_id)
            if self._on_instance_leave:
                await self._on_instance_leave(info)
    
    # ==================== 公共接口 ====================
    
    def get_instances(self) -> List[InstanceInfo]:
        """获取所有实例"""
        return list(self._instances.values())
    
    def get_healthy_instances(self) -> List[InstanceInfo]:
        """获取健康实例"""
        return [i for i in self._instances.values() if i.is_healthy]
    
    def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        """获取指定实例"""
        return self._instances.get(instance_id)
    
    def get_self_info(self) -> Optional[InstanceInfo]:
        """获取本实例信息"""
        return self._self_info
    
    def update_self_stats(
        self,
        active_slots: int = None,
        total_requests: int = None,
        avg_latency_ms: float = None,
    ):
        """更新本实例统计"""
        if self._self_info:
            if active_slots is not None:
                self._self_info.active_slots = active_slots
            if total_requests is not None:
                self._self_info.total_requests = total_requests
            if avg_latency_ms is not None:
                self._self_info.avg_latency_ms = avg_latency_ms
    
    def on_instance_join(self, callback: Callable):
        """注册实例加入回调"""
        self._on_instance_join = callback
    
    def on_instance_leave(self, callback: Callable):
        """注册实例离开回调"""
        self._on_instance_leave = callback
    
    def on_leader_change(self, callback: Callable):
        """注册领导者变更回调"""
        self._on_leader_change = callback
