"""
AGA 同步协议配置

定义 Portal 与 Runtime 之间的同步协议配置。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RedisConfig:
    """Redis 配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    # 发布/订阅
    channel_prefix: str = "aga:sync"
    
    # 连接池
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    # 重试
    retry_on_timeout: bool = True
    retry_max_attempts: int = 3
    
    @property
    def url(self) -> str:
        """生成 Redis URL"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class KafkaConfig:
    """Kafka 配置"""
    bootstrap_servers: str = "localhost:9092"
    topic: str = "aga-sync"
    group_id: str = "aga-consumers"
    
    # 生产者配置
    acks: str = "all"
    retries: int = 3
    batch_size: int = 16384
    linger_ms: int = 10
    
    # 消费者配置
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    
    # 安全配置
    security_protocol: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None


@dataclass
class SyncConfig:
    """
    同步协议配置
    
    支持多种后端：
    - redis: Redis Pub/Sub（推荐，低延迟）
    - kafka: Kafka（适合大规模部署）
    - memory: 内存队列（仅测试用）
    """
    backend: str = "redis"  # redis, kafka, memory
    
    # 后端配置
    redis: RedisConfig = field(default_factory=RedisConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    
    # 消息配置
    message_timeout: int = 30  # 消息超时（秒）
    max_message_size: int = 10 * 1024 * 1024  # 最大消息大小（10MB）
    
    # 批量配置
    batch_enabled: bool = True
    batch_size: int = 100
    batch_timeout: float = 1.0  # 秒
    
    # 确认机制
    require_ack: bool = True
    ack_timeout: int = 10  # 秒
    
    # 重试配置
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_delay: float = 1.0  # 秒
    retry_exponential_backoff: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            else:
                return obj
        
        return _to_dict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncConfig":
        """从字典创建"""
        return cls(
            backend=data.get("backend", "redis"),
            redis=RedisConfig(**data.get("redis", {})),
            kafka=KafkaConfig(**data.get("kafka", {})),
            message_timeout=data.get("message_timeout", 30),
            max_message_size=data.get("max_message_size", 10 * 1024 * 1024),
            batch_enabled=data.get("batch_enabled", True),
            batch_size=data.get("batch_size", 100),
            batch_timeout=data.get("batch_timeout", 1.0),
            require_ack=data.get("require_ack", True),
            ack_timeout=data.get("ack_timeout", 10),
            retry_enabled=data.get("retry_enabled", True),
            retry_max_attempts=data.get("retry_max_attempts", 3),
            retry_delay=data.get("retry_delay", 1.0),
            retry_exponential_backoff=data.get("retry_exponential_backoff", True),
        )
