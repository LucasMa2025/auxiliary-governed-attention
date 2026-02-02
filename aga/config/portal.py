"""
AGA Portal 配置

Portal 是无 GPU 依赖的 API 服务，负责：
- 知识元数据管理
- 生命周期状态管理
- 同步消息发布到 Runtime
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8081
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class PersistenceDBConfig:
    """持久化数据库配置"""
    type: str = "sqlite"  # sqlite, postgres
    # SQLite
    sqlite_path: str = "aga_portal.db"
    # PostgreSQL
    postgres_url: Optional[str] = None
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20


@dataclass
class MessagingConfig:
    """消息队列配置（同步到 Runtime）"""
    backend: str = "redis"  # redis, kafka, memory
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_channel: str = "aga:sync"
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "aga-sync"
    kafka_group_id: str = "aga-portal"


@dataclass
class RegistryConfig:
    """Runtime 注册表配置"""
    type: str = "redis"  # redis, memory
    key_prefix: str = "aga:runtime:"
    heartbeat_interval: int = 30  # 秒
    timeout: int = 90  # 秒


@dataclass
class GovernanceConfig:
    """内置治理配置（可选）"""
    enabled: bool = False
    auto_confirm_after_hits: int = 100
    auto_deprecate_after_days: int = 30
    auto_quarantine_on_error: bool = True


@dataclass
class PortalConfig:
    """
    AGA Portal 完整配置
    
    配置文件示例 (portal_config.yaml):
    ```yaml
    server:
      host: "0.0.0.0"
      port: 8081
      workers: 4
    
    persistence:
      type: "postgres"
      postgres_url: "postgresql://aga:password@localhost:5432/aga"
    
    messaging:
      backend: "redis"
      redis_host: "localhost"
      redis_port: 6379
      redis_channel: "aga:sync"
    
    registry:
      type: "redis"
      heartbeat_interval: 30
    
    governance:
      enabled: false
    ```
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    persistence: PersistenceDBConfig = field(default_factory=PersistenceDBConfig)
    messaging: MessagingConfig = field(default_factory=MessagingConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    
    # 元数据
    version: str = "3.2.0"
    environment: str = "development"  # development, staging, production
    
    @classmethod
    def for_development(cls) -> "PortalConfig":
        """开发环境配置"""
        return cls(
            server=ServerConfig(
                host="127.0.0.1",
                port=8081,
                workers=1,
                reload=True,
                log_level="debug",
            ),
            persistence=PersistenceDBConfig(
                type="sqlite",
                sqlite_path="aga_dev.db",
            ),
            messaging=MessagingConfig(
                backend="memory",
            ),
            registry=RegistryConfig(
                type="memory",
            ),
            environment="development",
        )
    
    @classmethod
    def for_production(cls, postgres_url: str, redis_host: str = "localhost") -> "PortalConfig":
        """生产环境配置"""
        return cls(
            server=ServerConfig(
                host="0.0.0.0",
                port=8081,
                workers=4,
                reload=False,
                log_level="warning",
            ),
            persistence=PersistenceDBConfig(
                type="postgres",
                postgres_url=postgres_url,
                postgres_pool_size=20,
            ),
            messaging=MessagingConfig(
                backend="redis",
                redis_host=redis_host,
            ),
            registry=RegistryConfig(
                type="redis",
            ),
            environment="production",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [_to_dict(item) for item in obj]
            else:
                return obj
        
        return _to_dict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortalConfig":
        """从字典创建"""
        return cls(
            server=ServerConfig(**data.get("server", {})),
            persistence=PersistenceDBConfig(**data.get("persistence", {})),
            messaging=MessagingConfig(**data.get("messaging", {})),
            registry=RegistryConfig(**data.get("registry", {})),
            governance=GovernanceConfig(**data.get("governance", {})),
            version=data.get("version", "3.2.0"),
            environment=data.get("environment", "development"),
        )
