"""
aga-knowledge 配置系统

提供 Portal 和知识管理的完整配置。
支持从 YAML 文件加载和环境变量覆盖。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

AGA_KNOWLEDGE_VERSION = "0.3.0"


# ==================== 子配置 ====================

@dataclass
class ServerConfig:
    """Portal 服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8081
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class PersistenceDBConfig:
    """持久化数据库配置"""
    type: str = "sqlite"                    # sqlite | postgres | memory | redis
    sqlite_path: str = "aga_knowledge.db"   # SQLite 数据库路径
    postgres_url: Optional[str] = None      # PostgreSQL 连接 URL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "aga_knowledge"
    postgres_user: str = "aga"
    postgres_password: Optional[str] = None
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    redis_url: Optional[str] = None         # Redis 连接 URL
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_key_prefix: str = "aga_knowledge"
    redis_ttl_days: int = 30
    redis_pool_size: int = 10
    enable_audit: bool = True               # 是否启用审计日志


@dataclass
class MessagingConfig:
    """消息队列配置（同步到 Runtime）"""
    backend: str = "redis"                  # redis | memory
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_channel: str = "aga:sync"


@dataclass
class RegistryConfig:
    """Runtime 注册表配置"""
    type: str = "memory"                    # redis | memory
    key_prefix: str = "aga:runtime:"
    heartbeat_interval: int = 30
    timeout: int = 90


@dataclass
class GovernanceConfig:
    """内置治理配置（可选）"""
    enabled: bool = False
    auto_confirm_after_hits: int = 100
    auto_deprecate_after_days: int = 30
    auto_quarantine_on_error: bool = True


@dataclass
class RetrieverConfig:
    """检索器配置"""
    index_backend: str = "brute"          # "brute" | "hnsw"
    bm25_enabled: bool = False
    bm25_weight: float = 0.3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    similarity_threshold: float = 0.0
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    hnsw_max_elements: int = 100000


@dataclass
class ImageHandlingConfig:
    """图片处理配置"""
    enabled: bool = False
    asset_dir: str = ""                    # 静态资源目录
    base_url: str = ""                     # Portal 资源 URL 前缀
    max_image_size_mb: int = 10
    supported_formats: List[str] = field(
        default_factory=lambda: ["png", "jpg", "jpeg", "gif", "webp", "svg"]
    )
    inline_description: bool = True
    description_template: str = "[图片: {alt}, 参见 {url}]"


@dataclass
class ChunkerAppConfig:
    """分片器配置"""
    strategy: str = "sliding_window"
    chunk_size: int = 300
    overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 500
    condition_mode: str = "title_context"   # first_sentence | title_context | keyword | summary
    language: str = "auto"


# ==================== 主配置 ====================

@dataclass
class PortalConfig:
    """
    AGA Knowledge Portal 完整配置

    配置文件示例 (portal_config.yaml):
    ```yaml
    server:
      host: "0.0.0.0"
      port: 8081
      workers: 4

    persistence:
      type: "sqlite"
      sqlite_path: "aga_knowledge.db"
      enable_audit: true

    messaging:
      backend: "redis"
      redis_host: "localhost"
      redis_port: 6379
      redis_channel: "aga:sync"

    registry:
      type: "memory"
      timeout: 90

    governance:
      enabled: false
    ```
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    persistence: PersistenceDBConfig = field(default_factory=PersistenceDBConfig)
    messaging: MessagingConfig = field(default_factory=MessagingConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    image_handling: ImageHandlingConfig = field(default_factory=ImageHandlingConfig)
    chunker: ChunkerAppConfig = field(default_factory=ChunkerAppConfig)

    # aga-core 对齐配置（从字典/YAML 加载）
    aga_core_alignment: Optional[Dict[str, Any]] = None

    version: str = field(default_factory=lambda: AGA_KNOWLEDGE_VERSION)
    environment: str = "development"

    def get_alignment(self):
        """
        获取 AGACoreAlignment 实例

        从 aga_core_alignment 字段创建 AGACoreAlignment 对象。
        如果未配置，返回默认值。
        """
        from .alignment import AGACoreAlignment
        if self.aga_core_alignment:
            # 支持引用 aga-core 配置文件
            config_path = self.aga_core_alignment.get("aga_core_config_path")
            if config_path:
                return AGACoreAlignment.from_aga_config_yaml(config_path)
            return AGACoreAlignment.from_dict(self.aga_core_alignment)
        return AGACoreAlignment()

    @classmethod
    def for_development(cls) -> "PortalConfig":
        """开发环境配置"""
        return cls(
            server=ServerConfig(
                host="127.0.0.1", port=8081, workers=1,
                reload=True, log_level="debug",
            ),
            persistence=PersistenceDBConfig(
                type="sqlite", sqlite_path="aga_dev.db",
            ),
            messaging=MessagingConfig(backend="memory"),
            registry=RegistryConfig(type="memory"),
            environment="development",
        )

    @classmethod
    def for_production(cls, postgres_url: str, redis_host: str = "localhost") -> "PortalConfig":
        """生产环境配置"""
        return cls(
            server=ServerConfig(
                host="0.0.0.0", port=8081, workers=4,
                reload=False, log_level="warning",
            ),
            persistence=PersistenceDBConfig(
                type="postgres", postgres_url=postgres_url,
                postgres_pool_size=20,
            ),
            messaging=MessagingConfig(
                backend="redis", redis_host=redis_host,
            ),
            registry=RegistryConfig(type="redis"),
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
            retriever=RetrieverConfig(**data.get("retriever", {})),
            image_handling=ImageHandlingConfig(**data.get("image_handling", {})),
            chunker=ChunkerAppConfig(**data.get("chunker", {})),
            aga_core_alignment=data.get("aga_core_alignment"),
            version=data.get("version", AGA_KNOWLEDGE_VERSION),
            environment=data.get("environment", "development"),
        )


def load_config(config_path: str) -> PortalConfig:
    """
    从 YAML 文件加载配置

    Args:
        config_path: YAML 配置文件路径

    Returns:
        PortalConfig 实例

    支持环境变量覆盖:
        AGA_PORTAL_HOST, AGA_PORTAL_PORT,
        AGA_PERSISTENCE_TYPE, AGA_PERSISTENCE_SQLITE_PATH,
        AGA_PERSISTENCE_POSTGRES_URL,
        AGA_MESSAGING_BACKEND, AGA_MESSAGING_REDIS_HOST,
        AGA_ENVIRONMENT
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("需要安装 PyYAML: pip install pyyaml")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    config = PortalConfig.from_dict(data)

    # 环境变量覆盖
    if os.environ.get("AGA_PORTAL_HOST"):
        config.server.host = os.environ["AGA_PORTAL_HOST"]
    if os.environ.get("AGA_PORTAL_PORT"):
        config.server.port = int(os.environ["AGA_PORTAL_PORT"])
    if os.environ.get("AGA_PERSISTENCE_TYPE"):
        config.persistence.type = os.environ["AGA_PERSISTENCE_TYPE"]
    if os.environ.get("AGA_PERSISTENCE_SQLITE_PATH"):
        config.persistence.sqlite_path = os.environ["AGA_PERSISTENCE_SQLITE_PATH"]
    if os.environ.get("AGA_PERSISTENCE_POSTGRES_URL"):
        config.persistence.postgres_url = os.environ["AGA_PERSISTENCE_POSTGRES_URL"]
    if os.environ.get("AGA_MESSAGING_BACKEND"):
        config.messaging.backend = os.environ["AGA_MESSAGING_BACKEND"]
    if os.environ.get("AGA_MESSAGING_REDIS_HOST"):
        config.messaging.redis_host = os.environ["AGA_MESSAGING_REDIS_HOST"]
    if os.environ.get("AGA_ENVIRONMENT"):
        config.environment = os.environ["AGA_ENVIRONMENT"]

    logger.info(f"Config loaded from {config_path}, environment={config.environment}")
    return config
