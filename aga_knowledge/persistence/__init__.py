"""
aga-knowledge 持久化层

提供统一的持久化接口，支持多种后端：
- MemoryAdapter: 内存存储（测试/开发）
- SQLiteAdapter: SQLite 存储（开发/小规模生产）
- PostgresAdapter: PostgreSQL 存储（生产环境）
- RedisAdapter: Redis 存储（热缓存层）

附加功能：
- VersionedKnowledgeStore: 知识版本控制
- TextCompressor / DecompressionCache: 文本压缩
"""

from .base import PersistenceAdapter, KnowledgeRecord
from .memory_adapter import MemoryAdapter
from .sqlite_adapter import SQLiteAdapter
from .versioning import (
    KnowledgeVersion,
    VersionDiff,
    VersionedKnowledgeStore,
)
from .compression import (
    CompressionAlgorithm,
    TextCompressionConfig,
    TextCompressor,
    DecompressionCache,
)

__all__ = [
    # 基类
    "PersistenceAdapter",
    "KnowledgeRecord",
    # 适配器
    "MemoryAdapter",
    "SQLiteAdapter",
    # 版本控制
    "KnowledgeVersion",
    "VersionDiff",
    "VersionedKnowledgeStore",
    # 压缩
    "CompressionAlgorithm",
    "TextCompressionConfig",
    "TextCompressor",
    "DecompressionCache",
    # 工厂
    "create_adapter",
]


def create_adapter(config: dict) -> PersistenceAdapter:
    """
    根据配置创建持久化适配器

    Args:
        config: 持久化配置字典
            type: "memory" | "sqlite" | "postgres" | "redis"
            sqlite_path: SQLite 数据库路径（type=sqlite 时）
            postgres_*: PostgreSQL 参数（type=postgres 时）
            redis_*: Redis 参数（type=redis 时）

    Returns:
        PersistenceAdapter 实例
    """
    adapter_type = config.get("type", "memory")

    if adapter_type == "memory":
        return MemoryAdapter(
            max_slots_per_namespace=config.get(
                "max_slots_per_namespace", 128
            ),
        )
    elif adapter_type == "sqlite":
        return SQLiteAdapter(
            db_path=config.get("sqlite_path", "aga_knowledge.db"),
            enable_audit=config.get("enable_audit", True),
        )
    elif adapter_type == "postgres":
        from .postgres_adapter import PostgresAdapter

        return PostgresAdapter(
            host=config.get("postgres_host", "localhost"),
            port=config.get("postgres_port", 5432),
            database=config.get("postgres_database", "aga_knowledge"),
            user=config.get("postgres_user", "aga"),
            password=config.get("postgres_password"),
            pool_size=config.get("postgres_pool_size", 5),
            max_overflow=config.get("postgres_max_overflow", 10),
            enable_audit=config.get("enable_audit", True),
            dsn=config.get("postgres_url"),
        )
    elif adapter_type == "redis":
        from .redis_adapter import RedisAdapter

        return RedisAdapter(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 0),
            password=config.get("redis_password"),
            key_prefix=config.get("redis_key_prefix", "aga_knowledge"),
            ttl_days=config.get("redis_ttl_days", 30),
            pool_size=config.get("redis_pool_size", 10),
            enable_audit=config.get("enable_audit", True),
            url=config.get("redis_url"),
        )
    else:
        raise ValueError(f"未知的持久化类型: {adapter_type}")
