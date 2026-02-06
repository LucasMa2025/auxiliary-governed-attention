"""
AGA 持久化层

提供多适配器持久化支持：
- SQLite: 开发/测试环境
- Redis: 热缓存层
- PostgreSQL: 冷存储层
- Composite: 分层组合适配器
- Memory: 测试/L0 缓存

版本: v3.2

v3.2 新增 Portal 扩展接口支持：
- get_namespaces(): 获取所有命名空间
- save_audit_log(): 保存审计日志
- query_audit_log(): 查询审计日志
- save_knowledge(): 简化知识保存接口
- load_knowledge(): 简化知识加载接口
- query_knowledge(): 查询知识列表
"""
from .base import (
    PersistenceAdapter,
    KnowledgeRecord,
    PersistenceError,
    ConnectionError,
    SerializationError,
    TrustTier,
    TRUST_TIER_PRIORITY,
)
from .sqlite_adapter import SQLiteAdapter
from .memory_adapter import MemoryAdapter
from .composite_adapter import CompositeAdapter
from .manager import PersistenceManager
from .pool import ConnectionPool, PoolConfig, PoolStats, PooledConnection
from .compression import (
    VectorCompressor,
    CompressionConfig,
    DecompressionCache,
    Precision,
    Compression,
)

# 可选适配器（需要额外依赖）
try:
    from .redis_adapter import RedisAdapter
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

try:
    from .postgres_adapter import PostgresAdapter
    _HAS_POSTGRES = True
except ImportError:
    _HAS_POSTGRES = False


def create_adapter(adapter_type: str, **kwargs) -> PersistenceAdapter:
    """
    工厂函数：创建持久化适配器
    
    Args:
        adapter_type: 适配器类型 (sqlite, memory, redis, postgres, composite)
        **kwargs: 适配器特定参数
    
    Returns:
        PersistenceAdapter 实例
    """
    if adapter_type == "sqlite":
        return SQLiteAdapter(**kwargs)
    elif adapter_type == "memory":
        return MemoryAdapter(**kwargs)
    elif adapter_type == "redis":
        if not _HAS_REDIS:
            raise ImportError("Redis adapter requires 'redis' package. Install with: pip install redis")
        return RedisAdapter(**kwargs)
    elif adapter_type == "postgres":
        if not _HAS_POSTGRES:
            raise ImportError("PostgreSQL adapter requires 'asyncpg' package. Install with: pip install asyncpg")
        return PostgresAdapter(**kwargs)
    elif adapter_type == "composite":
        return CompositeAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


# 兼容旧版 persistence.py 的导入
try:
    from ..persistence import (
        SQLitePersistence,
        AGAPersistenceManager,
        AGAPersistence,
        KnowledgeRecord as LegacyKnowledgeRecord,
    )
    _HAS_LEGACY = True
except ImportError:
    _HAS_LEGACY = False
    SQLitePersistence = None
    AGAPersistenceManager = None
    AGAPersistence = None


__all__ = [
    # 基类
    "PersistenceAdapter",
    "KnowledgeRecord",
    "PersistenceError",
    "ConnectionError",
    "SerializationError",
    # 信任层级
    "TrustTier",
    "TRUST_TIER_PRIORITY",
    # 适配器
    "SQLiteAdapter",
    "MemoryAdapter",
    "CompositeAdapter",
    # 管理器
    "PersistenceManager",
    # 连接池
    "ConnectionPool",
    "PoolConfig",
    "PoolStats",
    "PooledConnection",
    # 压缩
    "VectorCompressor",
    "CompressionConfig",
    "DecompressionCache",
    "Precision",
    "Compression",
    # 工厂
    "create_adapter",
    # 兼容旧版
    "SQLitePersistence",
    "AGAPersistenceManager",
    "AGAPersistence",
]
