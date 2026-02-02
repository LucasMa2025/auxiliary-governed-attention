"""
AGA 持久化层

提供多适配器持久化支持：
- SQLite: 开发/测试环境
- Redis: 热缓存层
- PostgreSQL: 冷存储层
- Composite: 分层组合适配器

版本: v3.0
"""
from .base import (
    PersistenceAdapter,
    KnowledgeRecord,
    PersistenceError,
    ConnectionError,
    SerializationError,
)
from .sqlite_adapter import SQLiteAdapter
from .memory_adapter import MemoryAdapter
from .composite_adapter import CompositeAdapter
from .manager import PersistenceManager

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


__all__ = [
    # 基类
    "PersistenceAdapter",
    "KnowledgeRecord",
    "PersistenceError",
    "ConnectionError",
    "SerializationError",
    # 适配器
    "SQLiteAdapter",
    "MemoryAdapter",
    "CompositeAdapter",
    # 管理器
    "PersistenceManager",
    # 工厂
    "create_adapter",
]
