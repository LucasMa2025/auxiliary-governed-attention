"""
AGA Production Runtime v1.0

Phase-1 生产级改造核心模块：
- 三段式 Gate (Gate-0/1/2)
- Slot Subspace Pool (物理隔离)
- Redis + PostgreSQL 持久化
- Fail-Open 机制
- 并发安全

设计原则：
🔒 不变量 1：推理可见知识规模 = O(1)
🔒 不变量 2：AGA 永远是"可绕过"的
🔒 不变量 3：治理、学习、评估永不进入热路径
"""

from .config import ProductionAGAConfig, GateConfig, SlotPoolConfig, PersistenceConfig
from .gate import GateChain, GateResult, Gate0, Gate1, Gate2
from .slot_pool import SlotSubspacePool, SlotPool, Slot, EvictionPolicy
from .persistence import (
    RedisPersistence,
    PostgreSQLPersistence,
    HybridPersistence,
    PersistenceManager,
)
from .operator import AGAOperator, ConcurrentAGAManager
from .writer import KnowledgeWriter, WriteRequest, WriteResult

__version__ = "1.0.0"

__all__ = [
    # Config
    "ProductionAGAConfig",
    "GateConfig",
    "SlotPoolConfig",
    "PersistenceConfig",
    # Gate
    "GateChain",
    "GateResult",
    "Gate0",
    "Gate1",
    "Gate2",
    # Slot Pool
    "SlotSubspacePool",
    "SlotPool",
    "Slot",
    "EvictionPolicy",
    # Persistence
    "RedisPersistence",
    "PostgreSQLPersistence",
    "HybridPersistence",
    "PersistenceManager",
    # Operator
    "AGAOperator",
    "ConcurrentAGAManager",
    # Writer
    "KnowledgeWriter",
    "WriteRequest",
    "WriteResult",
]

