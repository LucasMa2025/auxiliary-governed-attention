"""
AGA Production Runtime v1.1

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

⚠️ 部署模式说明 (v1.1)
========================

本模块专为 **单机部署模式** 设计，所有组件运行在同一进程/同一服务器：

    ┌─────────────────────────────────────────────┐
    │              单机部署架构                    │
    │                                             │
    │  ┌───────────┐    ┌──────────────────────┐ │
    │  │ 治理系统   │───▶│   KnowledgeWriter    │ │
    │  └───────────┘    │   (进程内写入)        │ │
    │                   └──────────┬───────────┘ │
    │                              │              │
    │  ┌───────────────────────────┼────────────┐│
    │  │        AGAOperator        │            ││
    │  │  ┌─────────────┐  ┌──────▼─────┐      ││
    │  │  │  GateChain  │  │ SlotPool   │      ││
    │  │  │ (三段门控)  │  │ (内存槽位) │      ││
    │  │  └─────────────┘  └────────────┘      ││
    │  │                                        ││
    │  │  ┌────────────────────────────────┐   ││
    │  │  │     PersistenceManager         │   ││
    │  │  │  Redis(热) + PostgreSQL(冷)    │   ││
    │  │  └────────────────────────────────┘   ││
    │  └────────────────────────────────────────┘│
    │                                             │
    │  GPU 服务器 (与 LLM 同机部署)               │
    └─────────────────────────────────────────────┘

模块特点：
- persistence.py: 使用同步 Redis/SQLAlchemy 客户端，适合同进程访问
- writer.py: 直接操作本地 AGA 实例和持久化层，无网络开销
- 所有状态在单机内存中，无分布式同步延迟

适用场景：
✅ 单 GPU 服务器部署
✅ 开发测试环境
✅ 小规模生产环境 (单实例)

不适用场景：
❌ 多服务器集群部署 (请使用 aga.portal + aga.runtime)
❌ API 服务与推理服务分离部署
❌ 需要 Portal 水平扩展的场景

如需分离部署，请参考：
- aga.portal: GPU 无关的 API 服务
- aga.runtime: GPU 依赖的推理运行时
- aga.sync: Portal ↔ Runtime 同步协议
- aga.client: 外部系统访问 Portal 的 HTTP 客户端
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

__version__ = "1.1.0"

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

