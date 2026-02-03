# AGA 分布式实施方案

**版本**: v1.0  
**日期**: 2026-02-01

---

## 目录

1. [AGA 分布式本质分析](#1-aga-分布式本质分析)
2. [分布式架构设计](#2-分布式架构设计)
3. [分布式实施模式](#3-分布式实施模式)
4. [一致性与同步](#4-一致性与同步)
5. [性能优化](#5-性能优化)
6. [实施路线图](#6-实施路线图)

---

## 1. AGA 分布式本质分析

### 1.1 AGA 作为 Transformer 插件的特性

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGA 作为 Transformer 插件的本质                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Transformer 层结构                                │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                    Layer N                                   │    │   │
│  │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │    │   │
│  │  │  │ Self-Attn   │───▶│    AGA      │───▶│    FFN      │      │    │   │
│  │  │  │ (Frozen)    │    │  (Plugin)   │    │  (Frozen)   │      │    │   │
│  │  │  └─────────────┘    └──────┬──────┘    └─────────────┘      │    │   │
│  │  │                            │                                 │    │   │
│  │  │                     ┌──────▼──────┐                          │    │   │
│  │  │                     │ Slot Pool   │ ← 可独立部署              │    │   │
│  │  │                     └─────────────┘                          │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  关键特性：                                                                  │
│  ✅ 无状态计算：AGA 前向传播是纯函数                                          │
│  ✅ 状态外置：知识存储在 SlotPool，可独立管理                                  │
│  ✅ 层级独立：每层 AGA 可独立配置和部署                                        │
│  ✅ 可选旁路：AGA 故障时可完全绕过                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 分布式可行性分析

| 特性           | 描述                       | 分布式影响          |
| -------------- | -------------------------- | ------------------- |
| **无状态计算** | AGA 前向传播不依赖历史状态 | ✅ 天然支持水平扩展 |
| **状态外置**   | SlotPool 可独立存储        | ✅ 支持共享存储     |
| **层级独立**   | 每层 AGA 独立运行          | ✅ 支持层级分布     |
| **Fail-Open**  | 故障时绕过                 | ✅ 高可用保证       |
| **跨层衰减**   | 需要跨层状态               | ⚠️ 需要请求级上下文 |

### 1.3 分布式挑战

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        分布式挑战矩阵                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐ │
│  │     挑战        │     难度        │     影响        │     方案        │ │
│  ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤ │
│  │ 知识一致性      │      中         │      高         │ 事件驱动同步    │ │
│  │ 跨层状态传递    │      高         │      中         │ 请求级上下文    │ │
│  │ 热点槽位竞争    │      中         │      中         │ 分片 + 缓存     │ │
│  │ 隔离命令传播    │      高         │      高         │ 分布式锁        │ │
│  │ 审计日志聚合    │      低         │      低         │ 集中式日志      │ │
│  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 分布式架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGA 分布式架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        客户端层                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │  Client 1   │  │  Client 2   │  │  Client N   │                  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │   │
│  └─────────┼────────────────┼────────────────┼──────────────────────────┘   │
│            │                │                │                              │
│            └────────────────┼────────────────┘                              │
│                             │                                               │
│  ┌──────────────────────────▼──────────────────────────────────────────┐   │
│  │                     负载均衡层                                        │   │
│  │                   (Nginx / K8s Ingress)                              │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│  ┌──────────────────────────▼──────────────────────────────────────────┐   │
│  │                     推理服务层                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                    LLM + AGA 实例集群                         │    │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │    │   │
│  │  │  │  Instance 1 │  │  Instance 2 │  │  Instance N │          │    │   │
│  │  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │          │    │   │
│  │  │  │  │  LLM  │  │  │  │  LLM  │  │  │  │  LLM  │  │          │    │   │
│  │  │  │  ├───────┤  │  │  ├───────┤  │  │  ├───────┤  │          │    │   │
│  │  │  │  │  AGA  │  │  │  │  AGA  │  │  │  │  AGA  │  │          │    │   │
│  │  │  │  │ Local │  │  │  │ Local │  │  │  │ Local │  │          │    │   │
│  │  │  │  │ Cache │  │  │  │ Cache │  │  │  │ Cache │  │          │    │   │
│  │  │  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │          │    │   │
│  │  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │    │   │
│  │  └─────────┼────────────────┼────────────────┼──────────────────┘    │   │
│  └────────────┼────────────────┼────────────────┼───────────────────────┘   │
│               │                │                │                           │
│               └────────────────┼────────────────┘                           │
│                                │                                            │
│  ┌─────────────────────────────▼────────────────────────────────────────┐  │
│  │                      消息总线层                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Kafka Cluster                                 │ │  │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │ │  │
│  │  │  │ knowledge   │  │ lifecycle   │  │ quarantine  │              │ │  │
│  │  │  │ _sync       │  │ _events     │  │ _commands   │              │ │  │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘              │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                │                                            │
│  ┌─────────────────────────────▼────────────────────────────────────────┐  │
│  │                      存储层                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │ Redis       │  │ PostgreSQL  │  │ Vector DB   │  │ Object      │ │  │
│  │  │ Cluster     │  │ (Primary)   │  │ (Optional)  │  │ Storage     │ │  │
│  │  │ (Hot Cache) │  │             │  │             │  │ (Backup)    │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 组件职责

| 组件           | 职责                  | 技术选型               |
| -------------- | --------------------- | ---------------------- |
| **推理实例**   | LLM 推理 + AGA 计算   | PyTorch + vLLM         |
| **本地缓存**   | 热点知识缓存          | 内存 Dict              |
| **消息总线**   | 事件广播和同步        | Kafka                  |
| **Redis**      | 共享热缓存 + 分布式锁 | Redis Cluster          |
| **PostgreSQL** | 持久化存储            | PostgreSQL + PgBouncer |
| **Vector DB**  | 大规模知识检索        | Milvus / Qdrant        |

### 2.3 数据流设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        数据流设计                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  推理请求流程：                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Request │ ──▶│ Local   │───▶│ Redis   │───▶│ AGA     │───▶│ Response│  │
│  │         │    │ Cache   │    │ (miss)  │    │ Compute │    │         │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                      │                                                      │
│                      │ hit                                                  │
│                      └──────────────────────────────────────▶               │
│                                                                             │
│  知识注入流程：                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Inject  │───▶│ Local   │───▶│ Redis   │───▶│ Kafka   │───▶│ Other   │  │
│  │ Request │    │ Pool    │    │ Write   │    │ Publish │    │ Instances│  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                                                    │                        │
│                                                    ▼                        │
│                                              ┌─────────┐                    │
│                                              │PostgreSQL│                    │
│                                              │ Persist │                    │
│                                              └─────────┘                    │
│                                                                             │
│  隔离命令流程（强一致）：                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │Quarantine│───▶│ Redis   │───▶│ Kafka   │───▶│ All     │───▶│ ACK     │  │
│  │ Command │    │ Lock    │    │ Sync    │    │ Instances│    │ Wait    │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 分布式实施模式

### 3.1 模式一：共享存储模式

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    模式一：共享存储模式                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  特点：所有实例共享同一个 SlotPool 存储                                       │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │  Instance 1 │    │  Instance 2 │    │  Instance N │                     │
│  │  ┌───────┐  │    │  ┌───────┐  │    │  ┌───────┐  │                     │
│  │  │  AGA  │  │    │  │  AGA  │  │    │  │  AGA  │  │                     │
│  │  └───┬───┘  │    │  └───┬───┘  │    │  └───┬───┘  │                     │
│  └──────┼──────┘    └──────┼──────┘    └──────┼──────┘                     │
│         │                  │                  │                             │
│         └──────────────────┼──────────────────┘                             │
│                            │                                                │
│                     ┌──────▼──────┐                                         │
│                     │   Redis     │  ← 共享 SlotPool                        │
│                     │   Cluster   │                                         │
│                     └─────────────┘                                         │
│                                                                             │
│  优点：                                                                      │
│  ✅ 实现简单                                                                 │
│  ✅ 强一致性                                                                 │
│  ✅ 无同步延迟                                                               │
│                                                                             │
│  缺点：                                                                      │
│  ❌ Redis 成为瓶颈                                                           │
│  ❌ 网络延迟影响推理性能                                                      │
│  ❌ 单点故障风险                                                             │
│                                                                             │
│  适用场景：                                                                  │
│  - 小规模部署（<10 实例）                                                    │
│  - 低延迟要求不高                                                            │
│  - 强一致性要求高                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 模式二：本地缓存 + 事件同步模式

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    模式二：本地缓存 + 事件同步模式                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  特点：每个实例维护本地缓存，通过事件同步                                      │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │  Instance 1 │    │  Instance 2 │    │  Instance N │                     │
│  │  ┌───────┐  │    │  ┌───────┐  │    │  ┌───────┐  │                     │
│  │  │  AGA  │  │    │  │  AGA  │  │    │  │  AGA  │  │                     │
│  │  ├───────┤  │    │  ├───────┤  │    │  ├───────┤  │                     │
│  │  │ Local │  │    │  │ Local │  │    │  │ Local │  │                     │
│  │  │ Cache │  │    │  │ Cache │  │    │  │ Cache │  │                     │
│  │  └───┬───┘  │    │  └───┬───┘  │    │  └───┬───┘  │                     │
│  └──────┼──────┘    └──────┼──────┘    └──────┼──────┘                     │
│         │                  │                  │                             │
│         └──────────────────┼──────────────────┘                             │
│                            │                                                │
│                     ┌──────▼──────┐                                         │
│                     │   Kafka     │  ← 事件同步                              │
│                     │   Cluster   │                                         │
│                     └─────────────┘                                         │
│                                                                             │
│  优点：                                                                      │
│  ✅ 推理延迟低（本地访问）                                                    │
│  ✅ 可水平扩展                                                               │
│  ✅ 容错性好                                                                 │
│                                                                             │
│  缺点：                                                                      │
│  ❌ 最终一致性（有同步延迟）                                                  │
│  ❌ 实现复杂                                                                 │
│  ❌ 内存占用高（每实例一份）                                                  │
│                                                                             │
│  适用场景：                                                                  │
│  - 大规模部署（10+ 实例）                                                    │
│  - 低延迟要求高                                                              │
│  - 可接受短暂不一致                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 模式三：分层缓存模式（推荐）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    模式三：分层缓存模式（推荐）                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  特点：L0 本地 + L1 Redis + L2 PostgreSQL                                   │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │  Instance 1 │    │  Instance 2 │    │  Instance N │                     │
│  │  ┌───────┐  │    │  ┌───────┐  │    │  ┌───────┐  │                     │
│  │  │  AGA  │  │    │  │  AGA  │  │    │  │  AGA  │  │                     │
│  │  ├───────┤  │    │  ├───────┤  │    │  ├───────┤  │                     │
│  │  │  L0   │  │    │  │  L0   │  │    │  │  L0   │  │  ← 热点缓存         │
│  │  │ (GPU) │  │    │  │ (GPU) │  │    │  │ (GPU) │  │    128 slots        │
│  │  └───┬───┘  │    │  └───┬───┘  │    │  └───┬───┘  │                     │
│  └──────┼──────┘    └──────┼──────┘    └──────┼──────┘                     │
│         │                  │                  │                             │
│         └──────────────────┼──────────────────┘                             │
│                            │                                                │
│                     ┌──────▼──────┐                                         │
│                     │     L1      │  ← 共享热缓存                            │
│                     │   Redis     │    1000+ slots                          │
│                     └──────┬──────┘                                         │
│                            │                                                │
│                     ┌──────▼──────┐                                         │
│                     │     L2      │  ← 持久化存储                            │
│                     │ PostgreSQL  │    无限容量                              │
│                     └─────────────┘                                         │
│                                                                             │
│  缓存策略：                                                                  │
│  - L0: LRU + 命中率优化，保留最热的 128 个槽位                               │
│  - L1: TTL + 访问频率，保留活跃的 1000+ 槽位                                 │
│  - L2: 全量持久化，按需加载                                                  │
│                                                                             │
│  优点：                                                                      │
│  ✅ 兼顾性能和一致性                                                         │
│  ✅ 内存效率高                                                               │
│  ✅ 可扩展性好                                                               │
│                                                                             │
│  缺点：                                                                      │
│  ❌ 实现最复杂                                                               │
│  ❌ 缓存失效处理复杂                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 模式选择指南

```python
"""
分布式模式选择器
"""
from dataclasses import dataclass
from enum import Enum


class DistributedMode(str, Enum):
    SHARED_STORAGE = "shared_storage"
    LOCAL_CACHE_SYNC = "local_cache_sync"
    HIERARCHICAL_CACHE = "hierarchical_cache"


@dataclass
class DeploymentRequirements:
    instance_count: int
    max_latency_ms: float
    consistency_requirement: str  # 'strong', 'eventual'
    knowledge_count: int
    budget: str  # 'low', 'medium', 'high'


def select_mode(requirements: DeploymentRequirements) -> DistributedMode:
    """选择分布式模式"""

    # 小规模 + 强一致性 → 共享存储
    if (requirements.instance_count < 10 and
        requirements.consistency_requirement == 'strong'):
        return DistributedMode.SHARED_STORAGE

    # 大规模 + 低延迟 → 本地缓存 + 同步
    if (requirements.instance_count >= 10 and
        requirements.max_latency_ms < 10):
        return DistributedMode.LOCAL_CACHE_SYNC

    # 默认推荐分层缓存
    return DistributedMode.HIERARCHICAL_CACHE
```

---

## 4. 一致性与同步

### 4.1 一致性模型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        一致性模型                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    操作一致性级别                                     │   │
│  │                                                                      │   │
│  │  ┌─────────────────┬─────────────────┬─────────────────────────────┐│   │
│  │  │     操作        │   一致性级别     │         实现方式            ││   │
│  │  ├─────────────────┼─────────────────┼─────────────────────────────┤│   │
│  │  │ 知识读取        │ 本地一致        │ 本地缓存优先                 ││   │
│  │  │ 知识注入        │ 最终一致        │ 异步广播                     ││   │
│  │  │ 生命周期更新    │ 最终一致        │ 异步广播                     ││   │
│  │  │ 隔离命令        │ 强一致          │ 分布式锁 + 同步广播          ││   │
│  │  │ 配置变更        │ 强一致          │ 分布式锁 + 同步广播          ││   │
│  │  └─────────────────┴─────────────────┴─────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  一致性保证：                                                                │
│  - 最终一致性延迟 < 1s（99%）                                               │
│  - 强一致性操作延迟 < 5s（99%）                                             │
│  - 隔离命令 100% 同步执行                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 同步协议

```python
"""
分布式同步协议
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio


class SyncProtocol(str, Enum):
    """同步协议类型"""
    FIRE_AND_FORGET = "fire_and_forget"  # 发送后不等待
    ASYNC_ACK = "async_ack"              # 异步确认
    SYNC_ACK = "sync_ack"                # 同步确认
    TWO_PHASE = "two_phase"              # 两阶段提交


@dataclass
class SyncConfig:
    """同步配置"""
    default_protocol: SyncProtocol = SyncProtocol.ASYNC_ACK
    ack_timeout_seconds: float = 5.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0


class DistributedSync:
    """
    分布式同步器

    支持多种同步协议。
    """

    def __init__(
        self,
        kafka_producer,
        redis_client,
        config: Optional[SyncConfig] = None,
    ):
        self.kafka = kafka_producer
        self.redis = redis_client
        self.config = config or SyncConfig()

        # 待确认消息
        self._pending_acks: Dict[str, asyncio.Event] = {}

    async def sync_knowledge_inject(
        self,
        namespace: str,
        lu_id: str,
        key_vector: List[float],
        value_vector: List[float],
        lifecycle_state: str,
    ):
        """同步知识注入（最终一致）"""
        message = {
            'type': 'knowledge_inject',
            'namespace': namespace,
            'lu_id': lu_id,
            'key_vector': key_vector,
            'value_vector': value_vector,
            'lifecycle_state': lifecycle_state,
            'timestamp': datetime.now().isoformat(),
        }

        await self._send(
            topic='aga_knowledge_sync',
            message=message,
            protocol=SyncProtocol.FIRE_AND_FORGET,
        )

    async def sync_quarantine(
        self,
        namespace: str,
        lu_id: str,
        reason: str,
    ) -> bool:
        """同步隔离命令（强一致）"""
        message_id = f"quarantine_{namespace}_{lu_id}_{datetime.now().timestamp()}"

        message = {
            'type': 'quarantine',
            'message_id': message_id,
            'namespace': namespace,
            'lu_id': lu_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        }

        # 使用两阶段提交
        return await self._two_phase_commit(
            topic='aga_quarantine_sync',
            message=message,
            message_id=message_id,
        )

    async def _send(
        self,
        topic: str,
        message: Dict[str, Any],
        protocol: SyncProtocol,
    ):
        """发送消息"""
        if protocol == SyncProtocol.FIRE_AND_FORGET:
            await self.kafka.send(topic, message)

        elif protocol == SyncProtocol.ASYNC_ACK:
            message_id = message.get('message_id', str(uuid.uuid4()))
            self._pending_acks[message_id] = asyncio.Event()

            await self.kafka.send(topic, message)

            # 异步等待确认
            asyncio.create_task(self._wait_for_ack(message_id))

        elif protocol == SyncProtocol.SYNC_ACK:
            message_id = message.get('message_id', str(uuid.uuid4()))
            self._pending_acks[message_id] = asyncio.Event()

            await self.kafka.send(topic, message)

            # 同步等待确认
            try:
                await asyncio.wait_for(
                    self._pending_acks[message_id].wait(),
                    timeout=self.config.ack_timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise SyncTimeoutError(f"Sync timeout for {message_id}")

    async def _two_phase_commit(
        self,
        topic: str,
        message: Dict[str, Any],
        message_id: str,
    ) -> bool:
        """两阶段提交"""
        # Phase 1: Prepare
        prepare_message = {**message, 'phase': 'prepare'}

        lock_key = f"aga:sync:lock:{message_id}"
        lock = await self.redis.lock(lock_key, timeout=30)

        try:
            async with lock:
                # 发送 prepare
                await self.kafka.send(topic, prepare_message)

                # 等待所有实例 prepare 确认
                prepare_acks = await self._wait_for_all_acks(
                    message_id,
                    phase='prepare',
                    timeout=self.config.ack_timeout_seconds,
                )

                if not prepare_acks:
                    # Prepare 失败，发送 abort
                    abort_message = {**message, 'phase': 'abort'}
                    await self.kafka.send(topic, abort_message)
                    return False

                # Phase 2: Commit
                commit_message = {**message, 'phase': 'commit'}
                await self.kafka.send(topic, commit_message)

                # 等待所有实例 commit 确认
                commit_acks = await self._wait_for_all_acks(
                    message_id,
                    phase='commit',
                    timeout=self.config.ack_timeout_seconds,
                )

                return commit_acks

        except Exception as e:
            # 发送 abort
            abort_message = {**message, 'phase': 'abort'}
            await self.kafka.send(topic, abort_message)
            raise

    async def _wait_for_ack(self, message_id: str):
        """等待确认"""
        try:
            await asyncio.wait_for(
                self._pending_acks[message_id].wait(),
                timeout=self.config.ack_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Ack timeout for {message_id}")
        finally:
            self._pending_acks.pop(message_id, None)

    async def _wait_for_all_acks(
        self,
        message_id: str,
        phase: str,
        timeout: float,
    ) -> bool:
        """等待所有实例确认"""
        # 实现略
        return True

    def receive_ack(self, message_id: str):
        """接收确认"""
        if message_id in self._pending_acks:
            self._pending_acks[message_id].set()
```

### 4.3 冲突解决

```python
"""
分布式冲突解决器
"""
from typing import Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class VersionedSlot:
    """带版本的槽位"""
    lu_id: str
    version: int
    updated_at: datetime
    lifecycle_state: str
    key_vector: List[float]
    value_vector: List[float]


class ConflictResolver:
    """
    冲突解决器

    策略：
    1. 版本号高的优先
    2. 版本号相同时，时间戳新的优先
    3. 隔离状态优先（安全第一）
    """

    def resolve(
        self,
        local: VersionedSlot,
        remote: VersionedSlot,
    ) -> VersionedSlot:
        """解决冲突"""
        # 隔离状态优先
        if remote.lifecycle_state == 'quarantined':
            return remote
        if local.lifecycle_state == 'quarantined':
            return local

        # 版本号比较
        if remote.version > local.version:
            return remote
        if local.version > remote.version:
            return local

        # 时间戳比较
        if remote.updated_at > local.updated_at:
            return remote

        return local

    def merge(
        self,
        local: VersionedSlot,
        remote: VersionedSlot,
    ) -> VersionedSlot:
        """
        合并冲突（高级策略）

        适用于可合并的更新。
        """
        winner = self.resolve(local, remote)

        # 合并元数据
        merged = VersionedSlot(
            lu_id=winner.lu_id,
            version=max(local.version, remote.version) + 1,
            updated_at=datetime.now(),
            lifecycle_state=winner.lifecycle_state,
            key_vector=winner.key_vector,
            value_vector=winner.value_vector,
        )

        return merged
```

---

## 5. 性能优化

### 5.1 缓存优化

```python
"""
分布式缓存优化器
"""
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import torch


@dataclass
class CacheConfig:
    """缓存配置"""
    l0_size: int = 128          # GPU 缓存大小
    l1_size: int = 1000         # Redis 缓存大小
    l0_ttl_seconds: int = 300   # L0 TTL
    l1_ttl_seconds: int = 3600  # L1 TTL
    prefetch_count: int = 10    # 预取数量


class HierarchicalCache:
    """
    分层缓存

    L0: GPU 内存（最热）
    L1: Redis（热）
    L2: PostgreSQL（冷）
    """

    def __init__(
        self,
        config: CacheConfig,
        redis_client,
        db_client,
    ):
        self.config = config
        self.redis = redis_client
        self.db = db_client

        # L0: LRU 缓存
        self.l0_cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

        # 访问统计
        self.access_stats: Dict[str, int] = {}

    async def get(
        self,
        lu_id: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取知识向量"""
        # L0 查找
        if lu_id in self.l0_cache:
            self.l0_cache.move_to_end(lu_id)
            self._record_access(lu_id)
            return self.l0_cache[lu_id]

        # L1 查找
        l1_result = await self._get_from_l1(lu_id)
        if l1_result:
            self._promote_to_l0(lu_id, l1_result)
            self._record_access(lu_id)
            return l1_result

        # L2 查找
        l2_result = await self._get_from_l2(lu_id)
        if l2_result:
            await self._promote_to_l1(lu_id, l2_result)
            self._promote_to_l0(lu_id, l2_result)
            self._record_access(lu_id)
            return l2_result

        return None

    async def put(
        self,
        lu_id: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
    ):
        """写入知识向量"""
        vectors = (key_vector, value_vector)

        # 写入所有层
        self._put_to_l0(lu_id, vectors)
        await self._put_to_l1(lu_id, vectors)
        await self._put_to_l2(lu_id, vectors)

    def _promote_to_l0(
        self,
        lu_id: str,
        vectors: Tuple[torch.Tensor, torch.Tensor],
    ):
        """提升到 L0"""
        # 检查容量
        while len(self.l0_cache) >= self.config.l0_size:
            # 淘汰最久未使用的
            self.l0_cache.popitem(last=False)

        self.l0_cache[lu_id] = vectors

    def _put_to_l0(
        self,
        lu_id: str,
        vectors: Tuple[torch.Tensor, torch.Tensor],
    ):
        """写入 L0"""
        self._promote_to_l0(lu_id, vectors)

    async def _get_from_l1(
        self,
        lu_id: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """从 L1 获取"""
        key = f"aga:cache:{lu_id}"
        data = await self.redis.hgetall(key)

        if data:
            key_vector = torch.tensor(eval(data[b'key_vector'].decode()))
            value_vector = torch.tensor(eval(data[b'value_vector'].decode()))
            return (key_vector, value_vector)

        return None

    async def _put_to_l1(
        self,
        lu_id: str,
        vectors: Tuple[torch.Tensor, torch.Tensor],
    ):
        """写入 L1"""
        key = f"aga:cache:{lu_id}"
        await self.redis.hset(key, mapping={
            'key_vector': str(vectors[0].tolist()),
            'value_vector': str(vectors[1].tolist()),
        })
        await self.redis.expire(key, self.config.l1_ttl_seconds)

    async def _promote_to_l1(
        self,
        lu_id: str,
        vectors: Tuple[torch.Tensor, torch.Tensor],
    ):
        """提升到 L1"""
        await self._put_to_l1(lu_id, vectors)

    async def _get_from_l2(
        self,
        lu_id: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """从 L2 获取"""
        # 从 PostgreSQL 获取
        # 实现略
        pass

    async def _put_to_l2(
        self,
        lu_id: str,
        vectors: Tuple[torch.Tensor, torch.Tensor],
    ):
        """写入 L2"""
        # 写入 PostgreSQL
        # 实现略
        pass

    def _record_access(self, lu_id: str):
        """记录访问"""
        self.access_stats[lu_id] = self.access_stats.get(lu_id, 0) + 1

    async def prefetch(self, predicted_lu_ids: List[str]):
        """预取"""
        for lu_id in predicted_lu_ids[:self.config.prefetch_count]:
            if lu_id not in self.l0_cache:
                result = await self._get_from_l1(lu_id)
                if result:
                    self._promote_to_l0(lu_id, result)
```

### 5.2 批量处理优化

```python
"""
批量处理优化器
"""
from typing import List, Dict, Any
import torch
import asyncio


class BatchProcessor:
    """
    批量处理器

    将多个请求合并处理，提高吞吐量。
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._pending_requests: List[Dict[str, Any]] = []
        self._pending_futures: List[asyncio.Future] = []
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()

    async def process(self, request: Dict[str, Any]) -> Any:
        """处理单个请求"""
        future = asyncio.Future()

        async with self._lock:
            self._pending_requests.append(request)
            self._pending_futures.append(future)

            if len(self._pending_requests) >= self.max_batch_size:
                self._batch_event.set()

        # 等待批量处理完成
        return await future

    async def _batch_loop(self):
        """批量处理循环"""
        while True:
            # 等待批量或超时
            try:
                await asyncio.wait_for(
                    self._batch_event.wait(),
                    timeout=self.max_wait_ms / 1000,
                )
            except asyncio.TimeoutError:
                pass

            # 获取当前批量
            async with self._lock:
                if not self._pending_requests:
                    self._batch_event.clear()
                    continue

                batch_requests = self._pending_requests
                batch_futures = self._pending_futures
                self._pending_requests = []
                self._pending_futures = []
                self._batch_event.clear()

            # 批量处理
            try:
                results = await self._process_batch(batch_requests)

                for future, result in zip(batch_futures, results):
                    future.set_result(result)
            except Exception as e:
                for future in batch_futures:
                    future.set_exception(e)

    async def _process_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Any]:
        """处理批量请求"""
        # 合并输入
        batch_hidden_states = torch.stack([
            r['hidden_states'] for r in requests
        ])

        # 批量计算
        # ...

        return results
```

### 5.3 网络优化

```python
"""
网络优化配置
"""
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """网络配置"""
    # Kafka 配置
    kafka_batch_size: int = 16384
    kafka_linger_ms: int = 5
    kafka_compression: str = 'lz4'

    # Redis 配置
    redis_pool_size: int = 10
    redis_socket_timeout: float = 1.0
    redis_retry_on_timeout: bool = True

    # gRPC 配置（可选）
    grpc_max_message_size: int = 4 * 1024 * 1024
    grpc_keepalive_time_ms: int = 10000
    grpc_keepalive_timeout_ms: int = 5000


def create_optimized_kafka_producer(config: NetworkConfig):
    """创建优化的 Kafka 生产者"""
    from aiokafka import AIOKafkaProducer

    return AIOKafkaProducer(
        bootstrap_servers=config.kafka_bootstrap_servers,
        batch_size=config.kafka_batch_size,
        linger_ms=config.kafka_linger_ms,
        compression_type=config.kafka_compression,
        acks='all',  # 确保持久性
    )


def create_optimized_redis_pool(config: NetworkConfig):
    """创建优化的 Redis 连接池"""
    import aioredis

    return aioredis.ConnectionPool.from_url(
        config.redis_url,
        max_connections=config.redis_pool_size,
        socket_timeout=config.redis_socket_timeout,
        retry_on_timeout=config.redis_retry_on_timeout,
    )
```

---

## 6. 实施路线图

### 6.1 阶段规划

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        实施路线图                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: 基础设施 (2 周)                                                   │
│  ├── 部署 Kafka 集群                                                        │
│  ├── 部署 Redis 集群                                                        │
│  └── 配置 PostgreSQL 主从                                                   │
│                                                                             │
│  Phase 2: 核心改造 (3 周)                                                   │
│  ├── 实现 DistributedSynchronizer                                          │
│  ├── 实现 HierarchicalCache                                                │
│  └── 实现 ConflictResolver                                                 │
│                                                                             │
│  Phase 3: 集成测试 (2 周)                                                   │
│  ├── 单元测试                                                               │
│  ├── 集成测试                                                               │
│  └── 性能测试                                                               │
│                                                                             │
│  Phase 4: 灰度上线 (2 周)                                                   │
│  ├── 10% 流量灰度                                                           │
│  ├── 50% 流量灰度                                                           │
│  └── 100% 流量                                                              │
│                                                                             │
│  Phase 5: 优化迭代 (持续)                                                   │
│  ├── 性能优化                                                               │
│  ├── 监控完善                                                               │
│  └── 文档更新                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 关键里程碑

| 里程碑           | 时间   | 交付物      | 验收标准       |
| ---------------- | ------ | ----------- | -------------- |
| M1: 基础设施就绪 | Week 2 | 集群部署    | 可用性 > 99%   |
| M2: 核心功能完成 | Week 5 | 代码 + 测试 | 测试覆盖 > 80% |
| M3: 性能达标     | Week 7 | 性能报告    | 延迟 < 10ms    |
| M4: 全量上线     | Week 9 | 生产部署    | 无故障运行     |

### 6.3 风险与缓解

| 风险           | 概率 | 影响 | 缓解措施        |
| -------------- | ---- | ---- | --------------- |
| Kafka 性能不足 | 低   | 高   | 预留扩容能力    |
| 一致性问题     | 中   | 高   | 完善测试 + 监控 |
| 网络分区       | 低   | 中   | 降级策略        |
| 数据丢失       | 低   | 高   | 多副本 + 备份   |

---

## 附录

### A. 配置模板

```yaml
# config/distributed.yaml
distributed:
    mode: hierarchical_cache # shared_storage, local_cache_sync, hierarchical_cache

    kafka:
        bootstrap_servers:
            - kafka-1:9092
            - kafka-2:9092
            - kafka-3:9092
        topics:
            knowledge_sync: aga_knowledge_sync
            lifecycle_events: aga_lifecycle_events
            quarantine_commands: aga_quarantine_commands
        producer:
            batch_size: 16384
            linger_ms: 5
            compression: lz4
        consumer:
            group_id: aga_sync_${INSTANCE_ID}
            auto_offset_reset: latest

    redis:
        cluster:
            - redis-1:6379
            - redis-2:6379
            - redis-3:6379
        pool_size: 10
        socket_timeout: 1.0

    cache:
        l0_size: 128
        l1_size: 1000
        l0_ttl_seconds: 300
        l1_ttl_seconds: 3600

    sync:
        default_protocol: async_ack
        ack_timeout_seconds: 5.0
        retry_count: 3
```

### B. 监控指标

```yaml
# prometheus/aga_distributed_metrics.yaml
metrics:
    - name: aga_sync_latency_seconds
      type: histogram
      help: Knowledge sync latency
      buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    - name: aga_cache_hit_ratio
      type: gauge
      help: Cache hit ratio by level
      labels: [level] # l0, l1, l2

    - name: aga_conflict_count
      type: counter
      help: Number of conflicts detected
      labels: [namespace, resolution]

    - name: aga_instance_count
      type: gauge
      help: Number of active instances
```

---

_文档结束_
