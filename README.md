# AGA - Auxiliary Governed Attention

<div align="center">

![AGA Logo](https://img.shields.io/badge/AGA-Auxiliary%20Governed%20Attention-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-3.4.0-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**热插拔式知识注入系统 | Zero-Training Knowledge Injection for Frozen Transformers**

[English](#english) | [中文](#中文)

</div>

---

## 中文

### 📖 背景

在大语言模型（LLM）部署后，如何动态整合新知识而不损害现有能力，是一个长期未解决的挑战：

| 现有方案         | 问题                                             |
| ---------------- | ------------------------------------------------ |
| **全量微调**     | 灾难性遗忘、计算成本高                           |
| **LoRA/Adapter** | 需要训练、超参敏感                               |
| **RAG**          | 外部检索而非内化能力，模型无法区分"知道"与"借用" |

**AGA (Auxiliary Governed Attention)** 提出了一种新范式：

> 在冻结的 Transformer 上附加一个可治理的辅助注意力模块，在推理时动态注入知识，同时保持主模型与辅助知识之间的**主权边界**。

### ⚠️ 重要说明：AGA 的定位

**AGA 是 Transformer 模型的热插拔知识管理器，不是完整的治理系统。**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      可控自学习系统 (Continuous Learning System)                 │
│                      - 知识生成、验证、审批                                       │
│                      - 产出：Learning Unit (LU)                                  │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            治理系统 (Governance System)                          │
│                      - 生命周期决策 (CONFIRM/DEPRECATE/QUARANTINE)               │
│                      - 冲突解决、质量评估                                         │
│                      - 产出：Governance Decision                                 │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Bridge (LU Transfer API Client)                         │
│                      - AGAClient / AsyncAGAClient                               │
│                      - HTTP/REST 通信                                           │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AGA Portal (API Server)                               │
│                      ★ 独立部署 - 无 GPU 依赖 ★                                │
│                      - 知识元数据管理 (CRUD)                                     │
│                      - 生命周期状态管理                                          │
│                      - 审计日志                                                 │
│                      - 通过消息队列同步到 Runtime                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│   [PostgreSQL/SQLite]      [Redis/Kafka]        [Runtime Registry]              │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ 同步协议 (Redis Pub-Sub / Kafka)
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   AGA Runtime #1    │  │   AGA Runtime #2    │  │   AGA Runtime #N    │
│   (GPU Server)      │  │   (GPU Server)      │  │   (GPU Server)      │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────────┐  │
│  │  LLM Model    │  │  │  │  LLM Model    │  │  │  │  LLM Model    │  │
│  │  (Frozen)     │  │  │  │  (Frozen)     │  │  │  │  (Frozen)     │  │
│  └───────┬───────┘  │  │  └───────┬───────┘  │  │  └───────┬───────┘  │
│          │          │  │          │          │  │          │          │
│  ┌───────▼───────┐  │  │  ┌───────▼───────┐  │  │  ┌───────▼───────┐  │
│  │  AGA Module   │  │  │  │  AGA Module   │  │  │  │  AGA Module   │  │
│  │  - SlotPool   │  │  │  │  - SlotPool   │  │  │  │  - SlotPool   │  │
│  │  - EntropyGate│  │  │  │  - EntropyGate│  │  │  │  - EntropyGate│  │
│  │  - Decay      │  │  │  │  - Decay      │  │  │  │  - Decay      │  │
│  └───────────────┘  │  │  └───────────────┘  │  │  └───────────────┘  │
│  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────────┐  │
│  │  Sync Agent   │  │  │  │  Sync Agent   │  │  │  │  Sync Agent   │  │
│  │  (订阅变更)    │  │  │  │  (订阅变更)   │  │  │  │  (订阅变更)    │  │
│  └───────────────┘  │  │  └───────────────┘  │  │  └───────────────┘  │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

**AGA 负责**：

-   ✅ 辅助注意力计算（熵门控、内部路由）
-   ✅ 知识存储与检索（Slot Pool、持久化）
-   ✅ 多实例同步（状态复制、事件广播）
-   ✅ 提供 Knowledge Transfer API

**AGA 不负责**（需外部系统提供）：

-   ❌ 知识生成（由持续自学习系统产出 Learning Unit）
-   ❌ 治理决策（何时 CONFIRM/DEPRECATE/QUARANTINE）
-   ❌ 冲突解决（矛盾知识如何处理）
-   ❌ 质量评估（知识的价值判断）
-   ❌ 传播策略（知识是否/何时传播）

> 详细的架构分层方案请参阅 [docs/Architecture_Separation.md](docs/Architecture_Separation.md)

### 🎯 核心特性

| 特性                       | 说明                                                    |
| -------------------------- | ------------------------------------------------------- |
| **零训练注入**             | 知识直接写入 buffer，无需梯度计算                       |
| **热插拔设计**             | 运行时动态添加/移除知识                                 |
| **生命周期支持**           | 支持 PROBATIONARY/CONFIRMED/DEPRECATED/QUARANTINED 状态 |
| **熵门控**                 | 主模型自信时不干预，不确定时才贡献                      |
| **即时隔离**               | 问题知识可立即移除影响                                  |
| **完整可追溯**             | 每个知识槽位绑定 LU ID                                  |
| **多适配器持久化**         | SQLite/Redis/PostgreSQL 分层缓存                        |
| **分布式同步**             | 多实例状态复制与事件广播                                |
| **Knowledge Transfer API** | 供外部治理系统集成                                      |

### 📁 项目结构

```
AGA/
├── aga/                           # 核心模块
│   ├── __init__.py                # 统一导出
│   ├── types.py                   # 共享类型定义
│   ├── unified_config.py          # 统一配置
│   ├── core.py                    # AGA 核心实现（向后兼容）
│   ├── decay.py                   # 持久化衰减
│   ├── entropy_gate.py            # 熵门控
│   ├── exceptions.py              # 异常处理
│   │
│   ├── config/                    # ★ 配置管理 (v3.2 新增)
│   │   ├── portal.py              # Portal 配置
│   │   ├── runtime.py             # Runtime 配置
│   │   ├── sync.py                # 同步协议配置
│   │   └── loader.py              # YAML 加载器
│   │
│   ├── portal/                    # ★ Portal API (v3.2 新增)
│   │   ├── app.py                 # FastAPI 应用工厂
│   │   ├── service.py             # 业务逻辑层（无 GPU）
│   │   ├── routes.py              # HTTP 路由
│   │   └── registry.py            # Runtime 注册表
│   │
│   ├── runtime/                   # ★ Runtime Agent (v3.2 新增)
│   │   ├── agent.py               # 同步代理
│   │   ├── aga_runtime.py         # Runtime 推理模块
│   │   └── cache.py               # 本地知识缓存
│   │
│   ├── sync/                      # ★ 同步协议 (v3.2 新增)
│   │   ├── protocol.py            # 消息协议定义
│   │   ├── publisher.py           # 消息发布器
│   │   ├── subscriber.py          # 消息订阅器
│   │   └── backends.py            # Redis/Kafka/Memory 后端
│   │
│   ├── client/                    # ★ 客户端库 (v3.2 新增)
│   │   └── portal_client.py       # 外部系统集成客户端
│   │
│   ├── api/                       # REST API（单体部署）
│   │   ├── app.py                 # FastAPI 应用
│   │   ├── service.py             # 服务层
│   │   ├── routes.py              # 路由层
│   │   └── client.py              # HTTP 客户端
│   │
│   ├── operator/                  # 算子层
│   │   ├── aga_operator.py        # 统一 AGA 算子
│   │   ├── manager.py             # 多实例管理器
│   │   └── transformer.py         # Transformer 集成
│   │
│   ├── persistence/               # 持久化层
│   │   ├── base.py                # 抽象基类
│   │   ├── memory_adapter.py      # 内存适配器 (L0)
│   │   ├── sqlite_adapter.py      # SQLite 适配器
│   │   ├── redis_adapter.py       # Redis 适配器 (L1)
│   │   ├── postgres_adapter.py    # PostgreSQL 适配器 (L2)
│   │   ├── composite_adapter.py   # 组合适配器
│   │   └── manager.py             # 持久化管理器
│   │
│   ├── distributed/               # 分布式同步层
│   │   ├── sync.py                # 分布式同步器
│   │   ├── coordinator.py         # 实例协调器
│   │   ├── lock.py                # 分布式锁
│   │   ├── governance.py          # 治理参考实现（建议外置）
│   │   └── partition.py           # ★ 网络分区处理 (v3.4.0 新增)
│   │
│   ├── monitoring/                # ★ 监控模块 (v3.4.0 新增)
│   │   └── alerts.py              # 告警规则与仪表盘
│   │
│   └── production/                # 产品化模块
│       ├── config.py              # 生产配置
│       ├── gate.py                # 三段式门控
│       ├── slot_pool.py           # 槽位池管理
│       └── operator.py            # 生产算子
│
├── configs/                       # ★ 配置文件模板 (v3.2 新增)
│   ├── portal_config.yaml         # Portal 配置示例
│   └── runtime_config.yaml        # Runtime 配置示例
│
├── llm/                           # LLM 适配器
│   └── adapters/
│       ├── base.py                # 适配器基类
│       ├── deepseek.py            # DeepSeek 适配器
│       ├── ollama.py              # Ollama 适配器
│       ├── vllm.py                # vLLM 适配器
│       └── openai_compat.py       # OpenAI 兼容适配器
│
├── aga_experiment_tool/           # Web 实验工具
│   ├── app.py                     # Flask 应用
│   └── config.yaml                # 配置文件
│
├── scripts/                       # 启动脚本
│   ├── start_portal.sh            # ★ Portal 启动 (Linux/macOS)
│   ├── start_portal.bat           # ★ Portal 启动 (Windows)
│   ├── start_experiment_tool.sh   # 实验工具 (Linux/macOS)
│   └── start_experiment_tool.bat  # 实验工具 (Windows)
│
├── tests/                         # 单元测试
│   ├── conftest.py           # 全局 fixtures 和配置
│   ├── pytest.ini            # pytest 配置
│   ├── unit/                 # 单元测试（纯函数 & 算子级）
│   │   ├── core/             # 核心模块测试
│   │   ├── entropy_gate/     # 熵门控测试
│   │   ├── entropy_gate/     # 熵门控测试
│   │   ├── decay/            # 衰减模块测试
│   ├── component/            # 组件测试（模块级）
│   │   └── compression/      # 压缩模块测试
│   ├── component/            # 组件测试（模块级）
│   ├── component/            # 组件测试（模块级）
│   ├── component/            # 组件测试（模块级）
│   │   ├── persistence/      # 持久化适配器测试
│   │   ├── slot_pool/        # 槽位池测试
│   │   └── production_gate/  # 生产门控测试
│   ├── integration/          # 集成测试
│   │   ├── test_single_node.py    # 单节点测试
│   │   └── test_multi_runtime.py  # 多 Runtime 测试（Mock）
│   ├── fault/                # 故障注入测试
│   │   ├── test_redis_down.py         # Redis 故障测试
│   │   ├── test_network_partition.py  # 网络分区测试
│   │   └── test_stale_version.py      # 版本过期测试
│   ├── performance/          # 性能测试
│   │   ├── latency.py        # 延迟测试
│   │   ├── memory_growth.py  # 内存增长测试
│   │   └── long_run.py       # 长期运行测试
│   ├── mocks/                # Mock 对象
│   │   ├── redis_mock.py     # Redis Mock
│   │   ├── postgres_mock.py  # PostgreSQL Mock
│   │   ├── kafka_mock.py     # Kafka Mock
│   │   └── http_mock.py      # HTTP Mock
│   └── fixtures/             # 测试数据
│
├── docs/                          # 文档
│   ├── AGA_Implementation_Analysis.md
│   ├── Architecture_Separation.md   # 架构分层方案（重要）
│   ├── Distributed_AGA_Architecture.md
│   ├── Governance_Framework.md
│   └── Multi_Instance_Deployment.md
│
└── README.md
```

### 🚀 快速开始

#### 1. 安装依赖

```bash
# 基础依赖
pip install torch transformers flask pyyaml aiosqlite

# Portal API（v3.2 新增）
pip install fastapi uvicorn httpx pydantic

# 生产环境（可选）
pip install redis asyncpg aiokafka
```

#### 2. 分离部署模式（v3.2 新增）

**v3.2 引入了 Portal + Runtime 分离部署架构，支持大规模生产环境。**

##### 2.1 启动 Portal（API 服务，无需 GPU）

```bash
# 开发模式
./scripts/start_portal.sh --dev

# 生产模式（使用 Redis + PostgreSQL）
./scripts/start_portal.sh --prod --redis localhost --postgres postgresql://...

# 或使用 Python
python -m aga.portal.app --host 0.0.0.0 --port 8081
```

Portal 提供 REST API，访问 `http://localhost:8081/docs` 查看 OpenAPI 文档。

##### 2.2 启动 Runtime（与 LLM 同部署，需要 GPU）

```python
from aga.runtime import RuntimeAgent
from aga.config import RuntimeConfig

# 创建配置
config = RuntimeConfig.for_production(
    instance_id="runtime-001",
    portal_url="http://portal:8081",
    redis_host="localhost",
    hidden_dim=4096,
    num_slots=100,
)

# 创建 Agent
agent = RuntimeAgent(config)

# 初始化并启动
await agent.initialize()
await agent.start()

# 附加到模型
aga_layer = agent.attach_to_layer(transformer_layer)

# 使用（推理循环中）
output, diagnostics = agent.get_runtime().forward(hidden_states, attention_mask)
```

##### 2.3 外部治理系统集成

```python
from aga.client import AGAClient

# 创建客户端
client = AGAClient("http://portal:8081")

# 注入知识
client.inject_knowledge(
    lu_id="knowledge_001",
    condition="当用户询问法国首都",
    decision="回答巴黎",
    key_vector=[...],  # 编码后的向量
    value_vector=[...],
    namespace="geography",
    lifecycle_state="probationary",
)

# 确认知识
client.confirm("knowledge_001", reason="验证通过")

# 隔离问题知识
client.quarantine("knowledge_002", reason="检测到错误")

# 查询统计
stats = client.get_statistics(namespace="geography")
```

#### 3. 单体部署模式（传统方式）

##### 3.1 启动实验工具

```bash
# Linux/macOS
./scripts/start_experiment_tool.sh

# Windows
scripts\start_experiment_tool.bat

# 或直接运行
python -m aga_experiment_tool.app --port 8765
```

访问 `http://localhost:8765`，默认密码：`aga_experiment_2026`

##### 3.2 代码使用（单体模式）

```python
from aga import AGAConfig, AGAOperator, AGAManager
from aga import LifecycleState
from aga.persistence import SQLiteAdapter, PersistenceManager

# 创建配置
config = AGAConfig()
config.slot_pool.hidden_dim = 768
config.slot_pool.max_slots = 100

# 创建 AGA 算子
aga = AGAOperator(config)
aga.eval()

# 注入知识
import torch
key_vector = torch.randn(64)
value_vector = torch.randn(768)

aga.inject_knowledge(
    slot_idx=0,
    key_vector=key_vector,
    value_vector=value_vector,
    lu_id="LU_001_paris",
    lifecycle_state=LifecycleState.PROBATIONARY,
    condition="capital of France",
    decision="Paris",
)

# 确认知识
aga.update_lifecycle(0, LifecycleState.CONFIRMED)

# 查看统计
print(aga.get_statistics())
```

### 🏗️ 部署架构选择

AGA v3.2 提供两种部署模式：

| 模式         | 适用场景           | 特点                               |
| ------------ | ------------------ | ---------------------------------- |
| **单体部署** | 开发测试、单机推理 | 简单、API 与 AGA 同进程            |
| **分离部署** | 多实例生产、云原生 | Portal 无 GPU、Runtime 按 LLM 扩展 |

#### 分离部署架构优势

```
┌────────────────────────────────────────────────────────────────────────┐
│                           分离部署 vs 单体部署                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  单体部署                        分离部署                               │
│  ┌──────────────────┐           ┌──────────────────┐                   │
│  │ API + AGA + LLM  │           │ Portal (API)     │ ← 无 GPU          │
│  │ (同一进程)       │           │ - 知识管理       │                    │
│  │                  │           │ - 审计日志       │                   │
│  └──────────────────┘           └────────┬─────────┘                  │
│  优点：简单                               │ Redis/Kafka                │
│  缺点：                          ┌───────┼───────┐                     │
│  - API 占用 GPU                  ▼       ▼       ▼                     │
│  - 难以水平扩展                ┌─────┐ ┌─────┐ ┌─────┐                  │
│  - 单点故障                    │RT-1│ │RT-2│ │RT-N│ ← GPU               │
│                                │+LLM│ │+LLM│ │+LLM│                    │
│                                └─────┘ └─────┘ └─────┘                 │
│                                优点：                                  │
│                                - Portal 独立扩展                       │
│                                - Runtime 按 LLM 扩展                   │
│                                - 故障隔离                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

| 组件    | 单体部署      | 分离部署           |
| ------- | ------------- | ------------------ |
| Portal  | 与 AGA 同进程 | 独立服务（无 GPU） |
| Runtime | N/A           | 与 LLM 同机（GPU） |
| 同步    | 内存直接访问  | Redis Pub/Sub      |
| 持久化  | 本地文件      | PostgreSQL         |
| 扩展性  | 垂直扩展      | 水平扩展           |

### 💾 多适配器持久化

AGA v3.0+ 支持分层缓存架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    多适配器持久化架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ L0: Memory  │─▶│ L1: Redis   │─▶│ L2: Postgres│         │
│  │ (128 slots) │  │ (1000 slots)│  │ (无限)      │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  读取策略：L0 → L1 → L2（miss 时向下查找并提升）               │
│  写入策略：write-through（写入所有层）                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
from aga.persistence import (
    MemoryAdapter,
    SQLiteAdapter,
    RedisAdapter,
    PostgresAdapter,
    CompositeAdapter,
    PersistenceManager,
)

# 开发环境：SQLite
adapter = SQLiteAdapter("aga_data.db")
await adapter.connect()

# 生产环境：分层缓存
composite = CompositeAdapter(
    l0_adapter=MemoryAdapter(max_slots_per_namespace=128),
    l1_adapter=RedisAdapter(host="redis-host"),
    l2_adapter=PostgresAdapter(host="pg-host"),
)
await composite.connect()

# 使用管理器
persistence = PersistenceManager(composite, namespace="production")
await persistence.save_aga_state(aga)
await persistence.load_aga_state(aga)
```

### 🌐 分布式部署

AGA v3.1 支持多实例部署：

```python
from aga.distributed import DistributedSynchronizer

# 创建同步器
sync = DistributedSynchronizer(
    instance_id="instance-1",
    namespace="production",
    backend="redis",
    config={"host": "redis-host"},
)
await sync.start()

# 同步知识注入（由外部治理系统调用）
await sync.sync_knowledge_inject(
    lu_id="LU_001",
    slot_idx=0,
    key_vector=key_vec.tolist(),
    value_vector=value_vec.tolist(),
    lifecycle_state=LifecycleState.PROBATIONARY,
)

# 同步生命周期更新
await sync.sync_lifecycle_update("LU_001", LifecycleState.CONFIRMED)

# 同步隔离
await sync.sync_quarantine("LU_001")
```

### 🔌 外部治理系统集成

AGA 提供 Knowledge Transfer API，供外部治理系统（如持续自学习系统）集成：

```python
# ==================== 外部治理系统调用 AGA ====================

from aga import AGAOperator, AGAConfig, LifecycleState
from aga.persistence import SQLiteAdapter, PersistenceManager

# 1. 初始化 AGA
config = AGAConfig()
aga = AGAOperator(config)

adapter = SQLiteAdapter("aga_data.db")
await adapter.connect()
pm = PersistenceManager(adapter, namespace="production")

# 2. 外部治理系统产出 Learning Unit 后，注入 AGA
def on_learning_unit_approved(lu: LearningUnit):
    """当持续自学习系统审批通过一个 Learning Unit 时调用"""

    # 编码 key/value（可选，AGA 也可以自动编码）
    key_vector = model.encode(lu.condition)
    value_vector = model.encode(lu.decision)

    # 注入知识
    slot_idx = aga.inject_knowledge(
        lu_id=lu.id,
        key_vector=key_vector,
        value_vector=value_vector,
        lifecycle_state=LifecycleState.PROBATIONARY,  # 新知识默认试用
        reliability=0.3,
        condition=lu.condition,
        decision=lu.decision,
        metadata={"source": lu.source, "version": lu.version},
    )

    # 持久化
    await pm.save_slot(aga, slot_idx)

# 3. 外部治理系统决定确认知识
def on_knowledge_confirmed(lu_id: str):
    """当治理系统确认知识可信时调用"""
    aga.update_lifecycle_by_lu_id(lu_id, LifecycleState.CONFIRMED)
    aga.update_reliability(lu_id, 1.0)

# 4. 外部治理系统决定隔离知识
def on_knowledge_quarantined(lu_id: str, reason: str):
    """当治理系统检测到问题知识时调用"""
    aga.quarantine_by_lu_id(lu_id)
    # 记录审计日志（由治理系统负责）
    audit_log.record(lu_id, "quarantine", reason)

# 5. 监听 AGA 事件，反馈给治理系统
@aga.on_event("knowledge_hit")
def on_hit(lu_id: str, hit_count: int, context: dict):
    """知识被命中时，反馈给治理系统"""
    governance_system.record_hit(lu_id, context)

@aga.on_event("low_confidence_query")
def on_low_confidence(query_context: dict):
    """当 AGA 无法找到匹配知识时，提示治理系统"""
    governance_system.suggest_new_knowledge(query_context)
```

#### 治理系统需要实现的能力

| 能力             | 说明                                  | AGA 提供的支持                  |
| ---------------- | ------------------------------------- | ------------------------------- |
| **知识生成**     | 从用户交互、反馈、外部源产生 LU       | `inject_knowledge()` API        |
| **知识验证**     | 验证知识的正确性、安全性              | -                               |
| **生命周期决策** | 决定何时 CONFIRM/DEPRECATE/QUARANTINE | `update_lifecycle()` API        |
| **冲突解决**     | 处理矛盾知识                          | `list_knowledge()` 获取相似知识 |
| **质量评估**     | 评估知识的价值                        | `get_statistics()` 获取命中统计 |
| **传播策略**     | 决定知识是否/何时传播                 | `sync_knowledge_inject()` API   |
| **审批流程**     | 人类审核高风险知识                    | `update_lifecycle()` API        |

#### 示例：与持续自学习系统集成

```
┌─────────────────────────────────────────────────────────────┐
│              持续自学习系统（Continuous Learning System）    │
│                                                            │
│  用户交互 ──┐                                               │
│            │      ┌─────────┐      ┌─────────┐             │
│  外部反馈 ──┼──▶  │ 知识    │ ──▶  │ 治理    │             │
│            │      │ 候选池  │       │ 审批    │             │
│  自动提取 ──┘      └─────────┘      └────┬────┘             │
│                                         │                   │
│                                         ▼                   │
│                               ┌─────────────────┐           │
│                               │ Learning Unit   │           │
│                               │ (审批通过)       │           │
│                               └────────┬────────┘           │
│                                        │                    │
└────────────────────────────────────────┼────────────────────┘
                                         │
                                         │ inject_knowledge()
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                        AGA 知识管理器                        │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Slot 0  │  │ Slot 1  │  │ Slot 2  │  │  ...    │         │
│  │ LU_001  │  │ LU_002  │  │ LU_003  │  │         │         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frozen Transformer                       │
└─────────────────────────────────────────────────────────────┘
```

> `aga/distributed/governance.py` 提供了治理逻辑的**参考实现**，
> 但在生产环境中，建议您在自己的持续自学习系统中实现这些逻辑。

### 📊 知识生命周期

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
    ┌────────────────────────┐                                │
    │     PROBATIONARY       │──── 使用指标正常   ─────────────┤
    │     (r = 0.3)          │                                │
    └───────────┬────────────┘                                │
                │ 治理审批                                     │
                ▼                                             │
    ┌────────────────────────┐                                │
    │      CONFIRMED         │──── 异常检测   ─────────────────┤
    │     (r = 1.0)          │                                │
    └───────────┬────────────┘                                │
                │ 弃用请求                                     │
                ▼                                             │
    ┌────────────────────────┐                                │
    │      DEPRECATED        │──── 保留期过期   ───────────────┤
    │     (r = 0.1)          │                                │
    └───────────┬────────────┘                                │
                │ 清理                                        │
                ▼                                             │
    ┌────────────────────────┐                                │
    │     QUARANTINED        │◀──────────────────────────────┘
    │     (r = 0.0)          │      （紧急隔离可直接跳转）
    └────────────────────────┘
```

### 🔌 LLM 适配器

| 适配器                | 说明              | 使用场景 |
| --------------------- | ----------------- | -------- |
| `OllamaAdapter`       | Ollama 本地模型   | 开发测试 |
| `VLLMAdapter`         | vLLM 高性能推理   | 生产部署 |
| `DeepSeekAdapter`     | DeepSeek API/本地 | API 调用 |
| `OpenAICompatAdapter` | OpenAI 兼容接口   | 通用     |

```python
from llm.adapters import OllamaAdapter

adapter = OllamaAdapter(
    base_url="http://localhost:11434",
    model="qwen2.5:7b"
)

response = adapter.chat([
    {"role": "user", "content": "Hello"}
])
print(response.content)
```

### 📜 核心原理

**知识检索机制**：

```
输入: hidden_states X ∈ ℝ^{n×d}

1. 查询投影:
   Q' = X · W_down    # [n, d] → [n, d_b]

2. 注意力分数（带可靠性掩码）:
   scores_ij = (Q'_i · K_j^T) / √d_b + log(r_j)

   其中 r_j 是槽位 j 的可靠性（隔离槽位 r=0 → log(0)=-∞）

3. Softmax 检索:
   α_ij = softmax(scores_i)    # 隔离槽位权重为 0
   O_s = Σ_j α_ij · V_j

4. 熵门控融合:
   gate = σ(w₁·H + b)          # H = 主注意力熵
   Ô = O_primary + gate ⊙ O_s
```

**三段式门控**：

```
Gate-0 (先验门控)     Gate-1 (置信门控)     Gate-2 (Top-k 路由)
      │                     │                     │
      ▼                     ▼                     ▼
  namespace/app_id      不确定性估计           Top-k 槽位选择
      │                     │                     │
      ▼                     ▼                     ▼
  DISABLED/REQUIRED    BYPASS/PASS            路由分数
```

### ✅ 已实现的规模化优化

| 优化方案                           | 状态        | 实现位置                                      |
| ---------------------------------- | ----------- | --------------------------------------------- |
| **分层知识存储 (L0/L1/L2)**        | ✅ 完全实现 | `aga/persistence/composite_adapter.py`        |
| **Write-through / Read-promotion** | ✅ 完全实现 | `CompositeAdapter`                            |
| **Top-k 路由优化**                 | ✅ 完全实现 | `aga/core.py::SlotRouter`                     |
| **分块计算避免 OOM**               | ✅ 完全实现 | `SlotRouter._chunked_top_k()`                 |
| **三段式门控 (Gate-0/1/2)**        | ✅ 完全实现 | `aga/production/gate.py`                      |
| **多源熵信号**                     | ✅ 完全实现 | `aga/entropy_gate.py::EntropySource`          |
| **自适应阈值**                     | ✅ 完全实现 | `EntropyGateConfig.enable_adaptive_threshold` |
| **持久化衰减**                     | ✅ 完全实现 | `aga/decay.py::PersistenceDecay`              |
| **硬重置机制**                     | ✅ 完全实现 | `DecayConfig.enable_hard_reset`               |
| **命中计数 / 连续未命中**          | ✅ 完全实现 | `Slot.hit_count`, `consecutive_misses`        |
| **命名空间隔离**                   | ✅ 完全实现 | `aga/production/slot_pool.py`                 |
| **Early Exit**                     | ✅ 完全实现 | `AGAConfig.enable_early_exit`                 |
| **KV 向量压缩**                    | ✅ 完全实现 | `aga/persistence/compression.py`              |
| **动态槽位扩展**                   | ✅ 完全实现 | `aga/production/dynamic_slots.py`             |
| **知识版本控制**                   | ✅ 完全实现 | `aga/persistence/versioning.py`               |
| **编码器缓存**                     | ✅ 完全实现 | `aga/encoder/cache.py`                        |
| **连接池管理**                     | ✅ 完全实现 | `aga/persistence/pool.py`                     |
| **知识冲突检测**                   | ✅ 完全实现 | `aga/api/conflict.py`                         |
| **分布式追踪**                     | ✅ 完全实现 | `aga/api/tracing.py`                          |
| **混合精度支持**                   | ✅ 完全实现 | `aga/operator/optimizations.py`               |
| **CUDA Graph 优化**                | ✅ 完全实现 | `aga/operator/optimizations.py`               |
| **多头注意力并行**                 | ✅ 完全实现 | `aga/operator/parallel_attention.py` (v3.4.1) |
| **FlashAttention 集成**            | ✅ 完全实现 | `aga/operator/parallel_attention.py` (v3.4.1) |
| **网络分区处理**                   | ✅ 完全实现 | `aga/distributed/partition.py` (v3.4.1)       |
| **向量时钟一致性**                 | ✅ 完全实现 | `aga/distributed/partition.py` (v3.4.1)       |
| **告警规则配置**                   | ✅ 完全实现 | `aga/monitoring/alerts.py` (v3.4.1)           |
| **Grafana 仪表盘**                 | ✅ 完全实现 | `aga/monitoring/alerts.py` (v3.4.1)           |

### 🏛️ 内部治理系统

AGA 提供**基础内部治理能力**，同时建议集成外部治理系统：

#### 内部治理（已实现）

```python
from aga.distributed import (
    GovernanceArbiter,      # 治理裁决器
    PropagationThrottler,   # 传播节流器
    TrustTier,              # 信任层级
    PropagationPolicy,      # 传播策略
)

# 创建治理裁决器
arbiter = GovernanceArbiter(
    instance_id="instance-1",
    quorum_size=2,           # 少数即生效
    risk_threshold=0.3,      # 风险阈值
)

# 注册槽位信任层级
arbiter.register_slot(
    lu_id="LU_001",
    trust_tier=TrustTier.S1_EXPERIENCE,  # 经验槽：可回滚
)

# 评估传播（默认拒绝未注册知识）
decision = await arbiter.evaluate_propagation("LU_001", "instance-2")
if decision.verdict == GovernanceVerdict.ALLOW:
    # 允许传播
    pass

# 评估隔离（quorum 机制）
decision = await arbiter.evaluate_quarantine("LU_001", "异常输出", "instance-1")
# 达到 quorum 后自动生效
```

#### 信任层级（语义主权分区）

| 层级           | 传播策略 | 说明                      |
| -------------- | -------- | ------------------------- |
| **S0: 加速槽** | 立即传播 | 推理缓存，可丢失可重建    |
| **S1: 经验槽** | 延迟传播 | 60 秒观察期后传播，可回滚 |
| **S2: 策略槽** | 门控传播 | 需要审批（2 票）后传播    |
| **S3: 禁止槽** | 禁止传播 | 只读，不传播到其他实例    |

#### 内部 vs 外部治理

| 能力         | 内部治理  | 外部治理（建议） |
| ------------ | --------- | ---------------- |
| 信任层级分区 | ✅ 已实现 | 可扩展           |
| 传播节流     | ✅ 已实现 | 可扩展           |
| Quorum 隔离  | ✅ 已实现 | 可扩展           |
| 生命周期决策 | ⚠️ 框架级 | **推荐外置**     |
| 知识生成验证 | ❌ 不提供 | **必须外置**     |
| 冲突解决     | ⚠️ 策略级 | **推荐外置**     |
| 质量评估     | ⚠️ 统计级 | **推荐外置**     |
| 人类审批流程 | ❌ 不提供 | **必须外置**     |

#### 建议的集成模式

```
┌─────────────────────────────────────────────────────────────┐
│              外部治理系统（持续自学习系统）                    │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 知识生成     │  │ 质量评估    │   │ 人类审批    │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│         ┌────────────────────────────────┐                  │
│         │        Learning Unit           │                  │
│         │  + lifecycle_state 决策        │                  │
│         │  + trust_tier 决策             │                  │
│         └────────────────┬───────────────┘                  │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           │ inject_knowledge()
                           │ update_lifecycle()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGA 内部治理层                             │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 传播节流     │  │ Quorum 隔离 │  │ 信任分区     │          │
│  │ (延迟/速率)  │  │ (少数即生效)│   │ (S0-S3)     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  职责：执行治理决策，不做治理判断                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> `aga/distributed/governance.py` 提供参考实现，
> 生产环境建议将治理决策逻辑外置到持续自学习系统。

### ⚠️ 持续可控学习系统的挑战与解决方案

AGA 作为 Transformer 模型的**热插拔式知识管理器**，在持续可控学习场景下面临以下核心挑战：

#### 挑战一：知识容量的天花板

**问题描述**：

-   AGA 的知识存储依赖显式槽位（Slot Pool）
-   槽位数量受限于 GPU 显存和注意力计算复杂度
-   当知识规模达到 10,000+ 槽位时，检索效率下降

**解决方案**：分层知识存储架构

```
┌─────────────────────────────────────────────────────────────┐
│  L0: GPU 显存池    热知识（256-512 槽位，纳秒级检索）          │
│         ↓ 动态换入换出                                       │
│  L1: CPU 内存/Redis 温知识（10,000+ 槽位，微秒级检索）         │
│         ↓ 按需加载                                           │
│  L2: PostgreSQL    冷知识（百万级槽位，毫秒级检索）            │
└─────────────────────────────────────────────────────────────┘
关键：知识始终保持独立槽位形态，不与模型参数混合
```

#### 挑战二：知识老化与遗忘

**问题描述**：

-   知识有时效性，过期知识需要被遗忘
-   AGA 的隔离机制只能"禁用"而非真正遗忘
-   长期运行后会积累大量 DEPRECATED/QUARANTINED 槽位

**解决方案**：优雅遗忘机制

```
遗忘流程（保持主权边界）：
1. DEPRECATED → 观察期（可回滚）
2. QUARANTINED → 隔离期（不参与推理）
3. 归档到冷存储（保留审计日志）
4. 释放槽位资源（可被新知识复用）

知识整合（非蒸馏）：
- 合并语义重叠的槽位
- 保持每个原始 LU ID 的追溯性
- 释放冗余槽位
```

#### 挑战三：知识一致性与冲突

**问题描述**：

-   多个槽位可能包含矛盾的知识
-   推理时如何决定采信哪个？
-   分布式场景下冲突更加复杂

**解决方案**：槽位级冲突解决

```
检测层：
  - 语义相似度 > 0.9 但 decision 不同 → 潜在冲突
  - 同一 condition 多个槽位 → 版本冲突

解决策略：
  1. 时间优先：保留最新知识
  2. 可靠性优先：保留 reliability 更高的槽位
  3. 治理优先：标记冲突，等待人类裁决
  4. 共存策略：保留所有版本，推理时按上下文选择

关键：冲突解决在槽位级别，不影响主模型
```

#### 挑战四：跨模型知识迁移

**问题描述**：

-   AGA 槽位的 key/value 向量是模型特定的
-   更换基座模型后，现有知识无法直接使用
-   这是否意味着需要"重新训练"？

**解决方案**：零训练知识迁移

```
保留的内容（可迁移）：
  - lu_id, condition, decision（语义描述）
  - lifecycle_state, reliability（治理状态）
  - hit_count, metadata（使用统计）

重新生成的内容：
  - key_vector = new_model.encode(condition)
  - value_vector = new_model.encode(decision)

迁移流程：
  1. 导出知识描述（JSON/Protobuf）
  2. 在新模型上重新编码 key/value
  3. 保持原有治理状态和生命周期
  4. 验证迁移后的语义保真度

关键：知识的"含义"迁移，而非"参数"迁移
```

#### 挑战五：推理延迟的累积

**问题描述**：

-   每层 Transformer 都有 AGA 开销
-   槽位数量增加 → 注意力计算增加
-   可能成为推理瓶颈

**解决方案**：推理优化（保持热插拔特性）

```
1. Early Exit（已实现）：Gate-0 直接旁路，跳过整个 AGA
2. 稀疏注意力：只计算 Top-k 相关槽位的注意力
3. 异步预取：预测下一层需要的槽位，提前加载
4. 批量槽位融合：相似槽位合并计算，减少重复

原则：优化是"实现层面"的，不改变"架构层面"
```

#### 挑战六：对抗性知识注入

**问题描述**：

-   恶意用户可能注入有害知识
-   AGA 的"零训练注入"使攻击成本更低
-   传统的安全机制（如 RLHF）不适用

**解决方案**：知识安全防护机制

```
注入时检查：
  - 语义安全过滤（检测有害内容）
  - 来源验证（知识来源可信度）
  - 格式校验（防止注入攻击）

运行时防护：
  - 默认 PROBATIONARY 状态（可靠性 0.3）
  - 异常检测（输出突变 → 自动隔离）
  - 影响范围限制（propagation_radius）

治理层保障：
  - 人类审批流程（S2/S3 级别知识）
  - quorum 投票隔离（少数即生效）
  - 完整审计日志

关键：安全是"治理问题"，不是"训练问题"
```

### 🔮 未来优化方向

#### 短期优化（v3.2）

1. **治理层增强**

    - 人类治理接口（审批工作流 UI）
    - 知识质量自动评估
    - 冲突检测和解决

2. **性能优化**

    - GPU 内存池化与动态换入换出
    - 批量推理优化
    - 缓存预热策略

3. **监控增强**
    - Prometheus 指标导出
    - 分布式追踪
    - 异常检测告警

#### 中期优化（v4.0）

1. **知识整合（非蒸馏）**

    - 语义重叠槽位合并
    - 知识压缩（保持槽位形态）
    - 冗余检测与清理

2. **多模态支持**

    - 图像知识注入
    - 跨模态检索
    - 多模态门控

3. **联邦知识共享**
    - 跨组织知识同步
    - 隐私保护（差分隐私槽位）
    - 知识产权标记

#### 长期愿景（v5.0+）

1. **认知架构**

    - 多层知识表示（事实/规则/策略）
    - 推理链路可解释
    - 元认知能力（知道自己不知道什么）

2. **自主治理**

    - 知识自我评估
    - 自动生命周期管理
    - 自我修复（检测到问题知识自动隔离）

3. **跨模型知识迁移**
    - 语义级知识导出/导入
    - 模型无关的知识表示
    - 知识版本控制

### ❌ 明确不采用的方向

以下方向与 AGA 核心理念冲突，**明确不纳入路线图**：

| 被排除的方向           | 排除原因                       |
| ---------------------- | ------------------------------ |
| **知识蒸馏到 LoRA**    | 违背"零训练"原则，破坏主权边界 |
| **微调基座模型**       | 回归预训练老路，丧失热插拔特性 |
| **将槽位融入模型参数** | 无法追溯和隔离单个知识         |
| **RLHF 式对齐**        | 需要训练，且无法针对单个知识   |

> **AGA 的核心承诺**：知识永远是"外挂"，永远可追溯，永远可隔离。

### 📄 相关论文

本项目基于论文《Auxiliary Governed Attention: A Governable, Inference-time Auxiliary Attention Mechanism with Sovereign Boundaries for Frozen Transformers》实现。

### 📊 代码统计

| 模块                   | 行数        | 说明                     |
| ---------------------- | ----------- | ------------------------ |
| `aga/` (核心)          | 4,500+      | 核心 AGA 实现            |
| `aga/api/`             | 2,800+      | 单体部署 REST API        |
| `aga/portal/`          | 1,500+      | Portal API 服务          |
| `aga/operator/`        | 2,400+      | 统一算子层 (+700 v3.4.0) |
| `aga/persistence/`     | 4,200+      | 多适配器持久化           |
| `aga/distributed/`     | 2,200+      | 分布式同步 (+700 v3.4.0) |
| `aga/production/`      | 4,500+      | 产品化模块               |
| `aga/runtime/`         | 900+        | Runtime Agent            |
| `aga/sync/`            | 1,200+      | 同步协议                 |
| `aga/config/`          | 700+        | 配置管理                 |
| `aga/client/`          | 600+        | 客户端库                 |
| `aga/encoder/`         | 1,500+      | 编码器模块               |
| `aga/monitoring/`      | 1,000+      | 监控告警 (v3.4.0 新增)   |
| `llm/`                 | 2,550       | LLM 适配器               |
| `aga_experiment_tool/` | 1,149       | Web 实验工具             |
| **总计**               | **32,000+** | (不含测试和示例)         |

---

## English

### 📖 Background

How to dynamically integrate new knowledge after LLM deployment without compromising existing capabilities has been a long-standing unresolved challenge:

| Existing Solutions   | Issues                                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Full Fine-tuning** | Catastrophic forgetting, high computational cost                                                            |
| **LoRA/Adapter**     | Requires training, hyperparameter sensitive                                                                 |
| **RAG**              | External retrieval rather than internalized capability, model cannot distinguish "knowing" from "borrowing" |

**AGA (Auxiliary Governed Attention)** proposes a new paradigm:

> Attach a governable auxiliary attention module to a frozen Transformer, dynamically inject knowledge during inference, while maintaining **sovereign boundaries** between the main model and auxiliary knowledge.

### ⚠️ Important Note: AGA's Positioning

**AGA is a hot-pluggable knowledge manager for Transformer models, not a complete governance system.**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   Controllable Self-Learning System (Continuous Learning System)│
│                      - Knowledge generation, validation, approval               │
│                      - Output: Learning Unit (LU)                               │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Governance System                                    │
│                      - Lifecycle decisions (CONFIRM/DEPRECATE/QUARANTINE)       │
│                      - Conflict resolution, quality assessment                  │
│                      - Output: Governance Decision                              │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Bridge (LU Transfer API Client)                         │
│                      - AGAClient / AsyncAGAClient                               │
│                      - HTTP/REST communication                                  │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AGA Portal (API Server)                               │
│                      ★ Independent deployment - No GPU dependency ★            │
│                      - Knowledge metadata management (CRUD)                     │
│                      - Lifecycle state management                               │
│                      - Audit logs                                               │
│                      - Sync to Runtime via message queue                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│   [PostgreSQL/SQLite]      [Redis/Kafka]        [Runtime Registry]              │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ Sync Protocol (Redis Pub-Sub / Kafka)
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   AGA Runtime #1    │  │   AGA Runtime #2    │  │   AGA Runtime #N    │
│   (GPU Server)      │  │   (GPU Server)      │  │   (GPU Server)      │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────────┐  │
│  │  LLM Model    │  │  │  │  LLM Model    │  │  │  │  LLM Model    │  │
│  │  (Frozen)     │  │  │  │  (Frozen)     │  │  │  │  (Frozen)     │  │
│  └───────┬───────┘  │  │  └───────┬───────┘  │  │  └───────┬───────┘  │
│          │          │  │          │          │  │          │          │
│  ┌───────▼───────┐  │  │  ┌───────▼───────┐  │  │  ┌───────▼───────┐  │
│  │  AGA Module   │  │  │  │  AGA Module   │  │  │  │  AGA Module   │  │
│  │  - SlotPool   │  │  │  │  - SlotPool   │  │  │  │  - SlotPool   │  │
│  │  - EntropyGate│  │  │  │  - EntropyGate│  │  │  │  - EntropyGate│  │
│  │  - Decay      │  │  │  │  - Decay      │  │  │  │  - Decay      │  │
│  └───────────────┘  │  │  └───────────────┘  │  │  └───────────────┘  │
│  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────────┐  │
│  │  Sync Agent   │  │  │  │  Sync Agent   │  │  │  │  Sync Agent   │  │
│  │  (Subscribe)  │  │  │  │  (Subscribe)  │  │  │  │  (Subscribe)  │  │
│  └───────────────┘  │  │  └───────────────┘  │  │  └───────────────┘  │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

**AGA is responsible for**:

-   ✅ Auxiliary attention computation (entropy gating, internal routing)
-   ✅ Knowledge storage and retrieval (Slot Pool, persistence)
-   ✅ Multi-instance synchronization (state replication, event broadcasting)
-   ✅ Providing Knowledge Transfer API

**AGA is NOT responsible for** (requires external systems):

-   ❌ Knowledge generation (produced by continuous self-learning system as Learning Units)
-   ❌ Governance decisions (when to CONFIRM/DEPRECATE/QUARANTINE)
-   ❌ Conflict resolution (how to handle contradictory knowledge)
-   ❌ Quality assessment (value judgment of knowledge)
-   ❌ Propagation strategy (whether/when to propagate knowledge)

> For detailed architecture layering, please refer to [docs/Architecture_Separation.md](docs/Architecture_Separation.md)

### 🎯 Core Features

| Feature                       | Description                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------- |
| **Zero-training Injection**   | Knowledge directly written to buffer, no gradient computation needed          |
| **Hot-pluggable Design**      | Dynamically add/remove knowledge at runtime                                   |
| **Lifecycle Support**         | Support for PROBATIONARY/CONFIRMED/DEPRECATED/QUARANTINED states              |
| **Entropy Gating**            | No intervention when main model is confident, contributes only when uncertain |
| **Instant Isolation**         | Problematic knowledge can be immediately removed from affecting inference     |
| **Complete Traceability**     | Each knowledge slot bound to LU ID                                            |
| **Multi-adapter Persistence** | SQLite/Redis/PostgreSQL layered caching                                       |
| **Distributed Sync**          | Multi-instance state replication and event broadcasting                       |
| **Knowledge Transfer API**    | For external governance system integration                                    |

### 📁 Project Structure

```
AGA/
├── aga/                           # Core modules
│   ├── __init__.py                # Unified exports
│   ├── types.py                   # Shared type definitions
│   ├── unified_config.py          # Unified configuration
│   ├── core.py                    # AGA core implementation (backward compatible)
│   ├── decay.py                   # Persistence decay
│   ├── entropy_gate.py            # Entropy gating
│   ├── exceptions.py              # Exception handling
│   │
│   ├── config/                    # ★ Configuration management (v3.2 new)
│   │   ├── portal.py              # Portal configuration
│   │   ├── runtime.py             # Runtime configuration
│   │   ├── sync.py                # Sync protocol configuration
│   │   └── loader.py              # YAML loader
│   │
│   ├── portal/                    # ★ Portal API (v3.2 new)
│   │   ├── app.py                 # FastAPI application factory
│   │   ├── service.py             # Business logic layer (no GPU)
│   │   ├── routes.py              # HTTP routes
│   │   └── registry.py            # Runtime registry
│   │
│   ├── runtime/                   # ★ Runtime Agent (v3.2 new)
│   │   ├── agent.py               # Sync agent
│   │   ├── aga_runtime.py         # Runtime inference module
│   │   └── cache.py               # Local knowledge cache
│   │
│   ├── sync/                      # ★ Sync protocol (v3.2 new)
│   │   ├── protocol.py            # Message protocol definition
│   │   ├── publisher.py           # Message publisher
│   │   ├── subscriber.py          # Message subscriber
│   │   └── backends.py            # Redis/Kafka/Memory backends
│   │
│   ├── client/                    # ★ Client library (v3.2 new)
│   │   └── portal_client.py       # External system integration client
│   │
│   ├── api/                       # REST API (monolithic deployment)
│   │   ├── app.py                 # FastAPI application
│   │   ├── service.py             # Service layer
│   │   ├── routes.py              # Route layer
│   │   └── client.py              # HTTP client
│   │
│   ├── operator/                  # Operator layer
│   │   ├── aga_operator.py        # Unified AGA operator
│   │   ├── manager.py             # Multi-instance manager
│   │   └── transformer.py         # Transformer integration
│   │
│   ├── persistence/               # Persistence layer
│   │   ├── base.py                # Abstract base class
│   │   ├── memory_adapter.py      # Memory adapter (L0)
│   │   ├── sqlite_adapter.py      # SQLite adapter
│   │   ├── redis_adapter.py       # Redis adapter (L1)
│   │   ├── postgres_adapter.py    # PostgreSQL adapter (L2)
│   │   ├── composite_adapter.py   # Composite adapter
│   │   └── manager.py             # Persistence manager
│   │
│   ├── distributed/               # Distributed sync layer
│   │   ├── sync.py                # Distributed synchronizer
│   │   ├── coordinator.py         # Instance coordinator
│   │   ├── lock.py                # Distributed lock
│   │   └── governance.py          # Governance reference implementation (recommended external)
│   │
│   └── production/                # Production modules
│       ├── config.py              # Production configuration
│       ├── gate.py                # Three-stage gating
│       ├── slot_pool.py           # Slot pool management
│       └── operator.py            # Production operator
│
├── configs/                       # ★ Configuration file templates (v3.2 new)
│   ├── portal_config.yaml         # Portal configuration example
│   └── runtime_config.yaml        # Runtime configuration example
│
├── llm/                           # LLM adapters
│   └── adapters/
│       ├── base.py                # Adapter base class
│       ├── deepseek.py            # DeepSeek adapter
│       ├── ollama.py              # Ollama adapter
│       ├── vllm.py                # vLLM adapter
│       └── openai_compat.py       # OpenAI compatible adapter
│
├── aga_experiment_tool/           # Web experiment tool
│   ├── app.py                     # Flask application
│   └── config.yaml                # Configuration file
│
├── scripts/                       # Startup scripts
│   ├── start_portal.sh            # ★ Portal startup (Linux/macOS)
│   ├── start_portal.bat           # ★ Portal startup (Windows)
│   ├── start_experiment_tool.sh   # Experiment tool (Linux/macOS)
│   └── start_experiment_tool.bat  # Experiment tool (Windows)
│
├── tests/                         # Unit tests
│   ├── conftest.py           # Global fixtures and configurations
│   ├── pytest.ini            # pytest configuration
│   ├── unit/                 # Unit tests (pure function & operator level)
│   │   ├── core/             # Core module tests
│   │   ├── entropy_gate/     # Entropy gate tests
│   │   ├── entropy_gate/     # Entropy gate tests
│   │   ├── decay/            # Decay module tests
│   ├── component/            # Component tests (module level)
│   │   └── compression/      # Compression module tests
│   ├── component/            # Component tests (module level)
│   ├── component/            # Component tests (module level)
│   ├── component/            # Component tests (module level)
│   │   ├── persistence/      # Persistence adapter tests
│   │   ├── slot_pool/        # Slot pool tests
│   │   └── production_gate/  # Production gate tests
│   ├── integration/          # Integration tests
│   │   ├── test_single_node.py    # Single node tests
│   │   └── test_multi_runtime.py  # Multi-Runtime tests (Mock)
│   ├── fault/                # Fault injection tests
│   │   ├── test_redis_down.py         # Redis fault tests
│   │   ├── test_network_partition.py  # Network partition tests
│   │   └── test_stale_version.py      # Stale version tests
│   ├── performance/          # Performance tests
│   │   ├── latency.py        # Latency tests
│   │   ├── memory_growth.py  # Memory growth tests
│   │   └── long_run.py       # Long-running tests
│   ├── mocks/                # Mock objects
│   │   ├── redis_mock.py     # Redis Mock
│   │   ├── postgres_mock.py  # PostgreSQL Mock
│   │   ├── kafka_mock.py     # Kafka Mock
│   │   └── http_mock.py      # HTTP Mock
│   └── fixtures/             # Test data
│
├── docs/                          # Documentation
│   ├── AGA_Implementation_Analysis.md
│   ├── Architecture_Separation.md   # Architecture layering (important)
│   ├── Distributed_AGA_Architecture.md
│   ├── Governance_Framework.md
│   └── Multi_Instance_Deployment.md
│
└── README.md
```

### 🚀 Quick Start

#### 1. Install Dependencies

```bash
# Basic dependencies
pip install torch transformers flask pyyaml aiosqlite

# Portal API (v3.2 new)
pip install fastapi uvicorn httpx pydantic

# Production environment (optional)
pip install redis asyncpg aiokafka
```

#### 2. Separated Deployment Mode (v3.2 new)

**v3.2 introduces Portal + Runtime separated deployment architecture, supporting large-scale production environments.**

##### 2.1 Start Portal (API service, no GPU required)

```bash
# Development mode
./scripts/start_portal.sh --dev

# Production mode (using Redis + PostgreSQL)
./scripts/start_portal.sh --prod --redis localhost --postgres postgresql://...

# Or use Python
python -m aga.portal.app --host 0.0.0.0 --port 8081
```

Portal provides REST API, visit `http://localhost:8081/docs` to view OpenAPI documentation.

##### 2.2 Start Runtime (co-deployed with LLM, requires GPU)

```python
from aga.runtime import RuntimeAgent
from aga.config import RuntimeConfig

# Create configuration
config = RuntimeConfig.for_production(
    instance_id="runtime-001",
    portal_url="http://portal:8081",
    redis_host="localhost",
    hidden_dim=4096,
    num_slots=100,
)

# Create Agent
agent = RuntimeAgent(config)

# Initialize and start
await agent.initialize()
await agent.start()

# Attach to model
aga_layer = agent.attach_to_layer(transformer_layer)

# Use (in inference loop)
output, diagnostics = agent.get_runtime().forward(hidden_states, attention_mask)
```

##### 2.3 External Governance System Integration

```python
from aga.client import AGAClient

# Create client
client = AGAClient("http://portal:8081")

# Inject knowledge
client.inject_knowledge(
    lu_id="knowledge_001",
    condition="When user asks about the capital of France",
    decision="Answer Paris",
    key_vector=[...],  # Encoded vector
    value_vector=[...],
    namespace="geography",
    lifecycle_state="probationary",
)

# Confirm knowledge
client.confirm("knowledge_001", reason="Validation passed")

# Quarantine problematic knowledge
client.quarantine("knowledge_002", reason="Error detected")

# Query statistics
stats = client.get_statistics(namespace="geography")
```

#### 3. Monolithic Deployment Mode (Traditional approach)

##### 3.1 Start Experiment Tool

```bash
# Linux/macOS
./scripts/start_experiment_tool.sh

# Windows
scripts\start_experiment_tool.bat

# Or run directly
python -m aga_experiment_tool.app --port 8765
```

Visit `http://localhost:8765`, default password: `aga_experiment_2026`

##### 3.2 Code Usage (Monolithic mode)

```python
from aga import AGAConfig, AGAOperator, AGAManager
from aga import LifecycleState
from aga.persistence import SQLiteAdapter, PersistenceManager

# Create configuration
config = AGAConfig()
config.slot_pool.hidden_dim = 768
config.slot_pool.max_slots = 100

# Create AGA operator
aga = AGAOperator(config)
aga.eval()

# Inject knowledge
import torch
key_vector = torch.randn(64)
value_vector = torch.randn(768)

aga.inject_knowledge(
    slot_idx=0,
    key_vector=key_vector,
    value_vector=value_vector,
    lu_id="LU_001_paris",
    lifecycle_state=LifecycleState.PROBATIONARY,
    condition="capital of France",
    decision="Paris",
)

# Confirm knowledge
aga.update_lifecycle(0, LifecycleState.CONFIRMED)

# View statistics
print(aga.get_statistics())
```

### 🏗️ Deployment Architecture Selection

AGA v3.2 provides two deployment modes:

| Mode                      | Use Case                                      | Characteristics                             |
| ------------------------- | --------------------------------------------- | ------------------------------------------- |
| **Monolithic Deployment** | Development testing, single-machine inference | Simple, API and AGA in same process         |
| **Separated Deployment**  | Multi-instance production, cloud-native       | Portal without GPU, Runtime scales with LLM |

#### Advantages of Separated Deployment Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                     Separated vs Monolithic Deployment                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Monolithic                      Separated                             │
│  ┌──────────────────┐           ┌──────────────────┐                   │
│  │ API + AGA + LLM  │           │ Portal (API)     │ ← No GPU          │
│  │ (same process)   │           │ - Knowledge mgmt │                   │
│  │                  │           │ - Audit logs     │                   │
│  └──────────────────┘           └────────┬─────────┘                   │
│  Pros: Simple                             │ Redis/Kafka                │
│  Cons:                           ┌───────┼───────┐                     │
│  - API uses GPU                  ▼       ▼       ▼                     │
│  - Hard to scale                ┌─────┐ ┌─────┐ ┌─────┐                │
│  - Single point                 │RT-1 │ │RT-2 │ │RT-N │ ← GPU          │
│    of failure                   │+LLM │ │+LLM │ │+LLM │                │
│                                 └─────┘ └─────┘ └─────┘                │
│                                 Pros:                                  │
│                                 - Portal scales independently          │
│                                 - Runtime scales with LLM              │
│                                 - Fault isolation                      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

| Component   | Monolithic           | Separated                    |
| ----------- | -------------------- | ---------------------------- |
| Portal      | Same process as AGA  | Independent service (no GPU) |
| Runtime     | N/A                  | Co-located with LLM (GPU)    |
| Sync        | Direct memory access | Redis Pub/Sub                |
| Persistence | Local files          | PostgreSQL                   |
| Scalability | Vertical scaling     | Horizontal scaling           |

### 💾 Multi-adapter Persistence

AGA v3.0+ supports layered caching architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                Multi-adapter Persistence Architecture       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ L0: Memory  │─▶│ L1: Redis  │─▶│ L2: Postgres│          │
│  │ (128 slots) │  │ (1000 slots)│  │ (unlimited) │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  Read strategy: L0 → L1 → L2 (promote on miss)              │
│  Write strategy: write-through (write to all layers)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
from aga.persistence import (
    MemoryAdapter,
    SQLiteAdapter,
    RedisAdapter,
    PostgresAdapter,
    CompositeAdapter,
    PersistenceManager,
)

# Development environment: SQLite
adapter = SQLiteAdapter("aga_data.db")
await adapter.connect()

# Production environment: Layered caching
composite = CompositeAdapter(
    l0_adapter=MemoryAdapter(max_slots_per_namespace=128),
    l1_adapter=RedisAdapter(host="redis-host"),
    l2_adapter=PostgresAdapter(host="pg-host"),
)
await composite.connect()

# Use manager
persistence = PersistenceManager(composite, namespace="production")
await persistence.save_aga_state(aga)
await persistence.load_aga_state(aga)
```

### 🌐 Distributed Deployment

AGA v3.1 supports multi-instance deployment:

```python
from aga.distributed import DistributedSynchronizer

# Create synchronizer
sync = DistributedSynchronizer(
    instance_id="instance-1",
    namespace="production",
    backend="redis",
    config={"host": "redis-host"},
)
await sync.start()

# Sync knowledge injection (called by external governance system)
await sync.sync_knowledge_inject(
    lu_id="LU_001",
    slot_idx=0,
    key_vector=key_vec.tolist(),
    value_vector=value_vec.tolist(),
    lifecycle_state=LifecycleState.PROBATIONARY,
)

# Sync lifecycle update
await sync.sync_lifecycle_update("LU_001", LifecycleState.CONFIRMED)

# Sync quarantine
await sync.sync_quarantine("LU_001")
```

### 🔌 External Governance System Integration

AGA provides Knowledge Transfer API for external governance system (such as continuous self-learning system) integration:

```python
# ==================== External Governance System Calls AGA ====================

from aga import AGAOperator, AGAConfig, LifecycleState
from aga.persistence import SQLiteAdapter, PersistenceManager

# 1. Initialize AGA
config = AGAConfig()
aga = AGAOperator(config)

adapter = SQLiteAdapter("aga_data.db")
await adapter.connect()
pm = PersistenceManager(adapter, namespace="production")

# 2. After external governance system approves Learning Unit, inject into AGA
def on_learning_unit_approved(lu: LearningUnit):
    """Called when continuous self-learning system approves a Learning Unit"""

    # Encode key/value (optional, AGA can auto-encode)
    key_vector = model.encode(lu.condition)
    value_vector = model.encode(lu.decision)

    # Inject knowledge
    slot_idx = aga.inject_knowledge(
        lu_id=lu.id,
        key_vector=key_vector,
        value_vector=value_vector,
        lifecycle_state=LifecycleState.PROBATIONARY,  # New knowledge defaults to probationary
        reliability=0.3,
        condition=lu.condition,
        decision=lu.decision,
        metadata={"source": lu.source, "version": lu.version},
    )

    # Persist
    await pm.save_slot(aga, slot_idx)

# 3. External governance system decides to confirm knowledge
def on_knowledge_confirmed(lu_id: str):
    """Called when governance system confirms knowledge is trustworthy"""
    aga.update_lifecycle_by_lu_id(lu_id, LifecycleState.CONFIRMED)
    aga.update_reliability(lu_id, 1.0)

# 4. External governance system decides to quarantine knowledge
def on_knowledge_quarantined(lu_id: str, reason: str):
    """Called when governance system detects problematic knowledge"""
    aga.quarantine_by_lu_id(lu_id)
    # Record audit log (governance system's responsibility)
    audit_log.record(lu_id, "quarantine", reason)

# 5. Listen to AGA events, feedback to governance system
@aga.on_event("knowledge_hit")
def on_hit(lu_id: str, hit_count: int, context: dict):
    """When knowledge is hit, feedback to governance system"""
    governance_system.record_hit(lu_id, context)

@aga.on_event("low_confidence_query")
def on_low_confidence(query_context: dict):
    """When AGA cannot find matching knowledge, prompt governance system"""
    governance_system.suggest_new_knowledge(query_context)
```

#### Capabilities Governance System Needs to Implement

| Capability               | Description                                                   | AGA Support                                 |
| ------------------------ | ------------------------------------------------------------- | ------------------------------------------- |
| **Knowledge Generation** | Generate LU from user interaction, feedback, external sources | `inject_knowledge()` API                    |
| **Knowledge Validation** | Validate knowledge correctness, safety                        | -                                           |
| **Lifecycle Decisions**  | Decide when to CONFIRM/DEPRECATE/QUARANTINE                   | `update_lifecycle()` API                    |
| **Conflict Resolution**  | Handle contradictory knowledge                                | `list_knowledge()` to get similar knowledge |
| **Quality Assessment**   | Evaluate knowledge value                                      | `get_statistics()` to get hit statistics    |
| **Propagation Strategy** | Decide whether/when to propagate knowledge                    | `sync_knowledge_inject()` API               |
| **Approval Workflow**    | Human review of high-risk knowledge                           | `update_lifecycle()` API                    |

#### Example: Integration with Continuous Self-Learning System

```
┌─────────────────────────────────────────────────────────────┐
│         Continuous Self-Learning System                     │
│                                                             │
│  User interaction ──┐                                       │
│                     │      ┌─────────┐      ┌──────────┐    │
│  External feedback ─┼──▶  │Knowledge│ ──▶  │Governance│    │
│                     │      │Candidate│      │Approval  │    │
│  Auto extraction  ──┘      │Pool     │      │          │    │
│                            └─────────┘      └──┬───────┘    │
│                                                │            │
│                                                ▼            │
│                               ┌─────────────────┐           │
│                               │ Learning Unit   │           │
│                               │ (Approved)      │           │
│                               └────────┬────────┘           │
│                                        │                    │
└────────────────────────────────────────┼────────────────────┘
                                         │
                                         │ inject_knowledge()
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGA Knowledge Manager                    │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Slot 0  │  │ Slot 1  │  │ Slot 2  │  │  ...    │         │
│  │ LU_001  │  │ LU_002  │  │ LU_003  │  │         │         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frozen Transformer                       │
└─────────────────────────────────────────────────────────────┘
```

> `aga/distributed/governance.py` provides a **reference implementation** of governance logic,
> but in production environments, it is recommended to implement this logic in your own continuous self-learning system.

### 📊 Knowledge Lifecycle

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
    ┌────────────────────────┐                                │
    │     PROBATIONARY       │──── Normal metrics  ───────────┤
    │     (r = 0.3)          │                                │
    └───────────┬────────────┘                                │
                │ Governance approval                         │
                ▼                                             │
    ┌────────────────────────┐                                │
    │      CONFIRMED         │──── Anomaly detection  ────────┤
    │     (r = 1.0)          │                                │
    └───────────┬────────────┘                                │
                │ Deprecation request                         │
                ▼                                             │
    ┌────────────────────────┐                                │
    │      DEPRECATED        │──── Retention expires  ────────┤
    │     (r = 0.1)          │                                │
    └───────────┬────────────┘                                │
                │ Cleanup                                     │
                ▼                                             │
    ┌────────────────────────┐                                │
    │     QUARANTINED        │ ◀─────────────────────────────┘
    │     (r = 0.0)          │      (Emergency isolation can jump directly)
    └────────────────────────┘
```

### 🔌 LLM Adapters

| Adapter               | Description                     | Use Case   |
| --------------------- | ------------------------------- | ---------- |
| `OllamaAdapter`       | Ollama local models             | Dev/test   |
| `VLLMAdapter`         | vLLM high-performance inference | Production |
| `DeepSeekAdapter`     | DeepSeek API/local              | API calls  |
| `OpenAICompatAdapter` | OpenAI compatible interface     | Universal  |

```python
from llm.adapters import OllamaAdapter

adapter = OllamaAdapter(
    base_url="http://localhost:11434",
    model="qwen2.5:7b"
)

response = adapter.chat([
    {"role": "user", "content": "Hello"}
])
print(response.content)
```

### 📜 Core Principles

**Knowledge Retrieval Mechanism**:

```
Input: hidden_states X ∈ ℝ^{n×d}

1. Query projection:
   Q' = X · W_down    # [n, d] → [n, d_b]

2. Attention scores (with reliability mask):
   scores_ij = (Q'_i · K_j^T) / √d_b + log(r_j)

   where r_j is reliability of slot j (quarantined slots r=0 → log(0)=-∞)

3. Softmax retrieval:
   α_ij = softmax(scores_i)    # Quarantined slots have weight 0
   O_s = Σ_j α_ij · V_j

4. Entropy-gated fusion:
   gate = σ(w₁·H + b)          # H = main attention entropy
   Ô = O_primary + gate ⊙ O_s
```

**Three-stage Gating**:

```
Gate-0 (Prior gating)     Gate-1 (Confidence gating)     Gate-2 (Top-k routing)
      │                           │                            │
      ▼                           ▼                            ▼
  namespace/app_id          Uncertainty estimation        Top-k slot selection
      │                           │                            │
      ▼                           ▼                            ▼
  DISABLED/REQUIRED          BYPASS/PASS                  Routing scores
```

### ✅ Implemented Scaling Optimizations

| Optimization                             | Status               | Implementation                                |
| ---------------------------------------- | -------------------- | --------------------------------------------- |
| **Layered knowledge storage (L0/L1/L2)** | ✅ Fully implemented | `aga/persistence/composite_adapter.py`        |
| **Write-through / Read-promotion**       | ✅ Fully implemented | `CompositeAdapter`                            |
| **Top-k routing optimization**           | ✅ Fully implemented | `aga/core.py::SlotRouter`                     |
| **Chunked computation to avoid OOM**     | ✅ Fully implemented | `SlotRouter._chunked_top_k()`                 |
| **Three-stage gating (Gate-0/1/2)**      | ✅ Fully implemented | `aga/production/gate.py`                      |
| **Multi-source entropy signals**         | ✅ Fully implemented | `aga/entropy_gate.py::EntropySource`          |
| **Adaptive thresholds**                  | ✅ Fully implemented | `EntropyGateConfig.enable_adaptive_threshold` |
| **Persistence decay**                    | ✅ Fully implemented | `aga/decay.py::PersistenceDecay`              |
| **Hard reset mechanism**                 | ✅ Fully implemented | `DecayConfig.enable_hard_reset`               |
| **Hit count / Consecutive misses**       | ✅ Fully implemented | `Slot.hit_count`, `consecutive_misses`        |
| **Namespace isolation**                  | ✅ Fully implemented | `aga/production/slot_pool.py`                 |
| **Early Exit**                           | ✅ Fully implemented | `AGAConfig.enable_early_exit`                 |

### 🏛️ Internal Governance System

AGA provides **basic internal governance capabilities**, while recommending integration with external governance systems:

#### Internal Governance (Implemented)

```python
from aga.distributed import (
    GovernanceArbiter,      # Governance arbiter
    PropagationThrottler,   # Propagation throttler
    TrustTier,              # Trust tier
    PropagationPolicy,      # Propagation policy
)

# Create governance arbiter
arbiter = GovernanceArbiter(
    instance_id="instance-1",
    quorum_size=2,           # Minority rule
    risk_threshold=0.3,      # Risk threshold
)

# Register slot trust tier
arbiter.register_slot(
    lu_id="LU_001",
    trust_tier=TrustTier.S1_EXPERIENCE,  # Experience slot: rollback-able
)

# Evaluate propagation (defaults to rejecting unregistered knowledge)
decision = await arbiter.evaluate_propagation("LU_001", "instance-2")
if decision.verdict == GovernanceVerdict.ALLOW:
    # Allow propagation
    pass

# Evaluate quarantine (quorum mechanism)
decision = await arbiter.evaluate_quarantine("LU_001", "Abnormal output", "instance-1")
# Automatically takes effect after reaching quorum
```

#### Trust Tiers (Semantic Sovereignty Partitioning)

| Tier                       | Propagation Policy    | Description                                              |
| -------------------------- | --------------------- | -------------------------------------------------------- |
| **S0: Acceleration slots** | Immediate propagation | Inference cache, lossy and rebuildable                   |
| **S1: Experience slots**   | Delayed propagation   | 60s observation period before propagation, rollback-able |
| **S2: Policy slots**       | Gated propagation     | Requires approval (2 votes) before propagation           |
| **S3: Prohibited slots**   | No propagation        | Read-only, not propagated to other instances             |

#### Internal vs External Governance

| Capability                      | Internal Governance | External Governance (Recommended) |
| ------------------------------- | ------------------- | --------------------------------- |
| Trust tier partitioning         | ✅ Implemented      | Extensible                        |
| Propagation throttling          | ✅ Implemented      | Extensible                        |
| Quorum quarantine               | ✅ Implemented      | Extensible                        |
| Lifecycle decisions             | ⚠️ Framework-level  | **Recommended external**          |
| Knowledge generation validation | ❌ Not provided     | **Must be external**              |
| Conflict resolution             | ⚠️ Policy-level     | **Recommended external**          |
| Quality assessment              | ⚠️ Statistics-level | **Recommended external**          |
| Human approval workflow         | ❌ Not provided     | **Must be external**              |

#### Recommended Integration Pattern

```
┌─────────────────────────────────────────────────────────────┐
│         External Governance System (Continuous Learning)    │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Knowledge   │  │ Quality     │  │ Human       │          │
│  │ Generation  │  │ Assessment  │  │ Approval    │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│         ┌────────────────────────────────┐                  │
│         │        Learning Unit           │                  │
│         │  + lifecycle_state decision    │                  │
│         │  + trust_tier decision         │                  │
│         └────────────────┬───────────────┘                  │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           │ inject_knowledge()
                           │ update_lifecycle()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGA Internal Governance                   │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Propagation │  │ Quorum      │  │ Trust       │          │
│  │ Throttling  │  │ Quarantine  │  │ Partitioning│          │
│  │ (delay/rate)│  │ (minority)  │  │ (S0-S3)     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  Responsibility: Execute governance decisions,              │
│                  not make governance judgments              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> `aga/distributed/governance.py` provides reference implementation,
> Production environments recommend externalizing governance decision logic to the continuous self-learning system.

### ⚠️ Challenges and Solutions for Continuous Controllable Learning Systems

AGA, as a **hot-pluggable knowledge manager** for Transformer models, faces the following core challenges in continuous controllable learning scenarios:

#### Challenge 1: Knowledge Capacity Ceiling

**Problem Description**:

-   AGA's knowledge storage relies on explicit slots (Slot Pool)
-   Slot count limited by GPU memory and attention computation complexity
-   When knowledge scale reaches 10,000+ slots, retrieval efficiency degrades

**Solution**: Layered knowledge storage architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  L0: GPU memory pool    Hot knowledge (256-512 slots, ns retrieval)    │
│         ↓ Dynamic swap in/out                                          │
│  L1: CPU memory/Redis   Warm knowledge (10,000+ slots, μs retrieval)   │
│         ↓ Load on demand                                               │
│  L2: PostgreSQL        Cold knowledge (million slots, ms retrieval)    │
└────────────────────────────────────────────────────────────────────────┘
Key: Knowledge always maintains independent slot form, never mixed with model parameters
```

#### Challenge 2: Knowledge Aging and Forgetting

**Problem Description**:

-   Knowledge has temporal validity, expired knowledge needs to be forgotten
-   AGA's isolation mechanism can only "disable" rather than truly forget
-   After long-term operation accumulates many DEPRECATED/QUARANTINED slots

**Solution**: Graceful forgetting mechanism

```
Forgetting flow (maintaining sovereign boundaries):
1. DEPRECATED → Observation period (rollback-able)
2. QUARANTINED → Isolation period (not participating in inference)
3. Archive to cold storage (retain audit logs)
4. Release slot resources (reusable by new knowledge)

Knowledge consolidation (non-distillation):
- Merge semantically overlapping slots
- Maintain traceability of each original LU ID
- Release redundant slots
```

#### Challenge 3: Knowledge Consistency and Conflicts

**Problem Description**:

-   Multiple slots may contain contradictory knowledge
-   How to decide which to trust during inference?
-   Conflicts are more complex in distributed scenarios

**Solution**: Slot-level conflict resolution

```
Detection layer:
  - Semantic similarity > 0.9 but different decisions → Potential conflict
  - Same condition multiple slots → Version conflict

Resolution strategies:
  1. Temporal priority: Keep newest knowledge
  2. Reliability priority: Keep slot with higher reliability
  3. Governance priority: Flag conflict, await human judgment
  4. Coexistence strategy: Keep all versions, select by context during inference

Key: Conflict resolution at slot level, doesn't affect main model
```

#### Challenge 4: Cross-model Knowledge Migration

**Problem Description**:

-   AGA slot key/value vectors are model-specific
-   After changing base model, existing knowledge cannot be directly used
-   Does this mean "retraining" is needed?

**Solution**: Zero-training knowledge migration

```
Retained content (migratable):
  - lu_id, condition, decision (semantic description)
  - lifecycle_state, reliability (governance state)
  - hit_count, metadata (usage statistics)

Regenerated content:
  - key_vector = new_model.encode(condition)
  - value_vector = new_model.encode(decision)

Migration flow:
  1. Export knowledge descriptions (JSON/Protobuf)
  2. Re-encode key/value on new model
  3. Maintain original governance state and lifecycle
  4. Verify semantic fidelity after migration

Key: Migrate knowledge "meaning", not "parameters"
```

#### Challenge 5: Inference Latency Accumulation

**Problem Description**:

-   Every Transformer layer has AGA overhead
-   Slot count increase → Attention computation increase
-   May become inference bottleneck

**Solution**: Inference optimization (maintaining hot-pluggable characteristics)

```
1. Early Exit (implemented): Gate-0 direct bypass, skip entire AGA
2. Sparse attention: Only compute attention for Top-k relevant slots
3. Async prefetch: Predict slots needed for next layer, load in advance
4. Batch slot fusion: Merge similar slots for computation, reduce redundancy

Principle: Optimization is at "implementation level", doesn't change "architecture level"
```

#### Challenge 6: Adversarial Knowledge Injection

**Problem Description**:

-   Malicious users may inject harmful knowledge
-   AGA's "zero-training injection" makes attack cost lower
-   Traditional safety mechanisms (like RLHF) don't apply

**Solution**: Knowledge security protection mechanism

```
Injection-time checks:
  - Semantic safety filtering (detect harmful content)
  - Source verification (knowledge source trustworthiness)
  - Format validation (prevent injection attacks)

Runtime protection:
  - Default PROBATIONARY state (reliability 0.3)
  - Anomaly detection (output mutation → auto-isolation)
  - Impact scope limitation (propagation_radius)

Governance layer safeguards:
  - Human approval workflow (S2/S3 level knowledge)
  - Quorum voting for isolation (minority rule)
  - Complete audit logs

Key: Security is a "governance problem", not a "training problem"
```

### 🔮 Future Optimization Directions

#### Short-term Optimization (v3.2)

1. **Governance Layer Enhancement**

    - Human governance interface (approval workflow UI)
    - Automated knowledge quality assessment
    - Conflict detection and resolution

2. **Performance Optimization**

    - GPU memory pooling and dynamic swap in/out
    - Batch inference optimization
    - Cache warming strategy

3. **Monitoring Enhancement**
    - Prometheus metrics export
    - Distributed tracing
    - Anomaly detection alerts

#### Mid-term Optimization (v4.0)

1. **Knowledge Consolidation (non-distillation)**

    - Merge semantically overlapping slots
    - Knowledge compression (maintaining slot form)
    - Redundancy detection and cleanup

2. **Multi-modal Support**

    - Image knowledge injection
    - Cross-modal retrieval
    - Multi-modal gating

3. **Federated Knowledge Sharing**
    - Cross-organization knowledge sync
    - Privacy protection (differential privacy slots)
    - Intellectual property marking

#### Long-term Vision (v5.0+)

1. **Cognitive Architecture**

    - Multi-layer knowledge representation (facts/rules/strategies)
    - Reasoning chain interpretability
    - Meta-cognitive capabilities (knowing what it doesn't know)

2. **Autonomous Governance**

    - Knowledge self-assessment
    - Automated lifecycle management
    - Self-repair (auto-isolate problematic knowledge when detected)

3. **Cross-model Knowledge Migration**
    - Semantic-level knowledge export/import
    - Model-agnostic knowledge representation
    - Knowledge version control

### ❌ Explicitly Rejected Directions

The following directions conflict with AGA's core philosophy and are **explicitly excluded from the roadmap**:

| Rejected Direction                   | Reason for Exclusion                                                    |
| ------------------------------------ | ----------------------------------------------------------------------- |
| **Knowledge distillation to LoRA**   | Violates "zero-training" principle, breaks sovereign boundaries         |
| **Fine-tune base model**             | Regresses to pre-training approach, loses hot-pluggable characteristics |
| **Fuse slots into model parameters** | Cannot trace and isolate individual knowledge                           |
| **RLHF-style alignment**             | Requires training, cannot target individual knowledge                   |

> **AGA's Core Promise**: Knowledge is always "external", always traceable, always isolatable.

### 📄 Related Papers

This project is based on the paper "Auxiliary Governed Attention: A Governable, Inference-time Auxiliary Attention Mechanism with Sovereign Boundaries for Frozen Transformers".

### 📊 Code Statistics

| Module                 | Lines      | Description               |
| ---------------------- | ---------- | ------------------------- |
| `aga/` (core)          | 4,103      | Core AGA implementation   |
| `aga/api/`             | 2,414      | Monolithic REST API       |
| `aga/portal/`          | 1,442      | Portal API service        |
| `aga/operator/`        | 1,111      | Unified operator layer    |
| `aga/persistence/`     | 3,130      | Multi-adapter persistence |
| `aga/distributed/`     | 1,462      | Distributed sync          |
| `aga/production/`      | 3,468      | Production modules        |
| `aga/runtime/`         | 874        | Runtime Agent             |
| `aga/sync/`            | 1,182      | Sync protocol             |
| `aga/config/`          | 653        | Configuration management  |
| `aga/client/`          | 540        | Client library            |
| `aga/encoder/`         | 1,211      | Encoder module            |
| `llm/`                 | 2,550      | LLM adapters              |
| `aga_experiment_tool/` | 1,149      | Web experiment tool       |
| **Total**              | **25,340** | (excl. tests & examples)  |

### 📄 License

MIT License

---

<div align="center">

**AGA** - _Making LLMs safely borrow knowledge while knowing the difference_

</div>
