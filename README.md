# AGA - Auxiliary Governed Attention

<div align="center">

![AGA Logo](https://img.shields.io/badge/AGA-Auxiliary%20Governed%20Attention-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-3.1-green?style=flat-square)
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
┌─────────────────────────────────────────────────────────────┐
│        外部治理系统（如：可控持续自学习系统）           │
│        - 知识生成、验证、审批                                │
│        - 生命周期决策                                        │
│        - 冲突解决、质量评估                                  │
│        - 产出：Learning Unit (LU)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ Knowledge Transfer API
                           ▼
┌─────────────────────────────────────────────────────────────┐
│        AGA 知识管理器（本项目）                              │
│        - 知识存储与检索                                      │
│        - 熵门控与路由                                        │
│        - 多实例同步                                          │
│        - 持久化存储                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│        Frozen Transformer（冻结的基座模型）                  │
└─────────────────────────────────────────────────────────────┘
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
│   │   └── governance.py          # 治理参考实现（建议外置）
│   │
│   └── production/                # 产品化模块
│       ├── config.py              # 生产配置
│       ├── gate.py                # 三段式门控
│       ├── slot_pool.py           # 槽位池管理
│       └── operator.py            # 生产算子
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
│   ├── start_experiment_tool.sh   # Linux/macOS
│   └── start_experiment_tool.bat  # Windows
│
├── tests/                         # 单元测试
│   ├── test_core.py
│   ├── test_decay.py
│   └── test_entropy_gate.py
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

# 生产环境（可选）
pip install redis asyncpg aiokafka
```

#### 2. 启动实验工具

```bash
# Linux/macOS
./scripts/start_experiment_tool.sh

# Windows
scripts\start_experiment_tool.bat

# 或直接运行
python -m aga_experiment_tool.app --port 8765
```

访问 `http://localhost:8765`，默认密码：`aga_experiment_2026`

#### 3. 代码使用

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

### 💾 多适配器持久化

AGA v3.0 支持分层缓存架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    多适配器持久化架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ L0: Memory  │─▶│ L1: Redis   │─▶│ L2: Postgres│         │
│  │ (128 slots) │  │ (1000 slots)│  │ (无限)      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  读取策略：L0 → L1 → L2（miss 时向下查找并提升）              │
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
│                                                             │
│  用户交互 ──┐                                               │
│            │      ┌─────────┐      ┌─────────┐             │
│  外部反馈 ──┼──▶  │ 知识    │ ──▶  │ 治理    │             │
│            │      │ 候选池  │      │ 审批    │             │
│  自动提取 ──┘      └─────────┘      └────┬────┘             │
│                                         │                   │
│                                         ▼                   │
│                               ┌─────────────────┐           │
│                               │ Learning Unit   │           │
│                               │ (审批通过)      │           │
│                               └────────┬────────┘           │
│                                        │                    │
└────────────────────────────────────────┼────────────────────┘
                                         │
                                         │ inject_knowledge()
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                        AGA 知识管理器                        │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Slot 0  │  │ Slot 1  │  │ Slot 2  │  │  ...    │        │
│  │ LU_001  │  │ LU_002  │  │ LU_003  │  │         │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frozen Transformer                        │
└─────────────────────────────────────────────────────────────┘
```

> `aga/distributed/governance.py` 提供了治理逻辑的**参考实现**，
> 但在生产环境中，建议您在自己的持续自学习系统中实现这些逻辑。

### 📊 知识生命周期

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
    ┌────────────────────────┐                               │
    │     PROBATIONARY       │──── 使用指标正常 ─────────────┤
    │     (r = 0.3)          │                               │
    └───────────┬────────────┘                               │
                │ 治理审批                                    │
                ▼                                             │
    ┌────────────────────────┐                               │
    │      CONFIRMED         │──── 异常检测 ─────────────────┤
    │     (r = 1.0)          │                               │
    └───────────┬────────────┘                               │
                │ 弃用请求                                    │
                ▼                                             │
    ┌────────────────────────┐                               │
    │      DEPRECATED        │──── 保留期过期 ───────────────┤
    │     (r = 0.1)          │                               │
    └───────────┬────────────┘                               │
                │ 清理                                        │
                ▼                                             │
    ┌────────────────────────┐                               │
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
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 知识生成    │  │ 质量评估    │  │ 人类审批    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         └────────────────┼────────────────┘                 │
│                          ▼                                   │
│         ┌────────────────────────────────┐                  │
│         │        Learning Unit           │                  │
│         │  + lifecycle_state 决策        │                  │
│         │  + trust_tier 决策             │                  │
│         └────────────────┬───────────────┘                  │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           │ inject_knowledge()
                           │ update_lifecycle()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGA 内部治理层                             │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 传播节流    │  │ Quorum 隔离 │  │ 信任分区    │         │
│  │ (延迟/速率) │  │ (少数即生效)│  │ (S0-S3)     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  职责：执行治理决策，不做治理判断                             │
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
│  L0: GPU 显存池    热知识（256-512 槽位，纳秒级检索）         │
│         ↓ 动态换入换出                                       │
│  L1: CPU 内存/Redis 温知识（10,000+ 槽位，微秒级检索）        │
│         ↓ 按需加载                                           │
│  L2: PostgreSQL    冷知识（百万级槽位，毫秒级检索）           │
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

| 模块               | 行数        | 说明           |
| ------------------ | ----------- | -------------- |
| `aga/core.py`      | ~1,260      | 核心 AGA 实现  |
| `aga/operator/`    | ~1,330      | 统一算子层     |
| `aga/persistence/` | ~3,100      | 多适配器持久化 |
| `aga/distributed/` | ~1,500      | 分布式支持     |
| `aga/production/`  | ~2,900      | 产品化模块     |
| **总计**           | **~12,900** |                |

---

## English

### 📖 Background

After deploying large language models (LLMs), dynamically integrating new knowledge without compromising existing capabilities remains a long-standing challenge:

| Existing Solutions   | Problems                                                                                                    |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Full Fine-tuning** | Catastrophic forgetting, high computational cost                                                            |
| **LoRA/Adapter**     | Requires training, hyperparameter sensitive                                                                 |
| **RAG**              | External retrieval rather than internalized capability; model cannot distinguish "knowing" from "borrowing" |

**AGA (Auxiliary Governed Attention)** proposes a new paradigm:

> Attach a governable auxiliary attention module to frozen Transformers, dynamically injecting knowledge at inference time while maintaining **sovereign boundaries** between the primary model and auxiliary knowledge.

### ⚠️ Important: AGA's Positioning

**AGA is a hot-swappable knowledge manager for Transformer models, not a complete governance system.**

```
┌─────────────────────────────────────────────────────────────┐
│    External Governance System (e.g., Controllable CLS)      │
│    - Knowledge generation, verification, approval           │
│    - Lifecycle decisions                                    │
│    - Conflict resolution, quality assessment                │
│    - Output: Learning Unit (LU)                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ Knowledge Transfer API
                           ▼
┌─────────────────────────────────────────────────────────────┐
│    AGA Knowledge Manager (This Project)                     │
│    - Knowledge storage and retrieval                        │
│    - Entropy gating and routing                             │
│    - Multi-instance synchronization                         │
│    - Persistent storage                                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│    Frozen Transformer (Base Model)                          │
└─────────────────────────────────────────────────────────────┘
```

**AGA is Responsible for**:

-   ✅ Auxiliary attention computation (entropy gating, internal routing)
-   ✅ Knowledge storage and retrieval (Slot Pool, persistence)
-   ✅ Multi-instance synchronization (state replication, event broadcasting)
-   ✅ Providing Knowledge Transfer API

**AGA is NOT Responsible for** (requires external system):

-   ❌ Knowledge generation (produced by continuous learning system as Learning Units)
-   ❌ Governance decisions (when to CONFIRM/DEPRECATE/QUARANTINE)
-   ❌ Conflict resolution (how to handle contradictory knowledge)
-   ❌ Quality assessment (value judgment of knowledge)
-   ❌ Propagation strategy (whether/when to propagate knowledge)

> For detailed architectural separation, see [docs/Architecture_Separation.md](docs/Architecture_Separation.md)

### 🎯 Core Features

| Feature                         | Description                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------- |
| **Zero-Training Injection**     | Knowledge written directly to buffers, no gradient computation required          |
| **Hot-Swappable Design**        | Dynamically add/remove knowledge at runtime                                      |
| **Lifecycle Support**           | Supports PROBATIONARY/CONFIRMED/DEPRECATED/QUARANTINED states                    |
| **Entropy Gating**              | No intervention when primary model is confident; contributes only when uncertain |
| **Instant Isolation**           | Problematic knowledge can be immediately removed from inference                  |
| **Complete Traceability**       | Each knowledge slot bound to LU ID                                               |
| **Multi-Adapter Persistence**   | SQLite/Redis/PostgreSQL layered caching                                          |
| **Distributed Synchronization** | Multi-instance state replication and event broadcasting                          |
| **Knowledge Transfer API**      | For external governance system integration                                       |

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
│   ├── distributed/               # Distributed synchronization layer
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
│   ├── start_experiment_tool.sh   # Linux/macOS
│   └── start_experiment_tool.bat  # Windows
│
├── tests/                         # Unit tests
│   ├── test_core.py
│   ├── test_decay.py
│   └── test_entropy_gate.py
│
├── docs/                          # Documentation
│   ├── AGA_Implementation_Analysis.md
│   ├── Architecture_Separation.md   # Architectural separation (important)
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

# Production environment (optional)
pip install redis asyncpg aiokafka
```

#### 2. Launch Experiment Tool

```bash
# Linux/macOS
./scripts/start_experiment_tool.sh

# Windows
scripts\start_experiment_tool.bat

# Or run directly
python -m aga_experiment_tool.app --port 8765
```

Visit `http://localhost:8765`, default password: `aga_experiment_2026`

#### 3. Code Usage

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

### 💾 Multi-Adapter Persistence

AGA v3.0 supports layered caching architecture:

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Adapter Persistence Architecture          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ L0: Memory  │─▶│ L1: Redis   │─▶│ L2: Postgres│         │
│  │ (128 slots) │  │ (1000 slots)│  │ (unlimited) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  Read strategy: L0 → L1 → L2 (search down on miss and promote) │
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

# Synchronize knowledge injection (called by external governance system)
await sync.sync_knowledge_inject(
    lu_id="LU_001",
    slot_idx=0,
    key_vector=key_vec.tolist(),
    value_vector=value_vec.tolist(),
    lifecycle_state=LifecycleState.PROBATIONARY,
)

# Synchronize lifecycle update
await sync.sync_lifecycle_update("LU_001", LifecycleState.CONFIRMED)

# Synchronize quarantine
await sync.sync_quarantine("LU_001")
```

### 🔌 External Governance System Integration

AGA provides Knowledge Transfer API for external governance systems (such as continuous learning systems) integration:

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

# 2. Inject to AGA after external governance system produces Learning Unit
def on_learning_unit_approved(lu: LearningUnit):
    """Called when continuous learning system approves a Learning Unit"""

    # Encode key/value (optional, AGA can also auto-encode)
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
    """When AGA cannot find matching knowledge, suggest to governance system"""
    governance_system.suggest_new_knowledge(query_context)
```

#### Capabilities Governance System Needs to Implement

| Capability                 | Description                                                    | AGA Support                                 |
| -------------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| **Knowledge Generation**   | Generate LU from user interactions, feedback, external sources | `inject_knowledge()` API                    |
| **Knowledge Verification** | Verify correctness and safety of knowledge                     | -                                           |
| **Lifecycle Decisions**    | Decide when to CONFIRM/DEPRECATE/QUARANTINE                    | `update_lifecycle()` API                    |
| **Conflict Resolution**    | Handle contradictory knowledge                                 | `list_knowledge()` to get similar knowledge |
| **Quality Assessment**     | Evaluate value of knowledge                                    | `get_statistics()` to get hit statistics    |
| **Propagation Strategy**   | Decide whether/when to propagate knowledge                     | `sync_knowledge_inject()` API               |
| **Approval Workflow**      | Human review for high-risk knowledge                           | `update_lifecycle()` API                    |

#### Example: Integration with Continuous Learning System

```
┌─────────────────────────────────────────────────────────────┐
│         Continuous Learning System                          │
│                                                             │
│  User Interaction ──┐                                       │
│                     │  ┌─────────┐      ┌─────────┐         │
│  External Feedback ──┼─▶│ Knowledge│ ──▶ │ Governance│       │
│                     │  │ Candidate│      │ Approval │       │
│  Auto Extraction ────┘  │ Pool     │      └────┬────┘       │
│                         └─────────┘           │             │
│                                               ▼             │
│                                   ┌─────────────────┐       │
│                                   │ Learning Unit   │       │
│                                   │ (Approved)      │       │
│                                   └────────┬────────┘       │
│                                            │                │
└────────────────────────────────────────────┼────────────────┘
                                             │
                                             │ inject_knowledge()
                                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGA Knowledge Manager                      │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Slot 0  │  │ Slot 1  │  │ Slot 2  │  │  ...    │        │
│  │ LU_001  │  │ LU_002  │  │ LU_003  │  │         │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frozen Transformer                        │
└─────────────────────────────────────────────────────────────┘
```

> `aga/distributed/governance.py` provides **reference implementation** of governance logic.
> In production environments, it is recommended to implement these logics in your own continuous learning system.

### 📊 Knowledge Lifecycle

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
    ┌────────────────────────┐                               │
    │     PROBATIONARY       │──── Usage metrics normal ─────┤
    │     (r = 0.3)          │                               │
    └───────────┬────────────┘                               │
                │ Governance approval                         │
                ▼                                             │
    ┌────────────────────────┐                               │
    │      CONFIRMED         │──── Anomaly detection ────────┤
    │     (r = 1.0)          │                               │
    └───────────┬────────────┘                               │
                │ Deprecation request                         │
                ▼                                             │
    ┌────────────────────────┐                               │
    │      DEPRECATED        │──── Retention period expired ─┤
    │     (r = 0.1)          │                               │
    └───────────┬────────────┘                               │
                │ Cleanup                                     │
                ▼                                             │
    ┌────────────────────────┐                               │
    │     QUARANTINED        │◀──────────────────────────────┘
    │     (r = 0.0)          │      (Emergency quarantine can jump directly)
    └────────────────────────┘
```

### 🔌 LLM Adapters

| Adapter               | Description                     | Use Case              |
| --------------------- | ------------------------------- | --------------------- |
| `OllamaAdapter`       | Ollama local model              | Development/Testing   |
| `VLLMAdapter`         | vLLM high-performance inference | Production deployment |
| `DeepSeekAdapter`     | DeepSeek API/local              | API calls             |
| `OpenAICompatAdapter` | OpenAI compatible interface     | General               |

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

   where r_j is reliability of slot j (quarantined slots: r=0 → log(0)=-∞)

3. Softmax retrieval:
   α_ij = softmax(scores_i)    # Quarantined slots get weight 0
   O_s = Σ_j α_ij · V_j

4. Entropy gating fusion:
   gate = σ(w₁·H + b)          # H = primary attention entropy
   Ô = O_primary + gate ⊙ O_s
```

**Three-Stage Gating**:

```
Gate-0 (Prior Gating)  Gate-1 (Confidence Gating)  Gate-2 (Top-k Routing)
      │                     │                           │
      ▼                     ▼                           ▼
  namespace/app_id      Uncertainty estimation      Top-k slot selection
      │                     │                           │
      ▼                     ▼                           ▼
  DISABLED/REQUIRED    BYPASS/PASS                 Routing scores
```

### ✅ Implemented Scaling Optimizations

| Optimization Solution                   | Status               | Implementation Location                       |
| --------------------------------------- | -------------------- | --------------------------------------------- |
| **Tiered Knowledge Storage (L0/L1/L2)** | ✅ Fully Implemented | `aga/persistence/composite_adapter.py`        |
| **Write-through / Read-promotion**      | ✅ Fully Implemented | `CompositeAdapter`                            |
| **Top-k Routing Optimization**          | ✅ Fully Implemented | `aga/core.py::SlotRouter`                     |
| **Chunked Computation to Avoid OOM**    | ✅ Fully Implemented | `SlotRouter._chunked_top_k()`                 |
| **Three-Stage Gating (Gate-0/1/2)**     | ✅ Fully Implemented | `aga/production/gate.py`                      |
| **Multi-Source Entropy Signals**        | ✅ Fully Implemented | `aga/entropy_gate.py::EntropySource`          |
| **Adaptive Threshold**                  | ✅ Fully Implemented | `EntropyGateConfig.enable_adaptive_threshold` |
| **Persistence Decay**                   | ✅ Fully Implemented | `aga/decay.py::PersistenceDecay`              |
| **Hard Reset Mechanism**                | ✅ Fully Implemented | `DecayConfig.enable_hard_reset`               |
| **Hit Count / Consecutive Misses**      | ✅ Fully Implemented | `Slot.hit_count`, `consecutive_misses`        |
| **Namespace Isolation**                 | ✅ Fully Implemented | `aga/production/slot_pool.py`                 |
| **Early Exit**                          | ✅ Fully Implemented | `AGAConfig.enable_early_exit`                 |

### 🏛️ Internal Governance System

AGA provides **basic internal governance capabilities** while recommending integration with external governance systems:

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
    quorum_size=2,           # Minority rule takes effect
    risk_threshold=0.3,      # Risk threshold
)

# Register slot trust tier
arbiter.register_slot(
    lu_id="LU_001",
    trust_tier=TrustTier.S1_EXPERIENCE,  # Experience slot: rollback-able
)

# Evaluate propagation (default deny unregistered knowledge)
decision = await arbiter.evaluate_propagation("LU_001", "instance-2")
if decision.verdict == GovernanceVerdict.ALLOW:
    # Allow propagation
    pass

# Evaluate quarantine (quorum mechanism)
decision = await arbiter.evaluate_quarantine("LU_001", "Anomalous output", "instance-1")
# Automatically takes effect after reaching quorum
```

#### Trust Tiers (Semantic Sovereignty Partitioning)

| Tier                      | Propagation Policy    | Description                                    |
| ------------------------- | --------------------- | ---------------------------------------------- |
| **S0: Acceleration Slot** | Immediate propagation | Inference cache, lossy and rebuildable         |
| **S1: Experience Slot**   | Delayed propagation   | 60-second observation period, rollback-able    |
| **S2: Policy Slot**       | Gated propagation     | Requires approval (2 votes) before propagation |
| **S3: Prohibited Slot**   | No propagation        | Read-only, no propagation to other instances   |

#### Internal vs External Governance

| Capability                        | Internal Governance | External Governance (Recommended) |
| --------------------------------- | ------------------- | --------------------------------- |
| Trust Tier Partitioning           | ✅ Implemented      | Extensible                        |
| Propagation Throttling            | ✅ Implemented      | Extensible                        |
| Quorum Quarantine                 | ✅ Implemented      | Extensible                        |
| Lifecycle Decisions               | ⚠️ Framework-level  | **Recommended External**          |
| Knowledge Generation/Verification | ❌ Not Provided     | **Must Be External**              |
| Conflict Resolution               | ⚠️ Policy-level     | **Recommended External**          |
| Quality Assessment                | ⚠️ Statistics-level | **Recommended External**          |
| Human Approval Workflow           | ❌ Not Provided     | **Must Be External**              |

#### Recommended Integration Pattern

```
┌─────────────────────────────────────────────────────────────┐
│         External Governance System (Continuous Learning)     │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Knowledge   │  │ Quality     │  │ Human       │         │
│  │ Generation  │  │ Assessment  │  │ Approval    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         └────────────────┼────────────────┘                 │
│                          ▼                                   │
│         ┌────────────────────────────────┐                  │
│         │        Learning Unit           │                  │
│         │  + lifecycle_state decision    │                  │
│         │  + trust_tier decision         │                  │
│         └────────────────┬───────────────┘                  │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           │ inject_knowledge()
                           │ update_lifecycle()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              AGA Internal Governance Layer                   │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Propagation │  │ Quorum      │  │ Trust       │         │
│  │ Throttling  │  │ Quarantine  │  │ Partitioning│         │
│  │ (delay/rate)│  │ (minority   │  │ (S0-S3)     │         │
│  │             │  │  rule)      │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  Responsibility: Execute governance decisions, not make     │
│                  governance judgments                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> `aga/distributed/governance.py` provides reference implementation.
> In production environments, it is recommended to externalize governance decision logic to continuous learning systems.

### ⚠️ Challenges and Solutions for Continuous Controllable Learning Systems

As a **hot-swappable knowledge manager** for Transformer models, AGA faces the following core challenges in continuous controllable learning scenarios:

#### Challenge 1: Knowledge Capacity Ceiling

**Problem Description**:

-   AGA's knowledge storage relies on explicit slots (Slot Pool)
-   Number of slots is limited by GPU memory and attention computation complexity
-   When knowledge scale reaches 10,000+ slots, retrieval efficiency degrades

**Solution**: Tiered Knowledge Storage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  L0: GPU Memory Pool    Hot knowledge (256-512 slots, ns retrieval) │
│         ↓ Dynamic swap in/out                                │
│  L1: CPU Memory/Redis   Warm knowledge (10,000+ slots, μs retrieval) │
│         ↓ On-demand loading                                  │
│  L2: PostgreSQL         Cold knowledge (million+ slots, ms retrieval) │
└─────────────────────────────────────────────────────────────┘
Key: Knowledge always maintains independent slot form, not mixed with model parameters
```

#### Challenge 2: Knowledge Aging and Forgetting

**Problem Description**:

-   Knowledge has time-sensitivity; outdated knowledge needs to be forgotten
-   AGA's isolation mechanism can only "disable" but not truly forget
-   Long-term operation accumulates large amounts of DEPRECATED/QUARANTINED slots

**Solution**: Graceful Forgetting Mechanism

```
Forgetting process (maintaining sovereign boundaries):
1. DEPRECATED → Observation period (rollback-able)
2. QUARANTINED → Isolation period (not participating in inference)
3. Archive to cold storage (retain audit logs)
4. Release slot resources (can be reused by new knowledge)

Knowledge consolidation (non-distillation):
- Merge semantically overlapping slots
- Maintain traceability of each original LU ID
- Release redundant slots
```

#### Challenge 3: Knowledge Consistency and Conflicts

**Problem Description**:

-   Multiple slots may contain contradictory knowledge
-   During inference, how to decide which to trust?
-   Conflicts are more complex in distributed scenarios

**Solution**: Slot-Level Conflict Resolution

```
Detection layer:
  - Semantic similarity > 0.9 but different decision → Potential conflict
  - Multiple slots for same condition → Version conflict

Resolution strategies:
  1. Time priority: Retain newest knowledge
  2. Reliability priority: Retain slots with higher reliability
  3. Governance priority: Mark conflict, await human adjudication
  4. Coexistence strategy: Retain all versions, choose by context during inference

Key: Conflict resolution at slot level, does not affect primary model
```

#### Challenge 4: Cross-Model Knowledge Migration

**Problem Description**:

-   AGA slot key/value vectors are model-specific
-   After changing base model, existing knowledge cannot be directly used
-   Does this mean "retraining" is needed?

**Solution**: Zero-Training Knowledge Migration

```
Retained content (migratable):
  - lu_id, condition, decision (semantic description)
  - lifecycle_state, reliability (governance state)
  - hit_count, metadata (usage statistics)

Regenerated content:
  - key_vector = new_model.encode(condition)
  - value_vector = new_model.encode(decision)

Migration process:
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

**Solution**: Inference Optimization (maintaining hot-swappable characteristics)

```
1. Early Exit (implemented): Gate-0 directly bypasses, skipping entire AGA
2. Sparse attention: Only compute attention for Top-k relevant slots
3. Asynchronous prefetch: Predict and preload slots needed by next layer
4. Batch slot fusion: Merge similar slots for computation, reduce redundancy

Principle: Optimization at "implementation level", does not change "architectural level"
```

#### Challenge 6: Adversarial Knowledge Injection

**Problem Description**:

-   Malicious users may inject harmful knowledge
-   AGA's "zero-training injection" makes attack cost lower
-   Traditional security mechanisms (such as RLHF) do not apply

**Solution**: Knowledge Security Protection Mechanism

```
Injection-time checks:
  - Semantic safety filtering (detect harmful content)
  - Source verification (knowledge source credibility)
  - Format validation (prevent injection attacks)

Runtime protection:
  - Default PROBATIONARY state (reliability 0.3)
  - Anomaly detection (output mutation → auto quarantine)
  - Impact scope limitation (propagation_radius)

Governance layer safeguards:
  - Human approval workflow (S2/S3 level knowledge)
  - Quorum voting for quarantine (minority rule takes effect)
  - Complete audit logs

Key: Security is a "governance issue", not a "training issue"
```

### 🔮 Future Optimization Directions

#### Short-term Optimization (v3.2)

1. **Governance Layer Enhancement**

    - Human governance interface (approval workflow UI)
    - Automatic knowledge quality assessment
    - Conflict detection and resolution

2. **Performance Optimization**

    - GPU memory pooling and dynamic swap
    - Batch inference optimization
    - Cache warming strategy

3. **Monitoring Enhancement**
    - Prometheus metric export
    - Distributed tracing
    - Anomaly detection alerts

#### Medium-term Optimization (v4.0)

1. **Knowledge Consolidation (Non-Distillation)**

    - Merge semantically overlapping slots
    - Knowledge compression (maintaining slot form)
    - Redundancy detection and cleanup

2. **Multi-Modal Support**

    - Image knowledge injection
    - Cross-modal retrieval
    - Multi-modal gating

3. **Federated Knowledge Sharing**
    - Cross-organization knowledge synchronization
    - Privacy protection (differential privacy slots)
    - Intellectual property marking

#### Long-term Vision (v5.0+)

1. **Cognitive Architecture**

    - Multi-layer knowledge representation (facts/rules/strategies)
    - Interpretable reasoning chains
    - Meta-cognitive capability (knowing what you don't know)

2. **Autonomous Governance**

    - Knowledge self-assessment
    - Automatic lifecycle management
    - Self-repair (auto quarantine detected problematic knowledge)

3. **Cross-Model Knowledge Migration**
    - Semantic-level knowledge export/import
    - Model-agnostic knowledge representation
    - Knowledge version control

### ❌ Explicitly Excluded Directions

The following directions conflict with AGA's core philosophy and are **explicitly excluded from the roadmap**:

| Excluded Direction                      | Exclusion Reason                                              |
| --------------------------------------- | ------------------------------------------------------------- |
| **Knowledge Distillation to LoRA**      | Violates "zero-training" principle, breaks sovereign boundary |
| **Fine-tuning Base Model**              | Returns to pre-training paradigm, loses hot-swappable nature  |
| **Merging Slots into Model Parameters** | Cannot trace or isolate individual knowledge                  |
| **RLHF-style Alignment**                | Requires training, cannot target individual knowledge         |

> **AGA's Core Promise**: Knowledge is always "external", always traceable, always isolatable.

### 📄 Related Papers

This project is based on the paper _"Auxiliary Governed Attention: A Governable, Inference-time Auxiliary Attention Mechanism with Sovereign Boundaries for Frozen Transformers"_.

### 📊 Code Statistics

| Module             | Lines       | Description               |
| ------------------ | ----------- | ------------------------- |
| `aga/core.py`      | ~1,260      | Core AGA implementation   |
| `aga/operator/`    | ~1,330      | Unified operator layer    |
| `aga/persistence/` | ~3,100      | Multi-adapter persistence |
| `aga/distributed/` | ~1,500      | Distributed support       |
| `aga/production/`  | ~2,900      | Production modules        |
| **Total**          | **~12,900** |                           |

### 📄 License

MIT License

---

<div align="center">

**AGA** - _Making LLMs safely borrow knowledge while knowing the difference_

</div>
