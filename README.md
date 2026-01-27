# AGA - Auxiliary Governed Attention

<div align="center">

![AGA Logo](https://img.shields.io/badge/AGA-Auxiliary%20Governed%20Attention-blue?style=for-the-badge)
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

| 现有方案 | 问题 |
|----------|------|
| **全量微调** | 灾难性遗忘、计算成本高 |
| **LoRA/Adapter** | 需要训练、超参敏感 |
| **RAG** | 外部检索而非内化能力，模型无法区分"知道"与"借用" |

**AGA (Auxiliary Governed Attention)** 提出了一种新范式：

> 在冻结的 Transformer 上附加一个可治理的辅助注意力模块，在推理时动态注入知识，同时保持主模型与辅助知识之间的**主权边界**。

### 🎯 核心特性

| 特性 | 说明 |
|------|------|
| **零训练注入** | 知识直接写入 buffer，无需梯度计算 |
| **热插拔设计** | 运行时动态添加/移除知识 |
| **治理控制** | 生命周期状态（试用→确认→弃用→隔离） |
| **熵门控** | 主模型自信时不干预，不确定时才贡献 |
| **即时隔离** | 问题知识可立即移除影响 |
| **完整可追溯** | 每个知识槽位绑定 LU ID |
| **数据持久化** | SQLite 存储，避免停机失效 |

### 📁 项目结构

```
AGA/
├── aga/                        # 核心模块
│   ├── __init__.py
│   ├── core.py                 # AGA 核心实现
│   └── persistence.py          # 数据持久化
│
├── llm/                        # LLM 适配器
│   ├── adapters/
│   │   ├── base.py             # 适配器基类
│   │   ├── deepseek.py         # DeepSeek 适配器
│   │   ├── ollama.py           # Ollama 适配器
│   │   ├── vllm.py             # vLLM 适配器
│   │   └── openai_compat.py    # OpenAI 兼容适配器
│   └── ...
│
├── aga_experiment_tool/        # Web 实验工具
│   ├── app.py                  # Flask 应用
│   ├── config.yaml             # 配置文件
│   └── requirements.txt
│
├── scripts/                    # 启动脚本
│   ├── start_ollama.sh
│   ├── start_vllm.sh
│   ├── start_experiment_tool.sh
│   └── start_experiment_tool.bat
│
└── README.md
```

### 🚀 快速开始

#### 1. 安装依赖

```bash
pip install torch transformers flask pyyaml
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
from aga import AGA, LifecycleState
from aga.persistence import SQLitePersistence, AGAPersistenceManager

# 创建 AGA
aga = AGA(hidden_dim=768, bottleneck_dim=64, num_slots=100)
aga.eval()

# 创建持久化管理器
persistence = SQLitePersistence("aga_data.db")
pm = AGAPersistenceManager(persistence, aga_id="my_model")

# 注入知识（带持久化）
import torch
key_vector = torch.randn(64)
value_vector = torch.randn(768)

slot_idx = pm.sync_knowledge(
    aga=aga,
    lu_id="LU_001_paris",
    condition="capital of France",
    decision="Paris",
    key_vector=key_vector,
    value_vector=value_vector,
    lifecycle_state=LifecycleState.PROBATIONARY,
)

# 确认知识
pm.sync_lifecycle_update(aga, "LU_001_paris", LifecycleState.CONFIRMED)

# 查看统计
print(aga.get_statistics())
```

### 💾 数据持久化

AGA 使用 SQLite 实现数据持久化，避免停机后知识丢失：

```python
from aga.persistence import SQLitePersistence, AGAPersistenceManager

# 初始化持久化
persistence = SQLitePersistence("aga_data.db")

# 创建管理器
pm = AGAPersistenceManager(persistence, aga_id="model_1")

# 保存 AGA 状态
pm.save_aga(aga)

# 重启后加载
pm.load_aga(aga)

# 查看数据库统计
print(persistence.get_statistics())
```

**数据库表结构**：
- `aga_states`: AGA 实例配置
- `knowledge`: 知识条目（key/value 向量、生命周期、元数据）
- `audit_log`: 审计日志

> **生产环境**: Demo 使用 SQLite，生产环境请自行实现 PostgreSQL/Redis 版本的 `AGAPersistence` 接口。

### 🔌 LLM 适配器

支持多种 LLM 后端：

| 适配器 | 说明 | 使用场景 |
|--------|------|----------|
| `OllamaAdapter` | Ollama 本地模型 | 开发测试 |
| `VLLMAdapter` | vLLM 高性能推理 | 生产部署 |
| `DeepSeekAdapter` | DeepSeek API/本地 | API 调用 |
| `OpenAICompatAdapter` | OpenAI 兼容接口 | 通用 |

```python
from llm.adapters import OllamaAdapter, LLMConfig

# Ollama
adapter = OllamaAdapter(
    base_url="http://localhost:11434",
    model="qwen2.5:7b"
)

# 聊天
response = adapter.chat([
    {"role": "user", "content": "Hello"}
])
print(response.content)
```

### 🔧 实验工具功能

| 功能 | 说明 |
|------|------|
| **模型加载** | 支持 GPT-2、Qwen、LLaMA、Mistral 等 |
| **知识注入** | 条件 → 决策，支持生命周期状态 |
| **生命周期管理** | 确认、弃用、隔离操作 |
| **推理测试** | 测试注入知识的效果 |
| **数据收集** | 批量测试并导出实验数据 |
| **统计监控** | 槽位使用、命中计数、状态分布 |

### 📊 知识生命周期

```
                    ┌────────────────────────────────────────┐
                    │                                        │
                    ▼                                        │
    ┌────────────────────────┐                               │
    │     PROBATIONARY       │──── 使用指标正常  ─────────────┤
    │     (r = 0.3)          │                               │
    └───────────┬────────────┘                               │
                │ 治理审批                                    │
                ▼                                            │
    ┌────────────────────────┐                               │
    │      CONFIRMED         │──── 异常检测  ─────────────────┤
    │     (r = 1.0)          │                               │
    └───────────┬────────────┘                               │
                │ 弃用请求                                    │
                ▼                                            │
    ┌────────────────────────┐                               │
    │      DEPRECATED        │──── 保留期过期  ───────────────┤
    │     (r = 0.1)          │                               │
    └───────────┬────────────┘                               │
                │ 清理                                       │
                ▼                                            │
    ┌────────────────────────┐                               │
    │     QUARANTINED        │◀─────────────────────────────┘
    │     (r = 0.0)          │      （紧急隔离可直接跳转）
    └────────────────────────┘
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

**熵门控原理**：
- 主模型自信（低熵）→ gate ≈ 0 → AGA 不干预
- 主模型不确定（高熵）→ gate 增大 → AGA 可贡献

### 📄 相关论文

本项目基于论文《Auxiliary Governed Attention: A Governable, Inference-time Auxiliary Attention Mechanism with Sovereign Boundaries for Frozen Transformers》实现。

---

## English

### 📖 Background

How to dynamically integrate new knowledge into deployed LLMs without compromising existing capabilities remains a long-standing challenge. **AGA (Auxiliary Governed Attention)** proposes a new paradigm:

> Attach a governable auxiliary attention module to frozen Transformers, dynamically injecting knowledge at inference time while maintaining **sovereign boundaries** between the primary model and auxiliary knowledge.

### 🎯 Core Features

- **Zero-Training Injection**: Knowledge written directly to buffers
- **Hot-Pluggable Design**: Dynamic add/remove at runtime
- **Governance Control**: Lifecycle states with reliability weighting
- **Entropy Gating**: Primary-first principle
- **Instant Isolation**: Immediate removal of problematic knowledge
- **Full Traceability**: Each slot bound to Learning Unit ID
- **Data Persistence**: SQLite storage to survive restarts

### 🚀 Quick Start

```bash
# Install dependencies
pip install torch transformers flask pyyaml

# Start experiment tool
python -m aga_experiment_tool.app --port 8765
```

Access `http://localhost:8765`, default password: `aga_experiment_2026`

### 💾 Data Persistence

```python
from aga.persistence import SQLitePersistence, AGAPersistenceManager

# Initialize persistence
persistence = SQLitePersistence("aga_data.db")
pm = AGAPersistenceManager(persistence, aga_id="my_model")

# Inject with persistence
slot_idx = pm.sync_knowledge(
    aga=aga,
    lu_id="LU_001",
    condition="capital of France",
    decision="Paris",
    key_vector=key_vec,
    value_vector=val_vec,
)

# Load on restart
pm.load_aga(aga)
```

> **Production**: Demo uses SQLite. For production, implement `AGAPersistence` interface with PostgreSQL/Redis.

### 🔌 LLM Adapters

| Adapter | Description |
|---------|-------------|
| `OllamaAdapter` | Local Ollama models |
| `VLLMAdapter` | High-performance vLLM |
| `DeepSeekAdapter` | DeepSeek API/local |
| `OpenAICompatAdapter` | Generic OpenAI-compatible |

### 📄 License

MIT License

---

<div align="center">

**AGA** - *Making LLMs safely borrow knowledge while knowing the difference*

</div>


