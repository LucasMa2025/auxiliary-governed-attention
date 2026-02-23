# AGA 插件生态系统

<p align="center">
  <strong>冻结 LLM 的无损能力扩展</strong><br/>
  注意力治理 · 知识管理 · 可观测性
</p>

<p align="center">
  <img src="https://img.shields.io/badge/aga--core-v4.4.0-blue" alt="aga-core"/>
  <img src="https://img.shields.io/badge/aga--knowledge-v0.3.0-green" alt="aga-knowledge"/>
  <img src="https://img.shields.io/badge/aga--observability-v1.0.0-orange" alt="aga-observability"/>
  <img src="https://img.shields.io/badge/python-3.9+-brightgreen" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license"/>
</p>

---

## 什么是 AGA？

**AGA（辅助注意力治理）** 是面向冻结大语言模型的**运行时注意力治理插件**。当 LLM 在推理过程中遇到知识空白（表现为高熵/不确定性）时，AGA 自动将外部知识注入到 Transformer 的注意力层中 — **不修改任何模型参数**。

**AGA 不是 RAG，不是 LoRA，不是 Prompt Engineering。** 它在注意力层层面、推理过程中工作，由模型自身的熵信号驱动原子事实注入。

```
Token → Transformer 层 → 自注意力 → [熵高？] → AGA 注入 → 融合输出
```

---

## 单仓库结构

本仓库包含三个独立但可互操作的 Python 包：

```
AGAPlugin/
├── aga/                    ← aga-core（必需）
│   ├── plugin.py           # AGAPlugin — 3 行集成
│   ├── config.py           # AGAConfig — 完整外部化配置
│   ├── kv_store.py         # GPU 常驻 KV 存储
│   ├── gate/               # 三级熵门控
│   ├── operator/           # 瓶颈注入器
│   ├── retriever/          # BaseRetriever 协议
│   ├── adapter/            # HuggingFace / vLLM 适配器
│   ├── streaming.py        # 流式生成会话
│   ├── distributed.py      # 张量并行支持
│   └── instrumentation/    # EventBus、指标、审计
│
├── aga_knowledge/          ← aga-knowledge（可选）
│   ├── portal/             # FastAPI REST API
│   ├── persistence/        # SQLite / PostgreSQL / Redis
│   ├── encoder/            # 文本→向量（SentenceTransformer）
│   ├── retriever/          # HNSW + BM25 + RRF 混合检索
│   ├── chunker/            # 文档 → 知识片段
│   ├── alignment.py        # AGACoreAlignment
│   └── sync/               # Redis Pub/Sub 同步
│
├── aga_observability/      ← aga-observability（可选）
│   ├── prometheus_exporter.py  # Prometheus 指标
│   ├── grafana_dashboard.py    # 自动生成仪表盘
│   ├── alert_manager.py        # SLO/SLI 告警
│   ├── audit_storage.py        # 持久化审计追踪
│   └── health.py               # 健康检查 HTTP 端点
│
├── configs/                # 示例配置文件
│   ├── portal_config.yaml  # aga-knowledge Portal 配置
│   └── runtime_config.yaml # aga-core 运行时配置
│
├── tests/                  # 所有单元测试
├── pyproject.toml          # 根包（aga-core）
└── README_zh.md            # 本文件
```

### 包依赖关系

```
+-----------------------------------------------------+
|                  AGA 生态系统                         |
|                                                      |
|  +-------------+                                     |
|  |  aga-core   | ← 必需（唯一依赖: torch）            |
|  |  v4.4.0     |                                     |
|  +------+------+                                     |
|         |                                            |
|  +------v------+  +--------------------+             |
|  |aga-knowledge|  | aga-observability  | ← 可选       |
|  | v0.3.0      |  | v1.0.0            |              |
|  | (不依赖     |  | (依赖 aga-core)   |              |
|  |  aga-core)  |  |                    |              |
|  +-------------+  +--------------------+             |
+-----------------------------------------------------+
```

- **aga-core** 可以**完全独立使用**。唯一依赖：`torch>=2.0.0`。
- **aga-knowledge** 独立运行 — 管理明文知识，通过 `BaseRetriever` 协议提供向量检索。
- **aga-observability** 依赖 `aga-core` — 订阅 `EventBus` 事件实现监控。

---

## 快速开始

### 3 行集成（仅 aga-core）

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)
output = model.generate(input_ids)  # AGA 自动工作
```

### 全栈集成（aga-core + aga-knowledge + aga-observability）

```python
from aga import AGAPlugin, AGAConfig
from aga_knowledge import KnowledgeManager, AGACoreAlignment
from aga_knowledge.config import PortalConfig
from aga_knowledge.encoder import create_encoder, EncoderConfig
from aga_knowledge.retriever import KnowledgeRetriever

# 1. 对齐配置
alignment = AGACoreAlignment(
    hidden_dim=4096, bottleneck_dim=64,
    key_norm_target=5.0, value_norm_target=3.0,
)

# 2. 知识管理
manager = KnowledgeManager(PortalConfig.for_development())
await manager.start()

# 3. 编码器 + 召回器
encoder = create_encoder(EncoderConfig.from_alignment(alignment))
retriever = KnowledgeRetriever(
    manager=manager, encoder=encoder,
    alignment=alignment, namespace="default",
)

# 4. 插件
config = AGAConfig(
    hidden_dim=4096, bottleneck_dim=64,
    observability_enabled=True,  # 自动检测 aga-observability
)
plugin = AGAPlugin(config, retriever=retriever)
plugin.attach(model)
```

---

## 安装

### 从源码安装（单仓库）

```bash
cd AGAPlugin

# 仅安装 aga-core
pip install -e .

# 安装 aga-knowledge
pip install -e ./aga_knowledge[all]

# 安装 aga-observability
pip install -e ./aga_observability[full]

# 安装全部
pip install -e .[all]
pip install -e ./aga_knowledge[all]
pip install -e ./aga_observability[full]
```

### 从 PyPI 安装（发布后）

```bash
pip install aga-core                          # 仅核心
pip install aga-core[knowledge,observability] # 全栈
```

---

## 文档

### aga-core

| 文档 | 语言 |
| --- | --- |
| [README (English)](aga/README_en.md) | English |
| [README (中文)](aga/README_zh.md) | 中文 |
| [Product Documentation (English)](aga/docs/product_doc_en.md) | English |
| [产品说明书 (中文)](aga/docs/product_doc_zh.md) | 中文 |
| [User Manual (English)](aga/docs/user_manual_en.md) | English |
| [用户手册 (中文)](aga/docs/user_manual_zh.md) | 中文 |

### aga-knowledge

| 文档 | 语言 |
| --- | --- |
| [README (English)](aga_knowledge/README_en.md) | English |
| [README (中文)](aga_knowledge/README_zh.md) | 中文 |
| [Product Documentation (English)](aga_knowledge/docs/product_doc_en.md) | English |
| [产品说明书 (中文)](aga_knowledge/docs/product_doc_zh.md) | 中文 |
| [User Manual (English)](aga_knowledge/docs/user_manual_en.md) | English |
| [用户手册 (中文)](aga_knowledge/docs/user_manual_zh.md) | 中文 |

### aga-observability

| 文档 | 语言 |
| --- | --- |
| [README (English)](aga_observability/README_en.md) | English |
| [README (中文)](aga_observability/README_zh.md) | 中文 |
| [User Manual (English)](aga_observability/docs/user_manual_en.md) | English |
| [用户手册 (中文)](aga_observability/docs/user_manual_zh.md) | 中文 |

---

## 配置

示例配置文件位于 `configs/` 目录：

| 文件 | 用途 | 使用方 |
| --- | --- | --- |
| [`configs/portal_config.yaml`](configs/portal_config.yaml) | 知识 Portal 服务器、持久化、消息队列和治理配置 | `aga-knowledge` Portal |
| [`configs/runtime_config.yaml`](configs/runtime_config.yaml) | AGA 运行时参数、熵门控、衰减、同步和设备配置 | `aga-core` AGAPlugin |

### 使用方式

```python
# aga-core：加载运行时配置
from aga import AGAPlugin
plugin = AGAPlugin.from_config("configs/runtime_config.yaml")

# aga-knowledge：加载 Portal 配置
from aga_knowledge.config import PortalConfig
config = PortalConfig.from_yaml("configs/portal_config.yaml")
```

---

## 各包核心特性

### aga-core v4.4.0

- **3 行集成** — `AGAPlugin(config).attach(model)`
- **三级熵门控** — Gate-0（命名空间）/ Gate-1（熵）/ Gate-2（置信度）
- **瓶颈注意力注入** — 查询投影 → Top-K 路由 → 值投影
- **GPU 常驻 KVStore** — LRU 淘汰 + 知识固定 + 命名空间隔离
- **流式生成** — 逐 token 生成过程中的知识注入
- **标准召回器协议** — `BaseRetriever` 接口支持可插拔检索
- **Slot 治理** — 预算控制、语义去重、冷却期、稳定性检测
- **HuggingFace + vLLM 适配器** — LLaMA、Qwen、Mistral、GPT-2、Phi、Gemma、Falcon
- **张量并行** — `TPManager` 支持多 GPU KVStore 广播
- **Fail-Open 安全** — 异常永不阻断推理

### aga-knowledge v0.3.0

- **明文知识** — `condition/decision` 对，人类可读
- **4 种持久化后端** — 内存、SQLite、PostgreSQL、Redis
- **Portal REST API** — 基于 FastAPI 的知识 CRUD + 图片资产服务
- **混合检索** — HNSW（稠密）+ BM25（稀疏）+ RRF（融合）
- **AGACoreAlignment** — 编码器-核心维度/范数强制对齐
- **文档分片** — 5 种策略 + DocumentChunker + ConditionGenerator + ImageHandler
- **跨实例同步** — Redis Pub/Sub 实时知识同步
- **版本控制** — 知识版本管理，支持回滚和差异比较

### aga-observability v1.0.0

- **Prometheus 导出器** — 15+ 指标（计数器、直方图、仪表盘）
- **Grafana 仪表盘** — 自动生成 5 组面板 JSON
- **SLO/SLI 告警** — 可配置规则，支持 Webhook/回调通道
- **结构化日志** — JSON/文本格式，支持文件轮转
- **持久化审计** — JSONL 或 SQLite，支持保留策略
- **健康检查** — HTTP 端点，支持 Kubernetes 探针
- **零侵入** — EventBus 订阅，不修改 aga-core

---

## 测试

```bash
# 全部测试
python -m pytest tests/ -v

# aga-core 测试
python -m pytest tests/test_aga/ -v

# aga-knowledge 测试
python -m pytest tests/test_knowledge/ -v

# aga-observability 测试
python -m pytest tests/test_observability/ -v
```

---

## 路线图

| 包 | 当前版本 | 下一个里程碑 |
| --- | --- | --- |
| **aga-core** | v4.4.0 — 召回器协议、Slot 治理、流式生成 | v5.0 — 分层知识、INT8 KVStore、自适应瓶颈 |
| **aga-knowledge** | v0.3.0 — HNSW+BM25+RRF、DocumentChunker、AGACoreAlignment | v0.4.x — 对比学习微调、分布式编码器、Prometheus |
| **aga-observability** | v1.0.0 — Prometheus、Grafana、告警、审计、健康检查 | v1.1.0 — OpenTelemetry 链路追踪、分布式聚合 |

---

## 文件说明

| 文件/目录 | 说明 |
| --- | --- |
| `pyproject.toml` | 根包配置，定义 aga-core 及其可选依赖 |
| `configs/` | 示例配置文件，供 aga-core 和 aga-knowledge 使用 |
| `aga_dev.db` | 开发模式下 aga-knowledge 的默认 SQLite 数据库（已加入 `.gitignore`） |
| `tests/` | 所有子项目的单元测试 |
| `Qustions/` | 技术分析文档（多 GPU、高熵语义、Slot 抖动等） |

---

## 许可证

MIT License

Copyright (c) 2024-2026 AGA Team

---

<p align="center">
  <strong>AGA — 让每一次推理都充满知识</strong>
</p>
