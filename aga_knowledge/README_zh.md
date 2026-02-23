# aga-knowledge — AGA 知识管理系统

> **冻结模型的外部知识大脑 — 注册、编码、检索、注入。**

`aga-knowledge` 是 **AGA (辅助注意力治理)** 生态系统的独立知识管理包。它提供从明文知识注册到向量编码检索的完整流水线，将领域知识与 `aga-core` 的注意力层注入机制桥接。

## 为什么需要 aga-knowledge？

冻结的 LLM 缺乏领域特定事实。`aga-core` 可以在高熵时刻将知识注入注意力层，但它需要一个**知识来源**。`aga-knowledge` 就是这个来源：

```
领域专家 → Portal API → 持久化 → 同步 → KnowledgeManager → 编码器 → 召回器 → aga-core
```

没有 `aga-knowledge`，用户必须手动构造 key/value 向量并调用 `plugin.register()`。有了 `aga-knowledge`，从明文 `condition/decision` 对到生产级向量检索的整个流水线都是自动化的。

## 架构

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          aga-knowledge v0.3.0                                  │
│                                                                                │
│  ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌────────────────┐         │
│  │  Portal  │──▶│   持久化     │──▶│   同步    │──▶│    知识        │         │
│  │  (REST)  │    │   适配器    │    │  (Redis)  │    │   管理器       │         │
│  │  +资产   │    └─────────────┘    └──────────┘     └──────┬────────┘         │
│  └──────────┘                                            │                     │
│                                                          │                     │
│  ┌───────────────────────┐    ┌─────────────┐    ┌───────▼──────────────────┐  │
│  │  分片器 (Chunker)      │──▶│  编码器     │──▶ │  KnowledgeRetriever     │  │
│  │  ├ DocumentChunker    │    │ (文本→向量)  │    │  ├ HNSW 稠密检索         │  │
│  │  ├ ConditionGenerator │    │  +对齐验证   │    │  ├ BM25 稀疏检索         │  │
│  │  ├ ImageHandler       │    └─────────────┘    │  └ RRF 融合              │  │
│  │  └ 4种基础策略         │                       │  (BaseRetriever 协议)    │  │
│  └───────────────────────┘                       └──────────┬───────────────┘  │
│                                                             │                  │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────┐        │                  │
│  │ 版本控制  │  │   文本压缩  │   │ AGACoreAlignment │       │                  │
│  └──────────┘  └─────────────┘  └──────────────────┘        │                  │
└─────────────────────────────────────────────────────────────┼──────────────────┘
                                                              │
                                              ┌───────────────▼────────┐
                                              │      aga-core          │
                                              │  (AGAPlugin.attach)    │
                                              │  高熵门控触发           │
                                              │  → retrieve() 调用     │
                                              │  → KV 注入             │
                                              └────────────────────────┘
```

## 特性列表

### 核心特性 (已实现 ✅)

| 类别           | 特性                       | 状态 | 说明                                                 |
| -------------- | -------------------------- | ---- | ---------------------------------------------------- |
| **知识注册**   | Portal REST API            | ✅   | 基于 FastAPI 的 HTTP 知识管理 API                    |
|                | 明文 condition/decision 对 | ✅   | 人类可读的知识格式                                   |
|                | 批量注入                   | ✅   | 批量知识注册                                         |
|                | 命名空间隔离               | ✅   | 多租户知识分离                                       |
|                | 生命周期管理               | ✅   | 试用期 → 已确认 → 已弃用 → 已隔离                    |
|                | 信任层级                   | ✅   | 系统级 / 已验证 / 标准 / 实验性 / 不可信             |
| **持久化**     | 内存适配器                 | ✅   | 内存存储（测试用）                                   |
|                | SQLite 适配器              | ✅   | 文件存储（开发用）                                   |
|                | PostgreSQL 适配器          | ✅   | 生产级关系存储（asyncpg）                            |
|                | Redis 适配器               | ✅   | 高性能缓存层（aioredis）                             |
|                | 适配器工厂                 | ✅   | 配置驱动的适配器创建                                 |
|                | 审计日志                   | ✅   | 所有 CRUD 操作均有审计记录                           |
| **同步**       | Redis Pub/Sub              | ✅   | 跨实例实时知识同步                                   |
|                | 内存后端                   | ✅   | 进程内同步（测试用）                                 |
|                | 消息协议                   | ✅   | 类型化消息，支持 ACK/NACK                            |
|                | 全量同步                   | ✅   | 按需全量状态同步                                     |
|                | 心跳检测                   | ✅   | 实例存活检测                                         |
| **知识管理器** | 本地缓存                   | ✅   | 内存缓存，快速读取                                   |
|                | 异步消息订阅               | ✅   | 自动从 Portal 同步                                   |
|                | 缓存穿透读取               | ✅   | 缓存未命中 → 持久化层回退                            |
|                | 命中计数追踪               | ✅   | 知识使用统计                                         |
|                | 生命周期感知过滤           | ✅   | 仅提供活跃知识                                       |
| **编码器**     | BaseEncoder 协议           | ✅   | 文本 → 向量的抽象接口                                |
|                | SentenceTransformerEncoder | ✅   | 生产编码器（语义嵌入 + 投影层）                      |
|                | SimpleHashEncoder          | ✅   | 确定性哈希编码器（测试用）                           |
|                | 编码缓存                   | ✅   | FIFO 缓存避免重复编码                                |
|                | 批量编码                   | ✅   | 高效批量处理                                         |
|                | 投影层                     | ✅   | 可训练的 Linear(embed_dim → key_dim/value_dim)       |
|                | 范数缩放                   | ✅   | 匹配 aga-core 的 key_norm_target / value_norm_target |
|                | AGACoreAlignment 对齐      | ✅   | EncoderConfig.from_alignment() 自动配置维度和范数    |
| **召回器**     | KnowledgeRetriever         | ✅   | BaseRetriever 协议适配器                             |
|                | HNSW 稠密检索              | ✅   | hnswlib ANN 索引（10K+ 知识场景）                    |
|                | BM25 稀疏检索              | ✅   | rank-bm25 关键词匹配                                 |
|                | RRF 融合                   | ✅   | 互惠排序融合稠密+稀疏结果                            |
|                | 余弦相似度回退             | ✅   | hnswlib 不可用时暴力搜索                             |
|                | AGACoreAlignment 验证      | ✅   | 构造时强制验证编码器与 aga-core 对齐                 |
|                | 增量索引更新               | ✅   | 无需全量重建即可添加/删除                            |
|                | 自动刷新                   | ✅   | 定期从 KnowledgeManager 刷新索引                     |
|                | 注入反馈                   | ✅   | 成功注入时更新命中计数                               |
|                | 线程安全检索               | ✅   | RLock 保护的并发访问                                 |
|                | Fail-Open 设计             | ✅   | 检索失败返回空列表，依赖可选降级                     |
| **分片器**     | FixedSizeChunker           | ✅   | 基于 token 数的固定大小分片                          |
|                | SentenceChunker            | ✅   | 句子边界感知分片                                     |
|                | SemanticChunker            | ✅   | 基于嵌入的语义分组                                   |
|                | SlidingWindowChunker       | ✅   | 重叠滑动窗口分片                                     |
|                | DocumentChunker            | ✅   | Markdown 结构感知分片（v0.3.0）                      |
|                | ConditionGenerator         | ✅   | 多策略条件生成（首句/标题/关键词/摘要）              |
|                | ImageHandler               | ✅   | 文档图片处理（Base64/URL/本地路径 →Portal 资产）     |
|                | 可配置策略                 | ✅   | YAML 驱动的分片器选择                                |
| **版本控制**   | 版本历史                   | ✅   | 每个知识单元的完整变更历史                           |
|                | 版本回滚                   | ✅   | 恢复到任意历史版本                                   |
|                | 差异比较                   | ✅   | 版本间并排比较                                       |
|                | 变更审计                   | ✅   | 谁在何时修改了什么                                   |
| **压缩**       | zlib 压缩                  | ✅   | 标准文本压缩                                         |
|                | LZ4 压缩                   | ✅   | 快速压缩（可选）                                     |
|                | Zstd 压缩                  | ✅   | 平衡压缩（可选）                                     |
|                | 解压缓存                   | ✅   | LRU 缓存热数据                                       |
| **配置**       | 数据类配置                 | ✅   | 类型安全的配置                                       |
|                | YAML 文件加载              | ✅   | 外部配置文件                                         |
|                | 环境变量覆盖               | ✅   | 12-factor 应用支持                                   |
|                | 开发/生产预设              | ✅   | 快速启动配置                                         |
|                | 配置适配器                 | ✅   | 桥接 aga-core 配置格式                               |

### 与 aga-core 的集成

| 集成点             | 状态 | 说明                                                 |
| ------------------ | ---- | ---------------------------------------------------- |
| BaseRetriever 协议 | ✅   | KnowledgeRetriever 实现 aga-core 的召回器接口        |
| AGACoreAlignment   | ✅   | 编码器-核心对齐验证（维度/范数强制匹配）             |
| 维度对齐           | ✅   | 编码器输出匹配 AGAConfig.bottleneck_dim / hidden_dim |
| 范数目标匹配       | ✅   | key_norm_target 和 value_norm_target 可配置          |
| Fail-Open 安全     | ✅   | 所有检索失败返回空结果                               |
| 插件构造器         | ✅   | `AGAPlugin(config, retriever=knowledge_retriever)`   |
| EventBus 集成      | ✅   | 知识事件流向 aga-observability                       |

## 快速开始

### 安装

```bash
# 仅核心（无外部依赖）
pip install aga-knowledge

# 含 Portal API
pip install aga-knowledge[portal]

# 含 PostgreSQL 持久化
pip install aga-knowledge[postgres]

# 含 Redis 同步 + 持久化
pip install aga-knowledge[redis]

# 含编码器（SentenceTransformer）
pip install aga-knowledge[encoder]

# 含检索增强（HNSW + BM25）
pip install aga-knowledge[retrieval]

# 完整安装
pip install aga-knowledge[all]
```

### 基本使用（配合 aga-core）

```python
import asyncio
from aga_knowledge import KnowledgeManager, AGACoreAlignment
from aga_knowledge.config import PortalConfig
from aga_knowledge.encoder import create_encoder, EncoderConfig
from aga_knowledge.retriever import KnowledgeRetriever
from aga.plugin import AGAPlugin
from aga.config import AGAConfig

async def main():
    # 1. 对齐配置（aga-core 与 aga-knowledge 的桥梁）
    alignment = AGACoreAlignment(
        hidden_dim=4096, bottleneck_dim=64,
        key_norm_target=5.0, value_norm_target=3.0,
    )

    # 2. 启动 KnowledgeManager
    km_config = PortalConfig.for_development()
    manager = KnowledgeManager(km_config)
    await manager.start()

    # 3. 创建对齐的编码器
    encoder_config = EncoderConfig.from_alignment(alignment)
    encoder = create_encoder(encoder_config)

    # 4. 创建混合召回器
    retriever = KnowledgeRetriever(
        manager=manager, encoder=encoder,
        alignment=alignment, namespace="default",
        index_backend="hnsw", bm25_enabled=True,
    )

    # 5. 创建带召回器的 AGAPlugin
    aga_config = AGAConfig(bottleneck_dim=64, hidden_dim=4096)
    plugin = AGAPlugin(aga_config, retriever=retriever)

    # 6. 挂载到模型
    plugin.attach(model)

asyncio.run(main())
```

### 文档分片流水线

```python
from aga_knowledge.chunker import (
    create_document_chunker, DocumentChunker,
    ConditionGenerator, ImageHandler,
    create_chunker, ChunkerConfig,
)

# 方式一：结构化文档分片（Markdown）
doc_chunker = create_document_chunker(
    condition_mode="title_context",
    portal_base_url="http://localhost:8081",
    assets_dir="./static/assets",
)
chunks = doc_chunker.chunk_document(
    markdown_text, source_id="doc_001", title="医疗指南"
)

# 方式二：基础分片
chunker = create_chunker(ChunkerConfig(
    strategy="sliding_window",
    chunk_size=300,
    overlap=50,
))
chunks = chunker.chunk_document(document, source_id="doc_001", title="医疗指南")

# 通过 Portal API 注册分片
for chunk in chunks:
    record = chunk.to_knowledge_record()
    # POST 到 /knowledge/inject-text
```

### 独立使用（不依赖 aga-core）

`aga-knowledge` 可以作为独立的知识管理系统使用：

```python
from aga_knowledge import KnowledgeManager
from aga_knowledge.config import PortalConfig

config = PortalConfig.for_development()
manager = KnowledgeManager(config)
await manager.start()

# 查询知识
records = await manager.get_active_knowledge("default")
record = await manager.get_knowledge("default", "rule_001")
```

## 配置

### YAML 配置

```yaml
# portal_config.yaml
server:
    host: "0.0.0.0"
    port: 8081
    workers: 4

persistence:
    type: "postgres" # memory | sqlite | postgres | redis
    postgres_host: "db.example.com"
    postgres_port: 5432
    postgres_database: "aga_knowledge"
    postgres_user: "aga"
    postgres_password: "${AGA_DB_PASSWORD}"
    postgres_pool_size: 20
    enable_audit: true

messaging:
    backend: "redis"
    redis_host: "redis.example.com"
    redis_port: 6379
    redis_channel: "aga:sync"

registry:
    type: "redis"
    heartbeat_interval: 30
    timeout: 90

governance:
    enabled: true
    auto_confirm_after_hits: 100
    auto_deprecate_after_days: 30
```

### 编码器配置

```yaml
encoder:
    backend: "sentence_transformer" # sentence_transformer | simple_hash
    model_name: "all-MiniLM-L6-v2"
    key_dim: 64 # 必须 == AGAConfig.bottleneck_dim
    value_dim: 4096 # 必须 == AGAConfig.hidden_dim
    device: "cpu"
    batch_size: 32
    normalize: true
    cache_enabled: true
    cache_max_size: 10000
    options:
        condition_prefix: "condition: "
        decision_prefix: "decision: "
        key_norm_target: 5.0
        value_norm_target: 3.0
        projection_path: null # 预训练投影层权重路径
```

### 分片器配置

```yaml
chunker:
    strategy: "sliding_window" # fixed_size | sentence | semantic | sliding_window | document
    chunk_size: 300 # 每片段目标 token 数
    overlap: 50 # 重叠 token 数（sliding_window）
    min_chunk_size: 50
    max_chunk_size: 500
    condition_mode: "first_sentence" # first_sentence | title_context | keyword | summary
    language: "auto" # auto | zh | en
```

### AGACoreAlignment 配置

```yaml
aga_core_alignment:
    hidden_dim: 4096 # == AGAConfig.hidden_dim
    bottleneck_dim: 64 # == AGAConfig.bottleneck_dim
    key_norm_target: 5.0 # == AGAConfig.key_norm_target
    value_norm_target: 3.0 # == AGAConfig.value_norm_target
```

## 包结构

```
aga_knowledge/
├── __init__.py                 # 包入口，导出 KnowledgeManager, AGACoreAlignment
├── alignment.py               # AGACoreAlignment 编码器-核心对齐
├── config.py                   # PortalConfig, PersistenceDBConfig 等
├── types.py                    # KnowledgeRecord, LifecycleState, TrustTier
├── exceptions.py               # 异常层次结构
├── pyproject.toml              # 包元数据
│
├── manager/                    # 面向 Runtime 的知识管理器
│   └── knowledge_manager.py    # 缓存、同步订阅、查询接口
│
├── portal/                     # REST API (FastAPI)
│   ├── app.py                  # 应用工厂（含 /assets 静态资源挂载）
│   ├── routes.py               # API 端点
│   ├── service.py              # 业务逻辑
│   └── registry.py             # Runtime 实例注册表
│
├── persistence/                # 存储后端
│   ├── base.py                 # PersistenceAdapter 抽象基类
│   ├── memory_adapter.py       # 内存存储（测试）
│   ├── sqlite_adapter.py       # SQLite 存储（开发）
│   ├── postgres_adapter.py     # PostgreSQL 存储（生产）
│   ├── redis_adapter.py        # Redis 存储（缓存层）
│   ├── versioning.py           # 版本历史与回滚
│   └── compression.py          # 文本压缩（zlib/lz4/zstd）
│
├── sync/                       # 跨实例同步
│   ├── protocol.py             # SyncMessage, MessageType
│   ├── publisher.py            # 消息发布
│   └── backends.py             # Redis/Memory 同步后端
│
├── encoder/                    # 文本 → 向量编码
│   ├── base.py                 # BaseEncoder 抽象基类, EncoderConfig
│   ├── sentence_transformer_encoder.py  # 生产编码器
│   └── simple_encoder.py       # 哈希测试编码器
│
├── retriever/                  # aga-core 召回器桥接
│   ├── knowledge_retriever.py  # KnowledgeRetriever (BaseRetriever 实现)
│   ├── hnsw_index.py           # HNSW 稠密向量索引
│   ├── bm25_index.py           # BM25 稀疏检索索引
│   └── fusion.py               # RRF 融合算法
│
├── chunker/                    # 文档 → 片段
│   ├── base.py                 # BaseChunker 抽象基类, ChunkerConfig
│   ├── fixed_size.py           # 固定 token 数分片
│   ├── sentence.py             # 句子边界分片
│   ├── semantic.py             # 基于嵌入的语义分片
│   ├── sliding_window.py       # 重叠滑动窗口分片
│   ├── document_chunker.py     # Markdown 结构感知分片
│   ├── condition_generator.py  # 多策略条件生成
│   └── image_handler.py        # 文档图片处理
│
├── scripts/                    # 数据库初始化脚本
│   ├── init_postgresql.sql     # PostgreSQL Schema
│   └── init_sqlite.sql         # SQLite Schema
│
└── config_adapter/             # 配置桥接
    └── adapter.py              # aga-core 配置 ↔ aga-knowledge 配置
```

## 路线图

### v0.2.x — 基础版本（已完成）

-   [x] 明文 condition/decision 知识模型
-   [x] 4 种持久化后端（Memory, SQLite, PostgreSQL, Redis）
-   [x] Portal REST API 完整 CRUD
-   [x] Redis Pub/Sub 同步
-   [x] KnowledgeManager 本地缓存
-   [x] 编码器模块（SentenceTransformer + SimpleHash）
-   [x] KnowledgeRetriever（BaseRetriever 协议）
-   [x] 4 种基础分片策略
-   [x] 版本控制和文本压缩
-   [x] 605+ 单元测试通过

### v0.3.0 — 当前版本（生产对齐）

-   [x] **AGACoreAlignment** — 编码器-核心对齐验证（维度/范数强制匹配）
-   [x] **HNSW 稠密检索** — hnswlib ANN 索引（10K+ 知识场景）
-   [x] **BM25 稀疏检索** — rank-bm25 关键词匹配
-   [x] **RRF 融合** — 互惠排序融合稠密+稀疏结果
-   [x] **DocumentChunker** — Markdown 结构感知分片
-   [x] **ConditionGenerator** — 多策略条件生成（首句/标题/关键词/摘要）
-   [x] **ImageHandler** — 文档图片处理（Base64/URL/本地路径 →Portal 资产）
-   [x] **Portal 图片资产服务** — StaticFiles 挂载 /assets
-   [x] **优雅降级** — hnswlib/rank-bm25 可选依赖，自动回退
-   [x] **数据库 Schema** — PostgreSQL + SQLite 初始化脚本

### v0.4.x — 生产强化（下一步）

-   [ ] **对比学习微调** — 投影层的对比学习训练（condition→key, decision→value）
-   [ ] **分布式编码器** — gRPC 编码器服务，支持多实例部署
-   [ ] **Key-Value 联合去重** — 超越 key-key 相似度，支持渐进细化
-   [ ] **嵌入版本管理** — 追踪编码器模型变更并重新编码
-   [ ] **Prometheus 指标** — 检索延迟、缓存命中率、索引大小
-   [ ] **速率限制** — Portal API 速率限制和配额管理

### v1.0.0 — 正式发布

-   [ ] API 稳定性保证
-   [ ] 全面安全审计
-   [ ] 性能基准测试（延迟、吞吐量、内存）
-   [ ] Kubernetes Helm Chart 部署
-   [ ] 官方文档站点

## 测试

```bash
# 运行所有测试
python -m pytest tests/test_knowledge/ -v

# 运行特定模块测试
python -m pytest tests/test_knowledge/test_encoder.py -v
python -m pytest tests/test_knowledge/test_chunker.py -v
python -m pytest tests/test_knowledge/test_document_chunker.py -v
python -m pytest tests/test_knowledge/test_postgres_adapter.py -v
python -m pytest tests/test_knowledge/test_redis_adapter.py -v
python -m pytest tests/test_knowledge/test_retriever.py -v
python -m pytest tests/test_knowledge/test_versioning.py -v
python -m pytest tests/test_knowledge/test_compression.py -v
```

## 许可证

MIT License

## 相关包

| 包                                         | 说明                            |
| ------------------------------------------ | ------------------------------- |
| [aga-core](../aga/)                        | 注意力治理插件 — 熵门控 KV 注入 |
| [aga-observability](../aga_observability/) | aga-core 的监控、告警和审计     |
