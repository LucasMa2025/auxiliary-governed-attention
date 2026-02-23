# aga-knowledge 用户手册

**版本**: 0.3.0  
**最后更新**: 2026-02-23

---

## 目录

1. [概述](#1-概述)
2. [安装](#2-安装)
3. [核心概念](#3-核心概念)
4. [编码器-核心对齐](#4-编码器-核心对齐)
5. [知识注册](#5-知识注册)
6. [持久化配置](#6-持久化配置)
7. [知识同步](#7-知识同步)
8. [编码器模块](#8-编码器模块)
9. [知识分片器](#9-知识分片器)
10. [知识召回器](#10-知识召回器)
11. [与 aga-core 集成](#11-与-aga-core-集成)
12. [Portal API 参考](#12-portal-api-参考)
13. [版本控制](#13-版本控制)
14. [文本压缩](#14-文本压缩)
15. [配置参考](#15-配置参考)
16. [常见问题](#16-常见问题)

---

## 1. 概述

`aga-knowledge` 是 AGA 生态系统的知识管理系统。它管理领域知识的完整生命周期 — 从注册和存储到编码和检索 — 使 `aga-core` 能够在推理过程中将相关事实注入冻结的 LLM。

### aga-knowledge 做什么

1. **存储** 明文 condition/decision 知识对
2. **同步** 知识到分布式运行时实例
3. **编码** 明文为与 aga-core 兼容的 key/value 向量（含对齐验证）
4. **检索** 通过混合搜索（HNSW + BM25 + RRF）在 aga-core 熵门控触发时检索相关知识
5. **管理** 知识生命周期（试用期 → 已确认 → 已弃用）
6. **分片** 将大文档拆分为结构感知的知识片段，自动生成条件
7. **处理** 文档中的图片信息，转换为 Portal 可访问的 URL

### aga-knowledge 不做什么

- 不执行推理或模型前向传播
- 不管理 GPU 资源
- 不替代 aga-core 的注意力注入机制
- 不需要 GPU（完全在 CPU 上运行）

---

## 2. 安装

```bash
# 最小安装
pip install aga-knowledge

# 含特定后端
pip install aga-knowledge[portal]      # FastAPI Portal
pip install aga-knowledge[postgres]    # PostgreSQL
pip install aga-knowledge[redis]       # Redis
pip install aga-knowledge[encoder]     # SentenceTransformer + PyTorch
pip install aga-knowledge[retrieval]   # hnswlib + rank-bm25 + numpy
pip install aga-knowledge[all]         # 全部
```

### 验证安装

```python
from aga_knowledge import KnowledgeManager, AGACoreAlignment, __version__
print(f"aga-knowledge v{__version__}")
```

---

## 3. 核心概念

### 知识记录

知识记录是一个 `condition/decision` 文本对：

| 字段 | 类型 | 说明 |
|------|------|------|
| `lu_id` | str | 唯一的学习单元 ID |
| `condition` | str | 触发条件（何时注入） |
| `decision` | str | 知识内容（注入什么） |
| `namespace` | str | 隔离命名空间（默认: "default"） |
| `lifecycle_state` | str | 当前生命周期状态 |
| `trust_tier` | str | 信任层级 |
| `hit_count` | int | 该知识被使用的次数 |
| `version` | int | 版本号 |

### 生命周期状态

```
probationary ──(确认)──▶ confirmed ──(弃用)──▶ deprecated
      │                      │
      └──(隔离)──▶ quarantined ◀──(隔离)──┘
```

| 状态 | 可靠性 | 说明 |
|------|--------|------|
| `probationary` | 0.3 | 新注册，观察期 |
| `confirmed` | 1.0 | 已验证，完整注入权重 |
| `deprecated` | 0.1 | 已过时，最小注入权重 |
| `quarantined` | 0.0 | 已禁用，排除在检索之外 |

### 信任层级

| 层级 | 优先级 | 说明 |
|------|--------|------|
| `system` | 100 | 核心规则，最高信任 |
| `verified` | 80 | 经人工审核确认 |
| `standard` | 50 | 默认层级 |
| `experimental` | 30 | 测试中 |
| `untrusted` | 10 | 需谨慎使用 |

### 命名空间

命名空间提供知识隔离。每个命名空间是一个独立的知识域：

```python
# 医疗知识
await manager.get_active_knowledge("medical")

# 法律知识
await manager.get_active_knowledge("legal")
```

---

## 4. 编码器-核心对齐（v0.3.0 新增）

`AGACoreAlignment` 数据类确保 aga-knowledge 的编码器生成的向量可以直接被 aga-core 使用。**不对齐会在启动时检测并抛出 `ConfigError`。**

### 创建对齐配置

```python
from aga_knowledge import AGACoreAlignment

# 方法 1: 手动配置（生产推荐）
alignment = AGACoreAlignment(
    hidden_dim=4096,
    bottleneck_dim=64,
    key_norm_target=5.0,
    value_norm_target=3.0,
)

# 方法 2: 从 aga-core YAML 配置文件
alignment = AGACoreAlignment.from_aga_config_yaml("/path/to/aga_config.yaml")

# 方法 3: 从 AGAConfig 实例（开发环境）
from aga.config import AGAConfig
aga_config = AGAConfig(bottleneck_dim=64, hidden_dim=4096)
alignment = AGACoreAlignment.from_aga_config(aga_config)
```

### 在编码器中使用对齐

```python
from aga_knowledge.encoder import EncoderConfig, create_encoder

# 创建对齐的编码器配置（推荐）
encoder_config = EncoderConfig.from_alignment(
    alignment,
    backend="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
)
# encoder_config.key_dim == 64 (来自 bottleneck_dim)
# encoder_config.value_dim == 4096 (来自 hidden_dim)

encoder = create_encoder(encoder_config)
```

### YAML 配置方式

```yaml
# 在 portal_config.yaml 中
aga_core_alignment:
    hidden_dim: 4096
    bottleneck_dim: 64
    key_norm_target: 5.0
    value_norm_target: 3.0

# 或引用 aga-core 的配置文件
aga_core_alignment:
    aga_core_config_path: "/path/to/aga_config.yaml"
```

---

## 5. 知识注册

### 通过 Portal API

启动 Portal 服务器：

```bash
aga-portal  # 或: python -m aga_knowledge.portal.app
```

通过 HTTP 注册知识：

```bash
# 单条注入
curl -X POST http://localhost:8081/knowledge/inject-text \
  -H "Content-Type: application/json" \
  -d '{
    "lu_id": "med_001",
    "condition": "患者出现胸痛和呼吸困难",
    "decision": "考虑急性冠脉综合征。开具心电图、肌钙蛋白和胸部X光检查。",
    "namespace": "medical",
    "lifecycle_state": "probationary",
    "trust_tier": "verified"
  }'

# 批量注入
curl -X POST http://localhost:8081/knowledge/batch-text \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "medical",
    "items": [
      {"lu_id": "med_001", "condition": "...", "decision": "..."},
      {"lu_id": "med_002", "condition": "...", "decision": "..."}
    ]
  }'
```

### 通过 KnowledgeManager（编程方式）

```python
from aga_knowledge.sync import SyncMessage

# 创建注入消息
msg = SyncMessage.inject(
    lu_id="rule_001",
    condition="当用户询问退货政策时",
    decision="我们的退货政策允许在30天内凭收据退货。",
    namespace="customer_service",
)

# 通过同步后端发布
await publisher.publish(msg)
```

---

## 6. 持久化配置

### 内存（测试）

```yaml
persistence:
  type: "memory"
```

### SQLite（开发）

```yaml
persistence:
  type: "sqlite"
  sqlite_path: "aga_knowledge.db"
  enable_audit: true
```

### PostgreSQL（生产）

```yaml
persistence:
  type: "postgres"
  postgres_host: "db.example.com"
  postgres_port: 5432
  postgres_database: "aga_knowledge"
  postgres_user: "aga"
  postgres_password: "secret"
  postgres_pool_size: 20
  postgres_max_overflow: 30
  enable_audit: true
```

或通过 DSN：

```yaml
persistence:
  type: "postgres"
  postgres_url: "postgresql://aga:secret@db.example.com:5432/aga_knowledge"
```

### Redis（缓存层）

```yaml
persistence:
  type: "redis"
  redis_host: "redis.example.com"
  redis_port: 6379
  redis_db: 0
  redis_password: "secret"
  redis_key_prefix: "aga_knowledge"
  redis_ttl_days: 30
  redis_pool_size: 10
  enable_audit: true
```

---

## 7. 知识同步

### 架构

```
Portal ──(发布)──▶ Redis Pub/Sub ──(订阅)──▶ KnowledgeManager (实例 1)
                                 ──(订阅)──▶ KnowledgeManager (实例 2)
                                 ──(订阅)──▶ KnowledgeManager (实例 N)
```

### 配置

```yaml
messaging:
  backend: "redis"          # redis | memory
  redis_host: "localhost"
  redis_port: 6379
  redis_channel: "aga:sync"
```

### 消息类型

| 类型 | 说明 |
|------|------|
| `INJECT` | 注册新知识 |
| `UPDATE` | 更新生命周期状态 |
| `QUARANTINE` | 禁用知识 |
| `DELETE` | 删除知识 |
| `BATCH_INJECT` | 批量注册 |
| `FULL_SYNC` | 请求全量同步 |
| `HEARTBEAT` | 实例存活检测 |
| `ACK` / `NACK` | 投递确认 |

---

## 8. 编码器模块

编码器将明文 `condition/decision` 转换为 aga-core 所需的 `key/value` 向量。

### SentenceTransformerEncoder（推荐）

```python
from aga_knowledge import AGACoreAlignment
from aga_knowledge.encoder import create_encoder, EncoderConfig

# 创建对齐的编码器（推荐）
alignment = AGACoreAlignment(hidden_dim=4096, bottleneck_dim=64)
encoder_config = EncoderConfig.from_alignment(alignment)
encoder = create_encoder(encoder_config)

# 或手动配置
encoder = create_encoder(EncoderConfig(
    backend="sentence_transformer",
    model_name="all-MiniLM-L6-v2",  # 384 维嵌入
    key_dim=64,                      # 必须匹配 AGAConfig.bottleneck_dim
    value_dim=4096,                  # 必须匹配 AGAConfig.hidden_dim
    device="cpu",
    normalize=True,
    options={
        "condition_prefix": "condition: ",
        "decision_prefix": "decision: ",
        "key_norm_target": 5.0,
        "value_norm_target": 3.0,
    },
))

# 编码单条知识
encoded = encoder.encode(
    condition="患者体温超过39度",
    decision="给予退热药物治疗",
    lu_id="med_001",
)
# encoded.key_vector: [64]
# encoded.value_vector: [4096]

# 批量编码
records = [{"condition": "...", "decision": "...", "lu_id": "..."}]
encoded_list = encoder.encode_batch(records)
```

### 编码流水线

```
condition → SentenceTransformer → [384] → key_proj(384→64) → 归一化 → 缩放(5.0) → key [64]
decision  → SentenceTransformer → [384] → value_proj(384→4096) → 归一化 → 缩放(3.0) → value [4096]
```

### 投影层训练

投影层（`key_proj`, `value_proj`）使用 Xavier 均匀初始化。为获得更好效果，可以微调：

```python
# 训练后保存投影层
encoder.save_projections("projections_v1.pt")

# 加载预训练投影层
encoder = create_encoder(EncoderConfig(
    options={"projection_path": "projections_v1.pt"}
))
```

### SimpleHashEncoder（仅测试用）

```python
encoder = create_encoder(EncoderConfig(
    backend="simple_hash",
    key_dim=64,
    value_dim=4096,
))
```

> **警告**: SimpleHashEncoder 没有语义理解能力。仅用于测试。

---

## 9. 知识分片器

分片器将大文档拆分为适合 AGA 注入的知识片段（100-500 tokens）。

### 基础策略

| 策略 | 适用场景 | 说明 |
|------|----------|------|
| `fixed_size` | 均匀文档 | 按 token 数分片 |
| `sentence` | 叙述性文本 | 在句子边界分片 |
| `semantic` | 技术文档 | 将语义相似的句子分组 |
| `sliding_window` | 通用场景 | 重叠窗口保持上下文连续性 |

### 基本使用

```python
from aga_knowledge.chunker import create_chunker, ChunkerConfig

chunker = create_chunker(ChunkerConfig(
    strategy="sliding_window",
    chunk_size=300,       # 每片段目标 token 数
    overlap=50,           # 片段间重叠
    min_chunk_size=50,
    max_chunk_size=500,
    condition_mode="first_sentence",
    language="auto",
))

# 分片文档
chunks = chunker.chunk_document(
    text=document_text,
    source_id="doc_001",
    title="医疗指南第三章",
)
```

### 文档级分片（v0.3.0 新增）

对于结构化文档（Markdown），使用 `DocumentChunker` 获得更好效果：

```python
from aga_knowledge.chunker import create_document_chunker, ChunkerConfig

doc_chunker = create_document_chunker(ChunkerConfig(
    strategy="sliding_window",
    chunk_size=300,
    overlap=50,
    condition_mode="title_context",  # 使用章节层级作为条件
))

chunks = doc_chunker.chunk_document(
    text=markdown_text,
    source_id="doc_001",
    title="医疗指南",
)

# 每个片段具有来自文档结构的增强条件
for chunk in chunks:
    print(f"条件: {chunk.condition}")
    print(f"内容: {chunk.decision[:100]}...")
    record = chunk.to_knowledge_record()
    # 通过 Portal API 或 KnowledgeManager 注册
```

### 条件生成策略

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `first_sentence` | 使用片段的第一句话 | 通用文本 |
| `title_context` | 章节标题层级作为条件 | 结构化文档 |
| `keyword` | 从文本中提取关键词 | 关键词密集内容 |
| `summary` | 基于关键词的摘要 | 长章节 |

### 文档中的图片处理（v0.3.0 新增）

处理知识文档中嵌入的图片：

```python
from aga_knowledge.chunker import ImageHandler, create_document_chunker, ChunkerConfig

# 创建图片处理器
image_handler = ImageHandler(
    asset_dir="/var/aga-knowledge/assets",
    base_url="http://portal:8081/assets",
    max_image_size_mb=10,
    supported_formats=["png", "jpg", "jpeg", "gif", "webp"],
    description_template="[图片: {alt}, 查看 {url}]",
)

# 创建带图片处理的文档分片器
doc_chunker = create_document_chunker(
    ChunkerConfig(strategy="sliding_window", chunk_size=300),
    image_handler=image_handler,
)

# 文档中的图片将被：
# 1. 提取（Base64/URL/本地路径）
# 2. 保存到 Portal 资产目录
# 3. 在 decision 文本中替换为可访问的 URL 描述
```

---

## 10. 知识召回器

`KnowledgeRetriever` 桥接 `aga-knowledge` 与 `aga-core` 的 `BaseRetriever` 协议。v0.3.0 支持 HNSW + BM25 + RRF 混合搜索。

### 设置

```python
from aga_knowledge import AGACoreAlignment
from aga_knowledge.retriever import KnowledgeRetriever

# 对齐配置是必需的
alignment = AGACoreAlignment(hidden_dim=4096, bottleneck_dim=64)

retriever = KnowledgeRetriever(
    manager=knowledge_manager,      # KnowledgeManager 实例
    encoder=encoder,                # BaseEncoder 实例
    alignment=alignment,            # AGACoreAlignment（必需）
    namespace="default",
    auto_refresh_interval=60,       # 每 60 秒刷新索引
    similarity_threshold=0.3,       # 最低余弦相似度
    # HNSW 配置
    index_backend="hnsw",           # "hnsw" 或 "brute"
    hnsw_m=16,
    hnsw_ef_construction=200,
    hnsw_ef_search=100,
    hnsw_max_elements=100000,
    # BM25 配置
    bm25_enabled=True,
    bm25_weight=0.3,                # RRF 融合中的权重
    bm25_k1=1.5,
    bm25_b=0.75,
)

# 传递给 aga-core
plugin = AGAPlugin(config, retriever=retriever)
```

### 工作原理

1. **对齐验证**: 构造时验证编码器维度是否通过 `AGACoreAlignment` 与 aga-core 匹配
2. **预热**: 从 KnowledgeManager 加载活跃知识，编码为向量，构建 HNSW + BM25 索引
3. **检索**: 当 aga-core 的熵门控触发时：
   - 接收注意力层的 `hidden_states`
   - 投影为查询向量（key_dim）
   - 通过 HNSW 进行稠密搜索（或暴力搜索回退）
   - 通过 BM25 进行稀疏搜索（如有文本查询）
   - RRF 融合组合两种结果
   - 返回 top-k 结果作为 `RetrievalResult` 对象
4. **反馈**: 注入后，aga-core 报告哪些知识被实际使用

### 优雅降级

| 条件 | 行为 |
|------|------|
| 未安装 hnswlib | 自动回退到暴力 O(N) 搜索 |
| 未安装 rank-bm25 | BM25 禁用，纯稠密检索 |
| 无文本查询 | BM25 跳过，纯稠密检索 |
| 检索错误 | Fail-Open：返回空列表 |

### 统计信息

```python
stats = retriever.get_stats()
# {
#   "type": "KnowledgeRetriever",
#   "namespace": "default",
#   "initialized": true,
#   "index_backend": "hnsw",
#   "index_size": 1500,
#   "alignment": "AGACoreAlignment(hidden_dim=4096, ...)",
#   "retrieve_count": 42,
#   "avg_retrieve_time_ms": 0.8,
#   "avg_results_per_query": 3.1,
#   "hnsw_searches": 35,
#   "bm25_searches": 30,
#   "brute_searches": 7,
#   "fused_searches": 30,
#   "feedback_used": 28,
#   "feedback_unused": 14,
#   "errors": 0,
# }
```

---

## 11. 与 aga-core 集成

### 完整流水线（v0.3.0）

```python
import asyncio
from aga_knowledge import KnowledgeManager, AGACoreAlignment
from aga_knowledge.config import PortalConfig
from aga_knowledge.encoder import create_encoder, EncoderConfig
from aga_knowledge.retriever import KnowledgeRetriever
from aga_knowledge.chunker import create_document_chunker, ChunkerConfig
from aga.plugin import AGAPlugin
from aga.config import AGAConfig

async def setup_aga_with_knowledge(model):
    # 1. 对齐配置（aga-core 与 aga-knowledge 的桥梁）
    alignment = AGACoreAlignment(
        hidden_dim=4096,
        bottleneck_dim=64,
        key_norm_target=5.0,
        value_norm_target=3.0,
    )

    # 2. 知识管理器
    config = PortalConfig.for_production(
        postgres_url="postgresql://aga:pass@db:5432/aga_knowledge",
        redis_host="redis",
    )
    manager = KnowledgeManager(config, namespaces=["medical"])
    await manager.start()

    # 3. 对齐的编码器
    encoder_config = EncoderConfig.from_alignment(
        alignment,
        options={"projection_path": "medical_projections.pt"},
    )
    encoder = create_encoder(encoder_config)

    # 4. 混合召回器
    retriever = KnowledgeRetriever(
        manager=manager,
        encoder=encoder,
        alignment=alignment,
        namespace="medical",
        auto_refresh_interval=300,
        similarity_threshold=0.4,
        index_backend="hnsw",
        bm25_enabled=True,
    )

    # 5. AGA 插件
    aga_config = AGAConfig(
        bottleneck_dim=64,
        hidden_dim=4096,
        max_slots=32,
    )
    plugin = AGAPlugin(aga_config, retriever=retriever)
    plugin.attach(model)

    return plugin, manager
```

### 维度对齐检查清单

| aga-core 参数 | aga-knowledge 参数 | 强制要求 |
|---------------|-------------------|----------|
| `AGAConfig.bottleneck_dim` | `EncoderConfig.key_dim` | 必须（不匹配抛出 ConfigError） |
| `AGAConfig.hidden_dim` | `EncoderConfig.value_dim` | 必须（不匹配抛出 ConfigError） |
| `AGAConfig.key_norm_target` | `EncoderConfig.options["key_norm_target"]` | 必须（不匹配抛出 ConfigError） |
| `AGAConfig.value_norm_target` | `EncoderConfig.options["value_norm_target"]` | 必须（不匹配抛出 ConfigError） |

---

## 12. Portal API 参考

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/knowledge/inject-text` | 注册单条知识 |
| POST | `/knowledge/batch-text` | 批量注册 |
| GET | `/knowledge/{ns}/{lu_id}` | 获取单条知识 |
| GET | `/knowledge/{ns}` | 查询知识列表 |
| DELETE | `/knowledge/{ns}/{lu_id}` | 删除知识 |
| PUT | `/lifecycle/update` | 更新生命周期状态 |
| POST | `/lifecycle/quarantine` | 隔离知识 |
| GET | `/statistics` | 全局统计 |
| GET | `/statistics/{ns}` | 命名空间统计 |
| GET | `/audit` | 审计日志 |
| GET | `/namespaces` | 命名空间列表 |
| GET | `/health` | 健康检查 |
| GET | `/health/ready` | 就绪检查 |
| GET | `/assets/{path}` | 提供图片资产（v0.3.0） |

---

## 13. 版本控制

```python
from aga_knowledge.persistence import VersionedKnowledgeStore

store = VersionedKnowledgeStore(max_versions=10)

# 保存版本
store.save_version(
    lu_id="rule_001",
    condition="更新后的条件",
    decision="更新后的决策",
    lifecycle_state="confirmed",
    trust_tier="verified",
    created_by="admin",
    change_reason="修正了药物剂量",
)

# 获取历史
history = store.get_history("rule_001")

# 回滚
old_version = store.rollback("rule_001", target_version=2)

# 比较版本
diff = store.diff("rule_001", version_a=1, version_b=3)
```

---

## 14. 文本压缩

```python
from aga_knowledge.persistence import TextCompressor, TextCompressionConfig, CompressionAlgorithm

compressor = TextCompressor(TextCompressionConfig(
    algorithm=CompressionAlgorithm.ZLIB,
    zlib_level=6,
))

# 压缩
compressed = compressor.compress_text("长文本知识内容...")
# 解压
original = compressor.decompress_text(compressed)

# 批量操作
texts = ["文本1", "文本2", "文本3"]
compressed_batch = compressor.compress_batch(texts)
```

---

## 15. 配置参考

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AGA_PORTAL_HOST` | Portal 服务器主机 | `0.0.0.0` |
| `AGA_PORTAL_PORT` | Portal 服务器端口 | `8081` |
| `AGA_PERSISTENCE_TYPE` | 存储后端 | `sqlite` |
| `AGA_PERSISTENCE_SQLITE_PATH` | SQLite 文件路径 | `aga_knowledge.db` |
| `AGA_PERSISTENCE_POSTGRES_URL` | PostgreSQL DSN | - |
| `AGA_MESSAGING_BACKEND` | 同步后端 | `redis` |
| `AGA_MESSAGING_REDIS_HOST` | Redis 主机 | `localhost` |
| `AGA_ENVIRONMENT` | 环境名称 | `development` |

### 编程配置

```python
from aga_knowledge.config import PortalConfig

# 开发环境
config = PortalConfig.for_development()

# 生产环境
config = PortalConfig.for_production(
    postgres_url="postgresql://...",
    redis_host="redis.example.com",
)

# 从 YAML 加载
from aga_knowledge.config import load_config
config = load_config("portal_config.yaml")
```

---

## 16. 常见问题

**问: ConfigError — 编码器配置与 aga-core 不对齐**
```
ConfigError: 编码器配置与 aga-core 不对齐:
  - key_dim (64) != AGAConfig.bottleneck_dim (128)
```
答: 使用 `EncoderConfig.from_alignment(alignment)` 创建编码器配置，或确保 `EncoderConfig.key_dim == AGAConfig.bottleneck_dim` 且 `EncoderConfig.value_dim == AGAConfig.hidden_dim`。

**问: KnowledgeRetriever 返回空结果**
- 检查知识是否已注册且处于活跃状态（未被隔离）
- 降低 `similarity_threshold`（调试时可设为 0.0）
- 验证编码器已初始化（`retriever.get_stats()["encoder"]["initialized"]`）
- 检查索引大小（`retriever.get_stats()["index_size"]`）
- 验证对齐配置正确（`retriever.get_stats()["alignment"]`）

**问: HNSW 未被使用**
- 确保安装了 `hnswlib`：`pip install hnswlib`
- 检查日志中是否有 "hnswlib 未安装" 警告
- 验证 `index_backend="hnsw"` 已设置

**问: BM25 结果未出现**
- 确保安装了 `rank-bm25`：`pip install rank-bm25`
- 确保 `bm25_enabled=True` 已设置
- BM25 需要在 `query.metadata["query_text"]` 中提供文本查询 — 如果仅有向量查询，BM25 会自动跳过

**问: Redis 同步不工作**
- 验证 Redis 正在运行且可访问
- 检查 `messaging.redis_channel` 在 Portal 和 Runtime 之间是否一致
- 确保 `messaging.backend` 设置为 `"redis"`（而非 `"memory"`）

**问: PostgreSQL 连接池耗尽**
- 增加 `postgres_pool_size` 和 `postgres_max_overflow`
- 检查连接泄漏（确保调用了 `await adapter.disconnect()`）

**问: 分片器产生过多/过少的片段**
- 调整 `chunk_size`（每片段目标 token 数）
- 对于中文文本，设置 `language: "zh"` 以获得准确的 token 估算
- 使用 `sliding_window` 策略配合 `overlap` 保持上下文连续性
- 对于结构化文档，使用 `DocumentChunker` 配合 `condition_mode="title_context"`