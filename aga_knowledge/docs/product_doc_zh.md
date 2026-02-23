# aga-knowledge 产品说明书

**版本**: 0.3.0  
**最后更新**: 2026-02-23  
**状态**: Beta — 核心功能完成，编码器-Core 对齐已强制执行，混合检索（HNSW+BM25+RRF）已实现

---

## 1. 产品定位

### 1.1 使命

`aga-knowledge` 解决一个核心问题：**冻结的 LLM 缺乏领域特定事实，而 aga-core 需要一个可靠的知识来源来注入它们。**

在 AGA 生态系统中：

-   **aga-core** 提供注入机制（熵门控 KV 注入到注意力层）
-   **aga-knowledge** 提供知识流水线（注册 → 存储 → 编码 → 检索）
-   **aga-observability** 提供运维可见性

### 1.2 设计哲学

1. **明文优先**: 知识以人类可读的 `condition/decision` 文本对存储。向量化发生在边界处（编码器模块），而非存储层。
2. **配置驱动**: 所有组件（持久化、编码器、分片器、召回器）通过 YAML/字典选择和配置，而非硬编码。
3. **协议化集成**: `KnowledgeRetriever` 实现 `aga-core` 的 `BaseRetriever` 协议。拥有现有基础设施（Chroma, Milvus, Elasticsearch）的用户可以实现自己的召回器，完全不需要使用 `aga-knowledge`。
4. **编码器-Core 对齐**: `AGACoreAlignment` 数据类显式强制 aga-knowledge 编码器与 aga-core 注意力层之间的维度和范数一致性。不对齐的配置在启动时以 `ConfigError` 拒绝。
5. **Fail-Open 安全**: 每次检索失败都返回空列表。知识管理问题永远不会导致推理崩溃。
6. **纯 CPU 运行**: 整个知识流水线在 CPU 上运行，为推理保留 GPU 显存。

### 1.3 目标用户

| 用户类型      | 使用场景                                              |
| ------------- | ----------------------------------------------------- |
| **研究人员**  | 使用 SQLite + SimpleHashEncoder 快速原型验证          |
| **ML 工程师** | 使用 PostgreSQL + SentenceTransformerEncoder 生产部署 |
| **平台团队**  | 使用 BaseRetriever 协议实现自定义召回器               |
| **领域专家**  | 通过 Portal REST API 注册知识（无需编码）             |

---

## 2. 系统架构

### 2.1 组件概览

```
                    ┌──────────────────────────────────────────┐
                    │         Domain Expert / Admin            │
                    └──────────────────┬───────────────────────┘
                                       │ REST API
                    ┌──────────────────▼───────────────────────┐
                    │          Portal (FastAPI)                │
                    │  inject-text / batch-text / lifecycle    │
                    │  + /assets (静态图片服务)                │
                    └──────────────────┬───────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────┐
                    │         Persistence Layer                │
                    │  Memory | SQLite | PostgreSQL | Redis    │
                    │  + Versioning + Text Compression         │
                    └──────────────────┬───────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────┐
                    │      Sync Layer (Redis Pub/Sub)          │
                    │  INJECT / UPDATE / QUARANTINE / DELETE   │
                    └──────────┬───────────────┬───────────────┘
                               │               │
              ┌────────────────▼───┐   ┌───────▼────────────────┐
              │  KnowledgeManager  │   │  KnowledgeManager      │
              │  (实例 1)          │   │  (实例 2)              │
              │  本地缓存          │   │  本地缓存              │
              └────────┬───────────┘   └───────┬────────────────┘
                       │                       │
              ┌────────▼───────────┐   ┌───────▼────────────────┐
              │  编码器             │   │  编码器               │
              │  (文本 → 向量)     │   │  (文本 → 向量)        │
              │  AGACoreAlignment  │   │  AGACoreAlignment      │
              └────────┬───────────┘   └───────┬────────────────┘
                       │                       │
              ┌────────▼───────────┐   ┌───────▼────────────────┐
              │ KnowledgeRetriever │   │ KnowledgeRetriever     │
              │ HNSW + BM25 + RRF │   │ HNSW + BM25 + RRF     │
              │ (BaseRetriever)    │   │ (BaseRetriever)        │
              └────────┬───────────┘   └───────┬────────────────┘
                       │                       │
              ┌────────▼───────────┐   ┌───────▼────────────────┐
              │  aga-core 插件     │   │  aga-core 插件         │
              │  (GPU 实例 1)      │   │  (GPU 实例 2)          │
              └────────────────────┘   └────────────────────────┘
```

### 2.2 数据流

```
1. 注册:   专家 → Portal API → 持久化 → 同步 → 所有 KnowledgeManager
2. 编码:   KnowledgeManager → 编码器 (含 AGACoreAlignment) → key/value 向量 → HNSW + BM25 索引
3. 检索:   aga-core 高熵 → hidden_states → KnowledgeRetriever → 混合检索 (HNSW+BM25+RRF) → top-k 结果
4. 注入:   RetrievalResult → aga-core KVStore → BottleneckInjector → 注意力层
5. 反馈:   aga-core → on_injection_feedback() → 命中计数更新
```

### 2.3 检索架构 (v0.3.0 新增)

```
    RetrievalQuery
         │
    ┌────▼────┐
    │ 查询路由 │
    └────┬────┘
         │
    ┌────┼────┐
    │         │
  稠密检索   稀疏检索
  (HNSW)    (BM25)
    │         │
    └────┬────┘
         │
    ┌────▼────┐
    │ RRF 融合 │
    └────┬────┘
         │
    List[RetrievalResult]
```

---

## 3. 功能完备性分析

### 3.1 功能覆盖

| 能力                | aga-core 是否需要 | 是否已实现                            | 完备度         |
| ------------------- | ----------------- | ------------------------------------- | -------------- |
| 知识 CRUD           | 是                | 是                                    | 完整           |
| 批量操作            | 是                | 是                                    | 完整           |
| 命名空间隔离        | 是                | 是                                    | 完整           |
| 生命周期管理        | 是                | 是                                    | 完整           |
| 多后端持久化        | 是                | 是（4 种后端）                        | 完整           |
| 跨实例同步          | 是                | 是（Redis Pub/Sub）                   | 完整           |
| 文本 → 向量编码     | 是                | 是（2 种编码器 + AGACoreAlignment）   | 生产对齐       |
| 语义检索            | 是                | 是（HNSW + BM25 + RRF 混合检索）     | 生产级         |
| BaseRetriever 协议  | 是                | 是                                    | 完整           |
| 文档分片            | 是                | 是（5 种策略 + DocumentChunker）      | 完整           |
| Condition 生成      | 是                | 是（4 种模式）                        | 完整           |
| 图片处理            | 可选              | 是（Base64/URL/本地 → Portal 资产）   | 完整           |
| 编码器-Core 对齐    | 是                | 是（AGACoreAlignment + 验证）         | 完整           |
| 版本控制            | 可选              | 是                                    | 完整           |
| 文本压缩            | 可选              | 是                                    | 完整           |
| 审计日志            | 可选              | 是                                    | 完整           |

### 3.2 差距分析 (v0.3.0)

| 差距           | 严重度 | 影响                                     | 缓解措施                                |
| -------------- | ------ | ---------------------------------------- | --------------------------------------- |
| 投影层未训练   | 中     | 编码质量依赖随机初始化                   | 用户可微调并通过 `projection_path` 加载 |
| 无分布式编码器 | 低     | 每个实例加载自己的编码器模型             | 当前规模可接受                          |
| 同步-异步桥接  | 低     | `_refresh_index_sync` 使用 `asyncio.run` | 可工作但不够优雅                        |

> **注意**: v0.2.0 中的暴力检索和缺乏 BM25 的问题已在 v0.3.0 中通过 HNSW 和 BM25 混合检索解决。

### 3.3 结论

**aga-knowledge v0.3.0 作为 aga-core 的知识管理系统已达到生产对齐。** 完整的流水线 — 注册、编码、对齐验证、混合检索和注入 — 端到端可工作。检索系统通过 HNSW ANN 索引支持 10K+ 知识条目。主要改进方向是编码器质量（投影层训练）。

---

## 4. 编码器系统 — 设计、实现、不足与改进方向

### 4.1 当前架构

```
                    ┌──────────────────────────────┐
                    │        BaseEncoder           │
                    │  (Abstract Protocol)         │
                    │                              │
                    │  encode(condition, decision) │
                    │  encode_batch(records)       │
                    │  warmup() / shutdown()       │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌───────▼─────┐
    │ SentenceTransf │  │ SimpleHash │  │  (Future)   │
    │ ormerEncoder   │  │ Encoder    │  │ HuggingFace │
    │                │  │            │  │ OpenAI API  │
    │ ST model       │  │ SHA-256    │  │ Custom      │
    │ + key_proj     │  │ hash→vec   │  │             │
    │ + value_proj   │  │            │  │             │
    └────────────────┘  └────────────┘  └─────────────┘

    ┌──────────────────────────────────────────────────┐
    │        AGACoreAlignment (v0.3 新增)              │
    │                                                  │
    │  EncoderConfig.from_alignment(alignment)         │
    │  EncoderConfig.validate_alignment(alignment)     │
    │  → 强制 key_dim = bottleneck_dim                │
    │  → 强制 value_dim = hidden_dim                  │
    │  → 强制 key/value 范数目标                      │
    └──────────────────────────────────────────────────┘
```

### 4.2 编码器-Core 对齐 (v0.3.0 新增)

`AGACoreAlignment` 数据类提供显式配置桥接:

| aga-core 参数                  | aga-knowledge 参数                        | 强制执行 |
|-------------------------------|------------------------------------------|----------|
| `AGAConfig.bottleneck_dim`     | `EncoderConfig.key_dim`                   | 是（强制） |
| `AGAConfig.hidden_dim`         | `EncoderConfig.value_dim`                 | 是（强制） |
| `AGAConfig.key_norm_target`    | `EncoderConfig.options["key_norm_target"]` | 是（强制） |
| `AGAConfig.value_norm_target`  | `EncoderConfig.options["value_norm_target"]` | 是（强制） |
| `AGAConfig.num_heads`          | `AGACoreAlignment.num_heads`              | 信息性 |
| `AGAConfig.value_bottleneck_dim` | `AGACoreAlignment.value_bottleneck_dim` | 信息性 |

三种创建对齐配置的方式:
1. **手动配置**（生产环境推荐）: `AGACoreAlignment(hidden_dim=4096, bottleneck_dim=64, ...)`
2. **从 YAML 加载**: `AGACoreAlignment.from_aga_config_yaml("aga_config.yaml")`
3. **从 AGAConfig 实例**（开发环境）: `AGACoreAlignment.from_aga_config(config)`

不对齐在 `KnowledgeRetriever` 构造时检测并抛出 `ConfigError`。

### 4.3 SentenceTransformerEncoder — 详细分析

#### 编码流水线

```
输入:
  condition = "患者体温超过39度"
  decision  = "给予退热药物治疗并持续监测"

步骤 1: 文本前缀
  condition → "condition: 患者体温超过39度"
  decision  → "decision: 给予退热药物治疗并持续监测"

步骤 2: SentenceTransformer 编码
  condition → ST("condition: ...") → cond_embedding [384]
  decision  → ST("decision: ...")  → dec_embedding  [384]

步骤 3: 投影
  cond_embedding → key_proj(384 → 64)   → key_raw   [64]
  dec_embedding  → value_proj(384 → 4096) → value_raw [4096]

步骤 4: 归一化 + 缩放
  key_raw   → L2_归一化 → 缩放(5.0)  → key   [64]
  value_raw → L2_归一化 → 缩放(3.0)  → value [4096]

输出:
  key:   [64]   — 用于 aga-core 的注意力匹配 (bottleneck_dim)
  value: [4096] — 用于 aga-core 的知识注入 (hidden_dim)
```

#### 优势

1. **语义感知**: SentenceTransformer 提供高质量的句子级嵌入，捕获语义含义。
2. **维度灵活**: 投影层可以将任意嵌入维度映射到 aga-core 所需的维度。
3. **CPU 友好**: 在 CPU 上运行，为推理保留 GPU 显存。
4. **确定性**: 相同输入始终产生相同输出。
5. **编码缓存**: FIFO 缓存避免重复知识的冗余编码。
6. **投影层持久化**: 训练后的投影层可以保存/加载。
7. **对齐验证** (v0.3.0 新增): `EncoderConfig.from_alignment()` 和 `validate_alignment()` 确保与 aga-core 的维度和范数一致性。

#### 不足与改进方向

##### L1: 投影层未训练（严重度: 高）

投影层使用 Xavier 均匀随机权重初始化。检索质量依赖后续微调。

改进方向: 对比学习微调、值重建损失、从 aga-core 蒸馏、少样本校准。

##### L2: condition/decision 独立编码（严重度: 中）

相同 condition 不同 decision 的知识单元将有相同的 key 向量。

改进方向: 联合编码、交叉注意力编码、Key-Value 联合去重。

##### L3: 固定范数缩放（严重度: 低）

范数目标（5.0/3.0）是启发式的。

改进方向: 自适应范数校准。

### 4.4 编码器改进路线图

| 优先级 | 改进                             | 工作量 | 影响                       |
| ------ | -------------------------------- | ------ | -------------------------- |
| P0     | 投影层对比学习微调脚本           | 中     | 高 — 直接提升检索质量      |
| P1     | HuggingFace 编码器后端           | 低     | 中 — 更广泛的模型选择      |
| P1     | 联合 condition+decision 编码选项 | 中     | 中 — 更好的 key-value 对齐 |
| P2     | OpenAI/API 编码器后端            | 低     | 中 — 云端部署支持          |
| P2     | 自适应范数校准                   | 中     | 低-中 — 更好的注入权重     |
| P3     | 交叉注意力编码                   | 高     | 中 — 最优但复杂            |

---

## 5. 知识检索系统 — 设计、实现与分析

### 5.1 当前架构 (v0.3.0 — 混合检索)

```
                    ┌──────────────────────────────────────────────┐
                    │      KnowledgeRetriever (v0.3.0)             │
                    │  (implements BaseRetriever)                  │
                    │                                              │
                    │  ┌─────────────────────────────────────────┐ │
                    │  │  AGACoreAlignment 对齐验证              │ │
                    │  │  (构造时强制执行)                       │ │
                    │  └─────────────────────────────────────────┘ │
                    │                                              │
                    │  ┌──────────────┐  ┌───────────────────────┐ │
                    │  │  HNSW 索引   │  │  BM25 索引            │ │
                    │  │  (hnswlib)   │  │  (rank-bm25)          │ │
                    │  │  稠密 ANN    │  │  稀疏关键词            │ │
                    │  └──────┬───────┘  └─────────┬─────────────┘ │
                    │         └──────┬──────────────┘               │
                    │                │                              │
                    │         ┌──────▼──────┐                      │
                    │         │ RRF 融合    │                      │
                    │         └──────┬──────┘                      │
                    │                │                              │
                    │  ┌─────────────▼─────────────────┐          │
                    │  │  暴力搜索回退                  │          │
                    │  │  (_key_matrix 余弦, O(N))      │          │
                    │  └───────────────────────────────┘          │
                    └──────────────────────────────────────────────┘
```

### 5.2 稠密检索 — HNSW

**为什么选择 HNSW 而非 FAISS IVF+PQ:**
- HNSW 原生支持增量插入/删除（AGA 知识是动态的）
- HNSW 对碎片化语义不敏感（AGA 知识 100-500 tokens 片段）
- HNSW 在 1K-1M 规模下召回率 >95%
- FAISS IVF+PQ 需要重训练质心，不适合动态数据

**性能:**

| 知识数量 | 预期延迟 (CPU) | 可接受？ |
|----------|---------------|----------|
| 1,000 | ~0.2ms | 是 |
| 10,000 | ~0.5ms | 是 |
| 50,000 | ~0.8ms | 是 |
| 100,000 | ~1.0ms | 是 |

**配置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hnsw_m` | 16 | 每层连接数（召回率 vs 内存） |
| `hnsw_ef_construction` | 200 | 构建时搜索宽度 |
| `hnsw_ef_search` | 100 | 查询时搜索宽度 |
| `hnsw_max_elements` | 100,000 | 最大索引容量 |

### 5.3 稀疏检索 — BM25

与稠密检索互补，提供关键词级精确匹配:

- 编码器投影层未充分训练时，BM25 关键词匹配是必要补充
- 无文本查询（仅向量查询）时，BM25 自动跳过
- 同时索引 `condition` 和 `decision` 文本

**配置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bm25_enabled` | false | 是否启用 BM25 |
| `bm25_weight` | 0.3 | RRF 融合中的权重 |
| `bm25_k1` | 1.5 | 词频饱和参数 |
| `bm25_b` | 0.75 | 文档长度归一化参数 |

### 5.4 融合 — 倒数排名融合 (RRF)

```
RRF_score(d) = Σ weight_i / (k + rank_i(d))
```

- `k = 60`（防止排名靠前的结果权重过大）
- `dense_weight = 0.7`, `sparse_weight = 0.3`（可配置）
- 备选方案: `weighted_score_fusion` 用于分数尺度一致的系统

### 5.5 优雅降级

| 条件 | 行为 |
|------|------|
| hnswlib 未安装 | 自动回退到暴力搜索 O(N) |
| rank-bm25 未安装 | BM25 禁用，纯稠密检索 |
| 无文本查询 | BM25 跳过，纯稠密检索 |
| 检索错误 | Fail-Open: 返回空列表 |
| 空索引 | 立即返回空列表 |

### 5.6 检索改进路线图

| 优先级 | 改进                               | 工作量 | 影响                   |
| ------ | ---------------------------------- | ------ | ---------------------- |
| P1     | 增量张量更新                       | 低     | 中 — 降低更新延迟      |
| P2     | 学习 hidden_states 投影            | 中     | 低-中 — 更好的回退质量 |
| P2     | 双缓冲索引                         | 中     | 低 — 更平滑的更新      |
| P3     | 实体感知检索                       | 高     | 中 — 领域特定改进      |

---

## 6. 知识分片系统

### 6.1 设计原理

AGA 以 100-500 token 的片段注入知识。大文档必须被拆分为满足以下条件的片段：

1. 语义连贯（不在句子中间切断）
2. 有意义的 `condition` 字段（用于检索匹配）
3. 跨片段边界保留上下文（重叠）

### 6.2 策略对比

| 策略          | 连贯性 | 上下文保留     | 速度             | 适用场景         |
| ------------- | ------ | -------------- | ---------------- | ---------------- |
| FixedSize     | 低     | 无             | 快               | 均匀、结构化数据 |
| Sentence      | 中     | 无             | 快               | 叙述性文本       |
| Semantic      | 高     | 隐式           | 慢（需要编码器） | 技术文档         |
| SlidingWindow | 中     | 显式（重叠）   | 快               | 通用场景         |
| Document      | 高     | 结构感知       | 快               | Markdown/HTML 文档 |

### 6.3 文档级分片 (v0.3.0 新增)

`DocumentChunker` 提供结构感知分片:

1. **Markdown 标题解析**: 识别 `#`, `##`, `###` 等标题，保留章节层级
2. **章节上下文继承**: 子章节继承父章节标题用于 condition 生成
3. **Condition 生成器**: 4 种策略生成高质量 condition 文本:
   - `first_sentence` — 使用片段首句
   - `summary` — 基于关键词的摘要
   - `title_context` — 章节标题层级作为 condition
   - `keyword` — 从片段文本提取的 Top 关键词
4. **图片处理**: 通过 `ImageHandler` 提取和处理文档中的图片

### 6.4 图片处理 (v0.3.0 新增)

文档中可能包含图片（Markdown 语法）。`ImageHandler` 处理方式:

| 来源类型 | 处理方式 | 输出 |
|----------|---------|------|
| Base64 嵌入 | 解码 → 保存到 Portal 静态资源目录 | Portal URL |
| 外部 URL | 保留原始 URL | 原始 URL |
| 本地文件路径 | 复制到 Portal 静态资源目录 | Portal URL |

图片引用在 `decision` 文本中被替换为可访问的 URL 和文本描述（可配置模板），确保编码器可以处理上下文。

### 6.5 推荐

大多数使用场景推荐 **SlidingWindow（overlap=50）**。对于结构化文档（Markdown/wiki），使用 **DocumentChunker** 配合 `condition_mode="title_context"` 以获得最佳上下文保留和检索匹配效果。

---

## 7. 持久化系统

### 7.1 后端对比

| 后端       | 延迟    | 持久性    | 可扩展性 | 适用场景 |
| ---------- | ------- | --------- | -------- | -------- |
| Memory     | ~0.01ms | 无        | 单进程   | 测试     |
| SQLite     | ~1ms    | 文件级    | 单实例   | 开发     |
| PostgreSQL | ~2-5ms  | 完整 ACID | 多实例   | 生产     |
| Redis      | ~0.5ms  | 可配置    | 多实例   | 热缓存   |

### 7.2 生产推荐

**PostgreSQL 作为主存储 + Redis 作为缓存层**。

### 7.3 数据库 Schema (v0.3.0)

超越原始 AGA 设计的新表:

| 表 | 用途 |
|----|------|
| `namespaces` | 租户/领域隔离 |
| `knowledge` | 明文 condition/decision 对（**不含向量字段**） |
| `knowledge_versions` | 完整版本历史（含自动触发器） |
| `document_sources` | 文档来源追踪 |
| `image_assets` | 图片元数据和 Portal URL |
| `encoder_versions` | 编码器投影层版本追踪 |
| `audit_log` | 所有 CRUD 操作审计记录 |

---

## 8. 安全考虑

| 方面     | 状态   | 备注                  |
| -------- | ------ | --------------------- |
| 认证     | 未实现 | Portal API 开放访问   |
| 授权     | 未实现 | 无基于角色的访问控制  |
| 静态加密 | 未实现 | 依赖数据库加密        |
| 输入验证 | 部分   | Pydantic 模型验证结构 |
| SQL 注入 | 已防护 | 使用参数化查询        |
| 审计日志 | 已实现 | 所有 CRUD 操作已记录  |

---

## 9. 性能特征

### 9.1 编码性能

| 编码器                    | 单条编码 | 批量（100 条） | 内存占用       |
| ------------------------- | -------- | -------------- | -------------- |
| SentenceTransformer (CPU) | ~15ms    | ~200ms         | ~500MB（模型） |
| SimpleHash                | ~0.01ms  | ~1ms           | ~0             |

### 9.2 检索性能

| 索引大小 | 暴力搜索 (CPU) | HNSW (CPU) | BM25 (CPU) |
| -------- | -------------- | ---------- | ---------- |
| 100      | < 0.1ms        | < 0.1ms    | < 0.1ms    |
| 1,000    | ~0.5ms         | ~0.2ms     | ~0.3ms     |
| 10,000   | ~5ms           | ~0.5ms     | ~1ms       |
| 100,000  | ~50ms          | ~1ms       | ~3ms       |

---

## 10. 与替代方案的对比

### 10.1 vs. 直接使用 aga-core register()

| 方面           | 直接 register() | aga-knowledge                |
| -------------- | --------------- | ---------------------------- |
| 设置复杂度     | 低              | 中                           |
| 知识持久化     | 无（仅内存）    | 完整（4 种后端）             |
| 多实例同步     | 手动            | 自动（Redis Pub/Sub）        |
| 编码器-Core 对齐 | 手动           | 强制执行（AGACoreAlignment） |
| 混合检索       | 无              | HNSW + BM25 + RRF            |
| 审计追踪       | 无              | 完整审计日志                 |
| 适用于         | 研究、原型验证  | 生产部署                     |

### 10.2 vs. RAG 系统（LangChain, LlamaIndex）

| 方面           | RAG 系统         | aga-knowledge                |
| -------------- | ---------------- | ---------------------------- |
| 注入方式       | 提示词拼接       | 注意力层 KV 注入             |
| 上下文窗口影响 | 消耗 token       | 零 token 开销                |
| 集成深度       | 表面级（提示词） | 深层（注意力机制）           |
| 检索           | 成熟（多种后端） | 混合（HNSW + BM25 + RRF）   |

---

## 11. 总结

`aga-knowledge` v0.3.0 为 `aga-core` 提供了**生产对齐**的知识管理流水线。核心流水线 — 注册、持久化、同步、编码、对齐验证和混合检索 — 端到端可工作，650+ 单元测试通过。

### 核心优势

1. 关注点清晰分离（明文存储 vs. 向量编码）
2. 协议化集成（BaseRetriever）
3. **强制编码器-Core 对齐**（AGACoreAlignment）— v0.3.0 新增
4. **混合检索**（HNSW + BM25 + RRF）— v0.3.0 新增
5. **文档级分片**（含 condition 生成和图片处理）— v0.3.0 新增
6. 配置驱动架构
7. 全面的持久化选项

### 主要改进方向

1. **编码器质量**: 投影层训练是影响最大的单一改进
2. **安全性**: Portal API 的认证和授权
3. **分布式编码器**: gRPC 编码器服务
