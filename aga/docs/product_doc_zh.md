# AGA-Core 产品文档

> **版本**: 4.4.0  
> **最后更新**: 2026-02-23  
> **适用范围**: aga-core 核心包

---

## 1. 产品概述

### 1.1 什么是 AGA

AGA (Auxiliary Governed Attention) 是一个面向冻结大语言模型 (LLM) 的**推理时注意力治理插件**。其核心目标是**为冻结模型提供无损的能力扩展** — 通过在 Transformer 的注意力层中动态注入外部知识，使冻结模型在推理过程中能够访问其参数中未编码的知识。

### 1.2 核心理念

AGA 的设计基于一个关键观察：**LLM 在遇到其参数中未编码的知识时，会在隐藏状态中表现出高熵（高不确定性）**。AGA 利用这一信号，仅在模型"不确定"时才注入外部知识，避免干扰模型已有的正确推理路径。

**设计原则：**

-   **效果优先于性能** — 正确的知识注入是首要目标，延迟是次要的
-   **Fail-Open 安全** — 任何异常都回退到原始模型输出，AGA 永远不会降低推理质量
-   **配置驱动** — 所有参数外置，调优无需修改代码
-   **标准协议** — 召回器接口、适配器接口、事件总线 — 全部可插拔

### 1.3 与现有技术的区别

| 技术                   | 介入时机   | 修改模型 | 知识粒度                    | 动态性       | 决策依据           |
| ---------------------- | ---------- | -------- | --------------------------- | ------------ | ------------------ |
| **RAG**                | 推理前     | 否       | 文档/段落                   | 静态检索     | 查询相似度         |
| **LoRA**               | 训练时     | 是       | 全局                        | 需重训       | 无                 |
| **Prompt Engineering** | 推理前     | 否       | 受限于上下文窗口            | 手动         | 无                 |
| **AGA**                | **推理中** | **否**   | **原子事实 (10-50 tokens)** | **实时增删** | **模型内部熵信号** |

### 1.4 适用场景

-   **垂直领域知识系统**: 医疗诊断辅助、法律条文查询、金融风控规则
-   **动态知识更新**: 新闻事件、政策法规、产品信息的实时更新
-   **多租户知识隔离**: SaaS 场景下不同客户的独立知识空间
-   **模型知识补丁**: 快速修复模型的事实性错误，无需重新训练
-   **流式生成场景**: 在 token-by-token 生成过程中持续注入知识
-   **研究实验**: 探索推理时知识注入的效果和机制

---

## 2. 技术架构

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                          AGAPlugin                               │
│                       (唯一入口类)                                │
│                                                                  │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────────┐  │
│  │   KVStore   │  │ EntropyGate    │  │ BottleneckInjector    │  │
│  │  GPU常驻存储 │  │  System        │  │  核心注入路径          │  │
│  │  预分配内存  │  │  三段式门控     │  │  Top-K路由            │  │
│  │  LRU淘汰    │  │  熵否决机制     │  │  Value投影             │  │
│  │  Pin/Unpin  │  │  Early Exit    │  │  可靠性偏置            │  │
│  │  命名空间    │  │                │  │                       │  │
│  └─────────────┘  └────────────────┘  └───────────────────────┘  │
│                                                                  │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────────┐  │
│  │ Persistence │  │   Adapter      │  │ Instrumentation       │  │
│  │   Decay     │  │  HuggingFace   │  │  EventBus             │  │
│  │  跨层衰减    │  │  vLLM          │  │  ForwardMetrics       │  │
│  │  硬重置      │  │  自定义        │  │  AuditLog             │  │
│  │  线程隔离    │  │  Hook注入      │  │                       │  │
│  └─────────────┘  └────────────────┘  └───────────────────────┘  │
│                                                                  │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────────┐  │
│  │  Retriever  │  │ StreamingSess  │  │ Distributed (TP)      │  │
│  │  标准协议    │  │  会话管理      │  │  TPManager            │  │
│  │  BaseRetr.  │  │  逐token诊断   │  │  参数广播              │  │
│  │  Slot治理    │  │  热更新        │  │  KV同步               │  │
│  └─────────────┘  └────────────────┘  └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 推理流程

AGA 在每个挂载的 Transformer 层的 `self_attn` 输出后执行以下流程：

1. **Gate-0 先验检查**: 基于 namespace 的零成本过滤
2. **Gate-1 熵计算 + 置信门控**:
    - 计算 hidden_states 的内部方差作为不确定性信号
    - 通过学习的投影网络增强不确定性估计
    - 应用三段式熵否决：
        - `H < τ_low`: 模型自信 → gate = 0（不注入）
        - `τ_low ≤ H ≤ τ_high`: 正常区间 → gate = σ(w₁·H + b)
        - `H > τ_high`: 模型极度不确定 → gate ≤ max_gate（限制注入）
3. **Early Exit**: 如果 gate 均值低于阈值，直接返回原始输出
4. **召回器调用**（如已配置）: 高熵时通过 `BaseRetriever` 协议查询外部知识源，受 Slot 治理约束（冷却期、预算、去重、稳定性）
5. **Bottleneck 注入**:
    - Query 投影: `hidden_states → [batch, seq, bottleneck_dim]`
    - Top-K 路由: 选择最相关的 K 个知识槽位
    - 注意力计算: `softmax(query @ keys.T / √d + log(reliability)) @ values`
    - Value 投影: 通过 delta subspace 控制注入幅度
6. **跨层衰减**: 防止辅助注意力跨层累积（线程隔离）
7. **门控融合**: `output = primary_output + gate × aux_output`

### 2.3 核心公式

**熵门控 (Eq. 2)**:

```
α = σ(w₁ · H + b)
```

其中 H 是基于 hidden_states 方差计算的不确定性信号。

**三段式否决 (Eq. 3)**:

```
α_effective =
  0,                    if H < τ_low
  α,                    if τ_low ≤ H ≤ τ_high
  min(α, max_gate),     if H > τ_high
```

**持久化衰减 (Eq. 5)**:

```
α^{ℓ+1}_effective = γ · α^ℓ_effective
```

**Bottleneck 注入**:

```
query = W_q · hidden_states                    # [batch, seq, d_bottleneck]
scores = query · keys^T / √d_bottleneck       # [batch, seq, K]
weights = softmax(scores + log(reliability))   # [batch, seq, K]
aux = weights · values                         # [batch, seq, d_hidden]
aux = W_up(GELU(W_down(aux)))                  # delta subspace
output = primary + gate · aux                  # 门控融合
```

---

## 3. 模块详解

### 3.1 AGAPlugin (plugin.py)

**职责**: 唯一入口类，封装所有功能。

**公共 API**:

| 方法                       | 签名                                                                    | 说明                   |
| -------------------------- | ----------------------------------------------------------------------- | ---------------------- |
| `__init__`                 | `(config: AGAConfig = None, retriever: BaseRetriever = None, **kwargs)` | 创建插件实例           |
| `from_config`              | `(config_source: str/Path/Dict) → AGAPlugin`                            | 从配置创建（类方法）   |
| `register`                 | `(id, key, value, reliability=1.0, metadata=None, pinned=None) → bool`  | 注册单条知识           |
| `register_batch`           | `(entries: List[Dict]) → int`                                           | 批量注册知识           |
| `unregister`               | `(id: str) → bool`                                                      | 移除知识               |
| `load_from`                | `(source: str, **kwargs) → int`                                         | 从外部加载知识         |
| `clear`                    | `(namespace: str = None)`                                               | 清空知识               |
| `attach`                   | `(model, layer_indices=None, adapter=None)`                             | 挂载到模型             |
| `detach`                   | `()`                                                                    | 从模型卸载             |
| `create_streaming_session` | `(**kwargs) → StreamingSession`                                         | 创建流式生成会话       |
| `get_diagnostics`          | `() → Dict`                                                             | 获取诊断信息           |
| `get_audit_trail`          | `(limit=100, operation=None) → List[Dict]`                              | 获取审计日志           |
| `get_store_stats`          | `() → Dict`                                                             | 获取存储统计           |
| `reset_decay_contexts`     | `()`                                                                    | 重置线程本地衰减上下文 |

**属性**:

| 属性              | 类型   | 说明         |
| ----------------- | ------ | ------------ |
| `knowledge_count` | `int`  | 当前知识数量 |
| `is_attached`     | `bool` | 是否已挂载   |

**v4.4 变更**:

-   `register()` 新增 `pinned` 参数（默认值: `config.pin_registered`）
-   `__init__` 新增 `retriever` 参数，支持外部知识检索
-   `get_diagnostics()` 新增 `pinned_count`、`unpinned_count`、`retriever_stats`、`slot_governance`
-   `_decay_contexts` 改为 `threading.local()` 实现线程隔离
-   forward 路径集成召回器，配合完整 Slot 治理

### 3.2 AGAConfig (config.py)

**职责**: 统一配置管理，所有参数外置支持运行时调节。

**配置分组**:

| 分组          | 参数                         | 默认值        | 说明                              |
| ------------- | ---------------------------- | ------------- | --------------------------------- |
| **模型维度**  | `hidden_dim`                 | 4096          | 必须匹配目标模型                  |
|               | `bottleneck_dim`             | 64            | 检索键维度                        |
|               | `num_heads`                  | 32            | 注意力头数                        |
|               | `value_bottleneck_dim`       | 256           | Value 投影瓶颈维度                |
| **容量**      | `max_slots`                  | 256           | 热知识槽位上限                    |
| **设备**      | `device`                     | "cuda"        | 计算设备                          |
| **熵门控**    | `tau_low`                    | 0.5           | 低熵阈值                          |
|               | `tau_high`                   | 2.0           | 高熵阈值                          |
|               | `max_gate`                   | 0.8           | 最大门控值                        |
|               | `gate2_top_k`                | 8             | Top-K 路由数量                    |
|               | `early_exit_threshold`       | 0.05          | Early Exit 阈值                   |
| **衰减**      | `decay_enabled`              | true          | 衰减开关                          |
|               | `decay_strategy`             | "exponential" | 衰减策略                          |
|               | `decay_gamma`                | 0.9           | 衰减系数                          |
| **安全**      | `fail_open`                  | true          | Fail-Open 开关                    |
| **范数**      | `enable_norm_clipping`       | true          | 范数裁剪开关                      |
|               | `key_norm_target`            | 5.0           | Key 目标范数                      |
|               | `value_norm_target`          | 3.0           | Value 目标范数                    |
| **召回器**    | `retriever_backend`          | "null"        | 后端: null/kv_store/chroma/custom |
|               | `retriever_endpoint`         | ""            | 外部召回器连接地址                |
|               | `retriever_collection`       | ""            | 知识集合名称                      |
|               | `retriever_top_k`            | 5             | 每次召回最大结果数                |
|               | `retriever_min_score`        | 0.3           | 最小相关性阈值                    |
|               | `retriever_query_source`     | "q_proj"      | 查询来源: hidden_states/q_proj    |
|               | `retriever_auto_inject`      | true          | 召回结果自动注入 KVStore          |
|               | `retriever_cache_ttl`        | 300           | 召回结果缓存 TTL（秒）            |
|               | `retriever_timeout_ms`       | 10            | 召回超时（毫秒）                  |
| **Slot 治理** | `pin_registered`             | true          | register() 知识自动锁定           |
|               | `retriever_slot_ratio`       | 0.3           | 召回器最大 slot 占比              |
|               | `retriever_slot_budget`      | 0             | 显式 slot 预算（0=使用 ratio）    |
|               | `retriever_cooldown_steps`   | 5             | 召回冷却期（forward 步数）        |
|               | `retriever_dedup_similarity` | 0.95          | 语义去重阈值（余弦相似度）        |
|               | `slot_stability_threshold`   | 0.5           | 每步最大 slot 变化比例            |
| **埋点**      | `instrumentation_enabled`    | true          | 埋点开关                          |
|               | `event_buffer_size`          | 10000         | 事件缓冲区大小                    |
|               | `audit_log_level`            | "INFO"        | 审计日志级别                      |
| **可观测性**  | `observability_enabled`      | true          | 外部可观测性开关                  |

**加载方式**:

```python
# 1. 直接创建
config = AGAConfig(hidden_dim=4096)

# 2. 从 YAML 加载（支持嵌套段落）
config = AGAConfig.from_yaml("aga_config.yaml")

# 3. 从字典创建（支持 gate, decay, retriever, slot_governance 嵌套展平）
config = AGAConfig.from_dict({
    "hidden_dim": 4096,
    "gate": {"tau_low": 0.3, "tau_high": 2.5},
    "decay": {"enabled": True, "gamma": 0.85},
    "retriever": {"backend": "chroma", "endpoint": "localhost:8000"},
    "slot_governance": {"pin_registered": True, "retriever_slot_ratio": 0.3},
})

# 4. 验证
errors = config.validate()
```

### 3.3 KVStore (kv_store.py)

**职责**: GPU 常驻 KV 存储，预分配内存，LRU 淘汰，知识锁定。

**设计要点**:

-   **预分配 GPU 内存**: 初始化时一次性分配所有槽位的 GPU 内存
-   **FP16 存储**: 所有 KV 数据以 FP16 格式存储，最小化 VRAM 占用
-   **LRU 淘汰 + Pin 保护**: 槽位满时自动淘汰最近最少使用的*未锁定*知识；锁定知识永不被淘汰
-   **线程安全**: 所有写操作加锁
-   **命名空间隔离**: 通过 metadata 中的 namespace 字段实现
-   **活跃缓存**: `get_active()` 结果被缓存，写操作时自动失效

**Pin/Unpin 机制** (v4.4):

-   `put(id, key, value, pinned=True)` — 注册并锁定知识
-   `pin(id)` / `unpin(id)` — 显式管理锁定状态
-   锁定知识受 LRU 淘汰保护
-   `pinned_count` / `unpinned_count` 属性用于监控
-   `source` 元数据字段标记来源（"register" vs "retriever"）

**VRAM 占用公式**:

```
VRAM = max_slots × (bottleneck_dim × 2 + hidden_dim × 2 + 3) bytes
```

### 3.4 EntropyGateSystem (gate/entropy_gate.py)

**职责**: 完整三段式熵门控系统。

**Gate-0 (先验门控)**: 零计算成本，基于 namespace 的静态过滤。

**Gate-1 (置信门控)**:

-   计算 hidden_states 的内部方差
-   通过学习的投影网络增强不确定性估计
-   兼容 FlashAttention（不需要 attention weights）

**三段式熵否决**:

-   `H < τ_low`: 模型自信，gate 强制为 0
-   `τ_low ≤ H ≤ τ_high`: 正常区间，gate = σ(w₁·H + b)
-   `H > τ_high`: 模型极度不确定，gate 限制在 max_gate

### 3.5 BottleneckInjector (operator/bottleneck_injector.py)

**职责**: 核心注入路径，延迟 <0.1ms。

**数学流程**:

1. Query 投影: `W_q: [hidden_dim → bottleneck_dim]`
2. Top-K 路由: 选择全局最相关的 K 个知识槽位
3. 注意力计算: `softmax(query @ keys.T / √d + log(reliability))`
4. 加权求和: `attn_weights @ values`
5. Value 投影: `W_up(GELU(W_down(aux_output)))` — delta subspace

**信息容量**:

-   每个 key: `[bottleneck_dim=64]` — 用于检索匹配
-   每个 value: `[hidden_dim=4096]` — 实际注入的知识向量
-   每个 value 可编码 10-50 tokens 的原子事实语义

### 3.6 PersistenceDecay (gate/decay.py)

**职责**: 防止辅助注意力跨层累积导致推理风格漂移。

**线程隔离** (v4.4): 衰减上下文存储在 `threading.local()` 中，确保并发请求（如 vLLM 连续批处理）维护独立的衰减状态。

**支持的衰减策略**:

-   `exponential`: α\_{l+1} = γ^l · α_l（默认）
-   `linear`: α\_{l+1} = α_l - δ
-   `adaptive`: 基于累积量动态调整
-   `none`: 不衰减

**硬重置机制**: 当累积 gate 超过阈值时，强制重置为 0。

### 3.7 召回器协议 (retriever/)

**职责**: 外部知识检索的标准协议。AGA 在检测到高熵时查询外部知识源。

**v4.4 新增功能**: 这是核心新增，使 AGA 能够从外部知识基础设施获取推理事实。

**BaseRetriever** (抽象基类):

| 方法                    | 签名                                              | 说明                     |
| ----------------------- | ------------------------------------------------- | ------------------------ |
| `retrieve`              | `(query: RetrievalQuery) → List[RetrievalResult]` | 核心检索（高熵时调用）   |
| `warmup`                | `() → None`                                       | 可选预热（索引、连接等） |
| `on_injection_feedback` | `(result_id, was_used, gate_value) → None`        | 可选反馈回调             |
| `get_stats`             | `() → Dict[str, Any]`                             | 召回器统计               |
| `shutdown`              | `() → None`                                       | 释放资源                 |

**RetrievalQuery** (输入):

| 字段              | 类型                       | 说明                                   |
| ----------------- | -------------------------- | -------------------------------------- |
| `hidden_states`   | `Tensor[batch, seq, dim]`  | 当前层的 hidden_states（语义查询信号） |
| `query_projected` | `Tensor[batch, seq, bdim]` | q_proj 输出（已对齐到知识空间）        |
| `entropy`         | `Tensor[batch, seq]`       | 当前熵值                               |
| `layer_idx`       | `int`                      | 当前 Transformer 层索引                |
| `namespace`       | `Optional[str]`            | 命名空间过滤                           |
| `top_k`           | `int`                      | 期望返回的最大结果数                   |

**RetrievalResult** (输出):

| 字段          | 类型                     | 说明                 |
| ------------- | ------------------------ | -------------------- |
| `id`          | `str`                    | 知识唯一标识         |
| `key`         | `Tensor[bottleneck_dim]` | 检索键向量           |
| `value`       | `Tensor[hidden_dim]`     | 知识值向量           |
| `reliability` | `float`                  | 可靠性分数 (0.0-1.0) |
| `score`       | `float`                  | 检索相关性分数       |
| `metadata`    | `Optional[Dict]`         | 可选元数据           |

**内置召回器**:

| 召回器             | 说明                                     |
| ------------------ | ---------------------------------------- |
| `NullRetriever`    | 默认 — 不进行外部检索（仅 KVStore 模式） |
| `KVStoreRetriever` | 在现有 KVStore 条目中按余弦相似度搜索    |

**自定义召回器示例**:

```python
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult

class ChromaRetriever(BaseRetriever):
    def __init__(self, collection_name: str, endpoint: str):
        import chromadb
        self.client = chromadb.HttpClient(host=endpoint)
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        query_vec = query.query_projected.mean(dim=[0, 1]).cpu().numpy()
        results = self.collection.query(query_embeddings=[query_vec], n_results=query.top_k)
        return [...]

plugin = AGAPlugin(config, retriever=ChromaRetriever("medical_kb", "localhost:8000"))
```

### 3.8 Slot 治理

**职责**: 防止高并发、slot 争抢场景下的 slot thrashing（槽位抖动）和 gate jitter（门控抖动）。

**v4.4 新增功能**: 当 `max_slots` 较小而召回知识较多时，快速淘汰和重新插入会导致不稳定。Slot 治理提供五道防线：

| 机制               | 配置参数                      | 说明                                         |
| ------------------ | ----------------------------- | -------------------------------------------- |
| **知识锁定**       | `pin_registered`              | `register()` 知识自动锁定（不可被 LRU 淘汰） |
| **Slot 预算守卫**  | `retriever_slot_ratio/budget` | 召回器只能使用总 slot 的一部分               |
| **语义去重**       | `retriever_dedup_similarity`  | 过滤语义相似的召回结果（余弦 > 阈值）        |
| **注入稳定性检测** | `slot_stability_threshold`    | KVStore 变化过快时暂停召回                   |
| **自适应冷却期**   | `retriever_cooldown_steps`    | 两次召回之间的最小 forward 步数              |

**Slot 生命周期**:

```
register() → 锁定（受淘汰保护）
retriever  → 未锁定（在预算内受 LRU 淘汰）
```

### 3.9 HuggingFaceAdapter (adapter/huggingface.py)

**职责**: 自动检测并挂载到 HuggingFace 模型。

**支持的模型架构**:

-   LLaMA / LLaMA-2 / LLaMA-3
-   Qwen / Qwen-2
-   Mistral / Mixtral
-   GPT-2 / GPT-J / GPT-NeoX
-   Phi / Phi-2 / Phi-3
-   Gemma / Falcon

**挂载方式**: 通过 `register_forward_hook` 在 `self_attn` 输出后注入 AGA。

### 3.10 VLLMAdapter (adapter/vllm.py)

**职责**: 将 AGA 注入到 vLLM 推理框架中的模型，无需 fork vLLM。

**核心能力**:

| 能力                    | 说明                                           |
| ----------------------- | ---------------------------------------------- |
| `extract_model()`       | 从 vLLM LLM/LLMEngine 实例中提取底层 nn.Module |
| `get_layers()`          | 递归搜索 Transformer 层（支持多种模型结构）    |
| `wrap_layer()`          | 在 attention 输出后注册 forward hook 注入 AGA  |
| `check_compatibility()` | 检查 vLLM 配置兼容性并生成报告                 |

**VLLMHookWorker**: 兼容 IBM vLLM-Hook 插件系统。如 vLLM-Hook 未安装，自动回退到直接 PyTorch hook 方式。

### 3.11 分布式支持 (distributed.py)

**职责**: Tensor Parallelism 场景下的多 GPU 部署支持。

**v4.4 新增功能**: `TPManager` 在 TP rank 之间同步 AGA 状态。

| 方法                       | 说明                                            |
| -------------------------- | ----------------------------------------------- |
| `broadcast_parameters()`   | 从 rank 0 同步学习参数（gate_system, injector） |
| `broadcast_knowledge()`    | 从 rank 0 同步整个 KVStore 内容                 |
| `broadcast_single_entry()` | 注册后同步单条知识                              |
| `is_primary`               | 当前 rank 是否为 rank 0                         |
| `is_enabled`               | torch.distributed 是否已初始化                  |

**使用方式**:

```python
plugin = AGAPlugin(config)
tp_manager = TPManager(plugin)
tp_manager.broadcast_parameters()

if tp_manager.is_primary:
    plugin.register("fact_001", key=k, value=v)
tp_manager.broadcast_knowledge()
```

### 3.12 StreamingSession (streaming.py)

**职责**: 流式生成过程中的会话管理和实时诊断。

**核心功能**:

-   逐 token 诊断 (`get_step_diagnostics()`)
-   会话统计 (`get_session_summary()`)
-   动态知识热更新 (`update_knowledge()` / `remove_knowledge()`)
-   自动衰减上下文管理
-   层事件过滤 (v4.4): 仅统计主监控层的事件，避免多层挂载时的重复计数

### 3.13 Instrumentation (instrumentation/)

**职责**: 内置埋点层，零外部依赖，始终可用。

**EventBus**:

-   内存环形缓冲区（默认 10000 条）
-   可插拔订阅者模式
-   事件发射 <1μs，不影响推理延迟
-   支持通配符订阅 (`*`)

**ForwardMetrics**:

-   激活率 (activation_rate)
-   平均门控值 (gate_mean_avg)
-   平均熵值 (entropy_mean_avg)
-   P50/P95/P99 延迟百分位
-   按层统计

**AuditLog**:

-   所有知识管理操作的完整审计轨迹
-   Python logging 集成（始终输出）
-   内存缓冲区可查询
-   通过 EventBus 发射审计事件

---

## 4. 安全机制

### 4.1 Fail-Open

AGA 采用 Fail-Open 设计原则：**任何异常都回退到原始模型输出**。适用于：

-   forward 路径异常
-   召回器失败（返回空列表）
-   召回器预热/关闭失败

### 4.2 范数控制

注册知识时自动进行范数裁剪，防止异常值影响推理。

### 4.3 熵否决

三段式熵否决确保 AGA 不会在模型已经确信的情况下强行注入。

### 4.4 跨层衰减

持久化衰减防止辅助注意力在多层累积。硬重置机制在累积超过阈值时强制归零。通过 `threading.local()` 实现线程隔离。

### 4.5 Slot 治理

通过预算限制、语义去重、知识锁定、稳定性检测和自适应冷却期，防止 slot thrashing 和 gate jitter。

---

## 5. 性能特征

### 5.1 延迟

| 操作            | 典型延迟  | 说明                       |
| --------------- | --------- | -------------------------- |
| Gate-0 检查     | <1 μs     | 纯 Python 字符串比较       |
| Gate-1 熵计算   | ~10 μs    | 方差计算 + 投影网络        |
| Early Exit      | ~15 μs    | Gate-0 + Gate-1 后直接返回 |
| Bottleneck 注入 | ~50-80 μs | 完整注入路径               |
| 召回器调用      | 1-10 ms   | 外部知识检索               |
| 事件发射        | <1 μs     | 内存写入                   |

### 5.2 VRAM

| max_slots | hidden_dim=4096 | hidden_dim=8192 |
| --------- | --------------- | --------------- |
| 256       | ~2.0 MB         | ~4.0 MB         |
| 1000      | ~8.1 MB         | ~16.1 MB        |
| 5000      | ~40.6 MB        | ~81.0 MB        |

### 5.3 吞吐量

AGA 的旁路率（Early Exit）在通用场景下通常 >60%，意味着大部分 token 不会触发完整注入路径，对整体吞吐量影响极小。

在垂直领域场景下，激活率可能达到 40-70%，此时每次注入的延迟 ~50-80μs 仍远低于 Transformer 层本身的计算时间。

召回器调用（1-10ms）频率很低（受冷却期和稳定性检查约束），且仅在第一个挂载层的高熵 token 处触发。

---

## 6. 异常体系

| 异常类           | 继承自      | 触发场景          |
| ---------------- | ----------- | ----------------- |
| `AGAError`       | `Exception` | AGA 基础异常      |
| `AttachError`    | `AGAError`  | 模型挂载/卸载失败 |
| `KVStoreError`   | `AGAError`  | KV 存储操作失败   |
| `ConfigError`    | `AGAError`  | 配置加载/验证失败 |
| `GateError`      | `AGAError`  | 门控系统异常      |
| `AdapterError`   | `AGAError`  | LLM 适配器异常    |
| `RetrieverError` | `AGAError`  | 召回器操作失败    |

---

## 7. 数据类型

### KnowledgeEntry

```python
@dataclass
class KnowledgeEntry:
    id: str                              # 知识唯一标识
    key: torch.Tensor                    # [bottleneck_dim] 检索键
    value: torch.Tensor                  # [hidden_dim] 知识向量
    reliability: float = 1.0             # 可靠性分数 (0.0-1.0)
    metadata: Optional[Dict] = None      # 可选元数据
```

### GateDiagnostics

```python
@dataclass
class GateDiagnostics:
    gate0_passed: bool = True            # Gate-0 是否通过
    entropy_mean: float = 0.0            # 平均熵值
    gate_mean: float = 0.0              # 平均门控值
    gate_max: float = 0.0              # 最大门控值
    early_exit: bool = False            # 是否 Early Exit
    veto_ratio: float = 0.0            # 否决比例
```

### RetrievalQuery / RetrievalResult

详见第 3.7 节完整字段说明。

---

## 8. JSONL 知识文件格式

`aga-core` 内置支持 JSONL 格式的知识文件加载，无需 `aga-knowledge`。

**文件格式**:

```jsonl
{"id": "fact_001", "key": [0.1, 0.2, ...], "value": [0.3, 0.4, ...], "reliability": 0.95, "metadata": {"source": "medical_kb"}}
{"id": "fact_002", "key": [0.5, 0.6, ...], "value": [0.7, 0.8, ...], "reliability": 0.9}
```

**字段说明**:

| 字段          | 类型    | 必需 | 说明                              |
| ------------- | ------- | ---- | --------------------------------- |
| `id`          | string  | 是   | 知识唯一标识                      |
| `key`         | float[] | 是   | 检索键向量，长度 = bottleneck_dim |
| `value`       | float[] | 是   | 知识值向量，长度 = hidden_dim     |
| `reliability` | float   | 否   | 可靠性分数，默认 1.0              |
| `metadata`    | object  | 否   | 可选元数据                        |

---

## 9. 与 aga-knowledge 集成

当安装了 `aga-knowledge` 后，`aga-core` 可以自动集成更丰富的知识管理能力：

```python
# 方式 1: 通过 load_from 加载非 JSONL 格式
plugin.load_from("postgresql://localhost/knowledge_db")

# 方式 2: 配置驱动自动加载
plugin = AGAPlugin.from_config({
    "hidden_dim": 4096,
    "knowledge_sources": [
        {"type": "jsonl", "path": "base_knowledge.jsonl"},
        {"type": "portal", "url": "http://portal:8000"},
    ]
})

# 方式 3: 直接使用 KnowledgeManager
from aga_knowledge import KnowledgeManager
manager = KnowledgeManager(config)
manager.sync_to_plugin(plugin)
```

**注意**: 有了标准 `BaseRetriever` 协议，`aga-knowledge` 不再是严格必需的。用户可以使用现有基础设施（Chroma、Milvus、Elasticsearch 等）实现自己的召回器，直接传递给 `AGAPlugin`。

---

## 10. 完整 YAML 配置参考

```yaml
aga:
    # ===== 模型维度 =====
    hidden_dim: 4096
    bottleneck_dim: 64
    num_heads: 32
    value_bottleneck_dim: 256

    # ===== 容量 =====
    max_slots: 256

    # ===== 设备 =====
    device: "cuda"

    # ===== 熵门控 =====
    gate:
        gate0_enabled: true
        gate0_disabled_namespaces: []
        gate1_enabled: true
        gate1_uncertainty_source: "hidden_variance"
        gate2_top_k: 8
        tau_low: 0.5
        tau_high: 2.0
        max_gate: 0.8
        early_exit_enabled: true
        early_exit_threshold: 0.05

    # ===== 衰减 =====
    decay:
        enabled: true
        strategy: "exponential"
        gamma: 0.9
        hard_reset_threshold: 3.0

    # ===== 安全 =====
    fail_open: true
    max_forward_timeout_ms: 50

    # ===== 范数控制 =====
    enable_norm_clipping: true
    key_norm_target: 5.0
    value_norm_target: 3.0

    # ===== 召回器（配置驱动知识检索） =====
    retriever:
        backend: "null" # null / kv_store / chroma / milvus / custom
        endpoint: "" # 外部召回器连接地址
        collection: "" # 知识集合名称
        top_k: 5 # 每次召回最大结果数
        min_score: 0.3 # 最小相关性阈值
        query_source: "q_proj" # hidden_states / q_proj
        auto_inject: true # 召回结果自动注入 KVStore
        cache_ttl: 300 # 召回结果缓存 TTL（秒）
        timeout_ms: 10 # 召回超时（毫秒）

    # ===== Slot 治理 =====
    slot_governance:
        pin_registered: true # register() 知识自动锁定
        retriever_slot_ratio: 0.3 # 召回器最大 slot 占比
        retriever_slot_budget: 0 # 显式预算（0=使用 ratio）
        retriever_cooldown_steps: 5 # 两次召回最小间隔步数
        retriever_dedup_similarity: 0.95 # 语义去重阈值
        slot_stability_threshold: 0.5 # 每步最大变化比例

    # ===== 埋点与审计 =====
    instrumentation:
        instrumentation_enabled: true
        event_buffer_size: 10000
        audit_log_level: "INFO"

    # ===== 可观测性（需安装 aga-observability）=====
    observability:
        observability_enabled: true
        prometheus_enabled: true
        prometheus_port: 9090

    # ===== 知识源 =====
    knowledge_sources:
        - type: jsonl
          path: "data/base_knowledge.jsonl"
```

---

## 11. 测试

`aga-core` 包含完整的单元测试套件：

```bash
# 运行所有测试
cd AGAPlugin
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_plugin.py -v
python -m pytest tests/test_kv_store.py -v
python -m pytest tests/test_gate.py -v
python -m pytest tests/test_retriever.py -v
python -m pytest tests/test_streaming.py -v
```

**测试覆盖**:

-   AGAPlugin: 初始化、配置驱动、知识管理、模型挂载、诊断、召回器集成、Slot 治理
-   KVStore: CRUD、LRU 淘汰、pin/unpin、命名空间、线程安全、VRAM 估算、活跃缓存
-   EntropyGateSystem: 三段式门控、熵否决、Early Exit
-   BottleneckInjector: 注入路径、Top-K 路由、Value 投影
-   PersistenceDecay: 衰减策略、硬重置、上下文传递、线程隔离
-   HuggingFaceAdapter: 架构检测、Hook 注入
-   VLLMAdapter: 模型提取、层搜索、Hook 注入、兼容性检查
-   VLLMHookWorker: vLLM-Hook 兼容、回退机制
-   Retriever: NullRetriever、KVStoreRetriever、BaseRetriever 协议、预算/去重/冷却
-   StreamingSession: 会话管理、逐 token 诊断、层事件过滤
-   EventBus: 事件发射、订阅、查询
-   ForwardMetrics: 指标收集、百分位计算
-   AuditLog: 审计记录、过滤查询
-   AGAConfig: YAML 加载、字典创建、验证、嵌套展平

---

## 附录 A: 术语表

| 术语           | 英文                   | 说明                                 |
| -------------- | ---------------------- | ------------------------------------ |
| 熵门控         | Entropy Gating         | 基于模型不确定性决定是否注入知识     |
| 瓶颈注入       | Bottleneck Injection   | 通过低维投影空间进行知识匹配和注入   |
| 知识槽位       | Knowledge Slot         | KVStore 中的一个 key-value 对        |
| 旁路           | Bypass                 | 模型确信时跳过 AGA 注入              |
| 持久化衰减     | Persistence Decay      | 防止辅助注意力跨层累积               |
| 硬重置         | Hard Reset             | 累积 gate 超过阈值时强制归零         |
| Fail-Open      | Fail-Open              | 异常时回退到原始模型输出             |
| delta subspace | Delta Subspace         | Value 投影的瓶颈层，控制注入幅度     |
| 召回器         | Retriever              | 高熵时查询的外部知识源               |
| Slot 治理      | Slot Governance        | 防止 slot thrashing 和 gate jitter   |
| 知识锁定       | Knowledge Pinning      | 保护核心知识不被 LRU 淘汰            |
| 语义去重       | Semantic Deduplication | 过滤语义相似的召回结果               |
| Slot Thrashing | Slot Thrashing         | 知识槽位的快速淘汰和重新插入         |
| Gate Jitter    | Gate Jitter            | KVStore 内容快速变化导致的门控不稳定 |

## 附录 B: 版本历史

| 版本  | 日期       | 主要变更                                            |
| ----- | ---------- | --------------------------------------------------- |
| 4.2.0 | 2026-02-20 | aga-core 初始发布，完整插件架构                     |
| 4.3.0 | 2026-02-22 | vLLM 适配器、流式生成、IBM vLLM-Hook 兼容           |
| 4.4.0 | 2026-02-23 | 召回器协议、Slot 治理、TP 支持、线程隔离、pin/unpin |
