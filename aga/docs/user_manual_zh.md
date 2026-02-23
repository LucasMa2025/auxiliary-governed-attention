# AGA-Core 用户手册

> **版本**: 4.4.0  
> **最后更新**: 2026-02-23

---

## 目录

1. [安装与环境](#1-安装与环境)
2. [快速入门](#2-快速入门)
3. [知识管理](#3-知识管理)
4. [模型挂载](#4-模型挂载)
5. [流式生成注入](#5-流式生成注入)
6. [外部召回器集成](#6-外部召回器集成)
7. [配置管理](#7-配置管理)
8. [诊断与监控](#8-诊断与监控)
9. [高级用法](#9-高级用法)
10. [生产部署](#10-生产部署)
11. [故障排除](#11-故障排除)
12. [FAQ](#12-faq)

---

## 1. 安装与环境

### 1.1 系统要求

| 要求     | 最低版本        | 推荐         |
| -------- | --------------- | ------------ |
| Python   | 3.9             | 3.10+        |
| PyTorch  | 2.0.0           | 2.1.0+       |
| CUDA     | 11.7 (可选)     | 12.0+        |
| GPU VRAM | 100 MB (AGA 自身) | 1 GB+ (含模型) |
| 系统内存 | 4 GB            | 16 GB+       |

### 1.2 安装

```bash
# 基础安装（仅依赖 torch）
pip install aga-core

# 从源码安装
git clone https://github.com/aga-project/aga-core.git
cd aga-core
pip install -e .

# 带 YAML 配置支持
pip install aga-core[yaml]

# 带知识管理系统
pip install aga-core[knowledge]

# 全部功能
pip install aga-core[all]

# 开发环境
pip install aga-core[dev]
```

### 1.3 验证安装

```python
import aga
print(f"AGA version: {aga.__version__}")
# 输出: AGA version: 4.4.0

from aga import AGAPlugin, AGAConfig
plugin = AGAPlugin(AGAConfig(hidden_dim=768, device="cpu"))
print(plugin)
# 输出: AGAPlugin(hidden_dim=768, slots=0/256, attached=False, device=cpu)
```

---

## 2. 快速入门

### 2.1 最简示例

```python
from aga import AGAPlugin, AGAConfig
import torch

# 步骤 1: 创建插件
plugin = AGAPlugin(AGAConfig(
    hidden_dim=4096,      # 必须匹配模型的 hidden_size
    device="cuda",
))

# 步骤 2: 注册知识
plugin.register(
    id="capital_france",
    key=torch.randn(64),       # bottleneck_dim=64
    value=torch.randn(4096),   # hidden_dim=4096
    reliability=0.95,
    pinned=True,               # 锁定核心知识（防止被淘汰）
)

# 步骤 3: 挂载到模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
plugin.attach(model)

# 步骤 4: 正常推理 — AGA 自动工作
output = model.generate(input_ids, max_new_tokens=100)

# 步骤 5: 查看效果
print(plugin.get_diagnostics())
```

### 2.2 配置驱动示例

**aga_config.yaml**:

```yaml
aga:
    hidden_dim: 4096
    bottleneck_dim: 64
    max_slots: 512
    device: "cuda"

    gate:
        tau_low: 0.5
        tau_high: 2.0
        max_gate: 0.8
        gate2_top_k: 8
        early_exit_threshold: 0.05

    decay:
        enabled: true
        strategy: "exponential"
        gamma: 0.9

    slot_governance:
        pin_registered: true
        retriever_dedup_similarity: 0.95

    knowledge_sources:
        - type: jsonl
          path: "data/medical_knowledge.jsonl"
```

```python
from aga import AGAPlugin

# 一行创建 + 自动加载知识
plugin = AGAPlugin.from_config("aga_config.yaml")
plugin.attach(model)
output = model.generate(input_ids)
```

---

## 3. 知识管理

### 3.1 知识格式

AGA 中的每条知识由以下部分组成：

| 字段          | 类型                     | 必需 | 说明                                      |
| ------------- | ------------------------ | ---- | ----------------------------------------- |
| `id`          | `str`                    | 是   | 唯一标识符                                |
| `key`         | `Tensor[bottleneck_dim]` | 是   | 检索键向量，用于与模型 hidden_states 匹配 |
| `value`       | `Tensor[hidden_dim]`     | 是   | 知识值向量，实际注入到注意力层的内容      |
| `reliability` | `float`                  | 否   | 可靠性分数 (0.0-1.0)，影响注入权重        |
| `pinned`      | `bool`                   | 否   | 若为 True，保护知识不被 LRU 淘汰 (v4.4.0) |
| `metadata`    | `Dict`                   | 否   | 元数据（namespace, source 等）            |

### 3.2 注册知识

```python
import torch

# 单条注册（支持锁定）
success = plugin.register(
    id="med_001",
    key=torch.randn(64),
    value=torch.randn(4096),
    reliability=0.95,
    pinned=True,                # 核心知识 — 保护不被淘汰
    metadata={"namespace": "cardiology", "source": "textbook"}
)
print(f"注册{'成功' if success else '失败'}")

# 批量注册
count = plugin.register_batch([
    {
        "id": "med_002",
        "key": torch.randn(64),
        "value": torch.randn(4096),
        "reliability": 0.9,
        "metadata": {"namespace": "cardiology"},
    },
    {
        "id": "med_003",
        "key": torch.randn(64),
        "value": torch.randn(4096),
        "reliability": 0.85,
    },
])
print(f"成功注册 {count} 条知识")
```

### 3.3 知识锁定 (v4.4.0)

锁定的知识受到保护，不会被 LRU 淘汰。这对于必须始终可用的核心领域知识至关重要。

```python
# 注册时锁定
plugin.register("core_fact", key=k, value=v, pinned=True)

# 锁定/解锁已有知识
plugin.store.pin("core_fact")
plugin.store.unpin("core_fact")

# 查看锁定统计
stats = plugin.get_store_stats()
print(f"已锁定: {stats['pinned_count']}, 未锁定: {stats['unpinned_count']}")
```

> **注意**: 当 `config.pin_registered = True`（默认）时，通过 `register()` 注册的所有知识自动锁定。由召回器注入的知识**不会**被自动锁定，确保核心知识始终优先。

### 3.4 从文件加载

**JSONL 文件格式**:

```jsonl
{"id": "fact_001", "key": [0.1, -0.2, 0.3, ...], "value": [0.4, 0.5, ...], "reliability": 0.95, "metadata": {"source": "wiki"}}
{"id": "fact_002", "key": [0.6, 0.7, ...], "value": [0.8, -0.9, ...], "reliability": 0.9}
```

> **注意**: `key` 数组长度必须等于 `bottleneck_dim`（默认 64），`value` 数组长度必须等于 `hidden_dim`（默认 4096）。

```python
# 从 JSONL 加载（内置，无需 aga-knowledge）
count = plugin.load_from("data/knowledge.jsonl")
print(f"加载了 {count} 条知识")
```

### 3.5 移除知识

```python
# 移除单条
success = plugin.unregister("med_001")

# 清空命名空间
plugin.clear(namespace="cardiology")

# 清空所有
plugin.clear()
```

### 3.6 知识查询

```python
# 检查知识是否存在
exists = plugin.store.contains("med_001")

# 获取单条知识
result = plugin.store.get("med_001")
if result:
    key, value, reliability = result

# 获取所有知识 ID
all_ids = plugin.store.get_all_ids()

# 获取知识元数据
meta = plugin.store.get_metadata("med_001")
```

### 3.7 知识准备建议

**如何准备 key 和 value 向量？**

在独立使用 `aga-core` 时，用户需要自行准备 KV 向量。以下是常见方法：

**方法 1: 使用 Sentence Transformer 编码**

```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

# 编码器
encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384 维

# 投影层（需要训练或手动初始化）
key_proj = nn.Linear(384, 64)    # -> bottleneck_dim
value_proj = nn.Linear(384, 4096)  # -> hidden_dim

# 编码知识文本
text = "法国的首都是巴黎"
embedding = torch.tensor(encoder.encode(text))

key = key_proj(embedding)
value = value_proj(embedding)

plugin.register("capital_france", key=key, value=value)
```

**方法 2: 使用模型自身的 hidden_states**

```python
# 通过模型自身编码知识文本
with torch.no_grad():
    outputs = model(tokenizer("法国的首都是巴黎", return_tensors="pt").input_ids)
    hidden = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]

# 投影到 key 空间
key = plugin.injector.q_proj(hidden).squeeze(0)  # [bottleneck_dim]
value = hidden.squeeze(0)  # [hidden_dim]

plugin.register("capital_france", key=key, value=value)
```

**方法 3: 使用 aga-knowledge（推荐用于生产环境）**

```python
# aga-knowledge 提供完整的文本 -> KV 编码管道
from aga_knowledge import KnowledgeManager
manager = KnowledgeManager(config)
manager.ingest_document("medical_textbook.pdf")
manager.sync_to_plugin(plugin)
```

---

## 4. 模型挂载

### 4.1 基本挂载

```python
# 默认挂载到最后 3 层
plugin.attach(model)

# 指定层索引（负数从后往前）
plugin.attach(model, layer_indices=[-1, -2, -3, -4, -5])

# 指定正索引
plugin.attach(model, layer_indices=[28, 29, 30, 31])
```

### 4.2 选择挂载层

**推荐策略**:

| 场景     | 推荐层数                 | 说明                   |
| -------- | ------------------------ | ---------------------- |
| 快速实验 | 最后 1 层 `[-1]`         | 最小开销               |
| 标准使用 | 最后 3 层 `[-1, -2, -3]` | 平衡效果和性能（默认） |
| 深度注入 | 最后 5-8 层              | 更强的知识影响力       |
| 全层注入 | 所有层                   | 最大影响力，但延迟最高 |

### 4.3 卸载

```python
# 卸载（移除所有 hooks）
plugin.detach()

# 卸载后可以重新挂载
plugin.attach(another_model)
```

### 4.4 vLLM 适配器

AGA 原生支持 vLLM 推理框架，**无需 fork vLLM**。

#### 4.4.1 基本用法

```python
from vllm import LLM, SamplingParams
from aga import AGAPlugin, AGAConfig
from aga.adapter.vllm import VLLMAdapter

# 1. 创建 vLLM 引擎（建议 enforce_eager=True）
llm = LLM(model="meta-llama/Llama-2-7b-hf", enforce_eager=True)

# 2. 从 vLLM 提取内部模型
model = VLLMAdapter.extract_model(llm)

# 3. 创建 AGA 插件并挂载
plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.register("fact_001", key=k, value=v)
plugin.attach(model, adapter=VLLMAdapter())

# 4. 正常使用 vLLM 推理 - AGA 自动介入
outputs = llm.generate(["法国的首都是什么？"], SamplingParams(max_tokens=100))
```

#### 4.4.2 兼容性检查

```python
report = VLLMAdapter.check_compatibility(llm)
print(f"兼容: {report['compatible']}")
print(f"模型: {report['model_type']}")
print(f"CUDA Graph: {report['cuda_graph']}")
print(f"张量并行: {report['tensor_parallel']}")

for warning in report['warnings']:
    print(f"  警告: {warning}")
for rec in report['recommendations']:
    print(f"  建议: {rec}")
```

#### 4.4.3 注意事项

| 项目 | 说明 |
| ---- | ---- |
| **PagedAttention** | AGA 在 attention 输出后注入，与 PagedAttention 兼容 |
| **连续批处理** | AGA 的 forward 逻辑正确处理 batch 维度 |
| **CUDA Graph** | 使用 `enforce_eager=True` 确保 hook 行为可预测 |
| **张量并行** | 使用 `TPManager` 在各 TP rank 间同步 KVStore |
| **推荐设置** | `enforce_eager=True` 以确保 hook 行为可预测 |

### 4.5 自定义适配器

如果您的模型不是标准 HuggingFace 架构：

```python
from aga.adapter.base import LLMAdapter

class MyCustomAdapter(LLMAdapter):
    def get_layers(self, model):
        return list(model.model.layers)

    def get_hidden_dim(self, model):
        return model.config.hidden_size

    def wrap_layer(self, model, layer_idx, aga_forward):
        layer = self.get_layers(model)[layer_idx]
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            input_hidden = input[0] if isinstance(input, tuple) else input
            fused = aga_forward(
                hidden_states=input_hidden,
                primary_attention_output=hidden,
            )
            if isinstance(output, tuple):
                return (fused,) + output[1:]
            return fused
        return layer.self_attn.register_forward_hook(hook_fn)

# 使用自定义适配器
plugin.attach(model, adapter=MyCustomAdapter())
```

---

## 5. 流式生成注入

### 5.1 概述

AGA 原生支持在 LLM 的自回归（token-by-token）生成过程中进行动态知识注入。由于 AGA 通过 `register_forward_hook` 挂载到 Transformer 层，它在每个 decode step 中自动评估每个 token 的熵值并决定是否注入知识。

`StreamingSession` 提供了流式生成过程中的**会话管理**和**实时诊断**能力。

### 5.2 基本用法

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)

# 创建流式会话
session = plugin.create_streaming_session()

# 模拟流式生成
for step in range(max_tokens):
    output = model.forward(current_token_ids)
    next_token = sample(output.logits)
    
    # 获取当前步骤的 AGA 诊断
    diag = session.get_step_diagnostics()
    print(f"Step {diag['step']}: applied={diag['aga_applied']}, "
          f"gate={diag['gate_mean']:.4f}, entropy={diag['entropy_mean']:.4f}")
    
    if next_token == eos_token:
        break

# 获取会话摘要
summary = session.get_session_summary()
print(f"总步数: {summary['total_steps']}")
print(f"注入次数: {summary['injection_count']}")
print(f"注入率: {summary['injection_rate']:.2%}")
print(f"平均门控值: {summary['avg_gate_mean']:.4f}")
print(f"平均熵值: {summary['avg_entropy_mean']:.4f}")

# 关闭会话（自动清理衰减上下文）
session.close()
```

### 5.3 动态知识热更新

在流式生成过程中，可以动态添加或移除知识：

```python
session = plugin.create_streaming_session()

for step in range(max_tokens):
    output = model.forward(current_token_ids)
    
    # 根据生成内容动态添加知识
    if detected_topic_change(output):
        session.update_knowledge(
            id="dynamic_fact",
            key=new_key_tensor,
            value=new_value_tensor,
            reliability=0.9,
        )
    
    # 移除不再需要的知识
    if topic_completed(output):
        session.remove_knowledge("dynamic_fact")

session.close()
```

### 5.4 与 HuggingFace generate() 集成

```python
from transformers import TextIteratorStreamer
from threading import Thread

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)

# 使用 HuggingFace 的 TextIteratorStreamer
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
session = plugin.create_streaming_session()

# 在后台线程中运行 generate
generation_kwargs = {
    "input_ids": input_ids,
    "max_new_tokens": 200,
    "streamer": streamer,
}
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 逐 token 读取输出
for text in streamer:
    print(text, end="", flush=True)

thread.join()
summary = session.get_session_summary()
print(f"\n--- AGA 注入率: {summary['injection_rate']:.2%} ---")
session.close()
```

### 5.5 流式生成中的多次激活

在垂直领域场景下，AGA 在一次完整的流式生成中可能被**多次激活**：

```
生成: "患者的诊断结果为..."
  Token "患者" -> 低熵 -> 旁路
  Token "的"   -> 低熵 -> 旁路
  Token "诊断" -> 高熵 -> AGA 注入医学知识
  Token "结果" -> 中熵 -> AGA 轻度注入
  Token "为"   -> 低熵 -> 旁路
  Token "急性" -> 高熵 -> AGA 注入疾病知识
  Token "心肌" -> 高熵 -> AGA 注入心脏病学知识
  Token "梗死" -> 中熵 -> AGA 轻度注入
```

---

## 6. 外部召回器集成

### 6.1 概述 (v4.4.0)

AGA 提供标准的 `BaseRetriever` 协议，允许连接到任何外部知识源（Chroma、Milvus、Elasticsearch、自定义数据库等）。当熵值较高时，AGA 自动查询召回器并将检索到的知识注入 KVStore。

### 6.2 实现自定义召回器

```python
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult
import torch

class ChromaRetriever(BaseRetriever):
    def __init__(self, collection_name: str):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_collection(collection_name)
    
    def retrieve(self, query: RetrievalQuery) -> list:
        # 将查询张量转换为嵌入向量进行搜索
        query_vec = query.query_projected.cpu().numpy().tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=query.top_k,
        )
        
        retrieval_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            retrieval_results.append(RetrievalResult(
                id=doc_id,
                key=torch.tensor(results['embeddings'][0][i][:64]),
                value=torch.tensor(results['embeddings'][0][i]),
                score=results['distances'][0][i],
                reliability=0.9,
            ))
        return retrieval_results
    
    def warmup(self):
        # 预热连接
        self.collection.count()
    
    def shutdown(self):
        pass
```

### 6.3 使用召回器

```python
from aga import AGAPlugin, AGAConfig

retriever = ChromaRetriever("medical_knowledge")
plugin = AGAPlugin(
    AGAConfig(hidden_dim=4096, retriever_auto_inject=True),
    retriever=retriever,
)
plugin.attach(model)

# AGA 在高熵时自动查询召回器
output = model.generate(input_ids)
```

### 6.4 内置召回器

| 召回器             | 说明                                          |
| ------------------ | --------------------------------------------- |
| `NullRetriever`    | 默认空操作召回器（返回空结果）                |
| `KVStoreRetriever` | 简单召回器，在现有 KVStore 中搜索             |

### 6.5 Slot 治理

使用外部召回器时，AGA 的 Slot 治理系统防止 Slot 抖动：

- **Slot 预算**: 限制召回器可使用的 Slot 数量（可配置比例）
- **语义去重**: 防止注入语义相似的知识（余弦相似度阈值）
- **冷却期**: 两次召回之间的最小步数间隔
- **稳定性检测**: 如果 KVStore 变化过快则暂停召回
- **知识锁定**: 通过 `register()` 注册的核心知识受保护，不会被召回器结果淘汰

```yaml
slot_governance:
    pin_registered: true              # 自动锁定注册的知识
    retriever_slot_ratio: 0.3         # 最多 30% 的 Slot 用于召回器
    retriever_cooldown_steps: 5       # 两次召回间最少 5 步
    retriever_dedup_similarity: 0.95  # 去重阈值
    slot_stability_threshold: 0.5     # 稳定性检测阈值
```

---

## 7. 配置管理

### 7.1 配置创建方式

```python
from aga import AGAConfig

# 方式 1: 直接创建（使用默认值）
config = AGAConfig(hidden_dim=4096)

# 方式 2: 从 YAML 文件加载
config = AGAConfig.from_yaml("aga_config.yaml")

# 方式 3: 从字典创建（支持嵌套）
config = AGAConfig.from_dict({
    "hidden_dim": 4096,
    "gate": {
        "tau_low": 0.3,
        "tau_high": 2.5,
        "gate2_top_k": 16,
    },
    "decay": {
        "enabled": True,
        "strategy": "exponential",
        "gamma": 0.85,
    },
    "retriever": {
        "backend": "null",
        "top_k": 5,
    },
    "slot_governance": {
        "pin_registered": True,
        "retriever_dedup_similarity": 0.95,
    },
})

# 方式 4: 关键字参数
plugin = AGAPlugin(
    hidden_dim=4096,
    bottleneck_dim=64,
    max_slots=512,
    tau_low=0.3,
)
```

### 7.2 配置验证

```python
config = AGAConfig(hidden_dim=4096, bottleneck_dim=8192)  # 错误: bottleneck > hidden
errors = config.validate()
if errors:
    for err in errors:
        print(f"配置错误: {err}")
```

### 7.3 运行时调节

```python
# 调节熵门控阈值
plugin.gate_system.update_thresholds(
    tau_low=0.3,
    tau_high=2.5,
    max_gate=0.7,
)

# 重置衰减上下文（新推理请求时）
plugin.reset_decay_contexts()
```

### 7.4 完整 YAML 配置参考

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

    # ===== 召回器 (v4.4.0) =====
    retriever:
        backend: "null"              # "null", "kvstore" 或自定义
        endpoint: ""                 # 远程召回器地址
        collection: ""               # 集合/索引名称
        top_k: 5
        min_score: 0.3
        query_source: "q_proj"       # "q_proj" 或 "hidden_states"
        auto_inject: true
        cache_ttl: 300
        timeout_ms: 10

    # ===== Slot 治理 (v4.4.0) =====
    slot_governance:
        pin_registered: true
        retriever_slot_ratio: 0.3
        retriever_slot_budget: 0     # 0 = 从 ratio 自动计算
        retriever_cooldown_steps: 5
        retriever_dedup_similarity: 0.95
        slot_stability_threshold: 0.5

    # ===== 安全 =====
    fail_open: true
    max_forward_timeout_ms: 50

    # ===== 范数控制 =====
    enable_norm_clipping: true
    key_norm_target: 5.0
    value_norm_target: 3.0

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

    # ===== 流式生成 =====
    streaming:
        diagnostics_buffer_size: 1000

    # ===== 知识源 =====
    knowledge_sources:
        - type: jsonl
          path: "data/base_knowledge.jsonl"
```

---

## 8. 诊断与监控

### 8.1 运行状态诊断

```python
diag = plugin.get_diagnostics()

# 基本状态
print(f"已挂载: {diag['attached']}")
print(f"知识数量: {diag['knowledge_count']}/{diag['max_slots']}")
print(f"激活率: {diag['activation_rate']:.2%}")
print(f"平均门控值: {diag['gate_mean_avg']:.4f}")
print(f"平均熵值: {diag['entropy_mean_avg']:.4f}")

# Slot 治理指标 (v4.4.0)
print(f"已锁定数量: {diag.get('pinned_count', 0)}")
print(f"召回器预算: {diag.get('retriever_budget', 0)}")
print(f"召回步数: {diag.get('retrieval_step_counter', 0)}")

# 延迟
if 'latency_p95_us' in diag:
    print(f"P95 延迟: {diag['latency_p95_us']:.1f} us")
```

### 8.2 审计日志

```python
# 获取最近 50 条审计日志
trail = plugin.get_audit_trail(limit=50)
for entry in trail:
    print(f"[{entry['operation']}] 成功={entry['success']} 详情={entry['details']}")

# 按操作类型过滤
registers = plugin.get_audit_trail(limit=100, operation="register")
```

### 8.3 KV 存储统计

```python
stats = plugin.get_store_stats()
print(f"知识数量: {stats['count']}")
print(f"已锁定: {stats['pinned_count']}")
print(f"未锁定: {stats['unpinned_count']}")
print(f"VRAM 占用: {stats['vram_bytes'] / 1024 / 1024:.2f} MB")
```

### 8.4 事件总线

```python
# 订阅事件（用于自定义监控）
def my_handler(event):
    if event.data.get("aga_applied"):
        print(f"AGA 注入! gate={event.data['gate_mean']:.4f}")

plugin.event_bus.subscribe("forward", my_handler)
```

### 8.5 与 aga-observability 集成

```python
# 自动检测并集成（无需额外代码）
plugin = AGAPlugin(AGAConfig(
    hidden_dim=4096,
    observability_enabled=True,
    prometheus_enabled=True,
    prometheus_port=9090,
))
# Prometheus 指标自动可用于: http://localhost:9090/metrics
```

---

## 9. 高级用法

### 9.1 多模型共享知识

```python
from aga import AGAPlugin, AGAConfig, KVStore

# 创建共享 KVStore
shared_store = KVStore(max_slots=1000, key_dim=64, value_dim=4096)
shared_store.put("fact_001", key_tensor, value_tensor, reliability=0.95)

# 创建多个插件共享同一存储
plugin_a = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin_a.store = shared_store
plugin_a.attach(model_a)

plugin_b = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin_b.store = shared_store
plugin_b.attach(model_b)
```

### 9.2 命名空间隔离

```python
plugin.register("med_001", key=k1, value=v1,
                metadata={"namespace": "cardiology"})
plugin.register("law_001", key=k2, value=v2,
                metadata={"namespace": "contract_law"})

# 清空特定命名空间
plugin.clear(namespace="cardiology")
```

### 9.3 动态知识更新

```python
# 更新知识（相同 ID 会覆盖）
plugin.register("fact_001", key=new_key, value=new_value)

# 移除过时知识
plugin.unregister("outdated_fact")

# 添加新知识
plugin.register("breaking_news", key=news_key, value=news_value, reliability=0.8)
```

### 9.4 张量并行 (v4.4.0)

```python
from aga.distributed import TPManager

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
tp_manager = TPManager(plugin)

# 在主 rank 上注册知识
if tp_manager.is_primary:
    plugin.register("fact_001", key=k, value=v)

# 广播到所有 rank
tp_manager.broadcast_knowledge()
tp_manager.broadcast_parameters()
```

### 9.5 自定义事件处理

```python
import json
from pathlib import Path

class FileAuditHandler:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event):
        with open(self.path, "a") as f:
            f.write(json.dumps({
                "type": event.type,
                "timestamp": event.timestamp,
                "data": event.data,
            }) + "\n")

handler = FileAuditHandler("logs/aga_audit.jsonl")
plugin.event_bus.subscribe("audit", handler)
```

---

## 10. 生产部署

### 10.1 部署检查清单

- [ ] 确认 `hidden_dim` 匹配目标模型
- [ ] 配置适当的 `max_slots`（根据知识量）
- [ ] 启用 `fail_open: true`（默认已启用）
- [ ] 启用 `enable_norm_clipping: true`（默认已启用）
- [ ] 调优 `tau_low` 和 `tau_high`（根据领域特征）
- [ ] 如使用外部知识源，配置召回器
- [ ] 如召回器负载较重，配置 Slot 治理参数
- [ ] 如使用 vLLM，运行 `VLLMAdapter.check_compatibility()` 检查兼容性
- [ ] 如使用 TP，设置 `TPManager` 进行知识同步
- [ ] 考虑安装 `aga-observability` 进行 Prometheus 监控

### 10.2 推荐配置

**通用场景**:

```yaml
aga:
    hidden_dim: 4096
    max_slots: 256
    fail_open: true
    gate:
        tau_low: 0.5
        tau_high: 2.0
        early_exit_threshold: 0.05
    decay:
        enabled: true
        gamma: 0.9
```

**垂直领域（高激活率）**:

```yaml
aga:
    hidden_dim: 4096
    max_slots: 1000
    gate:
        tau_low: 0.3
        tau_high: 3.0
        gate2_top_k: 16
        early_exit_threshold: 0.02
    decay:
        gamma: 0.95
    slot_governance:
        pin_registered: true
        retriever_slot_ratio: 0.4
        retriever_cooldown_steps: 3
```

**低延迟场景**:

```yaml
aga:
    hidden_dim: 4096
    max_slots: 128
    gate:
        gate2_top_k: 4
        early_exit_threshold: 0.1
    instrumentation:
        instrumentation_enabled: false
```

### 10.3 性能调优

| 参数                   | 增大效果             | 减小效果            |
| ---------------------- | -------------------- | ------------------- |
| `max_slots`            | 更多知识，更高 VRAM  | 更少知识，更低 VRAM |
| `gate2_top_k`          | 更精确匹配，更高延迟 | 更快匹配，可能遗漏  |
| `tau_low`              | 更少激活，更低延迟   | 更多激活，更多注入  |
| `tau_high`             | 更宽容的高熵处理     | 更严格的高熵限制    |
| `decay_gamma`          | 更慢衰减，更持久影响 | 更快衰减，更短影响  |
| `early_exit_threshold` | 更积极的 Early Exit  | 更多完整注入        |
| `retriever_cooldown`   | 更少召回，更稳定     | 更多召回，更新鲜    |
| `retriever_dedup_sim`  | 更多去重，更少多样性 | 更少去重，更多多样性 |

---

## 11. 故障排除

### 11.1 常见问题

**问题: AGA 从不激活（激活率 = 0%）**

可能原因:
1. 没有注册知识 -> 检查 `plugin.knowledge_count`
2. `tau_low` 设置过高 -> 降低 `tau_low`
3. 模型对当前输入很确信 -> 尝试更具挑战性的输入
4. `early_exit_threshold` 设置过高 -> 降低阈值

**问题: AGA 总是激活（激活率 ~ 100%）**

可能原因:
1. `tau_low` 设置过低 -> 提高 `tau_low`
2. 知识的 key 向量与所有输入都高度匹配 -> 检查 key 向量质量

**问题: 推理质量下降**

可能原因:
1. 知识的 value 向量质量差 -> 检查编码质量
2. `max_gate` 设置过高 -> 降低 `max_gate`
3. 衰减不足 -> 降低 `decay_gamma`
4. 召回器导致 Slot 抖动 -> 增加 `retriever_cooldown_steps` 或降低 `retriever_slot_ratio`

**问题: VRAM 不足**

解决方案:
1. 减少 `max_slots`
2. 减少挂载层数
3. 使用 CPU 设备（`device: "cpu"`）

### 11.2 日志调试

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("aga").setLevel(logging.DEBUG)
```

### 11.3 Fail-Open 行为

当 AGA 遇到任何异常时，如果 `fail_open=True`（默认），AGA 会：
1. 记录警告日志
2. 返回原始模型输出（不注入任何知识）
3. 继续正常工作

---

## 12. FAQ

### Q1: aga-core 可以独立使用吗？

**是的。** `aga-core` 的唯一依赖是 `torch>=2.0.0`。通过标准召回器协议（v4.4.0），您可以连接到任何外部知识源，无需 `aga-knowledge`。

### Q2: AGA 会修改模型参数吗？

**不会。** AGA 通过 `register_forward_hook` 挂载到模型，不修改任何模型权重。

### Q3: AGA 对推理速度的影响有多大？

在大多数场景下，AGA 的旁路率 >60%，对整体推理速度的影响 <5%。

### Q4: AGA 在一次推理中会被多次激活吗？

**是的。** AGA 在每个挂载层的每个 token 位置都会独立评估。在垂直领域场景下，AGA 可能在多个 token 位置和多个层上被激活。

### Q5: 流式生成中 AGA 如何工作？

AGA 通过 hook 机制自动在每个 decode step 中工作。`StreamingSession` 提供会话管理和实时诊断能力，但 AGA 的核心注入逻辑不需要特殊的流式 API。

### Q6: AGA 与 LoRA 可以同时使用吗？

**可以。** AGA 通过 hook 注入，LoRA 通过适配器权重注入，两者互不冲突。

### Q7: 支持哪些模型？

内置支持所有主流 HuggingFace 模型架构和 vLLM 推理框架。对于非标准架构，可以通过实现 `LLMAdapter` 接口来支持。

### Q8: AGA 支持 vLLM 吗？

**是的。** AGA 提供原生 `VLLMAdapter`，无需 fork vLLM。通过 `VLLMAdapter.extract_model()` 提取 vLLM 内部模型后，使用标准 `plugin.attach()` 即可挂载。

### Q9: 什么是知识锁定？

锁定的知识受到保护，不会被 LRU 淘汰。当 `pin_registered=True`（默认）时，通过 `register()` 注册的所有知识自动锁定。由召回器注入的知识不会被锁定，确保核心知识始终优先。

### Q10: 召回器协议如何工作？

实现 `BaseRetriever` 接口的 `retrieve()` 方法。AGA 在高熵时自动调用，受 Slot 治理规则（预算、冷却、去重）约束。这允许 AGA 连接到任何现有基础设施，无需 `aga-knowledge`。

### Q11: 如何选择 max_slots？

| 知识量       | 推荐 max_slots    | VRAM (hidden_dim=4096) |
| ------------ | ----------------- | ---------------------- |
| <100 条      | 128               | ~1 MB                  |
| 100-500 条   | 256-512           | ~2-4 MB                |
| 500-2000 条  | 1000              | ~8 MB                  |
| >5000 条     | 使用召回器 + slots | 按需加载               |
