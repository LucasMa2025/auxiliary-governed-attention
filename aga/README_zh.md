# AGA — Auxiliary Governed Attention

<p align="center">
  <strong>极简注意力治理插件</strong><br/>
  为冻结 LLM 提供推理时动态知识注入的无损能力扩展
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-4.4.0-blue" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license"/>
  <img src="https://img.shields.io/badge/torch-2.0+-red" alt="torch"/>
</p>

---

## 产品定位

**AGA (Auxiliary Governed Attention)** 是一个面向冻结大语言模型 (LLM) 的**推理时注意力治理插件**。其核心目标是**为冻结模型提供无损的能力扩展**。

在 LLM 推理过程中，当模型遇到其参数中未编码的知识（表现为高熵/高不确定性）时，AGA 会自动介入，将外部知识通过 Bottleneck Attention 机制注入到 Transformer 的注意力层中，从而在**不修改模型参数**的前提下，动态补充模型的知识盲区。

**AGA 不是 RAG，不是 LoRA，不是 Prompt Engineering。**

| 维度     | RAG                    | LoRA                 | AGA                             |
| -------- | ---------------------- | -------------------- | ------------------------------- |
| 介入时机 | 推理前（拼接 context） | 训练时（微调参数）   | 推理中（注意力层实时注入）      |
| 修改模型 | 否                     | 是（增加适配器权重） | 否（纯 hook，零参数修改）       |
| 知识粒度 | 文档/段落级            | 全局知识             | 原子事实级（10-50 tokens/slot） |
| 动态性   | 静态检索               | 需重新训练           | 实时增删，秒级生效              |
| 决策依据 | 用户查询相似度         | 无（始终生效）       | 模型内部熵信号（自适应）        |

**产品方向：** AGA 致力于成为 LLM 生态中一个**极简、纯粹、不可或缺的注意力治理标准组件**，特别适用于：

-   **垂直领域私有知识系统**：医疗、法律、金融等领域的专业知识实时注入
-   **动态知识更新场景**：新闻、政策、产品信息等需要实时更新的知识
-   **多租户知识隔离**：不同用户/租户拥有独立的知识空间
-   **模型知识补丁**：快速修复模型的事实性错误，无需重新训练
-   **流式生成场景**：在 token-by-token 生成过程中持续注入知识

---

## 核心价值

1. **推理时介入，零参数修改** — 通过 `register_forward_hook` 挂载到任意 HuggingFace 模型，不修改任何模型权重
2. **熵驱动自适应** — 三段式熵门控（Gate-0/1/2）确保只在模型不确定时才注入，避免干扰模型已有的正确推理
3. **极低延迟** — Bottleneck 注入路径 <0.1ms，对推理速度几乎无影响
4. **GPU 常驻** — KV 知识存储预分配在 GPU 上，256 slots 仅占 ~2MB VRAM
5. **Fail-Open 安全** — 任何异常自动回退到原始模型输出，保证生产安全
6. **3 行集成** — 最简使用只需 3 行代码
7. **流式生成支持** — 原生支持 token-by-token 流式生成中的动态知识注入
8. **标准召回器协议** — 通过 `BaseRetriever` 接口实现可插拔的外部知识检索
9. **Slot 治理** — 知识锁定、预算控制、语义去重、自适应召回，防止 Slot 抖动

---

## 快速开始

### 3 行集成

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)                    # 挂载到 HuggingFace 模型
output = model.generate(input_ids)      # AGA 自动工作
```

### 知识注册

```python
import torch

# 注册知识（pinned=True 保护核心知识不被淘汰）
plugin.register(
    id="fact_001",
    key=torch.randn(64),       # [bottleneck_dim] 检索键
    value=torch.randn(4096),   # [hidden_dim] 知识向量
    reliability=0.95,
    pinned=True,               # 锁定核心知识 (v4.4.0)
    metadata={"source": "medical_kb", "namespace": "cardiology"}
)

# 批量注册
plugin.register_batch([
    {"id": "fact_002", "key": k2, "value": v2, "reliability": 0.9},
    {"id": "fact_003", "key": k3, "value": v3, "reliability": 0.85},
])

# 从 JSONL 文件加载（内置，无需 aga-knowledge）
plugin.load_from("knowledge.jsonl")
```

### 流式生成注入

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)

# 流式生成 — AGA 在每个 token 生成时自动介入
streamer = plugin.create_streaming_session()
for token_output in model_generate_stream(input_ids):
    diag = streamer.get_step_diagnostics()
    if diag["aga_applied"]:
        print(f"Token {diag['step']}: AGA 注入, gate={diag['gate_mean']:.4f}")

summary = streamer.get_session_summary()
print(f"总 token 数: {summary['total_steps']}, 注入率: {summary['injection_rate']:.2%}")
```

### 外部召回器集成 (v4.4.0)

```python
from aga import AGAPlugin, AGAConfig
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult

# 实现自定义召回器（如基于 Chroma、Milvus 等）
class MyRetriever(BaseRetriever):
    def retrieve(self, query: RetrievalQuery) -> list:
        # 你的检索逻辑
        return [RetrievalResult(id="doc_1", key=k, value=v, score=0.95)]

plugin = AGAPlugin(AGAConfig(hidden_dim=4096), retriever=MyRetriever())
plugin.attach(model)
# AGA 在高熵时自动调用召回器获取知识
```

### 配置驱动

```python
# 从 YAML 文件创建
plugin = AGAPlugin.from_config("aga_config.yaml")
plugin.attach(model)

# 从字典创建
plugin = AGAPlugin.from_config({
    "hidden_dim": 4096,
    "bottleneck_dim": 64,
    "max_slots": 512,
    "device": "cuda",
    "retriever": {"backend": "null", "top_k": 5},
    "slot_governance": {"pin_registered": True, "retriever_dedup_similarity": 0.95},
})
```

---

## 架构概览

```
+-------------------------------------------------------------+
|                        AGAPlugin                             |
|  (唯一入口类)                                                 |
|                                                              |
|  +----------+  +--------------+  +------------------+        |
|  | KVStore  |  | EntropyGate  |  | BottleneckInject |        |
|  | GPU常驻   |  | 三段式门控   |  | 核心注入路径      |        |
|  | LRU+锁定  |  | Gate-0/1/2  |  | Top-K路由        |        |
|  | 命名空间  |  | 熵否决       |  | Value投影        |        |
|  +----------+  +--------------+  +------------------+        |
|                                                              |
|  +--------------+  +----------+  +------------------+        |
|  | Persistence  |  | Adapter  |  | Instrumentation  |        |
|  | Decay        |  | HF/vLLM  |  | EventBus         |        |
|  | 线程隔离     |  | 自动检测   |  | ForwardMetrics  |        |
|  | 硬重置       |  | Hook注入   |  | AuditLog        |        |
|  +--------------+  +----------+  +------------------+        |
|                                                              |
|  +--------------+  +----------+  +------------------+        |
|  | Retriever    |  | Slot治理  |  | Distributed      |        |
|  | BaseRetriever|  | 预算控制   |  | TPManager        |        |
|  | Null/KVStore |  | 语义去重   |  | 广播参数         |        |
|  | 可插拔       |  | 冷却期     |  | 广播KV           |        |
|  +--------------+  +----------+  +------------------+        |
|                                                              |
|  +------------------------------------------------------+   |
|  |              StreamingSession                          |   |
|  |  流式生成会话管理 . 逐 token 诊断 . 会话统计            |   |
|  +------------------------------------------------------+   |
+-------------------------------------------------------------+
```

### 推理流程

```
Token -> Transformer Layer -> Self-Attention Output
                                    |
                              +-----v-----+
                              |  Gate-0    | 先验检查 (namespace)
                              |  通过？     |
                              +-----+-----+
                                    | Yes
                              +-----v-----+
                              |  Gate-1    | 熵计算 + 置信门控
                              |  H > t_low?|
                              +-----+-----+
                                    | Yes
                              +-----v-----+
                              | Retriever  | 外部知识召回
                              | (配置驱动)  | (含 Slot 治理)
                              +-----+-----+
                                    |
                              +-----v-----+
                              | Bottleneck | Query投影 -> Top-K路由
                              | Injection  | -> 注意力计算 -> Value投影
                              +-----+-----+
                                    |
                              +-----v-----+
                              |  Decay     | 跨层衰减（线程隔离）
                              +-----+-----+
                                    |
                              +-----v-----+
                              |  Fusion    | output + gate x aux_output
                              +-----+-----+
                                    |
                              Final Output
```

---

## 已实现功能

### aga-core v4.4.0

| 模块                   | 功能                                           | 状态 |
| ---------------------- | ---------------------------------------------- | ---- |
| **AGAPlugin**          | 3 行集成入口类                                 | 完成 |
| **AGAPlugin**          | `from_config()` YAML/Dict 配置驱动             | 完成 |
| **AGAPlugin**          | `register()` 支持 `pinned` 参数                | 完成 |
| **AGAPlugin**          | `register_batch` / `load_from()` 知识加载      | 完成 |
| **AGAPlugin**          | `attach/detach` 模型挂载/卸载                  | 完成 |
| **AGAPlugin**          | `get_diagnostics()` 含 Slot 治理指标           | 完成 |
| **AGAPlugin**          | `get_audit_trail()` 审计日志查询               | 完成 |
| **AGAPlugin**          | Fail-Open 安全回退                             | 完成 |
| **AGAPlugin**          | 外部召回器集成 (`BaseRetriever`)               | 完成 |
| **AGAPlugin**          | Slot 治理（预算、去重、冷却、稳定性）          | 完成 |
| **StreamingSession**   | `create_streaming_session()` 流式会话管理      | 完成 |
| **StreamingSession**   | 逐 token 诊断（含层事件过滤）                  | 完成 |
| **StreamingSession**   | 会话统计 (`get_session_summary`)               | 完成 |
| **StreamingSession**   | 动态知识热更新 (`update_knowledge`)            | 完成 |
| **KVStore**            | GPU 预分配内存 + `get_active()` 缓存           | 完成 |
| **KVStore**            | LRU 淘汰 + 知识锁定 (`pin`/`unpin`)            | 完成 |
| **KVStore**            | 命名空间隔离、线程安全                         | 完成 |
| **EntropyGateSystem**  | Gate-0/1/2 三段式熵门控                        | 完成 |
| **EntropyGateSystem**  | Early Exit 优化                                | 完成 |
| **BottleneckInjector** | Query 投影 + Top-K 路由 + Value 投影           | 完成 |
| **PersistenceDecay**   | 指数/线性/自适应衰减（线程隔离）               | 完成 |
| **Retriever**          | `BaseRetriever` 标准协议                       | 完成 |
| **Retriever**          | `NullRetriever` / `KVStoreRetriever` 内置实现  | 完成 |
| **Distributed**        | `TPManager` 张量并行广播支持                   | 完成 |
| **HuggingFaceAdapter** | LLaMA/Qwen/Mistral/GPT-2/Phi/Gemma/Falcon 支持 | 完成 |
| **VLLMAdapter**        | vLLM 推理框架支持（不需要 fork）               | 完成 |
| **VLLMHookWorker**     | IBM vLLM-Hook 插件系统兼容                     | 完成 |
| **EventBus**           | 内存环形缓冲区、可插拔订阅者                   | 完成 |
| **ForwardMetrics**     | 激活率/门控值/熵值/延迟 + P50/P95/P99          | 完成 |
| **AuditLog**           | 知识操作审计轨迹（含过滤）                     | 完成 |
| **AGAConfig**          | 全参数外置 + 嵌套配置展平                      | 完成 |

---

## 独立使用说明

**`aga-core` 可以完全独立使用，无需安装 `aga-knowledge` 或 `aga-observability`。**

`aga-core` 的唯一依赖是 `torch>=2.0.0`。通过标准召回器协议（v4.4.0），用户可以将 AGA 连接到现有基础设施（Chroma、Milvus、Elasticsearch 等），无需 `aga-knowledge`。

| 能力                  | 独立使用 (aga-core only)                  | 配合 aga-knowledge |
| --------------------- | ----------------------------------------- | ------------------ |
| 知识注册 (KV 向量)    | `register(id, key, value, pinned=True)`   | 支持               |
| 外部召回器集成        | `BaseRetriever` 协议                      | 支持               |
| 批量注册 / JSONL 加载 | `register_batch()` / `load_from()`        | 支持               |
| 模型挂载              | `attach(model)`                           | 支持               |
| 熵门控推理            | 自动工作                                  | 支持               |
| 流式生成注入          | `create_streaming_session()`              | 支持               |
| 诊断与审计            | `get_diagnostics()` / `get_audit_trail()` | 支持               |
| Slot 治理             | 预算、去重、锁定、冷却                    | 支持               |
| 明文知识管理          | 不支持                                    | Portal API         |
| 多实例同步            | 不支持                                    | Redis/Kafka        |
| 持久化存储            | 不支持                                    | SQLite/PostgreSQL  |
| REST API              | 不支持                                    | FastAPI Portal     |

---

## 配置参考

### YAML 配置示例

```yaml
aga:
    # 模型维度（必须匹配目标模型）
    hidden_dim: 4096
    bottleneck_dim: 64
    num_heads: 32
    value_bottleneck_dim: 256

    # 容量
    max_slots: 256

    # 设备
    device: "cuda"

    # 熵门控
    gate:
        gate0_enabled: true
        gate1_enabled: true
        gate1_uncertainty_source: "hidden_variance"
        gate2_top_k: 8
        tau_low: 0.5
        tau_high: 2.0
        max_gate: 0.8
        early_exit_enabled: true
        early_exit_threshold: 0.05

    # 衰减
    decay:
        enabled: true
        strategy: "exponential"
        gamma: 0.9
        hard_reset_threshold: 3.0

    # 召回器 (v4.4.0)
    retriever:
        backend: "null" # "null", "kvstore" 或自定义
        top_k: 5
        min_score: 0.3
        auto_inject: true
        timeout_ms: 10

    # Slot 治理 (v4.4.0)
    slot_governance:
        pin_registered: true
        retriever_slot_ratio: 0.3
        retriever_cooldown_steps: 5
        retriever_dedup_similarity: 0.95
        slot_stability_threshold: 0.5

    # 安全
    fail_open: true
    max_forward_timeout_ms: 50

    # 范数控制
    enable_norm_clipping: true
    key_norm_target: 5.0
    value_norm_target: 3.0

    # 埋点与审计
    instrumentation:
        instrumentation_enabled: true
        event_buffer_size: 10000
        audit_log_level: "INFO"

    # 流式生成
    streaming:
        diagnostics_buffer_size: 1000
```

### 常用模型配置

| 模型       | hidden_dim | bottleneck_dim | num_heads | 推荐 max_slots |
| ---------- | ---------- | -------------- | --------- | -------------- |
| LLaMA-7B   | 4096       | 64             | 32        | 256            |
| LLaMA-13B  | 5120       | 64             | 40        | 256            |
| LLaMA-70B  | 8192       | 128            | 64        | 512            |
| Qwen-7B    | 4096       | 64             | 32        | 256            |
| Qwen-72B   | 8192       | 128            | 64        | 512            |
| Mistral-7B | 4096       | 64             | 32        | 256            |
| GPT-2      | 768        | 32             | 12        | 128            |
| Phi-2      | 2560       | 48             | 32        | 256            |

### VRAM 占用参考

| max_slots | hidden_dim=4096 | hidden_dim=8192 |
| --------- | --------------- | --------------- |
| 128       | ~1.0 MB         | ~2.0 MB         |
| 256       | ~2.0 MB         | ~4.0 MB         |
| 512       | ~4.1 MB         | ~8.1 MB         |
| 1000      | ~8.1 MB         | ~16.1 MB        |
| 5000      | ~40.6 MB        | ~81.0 MB        |

---

## 生态系统

AGA 采用三包分离架构，`aga-core` 是唯一必需包：

```
+-----------------------------------------------------+
|                   AGA 生态系统                        |
|                                                      |
|  +-------------+                                     |
|  |  aga-core   | <- 必需                              |
|  |  注意力治理   |    pip install aga-core             |
|  |  召回器协议   |    唯一依赖: torch                   |
|  |  知识注入     |                                     |
|  |  流式生成     |                                     |
|  +------+------+                                     |
|         |                                            |
|  +------v------+  +--------------------+             |
|  |aga-knowledge|  | aga-observability  |             |
|  | 知识管理     |  | 可观测性           | <- 可选      |
|  | Portal API  |  | Prometheus/Grafana |             |
|  | 持久化/同步  |  | SLO/SLI 监控      |             |
|  +-------------+  +--------------------+             |
+-----------------------------------------------------+
```

| 包                    | 用途                                    | 依赖                                |
| --------------------- | --------------------------------------- | ----------------------------------- |
| **aga-core**          | 注意力治理引擎 + 召回器协议 + 流式生成  | `torch>=2.0.0`                      |
| **aga-knowledge**     | 知识管理、Portal API、持久化、同步      | `aga-core`, `fastapi`, `sqlalchemy` |
| **aga-observability** | Prometheus 指标、Grafana 面板、SLO 监控 | `aga-core`, `prometheus-client`     |

---

## 适配器支持

### 已实现

| 适配器                 | 状态 | 说明                                                |
| ---------------------- | ---- | --------------------------------------------------- |
| **HuggingFaceAdapter** | 完成 | 支持所有主流 HF 模型（LLaMA/Qwen/Mistral/GPT-2 等） |
| **VLLMAdapter**        | 完成 | 支持 vLLM 推理框架，不需要 fork vLLM                |
| **VLLMHookWorker**     | 完成 | 兼容 IBM vLLM-Hook 插件系统                         |

### vLLM 适配器

**不需要 fork vLLM。** AGA 通过访问 vLLM 内部的 `nn.Module`，使用标准 `register_forward_hook` 注入。

```python
from vllm import LLM, SamplingParams
from aga import AGAPlugin, AGAConfig
from aga.adapter.vllm import VLLMAdapter

llm = LLM(model="meta-llama/Llama-3-8B", enforce_eager=True)
model = VLLMAdapter.extract_model(llm)

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model, adapter=VLLMAdapter())
outputs = llm.generate(["请解释量子纠缠"], SamplingParams(max_tokens=256))
```

**已知限制**:

| 限制                   | 解决方案                                         |
| ---------------------- | ------------------------------------------------ |
| **CUDA Graph**         | 使用 `enforce_eager=True`（推荐）                |
| **连续批处理**         | AGA 在 hidden_states 维度操作，自然支持 batch    |
| **Tensor Parallelism** | 使用 `TPManager` 在各 rank 间同步 KVStore        |
| **vLLM 版本**          | `extract_model` 支持多种路径，覆盖 vLLM >= 0.4.x |

### TensorRT-LLM 适配器

**可行性**: 有限 — TensorRT-LLM 将模型编译为静态图，不兼容 Python 级 hook。需要 C++/CUDA 自定义插件开发。建议使用 vLLM 替代。

---

## 未来目标

### v4.4.0 — 当前版本

-   [x] **召回器标准协议** — `BaseRetriever` 接口实现可插拔知识检索
-   [x] **Slot 治理** — 预算、去重、锁定、冷却、稳定性检测
-   [x] **知识锁定** — `pin()`/`unpin()` 保护核心知识不被淘汰
-   [x] **线程隔离衰减** — `threading.local()` 保证多线程推理安全
-   [x] **`get_active()` 缓存** — 避免 KVStore 重复创建张量
-   [x] **TPManager** — 张量并行支持，分布式部署
-   [x] **StreamingSession 层过滤** — 精确的逐 token 事件计数
-   [x] **vLLM Adapter** — 支持 vLLM 推理框架的适配器
-   [x] **IBM vLLM-Hook 兼容** — 无需 fork vLLM 的插件集成

### v5.0 — 下一版本

-   [ ] **知识编码器集成** — 内置轻量编码器，支持文本到 KV 的自动转换
-   [ ] **多层独立知识** — 不同 Transformer 层使用不同的知识子集
-   [ ] **INT8 量化 KVStore** — 进一步降低 VRAM 占用
-   [ ] **aga-observability v1.0** — Prometheus + Grafana 完整可观测性
-   [ ] **跨模型知识迁移** — 在不同模型间共享 AGA 知识
-   [ ] **自适应 Bottleneck** — 根据知识复杂度动态调整 bottleneck_dim

### 长期 (v6.0+)

-   [ ] **AGA 标准协议** — 定义 LLM 注意力治理的标准接口
-   [ ] **硬件加速** — CUDA Kernel 优化的注入路径
-   [ ] **多模态知识** — 支持图像/音频知识的注入
-   [ ] **自监督知识发现** — 自动从推理日志中发现需要补充的知识

---

## 安装

### 基础安装

```bash
# 仅安装核心（唯一依赖: torch）
pip install aga-core

# 从源码安装
cd AGAPlugin
pip install -e .
```

### 可选依赖

```bash
# YAML 配置支持
pip install aga-core[yaml]

# 知识管理系统
pip install aga-core[knowledge]

# 可观测性
pip install aga-core[observability]

# 全部安装
pip install aga-core[all]

# 开发环境
pip install aga-core[dev]
```

### 系统要求

-   Python >= 3.9
-   PyTorch >= 2.0.0
-   CUDA (推荐，CPU 也可运行但性能较低)

---

## 项目结构

```
aga/
├── __init__.py              # 包入口，导出所有公共 API
├── plugin.py                # AGAPlugin — 唯一入口类
├── streaming.py             # StreamingSession — 流式生成会话
├── config.py                # AGAConfig — 配置管理
├── kv_store.py              # KVStore — GPU 常驻 KV 存储（LRU + 锁定）
├── types.py                 # 数据类型定义
├── exceptions.py            # 异常体系
├── distributed.py           # TPManager — 张量并行支持
├── gate/
│   ├── entropy_gate.py      # EntropyGateSystem — 三段式熵门控
│   └── decay.py             # PersistenceDecay — 跨层衰减
├── operator/
│   └── bottleneck_injector.py  # BottleneckInjector — 核心注入器
├── retriever/
│   ├── base.py              # BaseRetriever — 标准召回器协议
│   ├── null_retriever.py    # NullRetriever — 默认空操作召回器
│   └── kvstore_retriever.py # KVStoreRetriever — 简单 KVStore 召回器
├── adapter/
│   ├── base.py              # LLMAdapter — 适配器抽象基类
│   ├── huggingface.py       # HuggingFaceAdapter — HF 适配器
│   └── vllm.py              # VLLMAdapter + VLLMHookWorker — vLLM 适配器
├── instrumentation/
│   ├── event_bus.py         # EventBus — 可插拔事件总线
│   ├── forward_metrics.py   # ForwardMetrics — 指标收集器
│   └── audit_log.py         # AuditLog — 审计日志
└── docs/
    ├── product_doc_zh.md    # 产品文档（中文）
    ├── product_doc_en.md    # 产品文档（English）
    ├── user_manual_zh.md    # 用户手册（中文）
    └── user_manual_en.md    # 用户手册（English）
```

---

## 许可证

MIT License

Copyright (c) 2024-2026 AGA Team

---

<p align="center">
  <strong>AGA — 让每一次推理都有知识的力量</strong>
</p>
