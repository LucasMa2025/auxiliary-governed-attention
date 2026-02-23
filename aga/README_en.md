# AGA — Auxiliary Governed Attention

<p align="center">
  <strong>Minimalist Attention Governance Plugin</strong><br/>
  Lossless capability extension for frozen LLMs via runtime dynamic knowledge injection
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-4.4.0-blue" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license"/>
  <img src="https://img.shields.io/badge/torch-2.0+-red" alt="torch"/>
</p>

---

## Product Vision

**AGA (Auxiliary Governed Attention)** is a **runtime attention governance plugin** for frozen Large Language Models (LLMs). Its core objective is to **provide lossless capability extension for frozen models**.

During LLM inference, when the model encounters knowledge not encoded in its parameters (manifested as high entropy/uncertainty), AGA automatically intervenes by injecting external knowledge into the Transformer's attention layers through a Bottleneck Attention mechanism — dynamically filling the model's knowledge gaps **without modifying any model parameters**.

**AGA is not RAG, not LoRA, not Prompt Engineering.**

| Dimension             | RAG                            | LoRA                  | AGA                                       |
| --------------------- | ------------------------------ | --------------------- | ----------------------------------------- |
| Intervention          | Pre-inference (context concat) | Training (fine-tune)  | Mid-inference (attention layer injection)  |
| Modifies Model        | No                             | Yes (adapter weights) | No (pure hooks, zero param changes)        |
| Knowledge Granularity | Document/passage               | Global knowledge      | Atomic facts (10-50 tokens/slot)           |
| Dynamism              | Static retrieval               | Requires retraining   | Real-time add/remove, instant effect       |
| Decision Basis        | Query similarity               | None (always active)  | Model internal entropy signal (adaptive)   |

**Product Direction:** AGA aims to become a **minimalist, pure, and indispensable attention governance standard component** in the LLM ecosystem, particularly suited for:

- **Vertical domain private knowledge systems**: Real-time injection of professional knowledge in healthcare, legal, finance, etc.
- **Dynamic knowledge update scenarios**: News, policies, product information requiring real-time updates
- **Multi-tenant knowledge isolation**: Independent knowledge spaces for different users/tenants
- **Model knowledge patching**: Quickly fix factual errors without retraining
- **Streaming generation**: Continuous knowledge injection during token-by-token generation

---

## Core Value

1. **Runtime intervention, zero parameter modification** — Attaches to any HuggingFace model via `register_forward_hook`, no weight changes
2. **Entropy-driven adaptive** — Three-stage entropy gating (Gate-0/1/2) ensures injection only when the model is uncertain
3. **Ultra-low latency** — Bottleneck injection path <0.1ms, negligible impact on inference speed
4. **GPU-resident** — KV knowledge store pre-allocated on GPU, 256 slots ~ 2MB VRAM
5. **Fail-Open safety** — Any exception automatically falls back to original model output
6. **3-line integration** — Minimal usage requires only 3 lines of code
7. **Streaming generation support** — Native support for dynamic knowledge injection during token-by-token streaming generation
8. **Standard Retriever protocol** — Pluggable external knowledge retrieval via `BaseRetriever` interface
9. **Slot Governance** — Knowledge pinning, budget control, semantic deduplication, and adaptive retrieval to prevent slot thrashing

---

## Quick Start

### 3-Line Integration

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)                    # Attach to HuggingFace model
output = model.generate(input_ids)      # AGA works automatically
```

### Knowledge Registration

```python
import torch

# Register knowledge (pinned=True protects from eviction)
plugin.register(
    id="fact_001",
    key=torch.randn(64),       # [bottleneck_dim] retrieval key
    value=torch.randn(4096),   # [hidden_dim] knowledge vector
    reliability=0.95,
    pinned=True,               # Pin core knowledge (v4.4.0)
    metadata={"source": "medical_kb", "namespace": "cardiology"}
)

# Batch registration
plugin.register_batch([
    {"id": "fact_002", "key": k2, "value": v2, "reliability": 0.9},
    {"id": "fact_003", "key": k3, "value": v3, "reliability": 0.85},
])

# Load from JSONL file (built-in, no aga-knowledge needed)
plugin.load_from("knowledge.jsonl")
```

### Streaming Generation Injection

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)

# Streaming generation — AGA automatically intervenes at each token
streamer = plugin.create_streaming_session()
for token_output in model_generate_stream(input_ids):
    diag = streamer.get_step_diagnostics()
    if diag["aga_applied"]:
        print(f"Token {diag['step']}: AGA injected, gate={diag['gate_mean']:.4f}")

summary = streamer.get_session_summary()
print(f"Total tokens: {summary['total_steps']}, Injection rate: {summary['injection_rate']:.2%}")
```

### External Retriever Integration (v4.4.0)

```python
from aga import AGAPlugin, AGAConfig
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult

# Implement your own retriever (e.g., backed by Chroma, Milvus, etc.)
class MyRetriever(BaseRetriever):
    def retrieve(self, query: RetrievalQuery) -> list:
        # Your retrieval logic here
        return [RetrievalResult(id="doc_1", key=k, value=v, score=0.95)]

plugin = AGAPlugin(AGAConfig(hidden_dim=4096), retriever=MyRetriever())
plugin.attach(model)
# AGA now automatically retrieves knowledge when entropy is high
```

### Config-Driven

```python
# Create from YAML file
plugin = AGAPlugin.from_config("aga_config.yaml")
plugin.attach(model)

# Create from dictionary
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

## Architecture

```
+-------------------------------------------------------------+
|                        AGAPlugin                             |
|  (Single Entry Point)                                        |
|                                                              |
|  +----------+  +--------------+  +------------------+        |
|  | KVStore  |  | EntropyGate  |  | BottleneckInject |        |
|  | GPU-res. |  | 3-Stage Gate |  | Core Injection   |        |
|  | LRU+Pin  |  | Gate-0/1/2   |  | Top-K Routing    |        |
|  | Namespace|  | Entropy Veto |  | Value Projection |        |
|  +----------+  +--------------+  +------------------+        |
|                                                              |
|  +--------------+  +----------+  +------------------+        |
|  | Persistence  |  | Adapter  |  | Instrumentation  |        |
|  | Decay        |  | HF/vLLM  |  | EventBus         |        |
|  | Thread-iso.  |  | Auto-det |  | ForwardMetrics   |        |
|  | Hard Reset   |  | Hook Inj |  | AuditLog         |        |
|  +--------------+  +----------+  +------------------+        |
|                                                              |
|  +--------------+  +----------+  +------------------+        |
|  | Retriever    |  | SlotGov  |  | Distributed      |        |
|  | BaseRetriever|  | Budget   |  | TPManager        |        |
|  | Null/KVStore |  | Dedup    |  | Broadcast Params |        |
|  | Pluggable    |  | Cooldown |  | Broadcast KV     |        |
|  +--------------+  +----------+  +------------------+        |
|                                                              |
|  +------------------------------------------------------+   |
|  |              StreamingSession                          |   |
|  |  Session Management . Per-token Diag . Summary         |   |
|  +------------------------------------------------------+   |
+-------------------------------------------------------------+
```

### Inference Flow

```
Token -> Transformer Layer -> Self-Attention Output
                                    |
                              +-----v-----+
                              |  Gate-0    | Prior check (namespace)
                              |  Pass?     |
                              +-----+-----+
                                    | Yes
                              +-----v-----+
                              |  Gate-1    | Entropy + confidence gate
                              |  H > t_low?|
                              +-----+-----+
                                    | Yes
                              +-----v-----+
                              | Retriever  | External knowledge retrieval
                              | (if config)| (with slot governance)
                              +-----+-----+
                                    |
                              +-----v-----+
                              | Bottleneck | Query proj -> Top-K routing
                              | Injection  | -> Attention -> Value proj
                              +-----+-----+
                                    |
                              +-----v-----+
                              |  Decay     | Cross-layer decay (thread-isolated)
                              +-----+-----+
                                    |
                              +-----v-----+
                              |  Fusion    | output + gate x aux_output
                              +-----+-----+
                                    |
                              Final Output
```

---

## Implemented Features

### aga-core v4.4.0

| Module                 | Feature                                                | Status      |
| ---------------------- | ------------------------------------------------------ | ----------- |
| **AGAPlugin**          | 3-line integration entry class                         | Complete |
| **AGAPlugin**          | `from_config()` YAML/Dict config-driven                | Complete |
| **AGAPlugin**          | `register()` with `pinned` parameter                   | Complete |
| **AGAPlugin**          | `register_batch` / `load_from()` knowledge loading     | Complete |
| **AGAPlugin**          | `attach/detach` model attachment                       | Complete |
| **AGAPlugin**          | `get_diagnostics()` with slot governance metrics       | Complete |
| **AGAPlugin**          | `get_audit_trail()` audit log query                    | Complete |
| **AGAPlugin**          | Fail-Open safety fallback                              | Complete |
| **AGAPlugin**          | External Retriever integration (`BaseRetriever`)       | Complete |
| **AGAPlugin**          | Slot Governance (budget, dedup, cooldown, stability)   | Complete |
| **StreamingSession**   | `create_streaming_session()` session management        | Complete |
| **StreamingSession**   | Per-token diagnostics with layer event filtering       | Complete |
| **StreamingSession**   | Session summary (`get_session_summary`)                | Complete |
| **StreamingSession**   | Dynamic knowledge hot-update (`update_knowledge`)      | Complete |
| **KVStore**            | GPU pre-allocated memory with `get_active()` caching   | Complete |
| **KVStore**            | LRU eviction + Knowledge Pinning (`pin`/`unpin`)       | Complete |
| **KVStore**            | Namespace isolation, thread-safe                       | Complete |
| **EntropyGateSystem**  | Gate-0/1/2 three-stage entropy gating                  | Complete |
| **EntropyGateSystem**  | Early Exit optimization                                | Complete |
| **BottleneckInjector** | Query projection + Top-K routing + Value projection    | Complete |
| **PersistenceDecay**   | Exponential/linear/adaptive decay (thread-isolated)    | Complete |
| **Retriever**          | `BaseRetriever` standard protocol                      | Complete |
| **Retriever**          | `NullRetriever` / `KVStoreRetriever` built-in          | Complete |
| **Distributed**        | `TPManager` for Tensor Parallelism broadcast           | Complete |
| **HuggingFaceAdapter** | LLaMA/Qwen/Mistral/GPT-2/Phi/Gemma/Falcon support     | Complete |
| **VLLMAdapter**        | vLLM framework support (no fork required)              | Complete |
| **VLLMHookWorker**     | IBM vLLM-Hook plugin system compatible                 | Complete |
| **EventBus**           | In-memory ring buffer, pluggable subscribers           | Complete |
| **ForwardMetrics**     | Activation rate/gate/entropy/latency + P50/P95/P99     | Complete |
| **AuditLog**           | Knowledge operation audit trail with filtering         | Complete |
| **AGAConfig**          | Full parameter externalization + nested config flatten  | Complete |

---

## Standalone Usage

**`aga-core` can be used completely standalone without `aga-knowledge` or `aga-observability`.**

The only dependency of `aga-core` is `torch>=2.0.0`. With the standard Retriever protocol (v4.4.0), users can connect AGA to their existing infrastructure (Chroma, Milvus, Elasticsearch, etc.) without needing `aga-knowledge`.

| Capability                          | Standalone (aga-core only)                   | With aga-knowledge       |
| ----------------------------------- | -------------------------------------------- | ------------------------ |
| Knowledge registration (KV vectors) | `register(id, key, value, pinned=True)`      | Yes                      |
| External retriever integration      | `BaseRetriever` protocol                     | Yes                      |
| Batch registration / JSONL loading  | `register_batch()` / `load_from()`           | Yes                      |
| Model attachment                    | `attach(model)`                              | Yes                      |
| Entropy-gated inference             | Automatic                                    | Yes                      |
| Streaming generation injection      | `create_streaming_session()`                 | Yes                      |
| Diagnostics & audit                 | `get_diagnostics()` / `get_audit_trail()`    | Yes                      |
| Slot Governance                     | Budget, dedup, pinning, cooldown             | Yes                      |
| Plaintext knowledge management      | No                                           | Portal API               |
| Multi-instance sync                 | No                                           | Redis/Kafka              |
| Persistent storage                  | No                                           | SQLite/PostgreSQL        |
| REST API                            | No                                           | FastAPI Portal           |

---

## Configuration

### YAML Configuration Example

```yaml
aga:
    # Model dimensions (must match target model)
    hidden_dim: 4096
    bottleneck_dim: 64
    num_heads: 32
    value_bottleneck_dim: 256

    # Capacity
    max_slots: 256

    # Device
    device: "cuda"

    # Entropy gating
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

    # Decay
    decay:
        enabled: true
        strategy: "exponential"
        gamma: 0.9
        hard_reset_threshold: 3.0

    # Retriever (v4.4.0)
    retriever:
        backend: "null"          # "null", "kvstore", or custom
        top_k: 5
        min_score: 0.3
        auto_inject: true
        timeout_ms: 10

    # Slot Governance (v4.4.0)
    slot_governance:
        pin_registered: true
        retriever_slot_ratio: 0.3
        retriever_cooldown_steps: 5
        retriever_dedup_similarity: 0.95
        slot_stability_threshold: 0.5

    # Safety
    fail_open: true
    max_forward_timeout_ms: 50

    # Norm control
    enable_norm_clipping: true
    key_norm_target: 5.0
    value_norm_target: 3.0

    # Instrumentation & audit
    instrumentation:
        instrumentation_enabled: true
        event_buffer_size: 10000
        audit_log_level: "INFO"

    # Streaming
    streaming:
        diagnostics_buffer_size: 1000
```

### Common Model Configurations

| Model      | hidden_dim | bottleneck_dim | num_heads | Recommended max_slots |
| ---------- | ---------- | -------------- | --------- | --------------------- |
| LLaMA-7B   | 4096       | 64             | 32        | 256                   |
| LLaMA-13B  | 5120       | 64             | 40        | 256                   |
| LLaMA-70B  | 8192       | 128            | 64        | 512                   |
| Qwen-7B    | 4096       | 64             | 32        | 256                   |
| Qwen-72B   | 8192       | 128            | 64        | 512                   |
| Mistral-7B | 4096       | 64             | 32        | 256                   |
| GPT-2      | 768        | 32             | 12        | 128                   |
| Phi-2      | 2560       | 48             | 32        | 256                   |

### VRAM Usage Reference

| max_slots | hidden_dim=4096 | hidden_dim=8192 |
| --------- | --------------- | --------------- |
| 128       | ~1.0 MB         | ~2.0 MB         |
| 256       | ~2.0 MB         | ~4.0 MB         |
| 512       | ~4.1 MB         | ~8.1 MB         |
| 1000      | ~8.1 MB         | ~16.1 MB        |
| 5000      | ~40.6 MB        | ~81.0 MB        |

---

## Ecosystem

AGA uses a three-package separation architecture. `aga-core` is the only required package:

```
+-----------------------------------------------------+
|                   AGA Ecosystem                      |
|                                                      |
|  +-------------+                                     |
|  |  aga-core   | <- Required                         |
|  |  Attention   |    pip install aga-core             |
|  |  Governance  |    Only dep: torch                  |
|  |  Retriever   |                                     |
|  |  Streaming   |                                     |
|  +------+------+                                     |
|         |                                            |
|  +------v------+  +--------------------+             |
|  |aga-knowledge|  | aga-observability  |             |
|  | Knowledge   |  | Observability      | <- Optional |
|  | Portal API  |  | Prometheus/Grafana |             |
|  | Persistence |  | SLO/SLI Monitoring |             |
|  +-------------+  +--------------------+             |
+-----------------------------------------------------+
```

| Package               | Purpose                                         | Dependency                          |
| --------------------- | ----------------------------------------------- | ----------------------------------- |
| **aga-core**          | Attention governance engine + retriever + stream | `torch>=2.0.0`                      |
| **aga-knowledge**     | Knowledge management, Portal API, persistence    | `aga-core`, `fastapi`, `sqlalchemy` |
| **aga-observability** | Prometheus metrics, Grafana dashboards, SLO      | `aga-core`, `prometheus-client`     |

---

## Adapter Support

### Implemented

| Adapter                | Status   | Description                                                  |
| ---------------------- | -------- | ------------------------------------------------------------ |
| **HuggingFaceAdapter** | Complete | Supports all major HF models (LLaMA/Qwen/Mistral/GPT-2 etc) |
| **VLLMAdapter**        | Complete | Supports vLLM inference framework, no fork required           |
| **VLLMHookWorker**     | Complete | Compatible with IBM vLLM-Hook plugin system                   |

### vLLM Adapter

**No fork of vLLM required.** AGA accesses vLLM's internal `nn.Module` and uses standard `register_forward_hook`.

```python
from vllm import LLM, SamplingParams
from aga import AGAPlugin, AGAConfig
from aga.adapter.vllm import VLLMAdapter

llm = LLM(model="meta-llama/Llama-3-8B", enforce_eager=True)
model = VLLMAdapter.extract_model(llm)

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model, adapter=VLLMAdapter())
outputs = llm.generate(["Explain quantum entanglement"], SamplingParams(max_tokens=256))
```

**Known limitations**:

| Limitation | Solution |
| ---------- | -------- |
| **CUDA Graph** | Use `enforce_eager=True` (recommended) |
| **Continuous Batching** | AGA operates on hidden_states dimension, naturally supports batch |
| **Tensor Parallelism** | Use `TPManager` for KVStore sync across ranks |
| **vLLM Version** | `extract_model` supports multiple paths, covering vLLM >= 0.4.x |

### TensorRT-LLM Adapter

**Feasibility**: Limited — TensorRT-LLM compiles models into static graphs, incompatible with Python-level hooks. Requires C++/CUDA custom plugin development. Recommended to use vLLM instead.

---

## Roadmap

### v4.4.0 — Current Version

- [x] **Retriever Standard Protocol** — `BaseRetriever` interface for pluggable knowledge retrieval
- [x] **Slot Governance** — Budget, dedup, pinning, cooldown, stability detection
- [x] **Knowledge Pinning** — `pin()`/`unpin()` to protect core knowledge from eviction
- [x] **Thread-Isolated Decay** — `threading.local()` for safe multi-thread inference
- [x] **`get_active()` Caching** — Avoid repeated tensor creation in KVStore
- [x] **TPManager** — Tensor Parallelism support for distributed deployment
- [x] **StreamingSession Layer Filtering** — Accurate per-token event counting
- [x] **vLLM Adapter** — Adapter for vLLM inference framework
- [x] **IBM vLLM-Hook Compatible** — Plugin integration without forking vLLM

### v5.0 — Next

- [ ] **Knowledge encoder integration** — Built-in lightweight encoder for text to KV conversion
- [ ] **Per-layer knowledge** — Different knowledge subsets for different Transformer layers
- [ ] **INT8 quantized KVStore** — Further reduce VRAM usage
- [ ] **aga-observability v1.0** — Complete Prometheus + Grafana observability
- [ ] **Cross-model knowledge transfer** — Share AGA knowledge across different models
- [ ] **Adaptive Bottleneck** — Dynamically adjust bottleneck_dim based on knowledge complexity

### Long-term (v6.0+)

- [ ] **AGA standard protocol** — Define standard interfaces for LLM attention governance
- [ ] **Hardware acceleration** — CUDA Kernel optimized injection path
- [ ] **Multi-modal knowledge** — Support for image/audio knowledge injection
- [ ] **Self-supervised knowledge discovery** — Automatically discover knowledge gaps from inference logs

---

## Installation

### Basic Installation

```bash
# Install core only (only dependency: torch)
pip install aga-core

# Install from source
cd AGAPlugin
pip install -e .
```

### Optional Dependencies

```bash
# YAML configuration support
pip install aga-core[yaml]

# Knowledge management system
pip install aga-core[knowledge]

# Observability
pip install aga-core[observability]

# All features
pip install aga-core[all]

# Development environment
pip install aga-core[dev]
```

### System Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA (recommended; CPU also works but with lower performance)

---

## Project Structure

```
aga/
├── __init__.py              # Package entry, exports all public APIs
├── plugin.py                # AGAPlugin — single entry point class
├── streaming.py             # StreamingSession — streaming generation session
├── config.py                # AGAConfig — configuration management
├── kv_store.py              # KVStore — GPU-resident KV storage (LRU + Pinning)
├── types.py                 # Data type definitions
├── exceptions.py            # Exception hierarchy
├── distributed.py           # TPManager — Tensor Parallelism support
├── gate/
│   ├── entropy_gate.py      # EntropyGateSystem — 3-stage entropy gating
│   └── decay.py             # PersistenceDecay — cross-layer decay
├── operator/
│   └── bottleneck_injector.py  # BottleneckInjector — core injector
├── retriever/
│   ├── base.py              # BaseRetriever — standard retriever protocol
│   ├── null_retriever.py    # NullRetriever — default no-op retriever
│   └── kvstore_retriever.py # KVStoreRetriever — simple KVStore retriever
├── adapter/
│   ├── base.py              # LLMAdapter — adapter abstract base class
│   ├── huggingface.py       # HuggingFaceAdapter — HF adapter
│   └── vllm.py              # VLLMAdapter + VLLMHookWorker — vLLM adapter
├── instrumentation/
│   ├── event_bus.py         # EventBus — pluggable event bus
│   ├── forward_metrics.py   # ForwardMetrics — metrics collector
│   └── audit_log.py         # AuditLog — audit log
└── docs/
    ├── product_doc_zh.md    # Product documentation (Chinese)
    ├── product_doc_en.md    # Product documentation (English)
    ├── user_manual_zh.md    # User manual (Chinese)
    └── user_manual_en.md    # User manual (English)
```

---

## License

MIT License

Copyright (c) 2024-2026 AGA Team

---

<p align="center">
  <strong>AGA — Empowering every inference with knowledge</strong>
</p>
