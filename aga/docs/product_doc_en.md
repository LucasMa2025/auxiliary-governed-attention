# AGA-Core Product Documentation

> **Version**: 4.4.0  
> **Last Updated**: 2026-02-23  
> **Scope**: aga-core package

---

## 1. Product Overview

### 1.1 What is AGA

AGA (Auxiliary Governed Attention) is a **runtime attention governance plugin** for frozen Large Language Models (LLMs). Its core objective is to **provide lossless capability extension for frozen models** — dynamically injecting external knowledge into the Transformer's attention layers during inference, enabling frozen models to access knowledge not encoded in their parameters.

### 1.2 Core Philosophy

AGA is built on a key observation: **when an LLM encounters knowledge not encoded in its parameters, it exhibits high entropy (high uncertainty) in its hidden states**. AGA leverages this signal to inject external knowledge only when the model is "uncertain," avoiding interference with the model's existing correct reasoning paths.

**Design Principles:**

-   **Effect over performance** — Correct knowledge injection is the primary goal; latency is secondary
-   **Fail-Open safety** — Any exception falls back to original model output; AGA never degrades inference
-   **Configuration-driven** — All parameters externalized; no code changes needed for tuning
-   **Standard protocols** — Retriever interface, adapter interface, event bus — all pluggable

### 1.3 Comparison with Existing Techniques

| Technique              | Intervention      | Modifies Model | Knowledge Granularity           | Dynamism                 | Decision Basis             |
| ---------------------- | ----------------- | -------------- | ------------------------------- | ------------------------ | -------------------------- |
| **RAG**                | Pre-inference     | No             | Document/passage                | Static retrieval         | Query similarity           |
| **LoRA**               | Training time     | Yes            | Global                          | Requires retraining      | None                       |
| **Prompt Engineering** | Pre-inference     | No             | Limited by context window       | Manual                   | None                       |
| **AGA**                | **Mid-inference** | **No**         | **Atomic facts (10-50 tokens)** | **Real-time add/remove** | **Model internal entropy** |

### 1.4 Target Scenarios

-   **Vertical domain knowledge systems**: Medical diagnosis, legal regulations, financial risk rules
-   **Dynamic knowledge updates**: News events, policy changes, product information
-   **Multi-tenant knowledge isolation**: Independent knowledge spaces in SaaS scenarios
-   **Model knowledge patching**: Quick fixes for factual errors without retraining
-   **Streaming generation**: Continuous knowledge injection during token-by-token generation
-   **Research experiments**: Exploring runtime knowledge injection mechanisms

---

## 2. Technical Architecture

### 2.1 Overall Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          AGAPlugin                               │
│                     (Single Entry Point)                         │
│                                                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │   KVStore    │  │ EntropyGate    │  │ BottleneckInjector    │ │
│  │  GPU-resident│  │  System        │  │  Core Injection Path  │ │
│  │  Pre-alloc   │  │  3-Stage Gate  │  │  Top-K Routing        │ │
│  │  LRU Evict   │  │  Entropy Veto  │  │  Value Projection     │ │
│  │  Pin/Unpin   │  │  Early Exit    │  │  Reliability Bias     │ │
│  │  Namespace   │  │                │  │                       │ │
│  └──────────────┘  └────────────────┘  └───────────────────────┘ │
│                                                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │ Persistence  │  │   Adapter      │  │ Instrumentation       │ │
│  │   Decay      │  │  HuggingFace   │  │  EventBus             │ │
│  │  Cross-layer │  │  vLLM          │  │  ForwardMetrics       │ │
│  │  Hard Reset  │  │  Custom        │  │  AuditLog             │ │
│  │  Thread-safe │  │  Hook Inject   │  │                       │ │
│  └──────────────┘  └────────────────┘  └───────────────────────┘ │
│                                                                  │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────────┐  │
│  │  Retriever  │  │ StreamingSess  │  │ Distributed (TP)      │  │
│  │  Protocol   │  │  Session Mgmt  │  │  TPManager            │  │
│  │  BaseRetr.  │  │  Per-token Diag│  │  Param Broadcast      │  │
│  │  Slot Gov.  │  │  Hot-Update    │  │  KV Sync              │  │
│  └─────────────┘  └────────────────┘  └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Inference Flow

AGA executes the following pipeline after each mounted Transformer layer's `self_attn` output:

1. **Gate-0 Prior Check**: Zero-cost namespace-based filtering
2. **Gate-1 Entropy Computation + Confidence Gating**:
    - Computes hidden_states internal variance as uncertainty signal
    - Enhances estimation via learned projection network
    - Applies three-stage entropy veto:
        - `H < τ_low`: Model confident → gate = 0 (no injection)
        - `τ_low ≤ H ≤ τ_high`: Normal range → gate = σ(w₁·H + b)
        - `H > τ_high`: Model extremely uncertain → gate ≤ max_gate (limited injection)
3. **Early Exit**: If gate mean is below threshold, return original output immediately
4. **Retriever Invocation** (if configured): On high entropy, query external knowledge source via `BaseRetriever` protocol, subject to Slot Governance (cooldown, budget, deduplication, stability)
5. **Bottleneck Injection**:
    - Query projection: `hidden_states → [batch, seq, bottleneck_dim]`
    - Top-K routing: Select K most relevant knowledge slots
    - Attention computation: `softmax(query @ keys.T / √d + log(reliability)) @ values`
    - Value projection: Control injection magnitude via delta subspace
6. **Cross-layer Decay**: Prevent auxiliary attention accumulation across layers (thread-isolated)
7. **Gated Fusion**: `output = primary_output + gate × aux_output`

### 2.3 Core Equations

**Entropy Gating (Eq. 2)**:

```
α = σ(w₁ · H + b)
```

where H is the uncertainty signal computed from hidden_states variance.

**Three-Stage Veto (Eq. 3)**:

```
α_effective =
  0,                    if H < τ_low
  α,                    if τ_low ≤ H ≤ τ_high
  min(α, max_gate),     if H > τ_high
```

**Persistence Decay (Eq. 5)**:

```
α^{ℓ+1}_effective = γ · α^ℓ_effective
```

**Bottleneck Injection**:

```
query = W_q · hidden_states                    # [batch, seq, d_bottleneck]
scores = query · keys^T / √d_bottleneck       # [batch, seq, K]
weights = softmax(scores + log(reliability))   # [batch, seq, K]
aux = weights · values                         # [batch, seq, d_hidden]
aux = W_up(GELU(W_down(aux)))                  # delta subspace
output = primary + gate · aux                  # gated fusion
```

---

## 3. Module Reference

### 3.1 AGAPlugin (plugin.py)

**Responsibility**: Single entry point class encapsulating all functionality.

**Public API**:

| Method                     | Signature                                                               | Description                         |
| -------------------------- | ----------------------------------------------------------------------- | ----------------------------------- |
| `__init__`                 | `(config: AGAConfig = None, retriever: BaseRetriever = None, **kwargs)` | Create plugin instance              |
| `from_config`              | `(config_source: str/Path/Dict) → AGAPlugin`                            | Create from config (classmethod)    |
| `register`                 | `(id, key, value, reliability=1.0, metadata=None, pinned=None) → bool`  | Register single knowledge entry     |
| `register_batch`           | `(entries: List[Dict]) → int`                                           | Batch register knowledge            |
| `unregister`               | `(id: str) → bool`                                                      | Remove knowledge                    |
| `load_from`                | `(source: str, **kwargs) → int`                                         | Load knowledge from external source |
| `clear`                    | `(namespace: str = None)`                                               | Clear knowledge                     |
| `attach`                   | `(model, layer_indices=None, adapter=None)`                             | Attach to model                     |
| `detach`                   | `()`                                                                    | Detach from model                   |
| `create_streaming_session` | `(**kwargs) → StreamingSession`                                         | Create streaming generation session |
| `get_diagnostics`          | `() → Dict`                                                             | Get diagnostic information          |
| `get_audit_trail`          | `(limit=100, operation=None) → List[Dict]`                              | Get audit log                       |
| `get_store_stats`          | `() → Dict`                                                             | Get storage statistics              |
| `reset_decay_contexts`     | `()`                                                                    | Reset thread-local decay contexts   |

**Properties**:

| Property          | Type   | Description                 |
| ----------------- | ------ | --------------------------- |
| `knowledge_count` | `int`  | Current knowledge count     |
| `is_attached`     | `bool` | Whether attached to a model |

**v4.4 Changes**:

-   `register()` now accepts `pinned` parameter (default: `config.pin_registered`)
-   `__init__` accepts `retriever` parameter for external knowledge retrieval
-   `get_diagnostics()` includes `pinned_count`, `unpinned_count`, `retriever_stats`, `slot_governance`
-   `_decay_contexts` replaced with `threading.local()` for thread isolation
-   Retriever integration in forward path with full Slot Governance

### 3.2 AGAConfig (config.py)

**Responsibility**: Unified configuration management with all parameters externalized.

**Configuration Groups**:

| Group                | Parameter                    | Default       | Description                          |
| -------------------- | ---------------------------- | ------------- | ------------------------------------ |
| **Model Dimensions** | `hidden_dim`                 | 4096          | Must match target model              |
|                      | `bottleneck_dim`             | 64            | Retrieval key dimension              |
|                      | `num_heads`                  | 32            | Attention heads                      |
|                      | `value_bottleneck_dim`       | 256           | Value projection bottleneck          |
| **Capacity**         | `max_slots`                  | 256           | Maximum hot knowledge slots          |
| **Device**           | `device`                     | "cuda"        | Compute device                       |
| **Entropy Gating**   | `tau_low`                    | 0.5           | Low entropy threshold                |
|                      | `tau_high`                   | 2.0           | High entropy threshold               |
|                      | `max_gate`                   | 0.8           | Maximum gate value                   |
|                      | `gate2_top_k`                | 8             | Top-K routing count                  |
|                      | `early_exit_threshold`       | 0.05          | Early exit threshold                 |
| **Decay**            | `decay_enabled`              | true          | Enable decay                         |
|                      | `decay_strategy`             | "exponential" | Decay strategy                       |
|                      | `decay_gamma`                | 0.9           | Decay coefficient                    |
| **Safety**           | `fail_open`                  | true          | Fail-Open mode                       |
| **Norm Control**     | `enable_norm_clipping`       | true          | Enable norm clipping                 |
|                      | `key_norm_target`            | 5.0           | Key target norm                      |
|                      | `value_norm_target`          | 3.0           | Value target norm                    |
| **Retriever**        | `retriever_backend`          | "null"        | Backend: null/kv_store/chroma/custom |
|                      | `retriever_endpoint`         | ""            | External retriever endpoint          |
|                      | `retriever_collection`       | ""            | Knowledge collection name            |
|                      | `retriever_top_k`            | 5             | Max results per retrieval            |
|                      | `retriever_min_score`        | 0.3           | Minimum relevance threshold          |
|                      | `retriever_query_source`     | "q_proj"      | Query source: hidden_states/q_proj   |
|                      | `retriever_auto_inject`      | true          | Auto-inject results into KVStore     |
|                      | `retriever_cache_ttl`        | 300           | Result cache TTL (seconds)           |
|                      | `retriever_timeout_ms`       | 10            | Retrieval timeout (ms)               |
| **Slot Governance**  | `pin_registered`             | true          | Auto-pin register() knowledge        |
|                      | `retriever_slot_ratio`       | 0.3           | Max slot ratio for retriever         |
|                      | `retriever_slot_budget`      | 0             | Explicit slot budget (0=use ratio)   |
|                      | `retriever_cooldown_steps`   | 5             | Cooldown period (forward steps)      |
|                      | `retriever_dedup_similarity` | 0.95          | Semantic dedup threshold (cosine)    |
|                      | `slot_stability_threshold`   | 0.5           | Max slot change rate per step        |
| **Instrumentation**  | `instrumentation_enabled`    | true          | Enable instrumentation               |
|                      | `event_buffer_size`          | 10000         | Event buffer size                    |
|                      | `audit_log_level`            | "INFO"        | Audit log level                      |
| **Observability**    | `observability_enabled`      | true          | Enable external observability        |

**Loading Methods**:

```python
# 1. Direct creation
config = AGAConfig(hidden_dim=4096)

# 2. From YAML (supports nested sections)
config = AGAConfig.from_yaml("aga_config.yaml")

# 3. From dict (supports nested flattening for gate, decay, retriever, slot_governance)
config = AGAConfig.from_dict({
    "hidden_dim": 4096,
    "gate": {"tau_low": 0.3, "tau_high": 2.5},
    "decay": {"enabled": True, "gamma": 0.85},
    "retriever": {"backend": "chroma", "endpoint": "localhost:8000"},
    "slot_governance": {"pin_registered": True, "retriever_slot_ratio": 0.3},
})

# 4. Validation
errors = config.validate()
```

### 3.3 KVStore (kv_store.py)

**Responsibility**: GPU-resident KV storage with pre-allocated memory, LRU eviction, and knowledge pinning.

**Design Principles**:

-   **Pre-allocated GPU memory**: All slot memory allocated at initialization
-   **FP16 storage**: Minimizes VRAM usage
-   **LRU eviction with pin protection**: Automatically evicts least recently used _unpinned_ knowledge when full; pinned knowledge is never evicted
-   **Thread-safe**: All write operations are locked
-   **Namespace isolation**: Via metadata namespace field
-   **Active cache**: `get_active()` results are cached and invalidated on writes

**Pin/Unpin Mechanism** (v4.4):

-   `put(id, key, value, pinned=True)` — Register and pin knowledge
-   `pin(id)` / `unpin(id)` — Explicitly manage pin status
-   Pinned knowledge is protected from LRU eviction
-   `pinned_count` / `unpinned_count` properties for monitoring
-   `source` metadata field tracks origin ("register" vs "retriever")

**VRAM Formula**:

```
VRAM = max_slots × (bottleneck_dim × 2 + hidden_dim × 2 + 3) bytes
```

### 3.4 EntropyGateSystem (gate/entropy_gate.py)

**Responsibility**: Complete three-stage entropy gating system.

**Gate-0 (Prior Gate)**: Zero-cost namespace-based filtering.

**Gate-1 (Confidence Gate)**:

-   Computes hidden_states internal variance
-   Enhanced by learned projection network
-   Compatible with FlashAttention (no attention weights needed)

**Three-Stage Entropy Veto**:

-   `H < τ_low`: Model confident → gate forced to 0
-   `τ_low ≤ H ≤ τ_high`: Normal range → gate = σ(w₁·H + b)
-   `H > τ_high`: Extremely uncertain → gate capped at max_gate

### 3.5 BottleneckInjector (operator/bottleneck_injector.py)

**Responsibility**: Core injection path, latency <0.1ms.

**Mathematical Flow**:

1. Query projection: `W_q: [hidden_dim → bottleneck_dim]`
2. Top-K routing: Select K most relevant slots
3. Attention: `softmax(query @ keys.T / √d + log(reliability))`
4. Weighted sum: `attn_weights @ values`
5. Value projection: `W_up(GELU(W_down(aux)))` — delta subspace

**Information Capacity**:

-   Each key: `[bottleneck_dim=64]` — for retrieval matching
-   Each value: `[hidden_dim=4096]` — actual injected knowledge vector
-   Each value encodes 10-50 tokens of atomic factual semantics

### 3.6 PersistenceDecay (gate/decay.py)

**Responsibility**: Prevent auxiliary attention accumulation across layers.

**Thread Isolation** (v4.4): Decay contexts are stored in `threading.local()`, ensuring concurrent requests (e.g., in vLLM continuous batching) maintain independent decay state.

**Supported Strategies**:

-   `exponential`: α\_{l+1} = γ^l · α_l (default)
-   `linear`: α\_{l+1} = α_l - δ
-   `adaptive`: Dynamic adjustment based on accumulation
-   `none`: No decay

**Hard Reset**: Forces gate to zero when accumulated gate exceeds threshold.

### 3.7 Retriever Protocol (retriever/)

**Responsibility**: Standard protocol for external knowledge retrieval. AGA queries external knowledge sources when high entropy is detected.

**v4.4 New Feature**: This is a core addition enabling AGA to acquire inference facts from external knowledge infrastructure.

**BaseRetriever** (Abstract Base Class):

| Method                  | Signature                                         | Description                               |
| ----------------------- | ------------------------------------------------- | ----------------------------------------- |
| `retrieve`              | `(query: RetrievalQuery) → List[RetrievalResult]` | Core retrieval (called on high entropy)   |
| `warmup`                | `() → None`                                       | Optional pre-heating (index, connections) |
| `on_injection_feedback` | `(result_id, was_used, gate_value) → None`        | Optional feedback callback                |
| `get_stats`             | `() → Dict[str, Any]`                             | Retriever statistics                      |
| `shutdown`              | `() → None`                                       | Release resources                         |

**RetrievalQuery** (Input):

| Field             | Type                       | Description                                    |
| ----------------- | -------------------------- | ---------------------------------------------- |
| `hidden_states`   | `Tensor[batch, seq, dim]`  | Current layer's hidden states (semantic query) |
| `query_projected` | `Tensor[batch, seq, bdim]` | q_proj output (aligned to knowledge space)     |
| `entropy`         | `Tensor[batch, seq]`       | Current entropy values                         |
| `layer_idx`       | `int`                      | Current Transformer layer index                |
| `namespace`       | `Optional[str]`            | Namespace filter                               |
| `top_k`           | `int`                      | Maximum results to return                      |

**RetrievalResult** (Output):

| Field         | Type                     | Description                 |
| ------------- | ------------------------ | --------------------------- |
| `id`          | `str`                    | Knowledge unique identifier |
| `key`         | `Tensor[bottleneck_dim]` | Retrieval key vector        |
| `value`       | `Tensor[hidden_dim]`     | Knowledge value vector      |
| `reliability` | `float`                  | Reliability score (0.0-1.0) |
| `score`       | `float`                  | Retrieval relevance score   |
| `metadata`    | `Optional[Dict]`         | Optional metadata           |

**Built-in Retrievers**:

| Retriever          | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `NullRetriever`    | Default — no external retrieval (KVStore-only mode)    |
| `KVStoreRetriever` | Searches existing KVStore entries by cosine similarity |

**Custom Retriever Example**:

```python
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult

class ChromaRetriever(BaseRetriever):
    def __init__(self, collection_name: str, endpoint: str):
        import chromadb
        self.client = chromadb.HttpClient(host=endpoint)
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        # Use q_proj output as query vector
        query_vec = query.query_projected.mean(dim=[0, 1]).cpu().numpy()
        results = self.collection.query(query_embeddings=[query_vec], n_results=query.top_k)
        # Convert to RetrievalResult...
        return [...]

plugin = AGAPlugin(config, retriever=ChromaRetriever("medical_kb", "localhost:8000"))
```

### 3.8 Slot Governance

**Responsibility**: Prevent slot thrashing and gate jitter in high-concurrency, slot-contention scenarios.

**v4.4 New Feature**: When `max_slots` is small and many knowledge items are recalled, rapid eviction and re-insertion can cause instability. Slot Governance provides five defense mechanisms:

| Mechanism                  | Config Parameter              | Description                                                  |
| -------------------------- | ----------------------------- | ------------------------------------------------------------ |
| **Knowledge Pinning**      | `pin_registered`              | `register()` knowledge is pinned (immune to LRU eviction)    |
| **Slot Budget Guard**      | `retriever_slot_ratio/budget` | Retriever can only use a fraction of total slots             |
| **Semantic Deduplication** | `retriever_dedup_similarity`  | Filter out semantically similar results (cosine > threshold) |
| **Injection Stability**    | `slot_stability_threshold`    | Pause retrieval if KVStore changes too rapidly               |
| **Adaptive Cooldown**      | `retriever_cooldown_steps`    | Minimum forward steps between retrievals                     |

**Slot Lifecycle**:

```
register() → Pinned (protected from eviction)
retriever  → Unpinned (subject to LRU eviction within budget)
```

### 3.9 HuggingFaceAdapter (adapter/huggingface.py)

**Responsibility**: Auto-detect and attach to HuggingFace models.

**Supported Architectures**:

-   LLaMA / LLaMA-2 / LLaMA-3
-   Qwen / Qwen-2
-   Mistral / Mixtral
-   GPT-2 / GPT-J / GPT-NeoX
-   Phi / Phi-2 / Phi-3
-   Gemma / Falcon

**Attachment**: Via `register_forward_hook` after `self_attn` output.

### 3.10 VLLMAdapter (adapter/vllm.py)

**Responsibility**: Inject AGA into models running within the vLLM inference framework, without forking vLLM.

**Core Capabilities**:

| Capability              | Description                                                              |
| ----------------------- | ------------------------------------------------------------------------ |
| `extract_model()`       | Extract underlying nn.Module from vLLM LLM/LLMEngine instance            |
| `get_layers()`          | Recursively search for Transformer layers (supports multiple structures) |
| `wrap_layer()`          | Register forward hook after attention output to inject AGA               |
| `check_compatibility()` | Check vLLM configuration compatibility and generate report               |

**VLLMHookWorker**: Compatible with IBM vLLM-Hook plugin system. Automatically falls back to direct PyTorch hook if vLLM-Hook is not installed.

### 3.11 Distributed Support (distributed.py)

**Responsibility**: Tensor Parallelism support for multi-GPU deployments.

**v4.4 New Feature**: `TPManager` synchronizes AGA's state across TP ranks.

| Method                     | Description                                                   |
| -------------------------- | ------------------------------------------------------------- |
| `broadcast_parameters()`   | Sync learnable parameters (gate_system, injector) from rank 0 |
| `broadcast_knowledge()`    | Sync entire KVStore content from rank 0                       |
| `broadcast_single_entry()` | Sync a single knowledge entry after registration              |
| `is_primary`               | Whether current rank is rank 0                                |
| `is_enabled`               | Whether torch.distributed is initialized                      |

**Usage**:

```python
plugin = AGAPlugin(config)
tp_manager = TPManager(plugin)
tp_manager.broadcast_parameters()

if tp_manager.is_primary:
    plugin.register("fact_001", key=k, value=v)
tp_manager.broadcast_knowledge()
```

### 3.12 StreamingSession (streaming.py)

**Responsibility**: Session management and real-time diagnostics during streaming generation.

**Key Features**:

-   Per-token diagnostics (`get_step_diagnostics()`)
-   Session summary (`get_session_summary()`)
-   Dynamic knowledge hot-update (`update_knowledge()` / `remove_knowledge()`)
-   Automatic decay context management
-   Layer event filtering (v4.4): Only counts events from the primary monitoring layer to avoid duplicate counting when multiple layers are hooked

### 3.13 Instrumentation (instrumentation/)

**Responsibility**: Built-in instrumentation layer, zero external dependencies.

**EventBus**:

-   In-memory ring buffer (default 10,000 events)
-   Pluggable subscriber pattern
-   Event emission <1μs, no inference latency impact
-   Wildcard subscription (`*`)

**ForwardMetrics**:

-   Activation rate
-   Average gate value
-   Average entropy value
-   P50/P95/P99 latency percentiles
-   Per-layer statistics

**AuditLog**:

-   Complete audit trail for all knowledge operations
-   Python logging integration (always outputs)
-   In-memory buffer queryable
-   Emits audit events via EventBus

---

## 4. Safety Mechanisms

### 4.1 Fail-Open

AGA follows the Fail-Open design principle: **any exception falls back to the original model output**. This applies to:

-   Forward path exceptions
-   Retriever failures (returns empty list)
-   Retriever warmup/shutdown failures

### 4.2 Norm Control

Automatic norm clipping during knowledge registration prevents outlier values from affecting inference.

### 4.3 Entropy Veto

Three-stage entropy veto ensures AGA never forcefully injects when the model is already confident.

### 4.4 Cross-Layer Decay

Persistence decay prevents auxiliary attention accumulation across layers. Hard reset mechanism forces zero when accumulation exceeds threshold. Thread-isolated via `threading.local()`.

### 4.5 Slot Governance

Prevents slot thrashing and gate jitter through budget limits, semantic deduplication, knowledge pinning, stability detection, and adaptive cooldown.

---

## 5. Performance Characteristics

### 5.1 Latency

| Operation      | Typical Latency | Notes                             |
| -------------- | --------------- | --------------------------------- |
| Gate-0 check   | <1 μs           | Pure Python string comparison     |
| Gate-1 entropy | ~10 μs          | Variance computation + projection |
| Early Exit     | ~15 μs          | After Gate-0 + Gate-1             |
| Full injection | ~50-80 μs       | Complete injection path           |
| Retriever call | 1-10 ms         | External knowledge retrieval      |
| Event emission | <1 μs           | Memory write                      |

### 5.2 VRAM

| max_slots | hidden_dim=4096 | hidden_dim=8192 |
| --------- | --------------- | --------------- |
| 256       | ~2.0 MB         | ~4.0 MB         |
| 1000      | ~8.1 MB         | ~16.1 MB        |
| 5000      | ~40.6 MB        | ~81.0 MB        |

### 5.3 Throughput

AGA's bypass rate (Early Exit) is typically >60% in general scenarios, meaning most tokens don't trigger the full injection path, with minimal impact on overall throughput.

In vertical domain scenarios, activation rate may reach 40-70%, where each injection's ~50-80μs latency is still far below the Transformer layer's own computation time.

Retriever calls (1-10ms) are infrequent (governed by cooldown and stability checks) and only occur on high-entropy tokens at the first hooked layer.

---

## 6. Exception Hierarchy

| Exception        | Inherits From | Trigger                        |
| ---------------- | ------------- | ------------------------------ |
| `AGAError`       | `Exception`   | Base AGA exception             |
| `AttachError`    | `AGAError`    | Model attach/detach failure    |
| `KVStoreError`   | `AGAError`    | KV store operation failure     |
| `ConfigError`    | `AGAError`    | Config load/validation failure |
| `GateError`      | `AGAError`    | Gate system exception          |
| `AdapterError`   | `AGAError`    | LLM adapter exception          |
| `RetrieverError` | `AGAError`    | Retriever operation failure    |

---

## 7. Data Types

### KnowledgeEntry

```python
@dataclass
class KnowledgeEntry:
    id: str                              # Unique knowledge identifier
    key: torch.Tensor                    # [bottleneck_dim] retrieval key
    value: torch.Tensor                  # [hidden_dim] knowledge vector
    reliability: float = 1.0             # Reliability score (0.0-1.0)
    metadata: Optional[Dict] = None      # Optional metadata
```

### GateDiagnostics

```python
@dataclass
class GateDiagnostics:
    gate0_passed: bool = True            # Gate-0 passed
    entropy_mean: float = 0.0            # Mean entropy
    gate_mean: float = 0.0              # Mean gate value
    gate_max: float = 0.0              # Max gate value
    early_exit: bool = False            # Early exit triggered
    veto_ratio: float = 0.0            # Veto ratio
```

### RetrievalQuery / RetrievalResult

See Section 3.7 for full field descriptions.

---

## 8. JSONL Knowledge File Format

`aga-core` has built-in support for JSONL knowledge file loading without `aga-knowledge`.

**File Format**:

```jsonl
{"id": "fact_001", "key": [0.1, 0.2, ...], "value": [0.3, 0.4, ...], "reliability": 0.95, "metadata": {"source": "medical_kb"}}
{"id": "fact_002", "key": [0.5, 0.6, ...], "value": [0.7, 0.8, ...], "reliability": 0.9}
```

**Field Reference**:

| Field         | Type    | Required | Description                                   |
| ------------- | ------- | -------- | --------------------------------------------- |
| `id`          | string  | Yes      | Unique knowledge identifier                   |
| `key`         | float[] | Yes      | Retrieval key vector, length = bottleneck_dim |
| `value`       | float[] | Yes      | Knowledge value vector, length = hidden_dim   |
| `reliability` | float   | No       | Reliability score, default 1.0                |
| `metadata`    | object  | No       | Optional metadata                             |

---

## 9. Integration with aga-knowledge

When `aga-knowledge` is installed, `aga-core` can automatically integrate richer knowledge management capabilities:

```python
# Method 1: Load non-JSONL formats via load_from
plugin.load_from("postgresql://localhost/knowledge_db")

# Method 2: Config-driven auto-loading
plugin = AGAPlugin.from_config({
    "hidden_dim": 4096,
    "knowledge_sources": [
        {"type": "jsonl", "path": "base_knowledge.jsonl"},
        {"type": "portal", "url": "http://portal:8000"},
    ]
})

# Method 3: Direct KnowledgeManager usage
from aga_knowledge import KnowledgeManager
manager = KnowledgeManager(config)
manager.sync_to_plugin(plugin)
```

**Note**: With the standard `BaseRetriever` protocol, `aga-knowledge` is no longer strictly necessary. Users can implement their own retriever using existing infrastructure (Chroma, Milvus, Elasticsearch, etc.) and pass it directly to `AGAPlugin`.

---

## 10. Complete YAML Configuration Reference

```yaml
aga:
    # ===== Model Dimensions =====
    hidden_dim: 4096
    bottleneck_dim: 64
    num_heads: 32
    value_bottleneck_dim: 256

    # ===== Capacity =====
    max_slots: 256

    # ===== Device =====
    device: "cuda"

    # ===== Entropy Gating =====
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

    # ===== Decay =====
    decay:
        enabled: true
        strategy: "exponential"
        gamma: 0.9
        hard_reset_threshold: 3.0

    # ===== Safety =====
    fail_open: true
    max_forward_timeout_ms: 50

    # ===== Norm Control =====
    enable_norm_clipping: true
    key_norm_target: 5.0
    value_norm_target: 3.0

    # ===== Retriever (Configuration-Driven Knowledge Retrieval) =====
    retriever:
        backend: "null" # null / kv_store / chroma / milvus / custom
        endpoint: "" # External retriever endpoint
        collection: "" # Knowledge collection name
        top_k: 5 # Max results per retrieval
        min_score: 0.3 # Minimum relevance threshold
        query_source: "q_proj" # hidden_states / q_proj
        auto_inject: true # Auto-inject results into KVStore
        cache_ttl: 300 # Result cache TTL (seconds)
        timeout_ms: 10 # Retrieval timeout (ms)

    # ===== Slot Governance =====
    slot_governance:
        pin_registered: true # Auto-pin register() knowledge
        retriever_slot_ratio: 0.3 # Max slot ratio for retriever
        retriever_slot_budget: 0 # Explicit budget (0=use ratio)
        retriever_cooldown_steps: 5 # Min steps between retrievals
        retriever_dedup_similarity: 0.95 # Semantic dedup threshold
        slot_stability_threshold: 0.5 # Max change rate per step

    # ===== Instrumentation & Audit =====
    instrumentation:
        instrumentation_enabled: true
        event_buffer_size: 10000
        audit_log_level: "INFO"

    # ===== Observability (requires aga-observability) =====
    observability:
        observability_enabled: true
        prometheus_enabled: true
        prometheus_port: 9090

    # ===== Knowledge Sources =====
    knowledge_sources:
        - type: jsonl
          path: "data/base_knowledge.jsonl"
```

---

## 11. Testing

`aga-core` includes a comprehensive unit test suite:

```bash
# Run all tests
cd AGAPlugin
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_plugin.py -v
python -m pytest tests/test_kv_store.py -v
python -m pytest tests/test_gate.py -v
python -m pytest tests/test_retriever.py -v
python -m pytest tests/test_streaming.py -v
```

**Test Coverage**:

-   AGAPlugin: Initialization, config-driven, knowledge management, model attachment, diagnostics, retriever integration, slot governance
-   KVStore: CRUD, LRU eviction, pin/unpin, namespace, thread safety, VRAM estimation, active cache
-   EntropyGateSystem: Three-stage gating, entropy veto, Early Exit
-   BottleneckInjector: Injection path, Top-K routing, Value projection
-   PersistenceDecay: Decay strategies, hard reset, context passing, thread isolation
-   HuggingFaceAdapter: Architecture detection, Hook injection
-   VLLMAdapter: Model extraction, layer search, Hook injection, compatibility check
-   VLLMHookWorker: vLLM-Hook compatibility, fallback mechanism
-   Retriever: NullRetriever, KVStoreRetriever, BaseRetriever protocol, budget/dedup/cooldown
-   StreamingSession: Session management, per-token diagnostics, layer event filtering
-   EventBus: Event emission, subscription, querying
-   ForwardMetrics: Metric collection, percentile computation
-   AuditLog: Audit recording, filtered querying
-   AGAConfig: YAML loading, dict creation, validation, nested flattening

---

## Appendix A: Glossary

| Term                   | Description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| Entropy Gating         | Deciding whether to inject knowledge based on model uncertainty           |
| Bottleneck Injection   | Knowledge matching and injection through low-dimensional projection space |
| Knowledge Slot         | A key-value pair in KVStore                                               |
| Bypass                 | Skipping AGA injection when model is confident                            |
| Persistence Decay      | Preventing auxiliary attention accumulation across layers                 |
| Hard Reset             | Forcing gate to zero when accumulation exceeds threshold                  |
| Fail-Open              | Falling back to original model output on any exception                    |
| Delta Subspace         | Value projection bottleneck layer controlling injection magnitude         |
| Retriever              | External knowledge source queried on high entropy                         |
| Slot Governance        | Mechanisms preventing slot thrashing and gate jitter                      |
| Knowledge Pinning      | Protecting core knowledge from LRU eviction                               |
| Semantic Deduplication | Filtering out semantically similar retrieval results                      |
| Slot Thrashing         | Rapid eviction and re-insertion of knowledge slots                        |
| Gate Jitter            | Unstable gate values from rapidly changing KVStore content                |

## Appendix B: Version History

| Version | Date       | Key Changes                                                                  |
| ------- | ---------- | ---------------------------------------------------------------------------- |
| 4.2.0   | 2026-02-20 | Initial aga-core release with full plugin architecture                       |
| 4.3.0   | 2026-02-22 | vLLM adapter, streaming generation, IBM vLLM-Hook compatibility              |
| 4.4.0   | 2026-02-23 | Retriever protocol, slot governance, TP support, thread isolation, pin/unpin |
