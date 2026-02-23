# AGA-Core User Manual

> **Version**: 4.4.0  
> **Last Updated**: 2026-02-23

---

## Table of Contents

1. [Installation & Environment](#1-installation--environment)
2. [Quick Start](#2-quick-start)
3. [Knowledge Management](#3-knowledge-management)
4. [Model Attachment](#4-model-attachment)
5. [Streaming Generation Injection](#5-streaming-generation-injection)
6. [External Retriever Integration](#6-external-retriever-integration)
7. [Configuration](#7-configuration)
8. [Diagnostics & Monitoring](#8-diagnostics--monitoring)
9. [Advanced Usage](#9-advanced-usage)
10. [Production Deployment](#10-production-deployment)
11. [Troubleshooting](#11-troubleshooting)
12. [FAQ](#12-faq)

---

## 1. Installation & Environment

### 1.1 System Requirements

| Requirement | Minimum         | Recommended    |
| ----------- | --------------- | -------------- |
| Python      | 3.9             | 3.10+          |
| PyTorch     | 2.0.0           | 2.1.0+         |
| CUDA        | 11.7 (optional) | 12.0+          |
| GPU VRAM    | 100 MB (AGA)    | 1 GB+ (w/model)|
| System RAM  | 4 GB            | 16 GB+         |

### 1.2 Installation

```bash
# Basic installation (only depends on torch)
pip install aga-core

# Install from source
git clone https://github.com/aga-project/aga-core.git
cd aga-core
pip install -e .

# With YAML configuration support
pip install aga-core[yaml]

# With knowledge management system
pip install aga-core[knowledge]

# All features
pip install aga-core[all]

# Development environment
pip install aga-core[dev]
```

### 1.3 Verify Installation

```python
import aga
print(f"AGA version: {aga.__version__}")
# Output: AGA version: 4.4.0

from aga import AGAPlugin, AGAConfig
plugin = AGAPlugin(AGAConfig(hidden_dim=768, device="cpu"))
print(plugin)
# Output: AGAPlugin(hidden_dim=768, slots=0/256, attached=False, device=cpu)
```

---

## 2. Quick Start

### 2.1 Minimal Example

```python
from aga import AGAPlugin, AGAConfig
import torch

# Step 1: Create plugin
plugin = AGAPlugin(AGAConfig(
    hidden_dim=4096,      # Must match model's hidden_size
    device="cuda",
))

# Step 2: Register knowledge
plugin.register(
    id="capital_france",
    key=torch.randn(64),       # bottleneck_dim=64
    value=torch.randn(4096),   # hidden_dim=4096
    reliability=0.95,
    pinned=True,               # Pin core knowledge (prevents eviction)
)

# Step 3: Attach to model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
plugin.attach(model)

# Step 4: Normal inference — AGA works automatically
output = model.generate(input_ids, max_new_tokens=100)

# Step 5: Check results
print(plugin.get_diagnostics())
```

### 2.2 Config-Driven Example

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

# One-line creation + auto-load knowledge
plugin = AGAPlugin.from_config("aga_config.yaml")
plugin.attach(model)
output = model.generate(input_ids)
```

---

## 3. Knowledge Management

### 3.1 Knowledge Format

Each knowledge entry in AGA consists of:

| Field         | Type                     | Required | Description                                    |
| ------------- | ------------------------ | -------- | ---------------------------------------------- |
| `id`          | `str`                    | Yes      | Unique identifier                              |
| `key`         | `Tensor[bottleneck_dim]` | Yes      | Retrieval key vector for matching hidden_states |
| `value`       | `Tensor[hidden_dim]`     | Yes      | Knowledge value vector injected into attention  |
| `reliability` | `float`                  | No       | Reliability score (0.0-1.0), affects weight     |
| `pinned`      | `bool`                   | No       | If True, protects from LRU eviction (v4.4.0)   |
| `metadata`    | `Dict`                   | No       | Metadata (namespace, source, etc.)             |

### 3.2 Register Knowledge

```python
import torch

# Single registration with pinning
success = plugin.register(
    id="med_001",
    key=torch.randn(64),
    value=torch.randn(4096),
    reliability=0.95,
    pinned=True,                # Core knowledge — protected from eviction
    metadata={"namespace": "cardiology", "source": "textbook"}
)

# Batch registration
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
print(f"Successfully registered {count} entries")
```

### 3.3 Knowledge Pinning (v4.4.0)

Pinned knowledge is protected from LRU eviction. This is critical for core domain knowledge that must always be available.

```python
# Register as pinned
plugin.register("core_fact", key=k, value=v, pinned=True)

# Pin/unpin existing knowledge
plugin.store.pin("core_fact")
plugin.store.unpin("core_fact")

# Check pinning stats
stats = plugin.get_store_stats()
print(f"Pinned: {stats['pinned_count']}, Unpinned: {stats['unpinned_count']}")
```

> **Note**: When `config.pin_registered = True` (default), all knowledge registered via `register()` is automatically pinned. Knowledge injected by the retriever is **not** pinned by default.

### 3.4 Load from File

**JSONL file format**:

```jsonl
{"id": "fact_001", "key": [0.1, -0.2, 0.3, ...], "value": [0.4, 0.5, ...], "reliability": 0.95, "metadata": {"source": "wiki"}}
{"id": "fact_002", "key": [0.6, 0.7, ...], "value": [0.8, -0.9, ...], "reliability": 0.9}
```

> **Note**: `key` array length must equal `bottleneck_dim` (default 64), `value` array length must equal `hidden_dim` (default 4096).

```python
# Load from JSONL (built-in, no aga-knowledge needed)
count = plugin.load_from("data/knowledge.jsonl")
print(f"Loaded {count} knowledge entries")
```

### 3.5 Remove Knowledge

```python
# Remove single entry
success = plugin.unregister("med_001")

# Clear namespace
plugin.clear(namespace="cardiology")

# Clear all
plugin.clear()
```

### 3.6 Query Knowledge

```python
# Check if knowledge exists
exists = plugin.store.contains("med_001")

# Get single entry
result = plugin.store.get("med_001")
if result:
    key, value, reliability = result

# Get all knowledge IDs
all_ids = plugin.store.get_all_ids()

# Get knowledge metadata
meta = plugin.store.get_metadata("med_001")
```

### 3.7 Knowledge Preparation Tips

**How to prepare key and value vectors?**

When using `aga-core` standalone, users need to prepare KV vectors themselves. Common methods:

**Method 1: Using Sentence Transformer**

```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

# Projection layers (need training or manual initialization)
key_proj = nn.Linear(384, 64)    # -> bottleneck_dim
value_proj = nn.Linear(384, 4096)  # -> hidden_dim

text = "The capital of France is Paris"
embedding = torch.tensor(encoder.encode(text))

key = key_proj(embedding)
value = value_proj(embedding)

plugin.register("capital_france", key=key, value=value)
```

**Method 2: Using the model's own hidden_states**

```python
with torch.no_grad():
    outputs = model(tokenizer("The capital of France is Paris", return_tensors="pt").input_ids)
    hidden = outputs.last_hidden_state.mean(dim=1)

key = plugin.injector.q_proj(hidden).squeeze(0)
value = hidden.squeeze(0)

plugin.register("capital_france", key=key, value=value)
```

**Method 3: Using aga-knowledge (recommended for production)**

```python
from aga_knowledge import KnowledgeManager
manager = KnowledgeManager(config)
manager.ingest_document("medical_textbook.pdf")
manager.sync_to_plugin(plugin)
```

---

## 4. Model Attachment

### 4.1 Basic Attachment

```python
# Default: attach to last 3 layers
plugin.attach(model)

# Specify layer indices (negative = from end)
plugin.attach(model, layer_indices=[-1, -2, -3, -4, -5])

# Specify positive indices
plugin.attach(model, layer_indices=[28, 29, 30, 31])
```

### 4.2 Choosing Layers

**Recommended strategies**:

| Scenario       | Recommended Layers       | Description                    |
| -------------- | ------------------------ | ------------------------------ |
| Quick test     | Last 1 `[-1]`           | Minimal overhead               |
| Standard use   | Last 3 `[-1, -2, -3]`   | Balance of effect & perf (default) |
| Deep injection | Last 5-8                 | Stronger knowledge influence   |
| Full injection | All layers               | Maximum influence, highest latency |

### 4.3 Detach

```python
# Detach (removes all hooks)
plugin.detach()

# Can re-attach after detaching
plugin.attach(another_model)
```

### 4.4 vLLM Adapter

AGA natively supports the vLLM inference framework — **no fork of vLLM required**.

#### 4.4.1 Basic Usage

```python
from vllm import LLM, SamplingParams
from aga import AGAPlugin, AGAConfig
from aga.adapter.vllm import VLLMAdapter

# 1. Create vLLM engine (enforce_eager=True recommended)
llm = LLM(model="meta-llama/Llama-2-7b-hf", enforce_eager=True)

# 2. Extract internal model from vLLM
model = VLLMAdapter.extract_model(llm)

# 3. Create AGA plugin and attach
plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.register("fact_001", key=k, value=v)
plugin.attach(model, adapter=VLLMAdapter())

# 4. Use vLLM normally — AGA works automatically
outputs = llm.generate(["What is the capital of France?"], SamplingParams(max_tokens=100))
```

#### 4.4.2 Compatibility Check

```python
report = VLLMAdapter.check_compatibility(llm)
print(f"Compatible: {report['compatible']}")
print(f"Model: {report['model_type']}")
print(f"CUDA Graph: {report['cuda_graph']}")
print(f"Tensor Parallel: {report['tensor_parallel']}")

for warning in report['warnings']:
    print(f"  Warning: {warning}")
for rec in report['recommendations']:
    print(f"  Recommendation: {rec}")
```

#### 4.4.3 Important Notes

| Item | Description |
| ---- | ----------- |
| **PagedAttention** | AGA injects after attention output, compatible with PagedAttention |
| **Continuous Batching** | AGA's forward logic correctly handles batch dimensions |
| **CUDA Graph** | Use `enforce_eager=True` for predictable hook behavior |
| **Tensor Parallelism** | Use `TPManager` to sync KVStore across TP ranks |
| **Recommended** | `enforce_eager=True` for predictable hook behavior |

### 4.5 Custom Adapter

For non-standard HuggingFace architectures:

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

plugin.attach(model, adapter=MyCustomAdapter())
```

---

## 5. Streaming Generation Injection

### 5.1 Overview

AGA natively supports dynamic knowledge injection during LLM's autoregressive (token-by-token) generation process. Since AGA attaches to Transformer layers via `register_forward_hook`, it automatically evaluates each token's entropy at every decode step and decides whether to inject knowledge.

`StreamingSession` provides **session management** and **real-time diagnostics** during the streaming generation process.

### 5.2 Basic Usage

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)

# Create streaming session
session = plugin.create_streaming_session()

# Simulate streaming generation
for step in range(max_tokens):
    output = model.forward(current_token_ids)
    next_token = sample(output.logits)
    
    # Get AGA diagnostics for current step
    diag = session.get_step_diagnostics()
    print(f"Step {diag['step']}: applied={diag['aga_applied']}, "
          f"gate={diag['gate_mean']:.4f}, entropy={diag['entropy_mean']:.4f}")
    
    if next_token == eos_token:
        break

# Get session summary
summary = session.get_session_summary()
print(f"Total steps: {summary['total_steps']}")
print(f"Injection count: {summary['injection_count']}")
print(f"Injection rate: {summary['injection_rate']:.2%}")
print(f"Avg gate mean: {summary['avg_gate_mean']:.4f}")
print(f"Avg entropy mean: {summary['avg_entropy_mean']:.4f}")

# Close session (auto-cleans decay contexts)
session.close()
```

### 5.3 Dynamic Knowledge Hot-Update

During streaming generation, you can dynamically add or remove knowledge:

```python
session = plugin.create_streaming_session()

for step in range(max_tokens):
    output = model.forward(current_token_ids)
    
    # Dynamically add knowledge based on generated content
    if detected_topic_change(output):
        session.update_knowledge(
            id="dynamic_fact",
            key=new_key_tensor,
            value=new_value_tensor,
            reliability=0.9,
        )
    
    # Remove knowledge no longer needed
    if topic_completed(output):
        session.remove_knowledge("dynamic_fact")

session.close()
```

### 5.4 Integration with HuggingFace generate()

```python
from transformers import TextIteratorStreamer
from threading import Thread

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
session = plugin.create_streaming_session()

generation_kwargs = {
    "input_ids": input_ids,
    "max_new_tokens": 200,
    "streamer": streamer,
}
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)

thread.join()
summary = session.get_session_summary()
print(f"\n--- AGA injection rate: {summary['injection_rate']:.2%} ---")
session.close()
```

### 5.5 Multiple Activations During Streaming

In vertical domain scenarios, AGA may be **activated multiple times** during a single streaming generation:

```
Generating: "The patient's diagnosis is..."
  Token "The"       -> Low entropy  -> Bypass
  Token "patient"   -> Low entropy  -> Bypass
  Token "diagnosis" -> High entropy -> AGA injects medical knowledge
  Token "is"        -> Low entropy  -> Bypass
  Token "acute"     -> High entropy -> AGA injects disease knowledge
  Token "myocardial"-> High entropy -> AGA injects cardiology knowledge
  Token "infarction"-> Medium entropy -> AGA light injection
```

---

## 6. External Retriever Integration

### 6.1 Overview (v4.4.0)

AGA provides a standard `BaseRetriever` protocol that allows connecting to any external knowledge source (Chroma, Milvus, Elasticsearch, custom databases, etc.). When entropy is high, AGA automatically queries the retriever and injects retrieved knowledge into the KVStore.

### 6.2 Implementing a Custom Retriever

```python
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult
import torch

class ChromaRetriever(BaseRetriever):
    def __init__(self, collection_name: str):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_collection(collection_name)
    
    def retrieve(self, query: RetrievalQuery) -> list:
        # Convert query tensor to embedding for search
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
        # Pre-warm connection
        self.collection.count()
    
    def shutdown(self):
        pass
```

### 6.3 Using a Retriever

```python
from aga import AGAPlugin, AGAConfig

retriever = ChromaRetriever("medical_knowledge")
plugin = AGAPlugin(
    AGAConfig(hidden_dim=4096, retriever_auto_inject=True),
    retriever=retriever,
)
plugin.attach(model)

# AGA automatically queries the retriever when entropy is high
output = model.generate(input_ids)
```

### 6.4 Built-in Retrievers

| Retriever          | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `NullRetriever`    | Default no-op retriever (returns empty results)       |
| `KVStoreRetriever` | Simple retriever that searches the existing KVStore   |

### 6.5 Slot Governance

When using an external retriever, AGA's Slot Governance system prevents slot thrashing:

- **Slot Budget**: Limits how many slots the retriever can use (configurable ratio)
- **Semantic Deduplication**: Prevents injecting semantically similar knowledge (cosine similarity threshold)
- **Cooldown**: Minimum steps between retrieval calls
- **Stability Detection**: Pauses retrieval if KVStore changes too rapidly
- **Knowledge Pinning**: Core knowledge (registered via `register()`) is protected from eviction by retriever results

```yaml
slot_governance:
    pin_registered: true              # Auto-pin registered knowledge
    retriever_slot_ratio: 0.3         # Max 30% of slots for retriever
    retriever_cooldown_steps: 5       # Min 5 steps between retrievals
    retriever_dedup_similarity: 0.95  # Dedup threshold
    slot_stability_threshold: 0.5     # Stability detection threshold
```

---

## 7. Configuration

### 7.1 Configuration Methods

```python
from aga import AGAConfig

# Method 1: Direct creation (with defaults)
config = AGAConfig(hidden_dim=4096)

# Method 2: Load from YAML file
config = AGAConfig.from_yaml("aga_config.yaml")

# Method 3: Create from dictionary (supports nesting)
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

# Method 4: Keyword arguments
plugin = AGAPlugin(
    hidden_dim=4096,
    bottleneck_dim=64,
    max_slots=512,
    tau_low=0.3,
)
```

### 7.2 Configuration Validation

```python
config = AGAConfig(hidden_dim=4096, bottleneck_dim=8192)
errors = config.validate()
if errors:
    for err in errors:
        print(f"Config error: {err}")
```

### 7.3 Runtime Adjustment

```python
# Adjust entropy gating thresholds
plugin.gate_system.update_thresholds(
    tau_low=0.3,
    tau_high=2.5,
    max_gate=0.7,
)

# Reset decay contexts (for new inference requests)
plugin.reset_decay_contexts()
```

### 7.4 Complete YAML Reference

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

    # ===== Retriever (v4.4.0) =====
    retriever:
        backend: "null"              # "null", "kvstore", or custom
        endpoint: ""                 # For remote retrievers
        collection: ""               # Collection/index name
        top_k: 5
        min_score: 0.3
        query_source: "q_proj"       # "q_proj" or "hidden_states"
        auto_inject: true
        cache_ttl: 300
        timeout_ms: 10

    # ===== Slot Governance (v4.4.0) =====
    slot_governance:
        pin_registered: true
        retriever_slot_ratio: 0.3
        retriever_slot_budget: 0     # 0 = auto-calculate from ratio
        retriever_cooldown_steps: 5
        retriever_dedup_similarity: 0.95
        slot_stability_threshold: 0.5

    # ===== Safety =====
    fail_open: true
    max_forward_timeout_ms: 50

    # ===== Norm Control =====
    enable_norm_clipping: true
    key_norm_target: 5.0
    value_norm_target: 3.0

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

    # ===== Streaming =====
    streaming:
        diagnostics_buffer_size: 1000

    # ===== Knowledge Sources =====
    knowledge_sources:
        - type: jsonl
          path: "data/base_knowledge.jsonl"
```

---

## 8. Diagnostics & Monitoring

### 8.1 Runtime Diagnostics

```python
diag = plugin.get_diagnostics()

print(f"Attached: {diag['attached']}")
print(f"Knowledge count: {diag['knowledge_count']}/{diag['max_slots']}")
print(f"Activation rate: {diag['activation_rate']:.2%}")
print(f"Avg gate value: {diag['gate_mean_avg']:.4f}")
print(f"Avg entropy: {diag['entropy_mean_avg']:.4f}")

# Slot governance metrics (v4.4.0)
print(f"Pinned count: {diag.get('pinned_count', 0)}")
print(f"Retriever budget: {diag.get('retriever_budget', 0)}")
print(f"Retrieval step: {diag.get('retrieval_step_counter', 0)}")

if 'latency_p95_us' in diag:
    print(f"P95 latency: {diag['latency_p95_us']:.1f} us")
```

### 8.2 Audit Trail

```python
trail = plugin.get_audit_trail(limit=50)
for entry in trail:
    print(f"[{entry['operation']}] success={entry['success']} details={entry['details']}")

# Filter by operation type
registers = plugin.get_audit_trail(limit=100, operation="register")
```

### 8.3 KV Store Statistics

```python
stats = plugin.get_store_stats()
print(f"Knowledge count: {stats['count']}")
print(f"Pinned count: {stats['pinned_count']}")
print(f"Unpinned count: {stats['unpinned_count']}")
print(f"VRAM usage: {stats['vram_bytes'] / 1024 / 1024:.2f} MB")
```

### 8.4 Event Bus

```python
def my_handler(event):
    if event.data.get("aga_applied"):
        print(f"AGA injected! gate={event.data['gate_mean']:.4f}")

plugin.event_bus.subscribe("forward", my_handler)
```

### 8.5 Integration with aga-observability

```python
plugin = AGAPlugin(AGAConfig(
    hidden_dim=4096,
    observability_enabled=True,
    prometheus_enabled=True,
    prometheus_port=9090,
))
# Prometheus metrics available at: http://localhost:9090/metrics
```

---

## 9. Advanced Usage

### 9.1 Shared Knowledge Across Models

```python
from aga import AGAPlugin, AGAConfig, KVStore

shared_store = KVStore(max_slots=1000, key_dim=64, value_dim=4096)
shared_store.put("fact_001", key_tensor, value_tensor, reliability=0.95)

plugin_a = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin_a.store = shared_store
plugin_a.attach(model_a)

plugin_b = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin_b.store = shared_store
plugin_b.attach(model_b)
```

### 9.2 Namespace Isolation

```python
plugin.register("med_001", key=k1, value=v1,
                metadata={"namespace": "cardiology"})
plugin.register("law_001", key=k2, value=v2,
                metadata={"namespace": "contract_law"})

plugin.clear(namespace="cardiology")
```

### 9.3 Dynamic Knowledge Updates

```python
# Update (same ID overwrites)
plugin.register("fact_001", key=new_key, value=new_value)

# Remove outdated
plugin.unregister("outdated_fact")

# Add new
plugin.register("breaking_news", key=news_key, value=news_value, reliability=0.8)
```

### 9.4 Tensor Parallelism (v4.4.0)

```python
from aga.distributed import TPManager

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
tp_manager = TPManager(plugin)

# On primary rank: register knowledge
if tp_manager.is_primary:
    plugin.register("fact_001", key=k, value=v)

# Broadcast to all ranks
tp_manager.broadcast_knowledge()
tp_manager.broadcast_parameters()
```

### 9.5 Custom Event Handling

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

## 10. Production Deployment

### 10.1 Deployment Checklist

- [ ] Confirm `hidden_dim` matches target model
- [ ] Configure appropriate `max_slots` (based on knowledge volume)
- [ ] Enable `fail_open: true` (enabled by default)
- [ ] Enable `enable_norm_clipping: true` (enabled by default)
- [ ] Tune `tau_low` and `tau_high` (based on domain characteristics)
- [ ] Configure retriever if using external knowledge source
- [ ] Configure slot governance for retriever-heavy workloads
- [ ] If using vLLM, run `VLLMAdapter.check_compatibility()` to verify
- [ ] If using TP, set up `TPManager` for knowledge synchronization
- [ ] Consider installing `aga-observability` for Prometheus monitoring

### 10.2 Recommended Configurations

**General scenario**:

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

**Vertical domain (high activation rate)**:

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

**Low latency scenario**:

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

### 10.3 Performance Tuning

| Parameter              | Increase Effect              | Decrease Effect            |
| ---------------------- | ---------------------------- | -------------------------- |
| `max_slots`            | More knowledge, higher VRAM  | Less knowledge, lower VRAM |
| `gate2_top_k`          | More precise, higher latency | Faster, may miss matches   |
| `tau_low`              | Less activation, lower lat.  | More activation, more inj. |
| `tau_high`             | More tolerant high-entropy   | Stricter high-entropy limit|
| `decay_gamma`          | Slower decay, longer effect  | Faster decay, shorter eff. |
| `early_exit_threshold` | More aggressive early exit   | More complete injections   |
| `retriever_cooldown`   | Less retrieval, more stable  | More retrieval, more fresh |
| `retriever_dedup_sim`  | More dedup, less diversity   | Less dedup, more diversity |

---

## 11. Troubleshooting

### 11.1 Common Issues

**Issue: AGA never activates (activation rate = 0%)**

Possible causes:
1. No knowledge registered -> Check `plugin.knowledge_count`
2. `tau_low` set too high -> Lower `tau_low`
3. Model is confident about current input -> Try more challenging inputs
4. `early_exit_threshold` set too high -> Lower threshold

**Issue: AGA always activates (activation rate ~ 100%)**

Possible causes:
1. `tau_low` set too low -> Increase `tau_low`
2. Key vectors match all inputs -> Check key vector quality

**Issue: Inference quality degradation**

Possible causes:
1. Poor value vector quality -> Check encoding quality
2. `max_gate` set too high -> Lower `max_gate`
3. Insufficient decay -> Lower `decay_gamma`
4. Slot thrashing from retriever -> Increase `retriever_cooldown_steps` or lower `retriever_slot_ratio`

**Issue: VRAM insufficient**

Solutions:
1. Reduce `max_slots`
2. Reduce number of hooked layers
3. Use CPU device (`device: "cpu"`)

### 11.2 Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("aga").setLevel(logging.DEBUG)
```

### 11.3 Fail-Open Behavior

When AGA encounters any exception with `fail_open=True` (default):
1. Logs a warning
2. Returns original model output (no injection)
3. Continues normal operation

---

## 12. FAQ

### Q1: Can aga-core be used standalone?

**Yes.** The only dependency is `torch>=2.0.0`. With the standard Retriever protocol (v4.4.0), you can connect to any external knowledge source without needing `aga-knowledge`.

### Q2: Does AGA modify model parameters?

**No.** AGA attaches via `register_forward_hook`, no model weights are modified.

### Q3: What is AGA's impact on inference speed?

In most scenarios, AGA's bypass rate is >60%, with <5% impact on overall inference speed.

### Q4: Can AGA be activated multiple times in one inference?

**Yes.** AGA independently evaluates at each token position in each hooked layer. In vertical domain scenarios, AGA may be activated at multiple token positions and layers.

### Q5: How does AGA work during streaming generation?

AGA works automatically at each decode step through the hook mechanism. `StreamingSession` provides session management and real-time diagnostics, but AGA's core injection logic doesn't require a special streaming API.

### Q6: Can AGA be used with LoRA?

**Yes.** AGA injects via hooks, LoRA injects via adapter weights — they don't conflict.

### Q7: Which models are supported?

Built-in support for all major HuggingFace model architectures and the vLLM inference framework. For non-standard architectures, implement the `LLMAdapter` interface.

### Q8: Does AGA support vLLM?

**Yes.** AGA provides a native `VLLMAdapter` — no fork of vLLM required. Use `VLLMAdapter.extract_model()` to extract vLLM's internal model, then attach with the standard `plugin.attach()`.

### Q9: What is Knowledge Pinning?

Pinned knowledge is protected from LRU eviction. When `pin_registered=True` (default), all knowledge registered via `register()` is automatically pinned. Knowledge injected by the retriever is not pinned, ensuring core knowledge always takes priority.

### Q10: How does the Retriever protocol work?

Implement the `BaseRetriever` interface with a `retrieve()` method. AGA automatically calls it when entropy is high, subject to Slot Governance rules (budget, cooldown, dedup). This allows AGA to connect to any existing infrastructure without needing `aga-knowledge`.

### Q11: How to choose max_slots?

| Knowledge Volume | Recommended max_slots | VRAM (hidden_dim=4096) |
| ---------------- | --------------------- | ---------------------- |
| <100 entries     | 128                   | ~1 MB                  |
| 100-500 entries  | 256-512               | ~2-4 MB                |
| 500-2000 entries | 1000                  | ~8 MB                  |
| >5000 entries    | Use retriever + slots | On-demand loading      |
