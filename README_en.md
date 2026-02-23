# AGA Plugin Ecosystem

<p align="center">
  <strong>Lossless Capability Extension for Frozen LLMs</strong><br/>
  Attention Governance · Knowledge Management · Observability
</p>

<p align="center">
  <img src="https://img.shields.io/badge/aga--core-v4.4.0-blue" alt="aga-core"/>
  <img src="https://img.shields.io/badge/aga--knowledge-v0.3.0-green" alt="aga-knowledge"/>
  <img src="https://img.shields.io/badge/aga--observability-v1.0.0-orange" alt="aga-observability"/>
  <img src="https://img.shields.io/badge/python-3.9+-brightgreen" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license"/>
</p>

---

## What is AGA?

**AGA (Auxiliary Governed Attention)** is a **runtime attention governance plugin** for frozen Large Language Models. When an LLM encounters knowledge gaps during inference (manifested as high entropy), AGA automatically injects external knowledge into the Transformer's attention layers — **without modifying any model parameters**.

**AGA is not RAG, not LoRA, not Prompt Engineering.** It operates at the attention layer level, mid-inference, providing atomic-fact injection driven by the model's own entropy signals.

```
Token → Transformer Layer → Self-Attention → [Entropy High?] → AGA Injection → Fused Output
```

---

## Mono-Repo Structure

This repository contains three independent but interoperable Python packages:

```
AGAPlugin/
├── aga/                    ← aga-core (required)
│   ├── plugin.py           # AGAPlugin — 3-line integration
│   ├── config.py           # AGAConfig — full externalization
│   ├── kv_store.py         # GPU-resident KV storage
│   ├── gate/               # 3-stage entropy gating
│   ├── operator/           # Bottleneck injection
│   ├── retriever/          # BaseRetriever protocol
│   ├── adapter/            # HuggingFace / vLLM adapters
│   ├── streaming.py        # Streaming generation session
│   ├── distributed.py      # Tensor Parallelism support
│   └── instrumentation/    # EventBus, metrics, audit
│
├── aga_knowledge/          ← aga-knowledge (optional)
│   ├── portal/             # FastAPI REST API
│   ├── persistence/        # SQLite / PostgreSQL / Redis
│   ├── encoder/            # Text→Vector (SentenceTransformer)
│   ├── retriever/          # HNSW + BM25 + RRF hybrid search
│   ├── chunker/            # Document → Knowledge fragments
│   ├── alignment.py        # AGACoreAlignment
│   └── sync/               # Redis Pub/Sub synchronization
│
├── aga_observability/      ← aga-observability (optional)
│   ├── prometheus_exporter.py  # Prometheus metrics
│   ├── grafana_dashboard.py    # Auto-generated dashboards
│   ├── alert_manager.py        # SLO/SLI alerting
│   ├── audit_storage.py        # Persistent audit trail
│   └── health.py               # Health check HTTP endpoint
│
├── configs/                # Example configuration files
│   ├── portal_config.yaml  # aga-knowledge Portal config
│   └── runtime_config.yaml # aga-core runtime config
│
├── tests/                  # All unit tests
├── pyproject.toml          # Root package (aga-core)
└── README_en.md            # This file
```

### Package Dependency

```
+-----------------------------------------------------+
|                   AGA Ecosystem                      |
|                                                      |
|  +-------------+                                     |
|  |  aga-core   | ← Required (only dep: torch)        |
|  |  v4.4.0     |                                     |
|  +------+------+                                     |
|         |                                            |
|  +------v------+  +--------------------+             |
|  |aga-knowledge|  | aga-observability  | ← Optional  |
|  | v0.3.0      |  | v1.0.0            |              |
|  | (no aga-core|  | (requires aga-core)|             |
|  |  dependency)|  |                    |             |
|  +-------------+  +--------------------+             |
+-----------------------------------------------------+
```

- **aga-core** can be used **completely standalone**. Only dependency: `torch>=2.0.0`.
- **aga-knowledge** is independent — it manages plaintext knowledge and provides vector retrieval via the `BaseRetriever` protocol.
- **aga-observability** requires `aga-core` — it subscribes to `EventBus` events for monitoring.

---

## Quick Start

### 3-Line Integration (aga-core only)

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)
output = model.generate(input_ids)  # AGA works automatically
```

### Full Stack (aga-core + aga-knowledge + aga-observability)

```python
from aga import AGAPlugin, AGAConfig
from aga_knowledge import KnowledgeManager, AGACoreAlignment
from aga_knowledge.config import PortalConfig
from aga_knowledge.encoder import create_encoder, EncoderConfig
from aga_knowledge.retriever import KnowledgeRetriever

# 1. Alignment
alignment = AGACoreAlignment(
    hidden_dim=4096, bottleneck_dim=64,
    key_norm_target=5.0, value_norm_target=3.0,
)

# 2. Knowledge Management
manager = KnowledgeManager(PortalConfig.for_development())
await manager.start()

# 3. Encoder + Retriever
encoder = create_encoder(EncoderConfig.from_alignment(alignment))
retriever = KnowledgeRetriever(
    manager=manager, encoder=encoder,
    alignment=alignment, namespace="default",
)

# 4. Plugin
config = AGAConfig(
    hidden_dim=4096, bottleneck_dim=64,
    observability_enabled=True,  # auto-detects aga-observability
)
plugin = AGAPlugin(config, retriever=retriever)
plugin.attach(model)
```

---

## Installation

### From Source (Mono-Repo)

```bash
cd AGAPlugin

# Install aga-core only
pip install -e .

# Install aga-knowledge
pip install -e ./aga_knowledge[all]

# Install aga-observability
pip install -e ./aga_observability[full]

# Install everything
pip install -e .[all]
pip install -e ./aga_knowledge[all]
pip install -e ./aga_observability[full]
```

### From PyPI (when published)

```bash
pip install aga-core                          # Core only
pip install aga-core[knowledge,observability] # Full stack
```

---

## Documentation

### aga-core

| Document | Language |
| --- | --- |
| [README (English)](aga/README_en.md) | English |
| [README (中文)](aga/README_zh.md) | 中文 |
| [Product Documentation (English)](aga/docs/product_doc_en.md) | English |
| [产品说明书 (中文)](aga/docs/product_doc_zh.md) | 中文 |
| [User Manual (English)](aga/docs/user_manual_en.md) | English |
| [用户手册 (中文)](aga/docs/user_manual_zh.md) | 中文 |

### aga-knowledge

| Document | Language |
| --- | --- |
| [README (English)](aga_knowledge/README_en.md) | English |
| [README (中文)](aga_knowledge/README_zh.md) | 中文 |
| [Product Documentation (English)](aga_knowledge/docs/product_doc_en.md) | English |
| [产品说明书 (中文)](aga_knowledge/docs/product_doc_zh.md) | 中文 |
| [User Manual (English)](aga_knowledge/docs/user_manual_en.md) | English |
| [用户手册 (中文)](aga_knowledge/docs/user_manual_zh.md) | 中文 |

### aga-observability

| Document | Language |
| --- | --- |
| [README (English)](aga_observability/README_en.md) | English |
| [README (中文)](aga_observability/README_zh.md) | 中文 |
| [User Manual (English)](aga_observability/docs/user_manual_en.md) | English |
| [用户手册 (中文)](aga_observability/docs/user_manual_zh.md) | 中文 |

---

## Configuration

Example configuration files are provided in the `configs/` directory:

| File | Purpose | Used By |
| --- | --- | --- |
| [`configs/portal_config.yaml`](configs/portal_config.yaml) | Knowledge Portal server, persistence, messaging, and governance configuration | `aga-knowledge` Portal |
| [`configs/runtime_config.yaml`](configs/runtime_config.yaml) | AGA runtime parameters, entropy gating, decay, sync, and device configuration | `aga-core` AGAPlugin |

### Usage

```python
# aga-core: load runtime config
from aga import AGAPlugin
plugin = AGAPlugin.from_config("configs/runtime_config.yaml")

# aga-knowledge: load portal config
from aga_knowledge.config import PortalConfig
config = PortalConfig.from_yaml("configs/portal_config.yaml")
```

---

## Key Features by Package

### aga-core v4.4.0

- **3-line integration** — `AGAPlugin(config).attach(model)`
- **3-stage entropy gating** — Gate-0 (namespace) / Gate-1 (entropy) / Gate-2 (confidence)
- **Bottleneck attention injection** — Query projection → Top-K routing → Value projection
- **GPU-resident KVStore** — LRU eviction + knowledge pinning + namespace isolation
- **Streaming generation** — Per-token knowledge injection during generation
- **Standard Retriever protocol** — `BaseRetriever` interface for pluggable retrieval
- **Slot Governance** — Budget control, semantic dedup, cooldown, stability detection
- **HuggingFace + vLLM adapters** — LLaMA, Qwen, Mistral, GPT-2, Phi, Gemma, Falcon
- **Tensor Parallelism** — `TPManager` for multi-GPU KVStore broadcast
- **Fail-Open safety** — Exceptions never block inference

### aga-knowledge v0.3.0

- **Plaintext knowledge** — `condition/decision` pairs, human-readable
- **4 persistence backends** — Memory, SQLite, PostgreSQL, Redis
- **Portal REST API** — FastAPI-based knowledge CRUD + image asset serving
- **Hybrid retrieval** — HNSW (dense) + BM25 (sparse) + RRF (fusion)
- **AGACoreAlignment** — Encoder-core dimension/norm mandatory alignment
- **Document chunking** — 5 strategies + DocumentChunker + ConditionGenerator + ImageHandler
- **Cross-instance sync** — Redis Pub/Sub real-time knowledge synchronization
- **Version control** — Knowledge versioning with rollback and diff

### aga-observability v1.0.0

- **Prometheus exporter** — 15+ metrics (counters, histograms, gauges)
- **Grafana dashboards** — Auto-generated 5-group dashboard JSON
- **SLO/SLI alerting** — Configurable rules with webhook/callback channels
- **Structured logging** — JSON/Text format with file rotation
- **Persistent audit** — JSONL or SQLite with retention policies
- **Health checking** — HTTP endpoint for Kubernetes probes
- **Zero intrusion** — EventBus subscription, no aga-core modification

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# aga-core tests
python -m pytest tests/test_aga/ -v

# aga-knowledge tests
python -m pytest tests/test_knowledge/ -v

# aga-observability tests
python -m pytest tests/test_observability/ -v
```

---

## Roadmap

| Package | Current | Next Milestone |
| --- | --- | --- |
| **aga-core** | v4.4.0 — Retriever protocol, slot governance, streaming | v5.0 — Per-layer knowledge, INT8 KVStore, adaptive bottleneck |
| **aga-knowledge** | v0.3.0 — HNSW+BM25+RRF, DocumentChunker, AGACoreAlignment | v0.4.x — Contrastive fine-tuning, distributed encoder, Prometheus |
| **aga-observability** | v1.0.0 — Prometheus, Grafana, alerting, audit, health | v1.1.0 — OpenTelemetry traces, distributed aggregation |

---

## License

MIT License

Copyright (c) 2024-2026 AGA Team

---

<p align="center">
  <strong>AGA — Empowering every inference with knowledge</strong>
</p>
