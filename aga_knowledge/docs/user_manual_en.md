# aga-knowledge User Manual

**Version**: 0.3.0  
**Last Updated**: 2026-02-23

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Core Concepts](#3-core-concepts)
4. [Encoder-Core Alignment](#4-encoder-core-alignment)
5. [Knowledge Registration](#5-knowledge-registration)
6. [Persistence Configuration](#6-persistence-configuration)
7. [Knowledge Synchronization](#7-knowledge-synchronization)
8. [Encoder Module](#8-encoder-module)
9. [Knowledge Chunker](#9-knowledge-chunker)
10. [Knowledge Retriever](#10-knowledge-retriever)
11. [Integration with aga-core](#11-integration-with-aga-core)
12. [Portal API Reference](#12-portal-api-reference)
13. [Version Control](#13-version-control)
14. [Text Compression](#14-text-compression)
15. [Configuration Reference](#15-configuration-reference)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Overview

`aga-knowledge` is the knowledge management system for the AGA ecosystem. It manages the full lifecycle of domain knowledge — from registration and storage to encoding and retrieval — enabling `aga-core` to inject relevant facts into frozen LLMs during inference.

### What aga-knowledge Does

1. **Stores** plaintext condition/decision knowledge pairs
2. **Synchronizes** knowledge across distributed runtime instances
3. **Encodes** plaintext into key/value vectors compatible with aga-core (with alignment validation)
4. **Retrieves** relevant knowledge using hybrid search (HNSW + BM25 + RRF) when aga-core's entropy gate triggers
5. **Manages** knowledge lifecycle (probationary → confirmed → deprecated)
6. **Chunks** large documents into knowledge fragments with structure-aware splitting and condition generation
7. **Processes** images in documents, converting them to Portal-accessible URLs

### What aga-knowledge Does NOT Do

- Does not perform inference or model forward passes
- Does not manage GPU resources
- Does not replace aga-core's attention injection mechanism
- Does not require GPU (runs entirely on CPU)

---

## 2. Installation

```bash
# Minimal installation
pip install aga-knowledge

# With specific backends
pip install aga-knowledge[portal]      # FastAPI Portal
pip install aga-knowledge[postgres]    # PostgreSQL
pip install aga-knowledge[redis]       # Redis
pip install aga-knowledge[encoder]     # SentenceTransformer + PyTorch
pip install aga-knowledge[retrieval]   # hnswlib + rank-bm25 + numpy
pip install aga-knowledge[all]         # Everything
```

### Verify Installation

```python
from aga_knowledge import KnowledgeManager, AGACoreAlignment, __version__
print(f"aga-knowledge v{__version__}")
```

---

## 3. Core Concepts

### Knowledge Record

A knowledge record is a `condition/decision` text pair:

| Field | Type | Description |
|-------|------|-------------|
| `lu_id` | str | Unique Learning Unit ID |
| `condition` | str | Trigger condition (when to inject) |
| `decision` | str | Knowledge content (what to inject) |
| `namespace` | str | Isolation namespace (default: "default") |
| `lifecycle_state` | str | Current lifecycle state |
| `trust_tier` | str | Trust level |
| `hit_count` | int | How many times this knowledge was used |
| `version` | int | Version number |

### Lifecycle States

```
probationary ──(confirm)──▶ confirmed ──(deprecate)──▶ deprecated
      │                         │
      └──(quarantine)──▶ quarantined ◀──(quarantine)──┘
```

| State | Reliability | Description |
|-------|-------------|-------------|
| `probationary` | 0.3 | Newly registered, under observation |
| `confirmed` | 1.0 | Validated, full injection weight |
| `deprecated` | 0.1 | Outdated, minimal injection weight |
| `quarantined` | 0.0 | Disabled, excluded from retrieval |

### Trust Tiers

| Tier | Priority | Description |
|------|----------|-------------|
| `system` | 100 | Core rules, highest trust |
| `verified` | 80 | Human-reviewed and confirmed |
| `standard` | 50 | Default tier |
| `experimental` | 30 | Under testing |
| `untrusted` | 10 | Use with caution |

### Namespaces

Namespaces provide knowledge isolation. Each namespace is an independent knowledge domain:

```python
# Medical knowledge
await manager.get_active_knowledge("medical")

# Legal knowledge
await manager.get_active_knowledge("legal")
```

---

## 4. Encoder-Core Alignment (New in v0.3.0)

The `AGACoreAlignment` dataclass ensures aga-knowledge's encoder generates vectors that are directly usable by aga-core. **Misalignment is detected at startup and raises `ConfigError`.**

### Creating Alignment

```python
from aga_knowledge import AGACoreAlignment

# Method 1: Manual configuration (recommended for production)
alignment = AGACoreAlignment(
    hidden_dim=4096,
    bottleneck_dim=64,
    key_norm_target=5.0,
    value_norm_target=3.0,
)

# Method 2: From aga-core YAML config file
alignment = AGACoreAlignment.from_aga_config_yaml("/path/to/aga_config.yaml")

# Method 3: From AGAConfig instance (development)
from aga.config import AGAConfig
aga_config = AGAConfig(bottleneck_dim=64, hidden_dim=4096)
alignment = AGACoreAlignment.from_aga_config(aga_config)
```

### Using Alignment with Encoder

```python
from aga_knowledge.encoder import EncoderConfig, create_encoder

# Create aligned encoder config (recommended)
encoder_config = EncoderConfig.from_alignment(
    alignment,
    backend="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
)
# encoder_config.key_dim == 64 (from bottleneck_dim)
# encoder_config.value_dim == 4096 (from hidden_dim)

encoder = create_encoder(encoder_config)
```

### YAML Configuration

```yaml
# In portal_config.yaml
aga_core_alignment:
    hidden_dim: 4096
    bottleneck_dim: 64
    key_norm_target: 5.0
    value_norm_target: 3.0

# Or reference aga-core's config file directly
aga_core_alignment:
    aga_core_config_path: "/path/to/aga_config.yaml"
```

---

## 5. Knowledge Registration

### Via Portal API

Start the Portal server:

```bash
aga-portal  # or: python -m aga_knowledge.portal.app
```

Register knowledge via HTTP:

```bash
# Single injection
curl -X POST http://localhost:8081/knowledge/inject-text \
  -H "Content-Type: application/json" \
  -d '{
    "lu_id": "med_001",
    "condition": "Patient presents with chest pain and shortness of breath",
    "decision": "Consider acute coronary syndrome. Order ECG, troponin, and chest X-ray.",
    "namespace": "medical",
    "lifecycle_state": "probationary",
    "trust_tier": "verified"
  }'

# Batch injection
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

### Via KnowledgeManager (Programmatic)

```python
from aga_knowledge.sync import SyncMessage

# Create injection message
msg = SyncMessage.inject(
    lu_id="rule_001",
    condition="When the user asks about return policy",
    decision="Our return policy allows returns within 30 days with receipt.",
    namespace="customer_service",
)

# Publish via sync backend
await publisher.publish(msg)
```

---

## 6. Persistence Configuration

### Memory (Testing)

```yaml
persistence:
  type: "memory"
```

### SQLite (Development)

```yaml
persistence:
  type: "sqlite"
  sqlite_path: "aga_knowledge.db"
  enable_audit: true
```

### PostgreSQL (Production)

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

Or via DSN:

```yaml
persistence:
  type: "postgres"
  postgres_url: "postgresql://aga:secret@db.example.com:5432/aga_knowledge"
```

### Redis (Cache Layer)

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

## 7. Knowledge Synchronization

### Architecture

```
Portal ──(publish)──▶ Redis Pub/Sub ──(subscribe)──▶ KnowledgeManager (Instance 1)
                                    ──(subscribe)──▶ KnowledgeManager (Instance 2)
                                    ──(subscribe)──▶ KnowledgeManager (Instance N)
```

### Configuration

```yaml
messaging:
  backend: "redis"          # redis | memory
  redis_host: "localhost"
  redis_port: 6379
  redis_channel: "aga:sync"
```

### Message Types

| Type | Description |
|------|-------------|
| `INJECT` | Register new knowledge |
| `UPDATE` | Update lifecycle state |
| `QUARANTINE` | Disable knowledge |
| `DELETE` | Remove knowledge |
| `BATCH_INJECT` | Bulk registration |
| `FULL_SYNC` | Request full state sync |
| `HEARTBEAT` | Instance liveness |
| `ACK` / `NACK` | Delivery confirmation |

---

## 8. Encoder Module

The encoder converts plaintext `condition/decision` into `key/value` vectors for aga-core.

### SentenceTransformerEncoder (Recommended)

```python
from aga_knowledge import AGACoreAlignment
from aga_knowledge.encoder import create_encoder, EncoderConfig

# Create aligned encoder (recommended)
alignment = AGACoreAlignment(hidden_dim=4096, bottleneck_dim=64)
encoder_config = EncoderConfig.from_alignment(alignment)
encoder = create_encoder(encoder_config)

# Or manual configuration
encoder = create_encoder(EncoderConfig(
    backend="sentence_transformer",
    model_name="all-MiniLM-L6-v2",  # 384-dim embeddings
    key_dim=64,                      # Must match AGAConfig.bottleneck_dim
    value_dim=4096,                  # Must match AGAConfig.hidden_dim
    device="cpu",
    normalize=True,
    options={
        "condition_prefix": "condition: ",
        "decision_prefix": "decision: ",
        "key_norm_target": 5.0,
        "value_norm_target": 3.0,
    },
))

# Encode single knowledge
encoded = encoder.encode(
    condition="Patient has fever above 39C",
    decision="Administer antipyretic medication",
    lu_id="med_001",
)
# encoded.key_vector: [64]
# encoded.value_vector: [4096]

# Batch encode
records = [{"condition": "...", "decision": "...", "lu_id": "..."}]
encoded_list = encoder.encode_batch(records)
```

### Encoding Pipeline

```
condition → SentenceTransformer → [384] → key_proj(384→64) → normalize → scale(5.0) → key [64]
decision  → SentenceTransformer → [384] → value_proj(384→4096) → normalize → scale(3.0) → value [4096]
```

### Projection Layer Training

The projection layers (`key_proj`, `value_proj`) are initialized with Xavier uniform. For better results, fine-tune them:

```python
# Save projections after training
encoder.save_projections("projections_v1.pt")

# Load pre-trained projections
encoder = create_encoder(EncoderConfig(
    options={"projection_path": "projections_v1.pt"}
))
```

### SimpleHashEncoder (Testing Only)

```python
encoder = create_encoder(EncoderConfig(
    backend="simple_hash",
    key_dim=64,
    value_dim=4096,
))
```

> **Warning**: SimpleHashEncoder has no semantic understanding. Use only for testing.

---

## 9. Knowledge Chunker

The chunker splits large documents into knowledge fragments suitable for AGA injection (100-500 tokens).

### Basic Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| `fixed_size` | Uniform documents | Split by token count |
| `sentence` | Narrative text | Split at sentence boundaries |
| `semantic` | Technical documents | Group semantically similar sentences |
| `sliding_window` | General purpose | Overlapping windows for context continuity |

### Basic Usage

```python
from aga_knowledge.chunker import create_chunker, ChunkerConfig

chunker = create_chunker(ChunkerConfig(
    strategy="sliding_window",
    chunk_size=300,       # Target tokens per chunk
    overlap=50,           # Overlap between chunks
    min_chunk_size=50,
    max_chunk_size=500,
    condition_mode="first_sentence",
    language="auto",
))

# Chunk a document
chunks = chunker.chunk_document(
    text=document_text,
    source_id="doc_001",
    title="Medical Guidelines Chapter 3",
)
```

### Document-Level Chunking (New in v0.3.0)

For structured documents (Markdown), use `DocumentChunker` for better results:

```python
from aga_knowledge.chunker import create_document_chunker, ChunkerConfig

doc_chunker = create_document_chunker(ChunkerConfig(
    strategy="sliding_window",
    chunk_size=300,
    overlap=50,
    condition_mode="title_context",  # Uses section hierarchy as condition
))

chunks = doc_chunker.chunk_document(
    text=markdown_text,
    source_id="doc_001",
    title="Medical Guidelines",
)

# Each chunk has enhanced condition from document structure
for chunk in chunks:
    print(f"Condition: {chunk.condition}")
    print(f"Decision: {chunk.decision[:100]}...")
    record = chunk.to_knowledge_record()
    # Register via Portal API or KnowledgeManager
```

### Condition Generation Strategies

| Mode | Description | Best For |
|------|-------------|----------|
| `first_sentence` | Uses the first sentence of the chunk | General text |
| `title_context` | Section title hierarchy as condition | Structured documents |
| `keyword` | Top keywords extracted from text | Keyword-heavy content |
| `summary` | Keyword-based summary | Long sections |

### Image Handling in Documents (New in v0.3.0)

Process images embedded in knowledge documents:

```python
from aga_knowledge.chunker import ImageHandler, create_document_chunker, ChunkerConfig

# Create image handler
image_handler = ImageHandler(
    asset_dir="/var/aga-knowledge/assets",
    base_url="http://portal:8081/assets",
    max_image_size_mb=10,
    supported_formats=["png", "jpg", "jpeg", "gif", "webp"],
    description_template="[Image: {alt}, see {url}]",
)

# Create document chunker with image handling
doc_chunker = create_document_chunker(
    ChunkerConfig(strategy="sliding_window", chunk_size=300),
    image_handler=image_handler,
)

# Images in the document are:
# 1. Extracted (Base64/URL/local path)
# 2. Saved to Portal assets directory
# 3. Replaced with accessible URL descriptions in decision text
```

---

## 10. Knowledge Retriever

`KnowledgeRetriever` bridges `aga-knowledge` with `aga-core`'s `BaseRetriever` protocol. In v0.3.0, it supports hybrid search with HNSW + BM25 + RRF fusion.

### Setup

```python
from aga_knowledge import AGACoreAlignment
from aga_knowledge.retriever import KnowledgeRetriever

# Alignment is mandatory
alignment = AGACoreAlignment(hidden_dim=4096, bottleneck_dim=64)

retriever = KnowledgeRetriever(
    manager=knowledge_manager,      # KnowledgeManager instance
    encoder=encoder,                # BaseEncoder instance
    alignment=alignment,            # AGACoreAlignment (required)
    namespace="default",
    auto_refresh_interval=60,       # Refresh index every 60s
    similarity_threshold=0.3,       # Minimum cosine similarity
    # HNSW configuration
    index_backend="hnsw",           # "hnsw" or "brute"
    hnsw_m=16,
    hnsw_ef_construction=200,
    hnsw_ef_search=100,
    hnsw_max_elements=100000,
    # BM25 configuration
    bm25_enabled=True,
    bm25_weight=0.3,                # Weight in RRF fusion
    bm25_k1=1.5,
    bm25_b=0.75,
)

# Pass to aga-core
plugin = AGAPlugin(config, retriever=retriever)
```

### How It Works

1. **Alignment Validation**: At construction, validates encoder dimensions match aga-core via `AGACoreAlignment`
2. **Warmup**: Loads active knowledge from KnowledgeManager, encodes to vectors, builds HNSW + BM25 indexes
3. **Retrieve**: When aga-core's entropy gate triggers:
   - Receives `hidden_states` from the attention layer
   - Projects to query vector (key_dim)
   - Dense search via HNSW (or brute-force fallback)
   - Sparse search via BM25 (if text query available)
   - RRF fusion combines both results
   - Returns top-k results as `RetrievalResult` objects
4. **Feedback**: After injection, aga-core reports which knowledge was actually used

### Graceful Degradation

| Condition | Behavior |
|-----------|----------|
| hnswlib not installed | Auto-fallback to brute-force O(N) search |
| rank-bm25 not installed | BM25 disabled, pure dense retrieval |
| No text query available | BM25 skipped, pure dense retrieval |
| Retrieval error | Fail-Open: return empty list |

### Statistics

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

## 11. Integration with aga-core

### Complete Pipeline (v0.3.0)

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
    # 1. Alignment (bridge between aga-core and aga-knowledge)
    alignment = AGACoreAlignment(
        hidden_dim=4096,
        bottleneck_dim=64,
        key_norm_target=5.0,
        value_norm_target=3.0,
    )

    # 2. Knowledge Manager
    config = PortalConfig.for_production(
        postgres_url="postgresql://aga:pass@db:5432/aga_knowledge",
        redis_host="redis",
    )
    manager = KnowledgeManager(config, namespaces=["medical"])
    await manager.start()

    # 3. Aligned Encoder
    encoder_config = EncoderConfig.from_alignment(
        alignment,
        options={"projection_path": "medical_projections.pt"},
    )
    encoder = create_encoder(encoder_config)

    # 4. Hybrid Retriever
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

    # 5. AGA Plugin
    aga_config = AGAConfig(
        bottleneck_dim=64,
        hidden_dim=4096,
        max_slots=32,
    )
    plugin = AGAPlugin(aga_config, retriever=retriever)
    plugin.attach(model)

    return plugin, manager
```

### Dimension Alignment Checklist

| aga-core Parameter | aga-knowledge Parameter | Enforcement |
|--------------------|------------------------|-------------|
| `AGAConfig.bottleneck_dim` | `EncoderConfig.key_dim` | Mandatory (ConfigError if mismatch) |
| `AGAConfig.hidden_dim` | `EncoderConfig.value_dim` | Mandatory (ConfigError if mismatch) |
| `AGAConfig.key_norm_target` | `EncoderConfig.options["key_norm_target"]` | Mandatory (ConfigError if mismatch) |
| `AGAConfig.value_norm_target` | `EncoderConfig.options["value_norm_target"]` | Mandatory (ConfigError if mismatch) |

---

## 12. Portal API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/knowledge/inject-text` | Register single knowledge |
| POST | `/knowledge/batch-text` | Batch register |
| GET | `/knowledge/{ns}/{lu_id}` | Get single knowledge |
| GET | `/knowledge/{ns}` | Query knowledge list |
| DELETE | `/knowledge/{ns}/{lu_id}` | Delete knowledge |
| PUT | `/lifecycle/update` | Update lifecycle state |
| POST | `/lifecycle/quarantine` | Quarantine knowledge |
| GET | `/statistics` | Global statistics |
| GET | `/statistics/{ns}` | Namespace statistics |
| GET | `/audit` | Audit log |
| GET | `/namespaces` | List namespaces |
| GET | `/health` | Health check |
| GET | `/health/ready` | Readiness check |
| GET | `/assets/{path}` | Serve image assets (v0.3.0) |

---

## 13. Version Control

```python
from aga_knowledge.persistence import VersionedKnowledgeStore

store = VersionedKnowledgeStore(max_versions=10)

# Save version
store.save_version(
    lu_id="rule_001",
    condition="Updated condition",
    decision="Updated decision",
    lifecycle_state="confirmed",
    trust_tier="verified",
    created_by="admin",
    change_reason="Corrected medical dosage",
)

# Get history
history = store.get_history("rule_001")

# Rollback
old_version = store.rollback("rule_001", target_version=2)

# Compare versions
diff = store.diff("rule_001", version_a=1, version_b=3)
```

---

## 14. Text Compression

```python
from aga_knowledge.persistence import TextCompressor, TextCompressionConfig, CompressionAlgorithm

compressor = TextCompressor(TextCompressionConfig(
    algorithm=CompressionAlgorithm.ZLIB,
    zlib_level=6,
))

# Compress
compressed = compressor.compress_text("Long knowledge text...")
# Decompress
original = compressor.decompress_text(compressed)

# Batch operations
texts = ["text1", "text2", "text3"]
compressed_batch = compressor.compress_batch(texts)
```

---

## 15. Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGA_PORTAL_HOST` | Portal server host | `0.0.0.0` |
| `AGA_PORTAL_PORT` | Portal server port | `8081` |
| `AGA_PERSISTENCE_TYPE` | Storage backend | `sqlite` |
| `AGA_PERSISTENCE_SQLITE_PATH` | SQLite file path | `aga_knowledge.db` |
| `AGA_PERSISTENCE_POSTGRES_URL` | PostgreSQL DSN | - |
| `AGA_MESSAGING_BACKEND` | Sync backend | `redis` |
| `AGA_MESSAGING_REDIS_HOST` | Redis host | `localhost` |
| `AGA_ENVIRONMENT` | Environment name | `development` |

### Programmatic Configuration

```python
from aga_knowledge.config import PortalConfig

# Development
config = PortalConfig.for_development()

# Production
config = PortalConfig.for_production(
    postgres_url="postgresql://...",
    redis_host="redis.example.com",
)

# From YAML
from aga_knowledge.config import load_config
config = load_config("portal_config.yaml")
```

---

## 16. Troubleshooting

### Common Issues

**Q: ConfigError: Encoder configuration not aligned with aga-core**
```
ConfigError: 编码器配置与 aga-core 不对齐:
  - key_dim (64) != AGAConfig.bottleneck_dim (128)
```
A: Use `EncoderConfig.from_alignment(alignment)` to create the encoder config, or ensure `EncoderConfig.key_dim == AGAConfig.bottleneck_dim` and `EncoderConfig.value_dim == AGAConfig.hidden_dim`.

**Q: KnowledgeRetriever returns empty results**
- Check that knowledge is registered and active (not quarantined)
- Lower `similarity_threshold` (try 0.0 for debugging)
- Verify encoder is initialized (`retriever.get_stats()["encoder"]["initialized"]`)
- Check index size (`retriever.get_stats()["index_size"]`)
- Verify alignment is correct (`retriever.get_stats()["alignment"]`)

**Q: HNSW not being used despite configuration**
- Ensure `hnswlib` is installed: `pip install hnswlib`
- Check logs for "hnswlib 未安装" warning
- Verify `index_backend="hnsw"` is set

**Q: BM25 results not appearing**
- Ensure `rank-bm25` is installed: `pip install rank-bm25`
- Ensure `bm25_enabled=True` is set
- BM25 requires text query in `query.metadata["query_text"]` — if only vector query is available, BM25 is automatically skipped

**Q: Redis sync not working**
- Verify Redis is running and accessible
- Check `messaging.redis_channel` matches between Portal and Runtime
- Ensure `messaging.backend` is set to `"redis"` (not `"memory"`)

**Q: PostgreSQL connection pool exhausted**
- Increase `postgres_pool_size` and `postgres_max_overflow`
- Check for connection leaks (ensure `await adapter.disconnect()` is called)

**Q: Chunker produces too many/few chunks**
- Adjust `chunk_size` (target tokens per chunk)
- For Chinese text, set `language: "zh"` for accurate token estimation
- Use `sliding_window` strategy with `overlap` for context continuity
- For structured documents, use `DocumentChunker` with `condition_mode="title_context"`