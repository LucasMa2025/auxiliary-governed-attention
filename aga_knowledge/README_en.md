# aga-knowledge — Knowledge Management System for AGA

> **The external knowledge brain for frozen LLMs — register, encode, retrieve, inject.**

`aga-knowledge` is the standalone knowledge management package for the **AGA (Auxiliary Governed Attention)** ecosystem. It provides the complete pipeline from plaintext knowledge registration to vector-encoded retrieval, bridging domain knowledge with `aga-core`'s attention-layer injection mechanism.

## Why aga-knowledge?

Frozen LLMs lack domain-specific facts. `aga-core` can inject knowledge into attention layers at high-entropy moments, but it needs a **knowledge source**. `aga-knowledge` is that source:

```
Domain Expert → Portal API → Persistence → Sync → KnowledgeManager → Encoder → Retriever → aga-core
```

Without `aga-knowledge`, users must manually construct key/value vectors and call `plugin.register()`. With `aga-knowledge`, the entire pipeline — from plaintext `condition/decision` pairs to production-grade vector retrieval — is automated.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          aga-knowledge v0.3.0                                  │
│                                                                                │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐          │
│  │  Portal  │──▶│ Persistence  │──▶ │   Sync   │──▶│  Knowledge   │          │
│  │  (REST)  │    │  Adapters    │    │  (Redis) │    │   Manager    │          │
│  │  +Assets │    └──────────────┘    └──────────┘    └──────┬───────┘          │
│  └──────────┘                                               │                  │
│                                                             │                  │
│  ┌───────────────────────┐    ┌─────────────┐    ┌──────────▼────────────────┐ │
│  │  Chunker              │──▶│  Encoder    │──▶ │ KnowledgeRetriever        │ │
│  │  ├ DocumentChunker    │    │(Text→Vec)   │    │ ├ HNSW dense search       │ │
│  │  ├ ConditionGenerator │    │ +Alignment  │    │ ├ BM25 sparse search      │ │
│  │  ├ ImageHandler       │    └─────────────┘    │ └ RRF fusion              │ │
│  │  └ 4 base strategies  │                       │(BaseRetriever protocol)   │ │
│  └───────────────────────┘                       └──────────┬────────────────┘ │
│                                                             │                  │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────┐        │                  │
│  │Versioning│  │ Compression │  │ AGACoreAlignment │        │                  │
│  └──────────┘  └─────────────┘  └──────────────────┘        │                  │
└─────────────────────────────────────────────────────────────┼──────────────────┘
                                                              │
                                              ┌───────────────▼────────┐
                                              │      aga-core          │
                                              │  (AGAPlugin.attach)    │
                                              │  High-entropy gate     │
                                              │  → retrieve() call     │
                                              │  → KV injection        │
                                              └────────────────────────┘
```

## Feature List

### Core Features (Implemented ✅)

| Category                   | Feature                            | Status | Description                                              |
| -------------------------- | ---------------------------------- | ------ | -------------------------------------------------------- |
| **Knowledge Registration** | Portal REST API                    | ✅     | FastAPI-based HTTP API for knowledge CRUD                |
|                            | Plaintext condition/decision pairs | ✅     | Human-readable knowledge format                          |
|                            | Batch injection                    | ✅     | Bulk knowledge registration                              |
|                            | Namespace isolation                | ✅     | Multi-tenant knowledge separation                        |
|                            | Lifecycle management               | ✅     | Probationary → Confirmed → Deprecated → Quarantined      |
|                            | Trust tiers                        | ✅     | System / Verified / Standard / Experimental / Untrusted  |
| **Persistence**            | Memory adapter                     | ✅     | In-memory storage for testing                            |
|                            | SQLite adapter                     | ✅     | File-based storage for development                       |
|                            | PostgreSQL adapter                 | ✅     | Production-grade relational storage (asyncpg)            |
|                            | Redis adapter                      | ✅     | High-performance cache layer (aioredis)                  |
|                            | Adapter factory                    | ✅     | Configuration-driven adapter creation                    |
|                            | Audit logging                      | ✅     | All CRUD operations are audited                          |
| **Synchronization**        | Redis Pub/Sub                      | ✅     | Real-time knowledge sync across instances                |
|                            | Memory backend                     | ✅     | In-process sync for testing                              |
|                            | Message protocol                   | ✅     | Typed messages with ACK/NACK support                     |
|                            | Full sync                          | ✅     | On-demand full state synchronization                     |
|                            | Heartbeat                          | ✅     | Instance liveness detection                              |
| **Knowledge Manager**      | Local cache                        | ✅     | In-memory cache for fast reads                           |
|                            | Async message subscription         | ✅     | Automatic sync from Portal                               |
|                            | Cache-through reads                | ✅     | Cache miss → persistence fallback                        |
|                            | Hit count tracking                 | ✅     | Knowledge usage statistics                               |
|                            | Lifecycle-aware filtering          | ✅     | Only active knowledge served                             |
| **Encoder**                | BaseEncoder protocol               | ✅     | Abstract interface for text→vector                       |
|                            | SentenceTransformerEncoder         | ✅     | Production encoder (semantic embeddings + projection)    |
|                            | SimpleHashEncoder                  | ✅     | Deterministic hash encoder for testing                   |
|                            | Encoding cache                     | ✅     | FIFO cache to avoid redundant encoding                   |
|                            | Batch encoding                     | ✅     | Efficient batch processing                               |
|                            | Projection layers                  | ✅     | Trainable Linear(embed_dim → key_dim/value_dim)          |
|                            | Norm scaling                       | ✅     | Match aga-core's key_norm_target / value_norm_target     |
|                            | AGACoreAlignment support           | ✅     | EncoderConfig.from_alignment() auto-configures dims      |
| **Retriever**              | KnowledgeRetriever                 | ✅     | BaseRetriever protocol adapter                           |
|                            | HNSW dense search                  | ✅     | hnswlib ANN index (10K+ knowledge)                       |
|                            | BM25 sparse search                 | ✅     | rank-bm25 keyword matching                               |
|                            | RRF fusion                         | ✅     | Reciprocal Rank Fusion of dense+sparse                   |
|                            | Cosine similarity fallback         | ✅     | Brute-force when hnswlib unavailable                     |
|                            | AGACoreAlignment validation        | ✅     | Mandatory encoder-core alignment at construction         |
|                            | Incremental index update           | ✅     | Add/remove without full rebuild                          |
|                            | Auto-refresh                       | ✅     | Periodic index refresh from KnowledgeManager             |
|                            | Injection feedback                 | ✅     | Hit count update on successful injection                 |
|                            | Thread-safe retrieval              | ✅     | RLock-protected concurrent access                        |
|                            | Fail-Open design                   | ✅     | Retrieval errors return empty list, optional degradation |
| **Chunker**                | FixedSizeChunker                   | ✅     | Token-count based splitting                              |
|                            | SentenceChunker                    | ✅     | Sentence-boundary aware splitting                        |
|                            | SemanticChunker                    | ✅     | Embedding-based semantic grouping                        |
|                            | SlidingWindowChunker               | ✅     | Overlapping window splitting                             |
|                            | DocumentChunker                    | ✅     | Markdown structure-aware chunking (v0.3.0)               |
|                            | ConditionGenerator                 | ✅     | Multi-strategy condition generation                      |
|                            | ImageHandler                       | ✅     | Document image processing (Base64/URL/local→Portal)      |
|                            | Configurable strategies            | ✅     | YAML-driven chunker selection                            |
| **Versioning**             | Version history                    | ✅     | Full change history per knowledge unit                   |
|                            | Rollback                           | ✅     | Restore to any previous version                          |
|                            | Diff comparison                    | ✅     | Side-by-side version comparison                          |
|                            | Change audit                       | ✅     | Who changed what and when                                |
| **Compression**            | zlib compression                   | ✅     | Standard compression for text                            |
|                            | LZ4 compression                    | ✅     | Fast compression (optional)                              |
|                            | Zstd compression                   | ✅     | Balanced compression (optional)                          |
|                            | Decompression cache                | ✅     | LRU cache for hot data                                   |
| **Configuration**          | Dataclass-based config             | ✅     | Type-safe configuration                                  |
|                            | YAML file loading                  | ✅     | External configuration files                             |
|                            | Environment variable override      | ✅     | 12-factor app support                                    |
|                            | Development/Production presets     | ✅     | Quick-start configurations                               |
|                            | Config adapter                     | ✅     | Bridge to aga-core config format                         |

### Integration with aga-core

| Integration Point      | Status | Description                                                  |
| ---------------------- | ------ | ------------------------------------------------------------ |
| BaseRetriever protocol | ✅     | KnowledgeRetriever implements aga-core's retriever interface |
| AGACoreAlignment       | ✅     | Encoder-core alignment validation (dim/norm mandatory)       |
| Dimension alignment    | ✅     | Encoder output matches AGAConfig.bottleneck_dim / hidden_dim |
| Norm target matching   | ✅     | key_norm_target and value_norm_target configurable           |
| Fail-Open safety       | ✅     | All retrieval failures return empty results                  |
| Plugin constructor     | ✅     | `AGAPlugin(config, retriever=knowledge_retriever)`           |
| EventBus integration   | ✅     | Knowledge events flow to aga-observability                   |

## Quick Start

### Installation

```bash
# Core only (no external dependencies)
pip install aga-knowledge

# With Portal API
pip install aga-knowledge[portal]

# With PostgreSQL persistence
pip install aga-knowledge[postgres]

# With Redis sync + persistence
pip install aga-knowledge[redis]

# With encoder (SentenceTransformer)
pip install aga-knowledge[encoder]

# With retrieval enhancements (HNSW + BM25)
pip install aga-knowledge[retrieval]

# Full installation
pip install aga-knowledge[all]
```

### Basic Usage (with aga-core)

```python
import asyncio
from aga_knowledge import KnowledgeManager, AGACoreAlignment
from aga_knowledge.config import PortalConfig
from aga_knowledge.encoder import create_encoder, EncoderConfig
from aga_knowledge.retriever import KnowledgeRetriever
from aga.plugin import AGAPlugin
from aga.config import AGAConfig

async def main():
    # 1. Alignment (bridge aga-core ↔ aga-knowledge)
    alignment = AGACoreAlignment(
        hidden_dim=4096, bottleneck_dim=64,
        key_norm_target=5.0, value_norm_target=3.0,
    )

    # 2. Start KnowledgeManager
    km_config = PortalConfig.for_development()
    manager = KnowledgeManager(km_config)
    await manager.start()

    # 3. Create Aligned Encoder
    encoder_config = EncoderConfig.from_alignment(alignment)
    encoder = create_encoder(encoder_config)

    # 4. Create Hybrid Retriever
    retriever = KnowledgeRetriever(
        manager=manager, encoder=encoder,
        alignment=alignment, namespace="default",
        index_backend="hnsw", bm25_enabled=True,
    )

    # 5. Create AGAPlugin with retriever
    aga_config = AGAConfig(bottleneck_dim=64, hidden_dim=4096)
    plugin = AGAPlugin(aga_config, retriever=retriever)

    # 6. Attach to model
    plugin.attach(model)

asyncio.run(main())
```

### Document Chunking Pipeline

```python
from aga_knowledge.chunker import (
    create_document_chunker, DocumentChunker,
    ConditionGenerator, ImageHandler,
    create_chunker, ChunkerConfig,
)

# Option 1: Structure-aware document chunking (Markdown)
doc_chunker = create_document_chunker(
    condition_mode="title_context",
    portal_base_url="http://localhost:8081",
    assets_dir="./static/assets",
)
chunks = doc_chunker.chunk_document(
    markdown_text, source_id="doc_001", title="Medical Guidelines"
)

# Option 2: Basic chunking
chunker = create_chunker(ChunkerConfig(
    strategy="sliding_window",
    chunk_size=300,
    overlap=50,
))
chunks = chunker.chunk_document(document, source_id="doc_001", title="Medical Guidelines")

# Register chunks via Portal API
for chunk in chunks:
    record = chunk.to_knowledge_record()
    # POST to /knowledge/inject-text
```

### Standalone Usage (without aga-core)

`aga-knowledge` can be used independently as a knowledge management system:

```python
from aga_knowledge import KnowledgeManager
from aga_knowledge.config import PortalConfig

config = PortalConfig.for_development()
manager = KnowledgeManager(config)
await manager.start()

# Query knowledge
records = await manager.get_active_knowledge("default")
record = await manager.get_knowledge("default", "rule_001")
```

## Configuration

### YAML Configuration

```yaml
# portal_config.yaml
server:
    host: "0.0.0.0"
    port: 8081
    workers: 4

persistence:
    type: "postgres" # memory | sqlite | postgres | redis
    postgres_host: "db.example.com"
    postgres_port: 5432
    postgres_database: "aga_knowledge"
    postgres_user: "aga"
    postgres_password: "${AGA_DB_PASSWORD}"
    postgres_pool_size: 20
    enable_audit: true

messaging:
    backend: "redis"
    redis_host: "redis.example.com"
    redis_port: 6379
    redis_channel: "aga:sync"

registry:
    type: "redis"
    heartbeat_interval: 30
    timeout: 90

governance:
    enabled: true
    auto_confirm_after_hits: 100
    auto_deprecate_after_days: 30
```

### Encoder Configuration

```yaml
encoder:
    backend: "sentence_transformer" # sentence_transformer | simple_hash
    model_name: "all-MiniLM-L6-v2"
    key_dim: 64 # Must == AGAConfig.bottleneck_dim
    value_dim: 4096 # Must == AGAConfig.hidden_dim
    device: "cpu"
    batch_size: 32
    normalize: true
    cache_enabled: true
    cache_max_size: 10000
    options:
        condition_prefix: "condition: "
        decision_prefix: "decision: "
        key_norm_target: 5.0
        value_norm_target: 3.0
        projection_path: null # Path to pre-trained projection weights
```

### Chunker Configuration

```yaml
chunker:
    strategy: "sliding_window" # fixed_size | sentence | semantic | sliding_window | document
    chunk_size: 300 # Target tokens per chunk
    overlap: 50 # Overlap tokens (sliding_window)
    min_chunk_size: 50
    max_chunk_size: 500
    condition_mode: "first_sentence" # first_sentence | title_context | keyword | summary
    language: "auto" # auto | zh | en
```

### AGACoreAlignment Configuration

```yaml
aga_core_alignment:
    hidden_dim: 4096 # == AGAConfig.hidden_dim
    bottleneck_dim: 64 # == AGAConfig.bottleneck_dim
    key_norm_target: 5.0 # == AGAConfig.key_norm_target
    value_norm_target: 3.0 # == AGAConfig.value_norm_target
```

## Package Structure

```
aga_knowledge/
├── __init__.py                 # Package entry, exports KnowledgeManager, AGACoreAlignment
├── alignment.py               # AGACoreAlignment encoder-core alignment
├── config.py                   # PortalConfig, PersistenceDBConfig, etc.
├── types.py                    # KnowledgeRecord, LifecycleState, TrustTier
├── exceptions.py               # Exception hierarchy
├── pyproject.toml              # Package metadata
│
├── manager/                    # Runtime-facing knowledge manager
│   └── knowledge_manager.py    # Cache, sync subscription, query interface
│
├── portal/                     # REST API (FastAPI)
│   ├── app.py                  # Application factory (with /assets static mount)
│   ├── routes.py               # API endpoints
│   ├── service.py              # Business logic
│   └── registry.py             # Runtime instance registry
│
├── persistence/                # Storage backends
│   ├── base.py                 # PersistenceAdapter ABC
│   ├── memory_adapter.py       # In-memory (testing)
│   ├── sqlite_adapter.py       # SQLite (development)
│   ├── postgres_adapter.py     # PostgreSQL (production)
│   ├── redis_adapter.py        # Redis (cache layer)
│   ├── versioning.py           # Version history & rollback
│   └── compression.py          # Text compression (zlib/lz4/zstd)
│
├── sync/                       # Inter-instance synchronization
│   ├── protocol.py             # SyncMessage, MessageType
│   ├── publisher.py            # Message publishing
│   └── backends.py             # Redis/Memory sync backends
│
├── encoder/                    # Text → Vector encoding
│   ├── base.py                 # BaseEncoder ABC, EncoderConfig
│   ├── sentence_transformer_encoder.py  # Production encoder
│   └── simple_encoder.py       # Hash-based test encoder
│
├── retriever/                  # aga-core retriever bridge
│   ├── knowledge_retriever.py  # KnowledgeRetriever (BaseRetriever impl)
│   ├── hnsw_index.py           # HNSW dense vector index
│   ├── bm25_index.py           # BM25 sparse retrieval index
│   └── fusion.py               # RRF fusion algorithm
│
├── chunker/                    # Document → Fragments
│   ├── base.py                 # BaseChunker ABC, ChunkerConfig
│   ├── fixed_size.py           # Fixed token-count chunks
│   ├── sentence.py             # Sentence-boundary chunks
│   ├── semantic.py             # Embedding-based semantic chunks
│   ├── sliding_window.py       # Overlapping window chunks
│   ├── document_chunker.py     # Markdown structure-aware chunking
│   ├── condition_generator.py  # Multi-strategy condition generation
│   └── image_handler.py        # Document image processing
│
├── scripts/                    # Database initialization scripts
│   ├── init_postgresql.sql     # PostgreSQL Schema
│   └── init_sqlite.sql         # SQLite Schema
│
└── config_adapter/             # Configuration bridge
    └── adapter.py              # aga-core config ↔ aga-knowledge config
```

## Roadmap

### v0.2.x — Foundation (Completed)

-   [x] Plaintext condition/decision knowledge model
-   [x] 4 persistence backends (Memory, SQLite, PostgreSQL, Redis)
-   [x] Portal REST API with full CRUD
-   [x] Redis Pub/Sub synchronization
-   [x] KnowledgeManager with local cache
-   [x] Encoder module (SentenceTransformer + SimpleHash)
-   [x] KnowledgeRetriever (BaseRetriever protocol)
-   [x] 4 base chunking strategies
-   [x] Version control and text compression
-   [x] 605+ unit tests passing

### v0.3.0 — Current (Production Alignment)

-   [x] **AGACoreAlignment** — Encoder-core alignment validation (dim/norm mandatory)
-   [x] **HNSW dense search** — hnswlib ANN index (10K+ knowledge)
-   [x] **BM25 sparse search** — rank-bm25 keyword matching
-   [x] **RRF fusion** — Reciprocal Rank Fusion of dense+sparse results
-   [x] **DocumentChunker** — Markdown structure-aware chunking
-   [x] **ConditionGenerator** — Multi-strategy condition generation
-   [x] **ImageHandler** — Document image processing (Base64/URL/local→Portal)
-   [x] **Portal image assets** — StaticFiles mount at /assets
-   [x] **Graceful degradation** — hnswlib/rank-bm25 optional, auto-fallback
-   [x] **Database Schema** — PostgreSQL + SQLite init scripts

### v0.4.x — Production Hardening (Next)

-   [ ] **Contrastive fine-tuning** for projection layers (condition→key, decision→value)
-   [ ] **Distributed encoder** — gRPC encoder service for multi-instance
-   [ ] **Key-Value joint deduplication** — beyond key-key similarity
-   [ ] **Embedding versioning** — track encoder model changes and re-encode
-   [ ] **Prometheus metrics** — retrieval latency, cache hit rate, index size
-   [ ] **Rate limiting** — Portal API rate limiting and quota management

### v1.0.0 — Production Release

-   [ ] API stability guarantee
-   [ ] Comprehensive security audit
-   [ ] Performance benchmarks (latency, throughput, memory)
-   [ ] Helm chart for Kubernetes deployment
-   [ ] Official documentation site

## Testing

```bash
# Run all tests
python -m pytest tests/test_knowledge/ -v

# Run specific module tests
python -m pytest tests/test_knowledge/test_encoder.py -v
python -m pytest tests/test_knowledge/test_chunker.py -v
python -m pytest tests/test_knowledge/test_document_chunker.py -v
python -m pytest tests/test_knowledge/test_postgres_adapter.py -v
python -m pytest tests/test_knowledge/test_redis_adapter.py -v
python -m pytest tests/test_knowledge/test_retriever.py -v
python -m pytest tests/test_knowledge/test_versioning.py -v
python -m pytest tests/test_knowledge/test_compression.py -v
```

## License

MIT License

## Related Packages

| Package                                    | Description                                              |
| ------------------------------------------ | -------------------------------------------------------- |
| [aga-core](../aga/)                        | Attention Governance Plugin — entropy-gated KV injection |
| [aga-observability](../aga_observability/) | Monitoring, alerting, and audit for aga-core             |
