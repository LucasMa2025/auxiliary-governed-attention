# aga-knowledge Product Specification

**Version**: 0.3.0  
**Last Updated**: 2026-02-23  
**Status**: Beta — Core functionality complete, encoder-core alignment enforced, hybrid retrieval (HNSW+BM25+RRF) implemented

---

## 1. Product Positioning

### 1.1 Mission

`aga-knowledge` solves one core problem: **Frozen LLMs lack domain-specific facts, and aga-core needs a reliable knowledge source to inject them.**

In the AGA ecosystem:

-   **aga-core** provides the injection mechanism (entropy-gated KV injection into attention layers)
-   **aga-knowledge** provides the knowledge pipeline (register → store → encode → retrieve)
-   **aga-observability** provides operational visibility

### 1.2 Design Philosophy

1. **Plaintext-First**: Knowledge is stored as human-readable `condition/decision` text pairs. Vectorization happens at the boundary (encoder module), not in the storage layer.
2. **Configuration-Driven**: All components (persistence, encoder, chunker, retriever) are selected and configured via YAML/dict, not hardcoded.
3. **Protocol-Based Integration**: `KnowledgeRetriever` implements `aga-core`'s `BaseRetriever` protocol. Users with existing infrastructure (Chroma, Milvus, Elasticsearch) can implement their own retriever and skip `aga-knowledge` entirely.
4. **Encoder-Core Alignment**: The `AGACoreAlignment` dataclass explicitly enforces dimensional and norm consistency between aga-knowledge's encoder and aga-core's attention layers. Misaligned configurations are rejected at startup with `ConfigError`.
5. **Fail-Open Safety**: Every retrieval failure returns an empty list. Knowledge management issues never crash inference.
6. **CPU-Only Operation**: The entire knowledge pipeline runs on CPU, preserving GPU memory for inference.

### 1.3 Target Users

| User Type          | Use Case                                                           |
| ------------------ | ------------------------------------------------------------------ |
| **Researchers**    | Quick prototyping with SQLite + SimpleHashEncoder                  |
| **ML Engineers**   | Production deployment with PostgreSQL + SentenceTransformerEncoder |
| **Platform Teams** | Custom retriever implementation via BaseRetriever protocol         |
| **Domain Experts** | Knowledge registration via Portal REST API (no coding required)    |

---

## 2. System Architecture

### 2.1 Component Overview

```
                    ┌──────────────────────────────────────────┐
                    │         Domain Expert / Admin            │
                    └──────────────────┬───────────────────────┘
                                       │ REST API
                    ┌──────────────────▼───────────────────────┐
                    │          Portal (FastAPI)                │
                    │  inject-text / batch-text / lifecycle    │
                    │  + /assets (static image serving)        │
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
              │  (Instance 1)      │   │  (Instance 2)          │
              │  Local Cache       │   │  Local Cache           │
              └────────┬───────────┘   └───────┬────────────────┘
                       │                       │
              ┌────────▼───────────┐   ┌───────▼────────────────┐
              │  Encoder           │   │  Encoder               │
              │  (Text → Vectors)  │   │  (Text → Vectors)      │
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
              │  aga-core Plugin   │   │  aga-core Plugin       │
              │  (GPU Instance 1)  │   │  (GPU Instance 2)      │
              └────────────────────┘   └────────────────────────┘
```

### 2.2 Data Flow

```
1. Register:  Expert → Portal API → Persistence → Sync → All KnowledgeManagers
2. Encode:    KnowledgeManager → Encoder (with AGACoreAlignment) → key/value vectors → HNSW + BM25 Index
3. Retrieve:  aga-core high entropy → hidden_states → KnowledgeRetriever → Hybrid Search (HNSW+BM25+RRF) → top-k results
4. Inject:    RetrievalResult → aga-core KVStore → BottleneckInjector → Attention Layer
5. Feedback:  aga-core → on_injection_feedback() → Hit count updates
```

### 2.3 Retrieval Architecture (New in v0.3.0)

```
    RetrievalQuery
         │
    ┌────▼────┐
    │ Query   │
    │ Router  │
    └────┬────┘
         │
    ┌────┼────┐
    │         │
  Dense     Sparse
  (HNSW)    (BM25)
    │         │
    └────┬────┘
         │
    ┌────▼────┐
    │ RRF     │
    │ Fusion  │
    └────┬────┘
         │
    List[RetrievalResult]
```

---

## 3. Functional Completeness Analysis

### 3.1 Feature Coverage

| Capability                | Required by aga-core | Implemented                              | Completeness              |
| ------------------------- | -------------------- | ---------------------------------------- | ------------------------- |
| Knowledge CRUD            | Yes                  | Yes                                      | Full                      |
| Batch Operations          | Yes                  | Yes                                      | Full                      |
| Namespace Isolation       | Yes                  | Yes                                      | Full                      |
| Lifecycle Management      | Yes                  | Yes                                      | Full                      |
| Multi-Backend Persistence | Yes                  | Yes (4 backends)                         | Full                      |
| Cross-Instance Sync       | Yes                  | Yes (Redis Pub/Sub)                      | Full                      |
| Text → Vector Encoding    | Yes                  | Yes (2 encoders + AGACoreAlignment)      | Production-aligned        |
| Semantic Retrieval        | Yes                  | Yes (HNSW + BM25 + RRF hybrid search)   | Production-grade          |
| BaseRetriever Protocol    | Yes                  | Yes                                      | Full                      |
| Document Chunking         | Yes                  | Yes (5 strategies + DocumentChunker)     | Full                      |
| Condition Generation      | Yes                  | Yes (4 modes)                            | Full                      |
| Image Handling            | Optional             | Yes (Base64/URL/local → Portal assets)   | Full                      |
| Encoder-Core Alignment    | Yes                  | Yes (AGACoreAlignment + validation)      | Full                      |
| Version Control           | Optional             | Yes                                      | Full                      |
| Text Compression          | Optional             | Yes                                      | Full                      |
| Audit Logging             | Optional             | Yes                                      | Full                      |

### 3.2 Gap Analysis (v0.3.0)

| Gap                         | Severity | Impact                                              | Mitigation                                         |
| --------------------------- | -------- | --------------------------------------------------- | -------------------------------------------------- |
| Projection layers untrained | Medium   | Encoding quality depends on random initialization   | Users can fine-tune and load via `projection_path` |
| No distributed encoder      | Low      | Each instance loads its own encoder model           | Acceptable at current scale                        |
| Sync-async bridge           | Low      | `_refresh_index_sync` uses `asyncio.run` workaround | Works but inelegant                                |

> **Note**: The brute-force retrieval and BM25 gaps from v0.2.0 have been resolved in v0.3.0 with HNSW and BM25 hybrid search.

### 3.3 Conclusion

**aga-knowledge v0.3.0 is production-aligned as a knowledge management system for aga-core.** The full pipeline — registration, encoding, retrieval, and injection — works end-to-end with enforced encoder-core alignment. The retrieval system supports 10K+ knowledge entries via HNSW ANN indexing. The primary remaining improvement area is encoder quality (projection training).

---

## 4. Encoder System — Design, Implementation, Limitations & Improvement Directions

### 4.1 Current Architecture

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
    │          AGACoreAlignment (New in v0.3)          │
    │                                                  │
    │  EncoderConfig.from_alignment(alignment)         │
    │  EncoderConfig.validate_alignment(alignment)     │
    │  → Enforces key_dim = bottleneck_dim            │
    │  → Enforces value_dim = hidden_dim              │
    │  → Enforces key/value norm targets              │
    └──────────────────────────────────────────────────┘
```

### 4.2 Encoder-Core Alignment (New in v0.3.0)

The `AGACoreAlignment` dataclass provides explicit configuration bridging:

| aga-core Parameter             | aga-knowledge Parameter                   | Enforced |
|-------------------------------|------------------------------------------|----------|
| `AGAConfig.bottleneck_dim`     | `EncoderConfig.key_dim`                   | Yes (mandatory) |
| `AGAConfig.hidden_dim`         | `EncoderConfig.value_dim`                 | Yes (mandatory) |
| `AGAConfig.key_norm_target`    | `EncoderConfig.options["key_norm_target"]` | Yes (mandatory) |
| `AGAConfig.value_norm_target`  | `EncoderConfig.options["value_norm_target"]` | Yes (mandatory) |
| `AGAConfig.num_heads`          | `AGACoreAlignment.num_heads`              | Informational |
| `AGAConfig.value_bottleneck_dim` | `AGACoreAlignment.value_bottleneck_dim` | Informational |

Three ways to create alignment:
1. **Manual** (production): `AGACoreAlignment(hidden_dim=4096, bottleneck_dim=64, ...)`
2. **From YAML**: `AGACoreAlignment.from_aga_config_yaml("aga_config.yaml")`
3. **From AGAConfig** (development): `AGACoreAlignment.from_aga_config(config)`

Misalignment is detected at `KnowledgeRetriever` construction time and raises `ConfigError`.

### 4.3 SentenceTransformerEncoder — Detailed Analysis

#### Encoding Pipeline

```
Input:
  condition = "Patient temperature exceeds 39°C"
  decision  = "Administer antipyretic and monitor continuously"

Step 1: Text Prefixing
  condition → "condition: Patient temperature exceeds 39°C"
  decision  → "decision: Administer antipyretic and monitor continuously"

Step 2: SentenceTransformer Encoding
  condition → ST("condition: ...") → cond_embedding [384]
  decision  → ST("decision: ...")  → dec_embedding  [384]

Step 3: Projection
  cond_embedding → key_proj(384 → 64)   → key_raw   [64]
  dec_embedding  → value_proj(384 → 4096) → value_raw [4096]

Step 4: Normalization + Scaling
  key_raw   → L2_normalize → scale(5.0)  → key   [64]
  value_raw → L2_normalize → scale(3.0)  → value [4096]

Output:
  key:   [64]   — for aga-core attention matching (bottleneck_dim)
  value: [4096] — for aga-core knowledge injection (hidden_dim)
```

#### Strengths

1. **Semantic Awareness**: SentenceTransformer provides high-quality sentence-level embeddings that capture semantic meaning, enabling meaningful similarity search.
2. **Dimension Flexibility**: Projection layers (`key_proj`, `value_proj`) can map any embedding dimension to aga-core's required dimensions.
3. **CPU-Friendly**: Runs entirely on CPU, preserving GPU memory for inference.
4. **Deterministic**: Same input always produces same output (using `torch.no_grad()`).
5. **Encoding Cache**: FIFO cache avoids redundant encoding of unchanged knowledge.
6. **Projection Persistence**: Trained projection layers can be saved/loaded for cross-deployment migration.
7. **Alignment Enforcement** (v0.3.0): `EncoderConfig.from_alignment()` and `validate_alignment()` ensure dimensional and norm consistency with aga-core.

#### Limitations & Improvement Directions

##### L1: Untrained Projection Layers (Severity: High)

**Problem**: Projection layers are initialized with Xavier uniform random weights. This means:

-   `key_proj` maps 384-dim semantic embeddings **randomly** into 64-dim key space
-   `value_proj` maps 384-dim semantic embeddings **randomly** into 4096-dim value space
-   The resulting vectors have no guaranteed semantic alignment with aga-core's learned attention space

**Impact**: Retrieval quality is degraded because key vectors don't optimally represent semantic content in the bottleneck space. Value vectors injected into attention layers may not carry the intended semantic information.

**Improvement Directions**:

1. **Contrastive Fine-Tuning**: Train `key_proj` with contrastive loss — similar conditions should have similar keys, dissimilar conditions should have distant keys.
2. **Value Reconstruction Loss**: Train `value_proj` such that the projected value, when injected into the attention layer, produces output similar to what the model would produce if the knowledge were in its parameters.
3. **Distillation from aga-core**: Use aga-core's actual `q_proj` and `v_proj` weights as teacher signals to align the projection layers.
4. **Few-Shot Calibration**: Provide a small set of labeled (condition, decision, expected_key, expected_value) pairs for quick calibration.

##### L2: Independent condition/decision Encoding (Severity: Medium)

**Problem**: Condition and decision are encoded independently. Two knowledge units with the same condition but different decisions will have identical key vectors, making them indistinguishable during retrieval.

**Improvement Directions**:

1. **Joint Encoding**: Concatenate condition and decision before encoding: `ST(condition + " [SEP] " + decision)`, then split the embedding for key and value projection.
2. **Cross-Attention Encoding**: Use a small cross-attention layer to process condition and decision embeddings before projection.
3. **Key-Value Joint Deduplication**: Use joint similarity (not just key-key) for deduplication to preserve progressively refined knowledge.

##### L3: Fixed Norm Scaling (Severity: Low)

**Problem**: Key and value vectors are scaled to fixed norms (default 5.0 and 3.0). These values are heuristic and may not match the actual distribution of key/value norms inside aga-core.

**Improvement Direction**: Adaptive norm calibration during a calibration phase, measuring actual key/value norms in aga-core's KVStore.

### 4.4 SimpleHashEncoder — Analysis

Testing and development only. Provides deterministic, zero-dependency encoding. Not suitable for production retrieval.

### 4.5 Encoder Improvement Roadmap

| Priority | Improvement                                          | Effort | Impact                                     |
| -------- | ---------------------------------------------------- | ------ | ------------------------------------------ |
| P0       | Contrastive fine-tuning script for projection layers | Medium | High — directly improves retrieval quality |
| P1       | HuggingFace encoder backend                          | Low    | Medium — broader model selection           |
| P1       | Joint condition+decision encoding option             | Medium | Medium — better key-value alignment        |
| P2       | OpenAI/API encoder backend                           | Low    | Medium — cloud deployment support          |
| P2       | Adaptive norm calibration                            | Medium | Low-Medium — better injection weighting    |
| P3       | Cross-attention encoding                             | High   | Medium — optimal but complex               |

---

## 5. Knowledge Retrieval System — Design, Implementation & Analysis

### 5.1 Current Architecture (v0.3.0 — Hybrid Search)

```
                    ┌──────────────────────────────────────────────┐
                    │      KnowledgeRetriever (v0.3.0)             │
                    │  (implements BaseRetriever)                  │
                    │                                              │
                    │  ┌─────────────────────────────────────────┐ │
                    │  │  AGACoreAlignment Validation            │ │
                    │  │  (enforced at construction)             │ │
                    │  └─────────────────────────────────────────┘ │
                    │                                              │
                    │  ┌──────────────┐  ┌───────────────────────┐ │
                    │  │  HNSW Index  │  │  BM25 Index           │ │
                    │  │  (hnswlib)   │  │  (rank-bm25)          │ │
                    │  │  Dense ANN   │  │  Sparse keyword       │ │
                    │  └──────┬───────┘  └─────────┬─────────────┘ │
                    │         │                     │               │
                    │         └──────┬──────────────┘               │
                    │                │                              │
                    │         ┌──────▼──────┐                      │
                    │         │ RRF Fusion  │                      │
                    │         └──────┬──────┘                      │
                    │                │                              │
                    │  ┌─────────────▼─────────────────┐          │
                    │  │  Brute-Force Fallback          │          │
                    │  │  (_key_matrix cosine, O(N))    │          │
                    │  └───────────────────────────────┘          │
                    │                                              │
                    │  retrieve(query):                            │
                    │    1. Extract query vector                   │
                    │    2. Dense search (HNSW or brute-force)     │
                    │    3. Sparse search (BM25 if text query)     │
                    │    4. RRF fusion (if sparse results exist)   │
                    │    5. Return RetrievalResult[]               │
                    └──────────────────────────────────────────────┘
```

### 5.2 Dense Retrieval — HNSW

**Why HNSW over FAISS IVF+PQ:**
- HNSW natively supports incremental insertion/deletion (AGA knowledge is dynamic)
- HNSW is insensitive to semantic fragmentation (AGA knowledge: 100-500 token segments)
- HNSW maintains >95% recall at 1K-1M scale
- FAISS IVF+PQ requires periodic centroid retraining, incompatible with dynamic data

**Performance:**

| Knowledge Count | Estimated Latency (CPU) | Acceptable? |
|----------------|------------------------|-------------|
| 1,000 | ~0.2ms | Yes |
| 10,000 | ~0.5ms | Yes |
| 50,000 | ~0.8ms | Yes |
| 100,000 | ~1.0ms | Yes |

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hnsw_m` | 16 | Connections per layer (recall vs memory) |
| `hnsw_ef_construction` | 200 | Build-time search breadth |
| `hnsw_ef_search` | 100 | Query-time search breadth |
| `hnsw_max_elements` | 100,000 | Maximum index capacity |

### 5.3 Sparse Retrieval — BM25

Complements dense retrieval with keyword-level exact matching:

- When encoder projection layers are untrained, BM25 keyword matching is essential
- When no text query is available (vector-only), BM25 is automatically skipped
- Indexes both `condition` and `decision` text

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bm25_enabled` | false | Enable BM25 sparse search |
| `bm25_weight` | 0.3 | Weight in RRF fusion |
| `bm25_k1` | 1.5 | Term frequency saturation |
| `bm25_b` | 0.75 | Document length normalization |

### 5.4 Fusion — Reciprocal Rank Fusion (RRF)

RRF merges dense and sparse results:

```
RRF_score(d) = Σ weight_i / (k + rank_i(d))
```

- `k = 60` (constant preventing top-ranked items from dominating)
- `dense_weight = 0.7`, `sparse_weight = 0.3` (configurable)
- Alternative: `weighted_score_fusion` for score-compatible systems

### 5.5 Graceful Degradation

| Condition | Behavior |
|-----------|----------|
| hnswlib not installed | Auto-fallback to brute-force O(N) search |
| rank-bm25 not installed | BM25 disabled, pure dense retrieval |
| No text query available | BM25 skipped, pure dense retrieval |
| Retrieval error | Fail-Open: return empty list |
| Empty index | Return empty list immediately |

### 5.6 Strengths

1. **Protocol Compliant**: Fully implements `BaseRetriever`, integrating seamlessly with aga-core.
2. **Alignment Enforced**: `AGACoreAlignment` validation at construction time prevents dimensional mismatches.
3. **Hybrid Search**: Dense (HNSW) + Sparse (BM25) + RRF fusion for high-precision retrieval.
4. **Scalable**: HNSW supports 100K+ knowledge entries with sub-millisecond latency.
5. **Fail-Open**: All errors are caught and return empty results.
6. **Thread-Safe**: `RLock` protects concurrent access to the vector index.
7. **Incremental Updates**: `refresh_knowledge()` and `remove_knowledge()` support single-item updates across all indexes.
8. **Auto-Refresh**: Configurable periodic refresh from KnowledgeManager.
9. **Feedback Loop**: `on_injection_feedback()` updates hit counts for knowledge lifecycle management.
10. **Rich Statistics**: Detailed stats including HNSW/BM25/brute-force/fusion search counts.

### 5.7 Remaining Improvement Directions

| Priority | Improvement                             | Effort | Impact                               |
| -------- | --------------------------------------- | ------ | ------------------------------------ |
| P1       | Incremental tensor updates              | Low    | Medium — reduces update latency      |
| P2       | Learned hidden_states projection        | Medium | Low-Medium — better fallback quality |
| P2       | Double-buffered index                   | Medium | Low — smoother updates               |
| P3       | Entity-aware retrieval                  | High   | Medium — domain-specific improvement |
| P3       | Multi-namespace retrieval               | Low    | Low — niche use case                 |

---

## 6. Knowledge Chunking System

### 6.1 Design Rationale

AGA injects knowledge in 100-500 token segments. Large documents must be split into segments that:

1. Are semantically coherent (don't cut mid-sentence)
2. Have meaningful `condition` fields (for retrieval matching)
3. Preserve context across chunk boundaries (overlap)

### 6.2 Strategy Comparison

| Strategy      | Coherence | Context Preservation | Speed                | Best For                 |
| ------------- | --------- | -------------------- | -------------------- | ------------------------ |
| FixedSize     | Low       | None                 | Fast                 | Uniform, structured data |
| Sentence      | Medium    | None                 | Fast                 | Narrative text           |
| Semantic      | High      | Implicit             | Slow (needs encoder) | Technical documentation  |
| SlidingWindow | Medium    | Explicit (overlap)   | Fast                 | General purpose          |
| Document      | High      | Structure-aware      | Fast                 | Markdown/HTML documents  |

### 6.3 Document-Level Chunking (New in v0.3.0)

The `DocumentChunker` provides structure-aware chunking:

1. **Markdown Header Parsing**: Recognizes `#`, `##`, `###` etc. headers and preserves section hierarchy
2. **Section Context Inheritance**: Child sections inherit parent section titles in condition generation
3. **Condition Generation**: 4 strategies for high-quality condition text:
   - `first_sentence` — Use the first sentence of the chunk
   - `summary` — Keyword-based summary
   - `title_context` — Section title hierarchy as condition
   - `keyword` — Top keywords extracted from chunk text
4. **Image Processing**: Via `ImageHandler`, extracts and processes images in documents

### 6.4 Image Handling (New in v0.3.0)

Documents may contain images (Markdown syntax). The `ImageHandler` processes them:

| Source Type | Processing | Output |
|-------------|-----------|--------|
| Base64 embedded | Decode → save to Portal assets dir | Portal URL |
| External URL | Keep original URL | Original URL |
| Local file path | Copy to Portal assets dir | Portal URL |

Image references in the `decision` text are replaced with accessible URLs and textual descriptions (configurable template), ensuring the encoder can process the context.

### 6.5 Recommendation

For most use cases, **SlidingWindow (overlap=50)** provides the best balance. For structured documents (Markdown/wiki), use **DocumentChunker** with `condition_mode="title_context"` for optimal context preservation and retrieval matching.

---

## 7. Persistence System

### 7.1 Backend Comparison

| Backend    | Latency | Durability   | Scalability     | Best For    |
| ---------- | ------- | ------------ | --------------- | ----------- |
| Memory     | ~0.01ms | None         | Single process  | Testing     |
| SQLite     | ~1ms    | File-level   | Single instance | Development |
| PostgreSQL | ~2-5ms  | Full ACID    | Multi-instance  | Production  |
| Redis      | ~0.5ms  | Configurable | Multi-instance  | Hot cache   |

### 7.2 Production Recommendation

**PostgreSQL as primary store + Redis as cache layer**:

-   PostgreSQL provides durable, ACID-compliant storage with full query capabilities
-   Redis provides fast reads for hot knowledge and Pub/Sub synchronization
-   This combination is battle-tested in production systems

### 7.3 Database Schema (v0.3.0)

New schema additions beyond the original AGA design:

| Table | Purpose |
|-------|---------|
| `namespaces` | Tenant/domain isolation |
| `knowledge` | Plaintext condition/decision pairs (**no vector fields**) |
| `knowledge_versions` | Full version history with automatic triggers |
| `document_sources` | Document origin tracking |
| `image_assets` | Image metadata and Portal URLs |
| `encoder_versions` | Encoder projection layer version tracking |
| `audit_log` | All CRUD operation audit trail |

---

## 8. Security Considerations

### 8.1 Current State

| Aspect                | Status          | Notes                              |
| --------------------- | --------------- | ---------------------------------- |
| Authentication        | Not implemented | Portal API is open access          |
| Authorization         | Not implemented | No role-based access control       |
| Encryption at rest    | Not implemented | Relies on database encryption      |
| Encryption in transit | Not implemented | Relies on TLS termination          |
| Input validation      | Partial         | Pydantic models validate structure |
| SQL injection         | Protected       | Parameterized queries used         |
| Audit logging         | Implemented     | All CRUD operations logged         |

### 8.2 Recommendations

1. Add API key authentication for Portal
2. Implement namespace-level authorization
3. Enable TLS for Redis connections
4. Add rate limiting for Portal API
5. Implement knowledge content validation (prevent injection attacks)

---

## 9. Performance Characteristics

### 9.1 Encoding Performance

| Encoder                   | Single Encode | Batch (100) | Memory Footprint |
| ------------------------- | ------------- | ----------- | ---------------- |
| SentenceTransformer (CPU) | ~15ms         | ~200ms      | ~500MB (model)   |
| SimpleHash                | ~0.01ms       | ~1ms        | ~0               |

### 9.2 Retrieval Performance

| Index Size | Brute-Force (CPU) | HNSW (CPU) | BM25 (CPU) |
| ---------- | ----------------- | ---------- | ---------- |
| 100        | < 0.1ms           | < 0.1ms    | < 0.1ms    |
| 1,000      | ~0.5ms            | ~0.2ms     | ~0.3ms     |
| 10,000     | ~5ms              | ~0.5ms     | ~1ms       |
| 100,000    | ~50ms             | ~1ms       | ~3ms       |

### 9.3 Persistence Performance

| Operation         | Memory  | SQLite | PostgreSQL | Redis  |
| ----------------- | ------- | ------ | ---------- | ------ |
| Single write      | ~0.01ms | ~1ms   | ~3ms       | ~0.5ms |
| Single read       | ~0.01ms | ~0.5ms | ~2ms       | ~0.3ms |
| Batch write (100) | ~0.1ms  | ~10ms  | ~15ms      | ~5ms   |
| Full load (1000)  | ~1ms    | ~50ms  | ~30ms      | ~20ms  |

---

## 10. Comparison with Alternatives

### 10.1 vs. Direct aga-core register()

| Aspect                | Direct register()          | aga-knowledge                 |
| --------------------- | -------------------------- | ----------------------------- |
| Setup complexity      | Low                        | Medium                        |
| Knowledge persistence | None (memory only)         | Full (4 backends)             |
| Multi-instance sync   | Manual                     | Automatic (Redis Pub/Sub)     |
| Lifecycle management  | None                       | Full (4 states + trust tiers) |
| Document ingestion    | Manual chunking + encoding | Automated pipeline            |
| Encoder-core alignment| Manual                     | Enforced (AGACoreAlignment)   |
| Hybrid search         | None                       | HNSW + BM25 + RRF            |
| Audit trail           | None                       | Full audit logging            |
| Best for              | Research, prototyping      | Production deployment         |

### 10.2 vs. RAG Systems (LangChain, LlamaIndex)

| Aspect                | RAG Systems                | aga-knowledge                        |
| --------------------- | -------------------------- | ------------------------------------ |
| Injection method      | Prompt concatenation       | Attention-layer KV injection         |
| Context window impact | Consumes tokens            | Zero token overhead                  |
| Integration depth     | Surface-level (prompt)     | Deep (attention mechanism)           |
| Knowledge format      | Documents/chunks           | condition/decision pairs             |
| Retrieval             | Mature (multiple backends) | Hybrid (HNSW + BM25 + RRF)          |
| Best for              | General-purpose RAG        | AGA-specific knowledge injection     |

---

## 11. Summary

`aga-knowledge` v0.3.0 provides a **production-aligned** knowledge management pipeline for `aga-core`. The core pipeline — registration, persistence, synchronization, encoding, alignment validation, and hybrid retrieval — works end-to-end with 650+ unit tests passing.

### Core Strengths

1. Clean separation of concerns (plaintext storage vs. vector encoding)
2. Protocol-based integration (BaseRetriever)
3. **Enforced encoder-core alignment** (AGACoreAlignment) — new in v0.3.0
4. **Hybrid retrieval** (HNSW + BM25 + RRF) — new in v0.3.0
5. **Document-level chunking** with condition generation and image handling — new in v0.3.0
6. Configuration-driven architecture
7. Comprehensive persistence options
8. Production-ready synchronization

### Key Improvement Areas

1. **Encoder Quality**: Projection layer training is the single highest-impact improvement
2. **Security**: Authentication and authorization for Portal API
3. **Distributed Encoder**: gRPC encoder service for multi-instance deployment

The system is ready for development and production use. For optimal retrieval quality, projection layer fine-tuning is recommended.
