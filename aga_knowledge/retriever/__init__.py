"""
aga-knowledge 召回器模块

提供符合 aga-core BaseRetriever 协议的知识检索实现。
将 aga-knowledge 的明文 condition/decision 知识转换为
aga-core 可注入的 key/value 向量。

架构:
    aga-core (BaseRetriever) <-- KnowledgeRetriever --> KnowledgeManager + Encoder

检索架构:
    - 稠密检索: HNSW（hnswlib），O(log N) 查询
    - 稀疏检索: BM25（rank-bm25），关键词匹配
    - 融合: RRF（倒数排名融合），合并两路结果

使用:
    from aga_knowledge.alignment import AGACoreAlignment
    from aga_knowledge.encoder import create_encoder, EncoderConfig
    from aga_knowledge.retriever import KnowledgeRetriever

    alignment = AGACoreAlignment.from_aga_config_yaml("aga_config.yaml")
    encoder_config = EncoderConfig.from_alignment(alignment)
    encoder = create_encoder(encoder_config)

    retriever = KnowledgeRetriever(
        manager=manager,
        encoder=encoder,
        alignment=alignment,
        index_backend="hnsw",
        bm25_enabled=True,
    )

    # 传递给 aga-core
    plugin = AGAPlugin(config, retriever=retriever)
"""

from .knowledge_retriever import KnowledgeRetriever, ConfigError
from .fusion import reciprocal_rank_fusion, weighted_score_fusion

__all__ = [
    "KnowledgeRetriever",
    "ConfigError",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]

# 可选导出（依赖 hnswlib / rank-bm25）
try:
    from .hnsw_index import HNSWIndex
    __all__.append("HNSWIndex")
except ImportError:
    pass

try:
    from .bm25_index import BM25Index
    __all__.append("BM25Index")
except ImportError:
    pass
