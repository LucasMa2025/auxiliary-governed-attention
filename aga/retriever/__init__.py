"""
aga/retriever/ — 标准召回器协议

提供配置驱动的知识检索接口，允许用户使用自己的知识基础设施
（Chroma, Milvus, Elasticsearch, 自定义等）。

设计原则:
  1. 效果优先于性能 — 高熵触发频率低（每次推理 1-10 次），可容忍 ms 级延迟
  2. 外部实现 — aga-core 只定义协议，具体实现由用户或 aga-knowledge 提供
  3. 标准协议 — BaseRetriever 是唯一接口，所有后端必须实现
  4. 内置简单实现 — NullRetriever（默认）和 KVStoreRetriever（本地）
  5. 按需召回 — 只在高熵触发时调用，不影响低熵旁路的零开销
  6. Fail-Open — 召回失败时回退到 KVStore 已有知识，不中断推理
"""

from .base import BaseRetriever, RetrievalQuery, RetrievalResult
from .null_retriever import NullRetriever
from .kv_store_retriever import KVStoreRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalQuery",
    "RetrievalResult",
    "NullRetriever",
    "KVStoreRetriever",
]
