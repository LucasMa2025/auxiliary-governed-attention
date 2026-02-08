"""
AGA 检索模块 - 大规模知识库支持

提供 ANN (Approximate Nearest Neighbor) 索引和动态知识加载能力，
支持 100K+ 到百万级知识槽位的高效检索。

核心组件:
- ANNIndex: ANN 索引抽象基类
- FAISSIndex: 基于 FAISS 的 ANN 索引实现
- HNSWIndex: 基于 HNSW 的 ANN 索引实现
- DynamicKnowledgeLoader: 动态知识加载器
- ANNIndexConfig: ANN 索引配置
- DynamicLoaderConfig: 动态加载器配置

版本: v1.0
"""

from .ann_index import (
    BaseANNIndex,
    FAISSIndex,
    HNSWIndex,
    ANNIndexConfig,
)
from .dynamic_loader import (
    DynamicKnowledgeLoader,
    DynamicLoaderConfig,
    LoadResult,
)

__all__ = [
    # ANN 索引
    "BaseANNIndex",
    "FAISSIndex",
    "HNSWIndex",
    "ANNIndexConfig",
    # 动态加载
    "DynamicKnowledgeLoader",
    "DynamicLoaderConfig",
    "LoadResult",
]
