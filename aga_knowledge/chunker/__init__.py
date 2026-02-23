"""
aga-knowledge 知识分片器

将大文档分割为适合 AGA 注入的知识片段（100-500 tokens）。

支持多种分片策略:
  - FixedSizeChunker: 固定大小分片（按 token 数）
  - SentenceChunker: 按句子边界分片
  - SemanticChunker: 语义段落分片（按标题/段落结构）
  - SlidingWindowChunker: 滑动窗口分片（带重叠）

高级功能:
  - DocumentChunker: 文档级分片器（结构感知 + condition 增强）
  - ConditionGenerator: 高质量 condition 生成器
  - ImageHandler: 文档图片处理器（上下文对齐）

使用:
    # 基础分片
    from aga_knowledge.chunker import create_chunker, ChunkerConfig

    chunker = create_chunker(ChunkerConfig(
        strategy="sliding_window",
        chunk_size=300,
        overlap=50,
    ))
    chunks = chunker.chunk(document_text)

    # 文档级分片（推荐用于知识注册）
    from aga_knowledge.chunker import create_document_chunker

    doc_chunker = create_document_chunker(ChunkerConfig(
        strategy="sliding_window",
        chunk_size=300,
        overlap=50,
        condition_mode="title_context",
    ))
    chunks = doc_chunker.chunk_document(
        text=markdown_text,
        source_id="doc_001",
        title="心脏解剖学",
    )
"""

from .base import (
    BaseChunker,
    ChunkerConfig,
    KnowledgeChunk,
)
from .fixed_size import FixedSizeChunker
from .sentence import SentenceChunker
from .semantic import SemanticChunker
from .sliding_window import SlidingWindowChunker
from .condition_generator import ConditionGenerator
from .document_chunker import DocumentChunker, DocumentSection
from .image_handler import ImageHandler, ImageAsset

__all__ = [
    # 基础
    "BaseChunker",
    "ChunkerConfig",
    "KnowledgeChunk",
    # 分片策略
    "FixedSizeChunker",
    "SentenceChunker",
    "SemanticChunker",
    "SlidingWindowChunker",
    # 文档级分片
    "DocumentChunker",
    "DocumentSection",
    "ConditionGenerator",
    # 图片处理
    "ImageHandler",
    "ImageAsset",
    # 工厂函数
    "create_chunker",
    "create_document_chunker",
]


def create_chunker(config: ChunkerConfig) -> BaseChunker:
    """
    根据配置创建基础分片器

    Args:
        config: 分片器配置

    Returns:
        BaseChunker 实例
    """
    strategy = config.strategy

    if strategy == "fixed_size":
        return FixedSizeChunker(config)
    elif strategy == "sentence":
        return SentenceChunker(config)
    elif strategy == "semantic":
        return SemanticChunker(config)
    elif strategy == "sliding_window":
        return SlidingWindowChunker(config)
    else:
        raise ValueError(
            f"未知的分片策略: {strategy}。"
            f"支持: fixed_size, sentence, semantic, sliding_window"
        )


def create_document_chunker(
    config: ChunkerConfig,
    image_handler: "ImageHandler | None" = None,
) -> DocumentChunker:
    """
    根据配置创建文档级分片器

    文档级分片器在基础分片策略之上，增加:
    - 文档结构感知（Markdown 标题层级）
    - 高质量 condition 生成（title_context 模式）
    - 可选的图片处理

    Args:
        config: 分片器配置（strategy 决定基础分片策略）
        image_handler: 可选的图片处理器实例

    Returns:
        DocumentChunker 实例

    使用:
        config = ChunkerConfig(
            strategy="sliding_window",
            chunk_size=300,
            overlap=50,
            condition_mode="title_context",
        )
        chunker = create_document_chunker(config)
        chunks = chunker.chunk_document(text, source_id="doc_001", title="文档标题")
    """
    base_chunker = create_chunker(config)
    return DocumentChunker(
        config=config,
        base_chunker=base_chunker,
        image_handler=image_handler,
    )
