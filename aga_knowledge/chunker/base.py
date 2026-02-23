"""
知识分片器抽象基类

定义分片协议和核心数据结构。

AGA 注入的知识片段推荐大小: 100-500 tokens
  - 过小: 语义不完整，注入效果差
  - 过大: 占用过多 KVStore 容量，稀释注意力

分片器负责将大文档拆分为适合 AGA 注入的 condition/decision 对。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import hashlib
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    """
    分片器配置

    配置示例 (YAML):
    ```yaml
    chunker:
      strategy: "sliding_window"
      chunk_size: 300          # 目标 token 数
      overlap: 50              # 重叠 token 数（sliding_window）
      min_chunk_size: 50       # 最小 chunk 大小
      max_chunk_size: 500      # 最大 chunk 大小
      separator: "\\n\\n"      # 段落分隔符（semantic）
      condition_mode: "first_sentence"  # condition 生成模式
      language: "auto"         # 语言（auto, zh, en）
    ```
    """
    strategy: str = "sliding_window"
    chunk_size: int = 300       # 目标 token 数
    overlap: int = 50           # 重叠 token 数
    min_chunk_size: int = 50    # 最小 chunk 大小
    max_chunk_size: int = 500   # 最大 chunk 大小
    separator: str = "\n\n"     # 段落分隔符
    condition_mode: str = "first_sentence"  # condition 生成模式
    language: str = "auto"      # 语言
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkerConfig":
        """从字典创建"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        if self.chunk_size <= 0:
            errors.append("chunk_size 必须大于 0")
        if self.overlap < 0:
            errors.append("overlap 不能为负数")
        if self.overlap >= self.chunk_size:
            errors.append("overlap 必须小于 chunk_size")
        if self.min_chunk_size <= 0:
            errors.append("min_chunk_size 必须大于 0")
        if self.max_chunk_size < self.chunk_size:
            errors.append("max_chunk_size 不能小于 chunk_size")
        valid_strategies = {"fixed_size", "sentence", "semantic", "sliding_window", "document"}
        if self.strategy not in valid_strategies:
            errors.append(f"strategy 必须是 {valid_strategies} 之一")
        return errors


@dataclass
class KnowledgeChunk:
    """
    知识片段

    分片器的输出，可直接用于 aga-knowledge 的知识注册。

    Attributes:
        chunk_id: 片段唯一 ID（基于内容哈希）
        condition: 触发条件（通常是片段的摘要或首句）
        decision: 决策内容（片段的完整文本）
        source_id: 源文档 ID
        chunk_index: 在源文档中的序号
        total_chunks: 源文档的总片段数
        token_count: 估算的 token 数
        metadata: 额外元数据
    """
    chunk_id: str
    condition: str
    decision: str
    source_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_knowledge_record(self) -> Dict[str, Any]:
        """转换为 aga-knowledge 知识记录格式"""
        return {
            "lu_id": self.chunk_id,
            "condition": self.condition,
            "decision": self.decision,
            "metadata": {
                "source_id": self.source_id,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                "token_count": self.token_count,
                **(self.metadata or {}),
            },
        }


class BaseChunker(ABC):
    """
    分片器抽象基类

    所有分片策略必须实现此接口。

    生命周期:
        1. __init__(config): 初始化
        2. chunk(text, source_id): 分片
        3. chunk_document(text, source_id, title): 分片文档（带标题）
    """

    def __init__(self, config: ChunkerConfig):
        self.config = config

    @abstractmethod
    def chunk(
        self,
        text: str,
        source_id: str = "",
    ) -> List[KnowledgeChunk]:
        """
        将文本分片

        Args:
            text: 待分片文本
            source_id: 源文档 ID

        Returns:
            KnowledgeChunk 列表
        """
        ...

    def chunk_document(
        self,
        text: str,
        source_id: str = "",
        title: str = "",
    ) -> List[KnowledgeChunk]:
        """
        分片文档（带标题上下文）

        如果提供了标题，会将标题作为 condition 的前缀。

        Args:
            text: 文档文本
            source_id: 源文档 ID
            title: 文档标题

        Returns:
            KnowledgeChunk 列表
        """
        chunks = self.chunk(text, source_id)

        if title:
            for chunk in chunks:
                chunk.condition = f"[{title}] {chunk.condition}"

        return chunks

    # ==================== 工具方法 ====================

    @staticmethod
    def estimate_tokens(text: str, language: str = "auto") -> int:
        """
        估算文本的 token 数

        粗略估算:
        - 英文: ~4 字符/token
        - 中文: ~1.5 字符/token
        - 混合: ~2.5 字符/token
        """
        if not text:
            return 0

        if language == "auto":
            # 检测中文字符比例
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            total_chars = len(text)
            chinese_ratio = chinese_chars / max(1, total_chars)

            if chinese_ratio > 0.3:
                language = "zh"
            else:
                language = "en"

        if language == "zh":
            return max(1, int(len(text) / 1.5))
        elif language == "en":
            return max(1, int(len(text) / 4))
        else:
            return max(1, int(len(text) / 2.5))

    @staticmethod
    def generate_chunk_id(text: str, source_id: str = "", index: int = 0) -> str:
        """生成基于内容的唯一 chunk ID"""
        content = f"{source_id}:{index}:{text[:200]}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def extract_first_sentence(text: str) -> str:
        """提取第一句话作为 condition"""
        if not text:
            return ""

        # 中英文句子结束符
        sentence_endings = re.compile(r'[。！？.!?]\s*')
        match = sentence_endings.search(text)

        if match and match.end() < len(text) * 0.5:
            return text[:match.end()].strip()

        # 如果没有找到句子结束符，取前 80 个字符
        if len(text) > 80:
            # 尝试在词边界截断
            truncated = text[:80]
            last_space = truncated.rfind(' ')
            if last_space > 40:
                return truncated[:last_space] + "..."
            return truncated + "..."

        return text

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """将文本分割为句子列表"""
        if not text:
            return []

        # 中英文句子分割
        # 保留分隔符
        parts = re.split(r'([。！？.!?]+\s*)', text)

        sentences = []
        current = ""
        for part in parts:
            current += part
            if re.match(r'[。！？.!?]+\s*$', part):
                sentence = current.strip()
                if sentence:
                    sentences.append(sentence)
                current = ""

        # 处理最后一个没有句号的部分
        if current.strip():
            sentences.append(current.strip())

        return sentences

    def get_stats(self) -> Dict[str, Any]:
        """获取分片器统计"""
        return {
            "type": self.__class__.__name__,
            "strategy": self.config.strategy,
            "chunk_size": self.config.chunk_size,
            "overlap": self.config.overlap,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.config.strategy!r}, "
            f"chunk_size={self.config.chunk_size})"
        )
