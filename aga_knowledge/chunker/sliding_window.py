"""
滑动窗口分片器

使用滑动窗口策略分片，相邻 chunk 之间有重叠区域。
重叠确保了跨 chunk 边界的语义连续性。

这是推荐的默认分片策略。
"""

import logging
from typing import List

from .base import BaseChunker, ChunkerConfig, KnowledgeChunk

logger = logging.getLogger(__name__)


class SlidingWindowChunker(BaseChunker):
    """
    滑动窗口分片器

    使用固定大小的滑动窗口，相邻 chunk 之间有 overlap 个 token 的重叠。
    在句子边界处对齐窗口，避免截断句子。

    参数:
    - chunk_size: 窗口大小（token 数）
    - overlap: 重叠大小（token 数）

    适用场景:
    - 通用文本分片（推荐默认策略）
    - 需要跨 chunk 语义连续性的场景
    - 长文档处理

    示例:
        文本: [S1 S2 S3 S4 S5 S6 S7 S8]
        chunk_size=4, overlap=1

        Chunk 0: [S1 S2 S3 S4]
        Chunk 1: [S4 S5 S6 S7]     ← S4 重叠
        Chunk 2: [S7 S8]           ← S7 重叠
    """

    def chunk(
        self,
        text: str,
        source_id: str = "",
    ) -> List[KnowledgeChunk]:
        if not text or not text.strip():
            return []

        language = self.config.language
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        min_size = self.config.min_chunk_size

        # 分割句子
        sentences = self.split_sentences(text)
        if not sentences:
            # 如果无法分割句子，回退到按字符分割
            return self._chunk_by_chars(text, source_id, language)

        # 计算每个句子的 token 数
        sentence_tokens = [
            self.estimate_tokens(s, language) for s in sentences
        ]

        # 滑动窗口
        chunks = []
        start_idx = 0
        index = 0

        while start_idx < len(sentences):
            # 从 start_idx 开始累积句子直到达到 chunk_size
            end_idx = start_idx
            current_tokens = 0

            while end_idx < len(sentences):
                if current_tokens + sentence_tokens[end_idx] > chunk_size:
                    if end_idx == start_idx:
                        # 单个句子超过 chunk_size，强制包含
                        end_idx += 1
                    break
                current_tokens += sentence_tokens[end_idx]
                end_idx += 1

            # 构建 chunk 文本
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences).strip()

            if not chunk_text:
                start_idx = end_idx
                continue

            token_count = self.estimate_tokens(chunk_text, language)

            # 如果太短且不是最后一个，跳过（会被下一个窗口覆盖）
            if token_count < min_size and end_idx < len(sentences):
                start_idx = end_idx
                continue

            # 如果太短且是最后一个，合并到上一个 chunk
            if token_count < min_size and chunks:
                last = chunks[-1]
                last.decision = last.decision + " " + chunk_text
                last.token_count = self.estimate_tokens(last.decision, language)
                break

            chunk_id = self.generate_chunk_id(chunk_text, source_id, index)
            condition = self.extract_first_sentence(chunk_text)

            chunks.append(KnowledgeChunk(
                chunk_id=chunk_id,
                condition=condition,
                decision=chunk_text,
                source_id=source_id,
                chunk_index=index,
                token_count=token_count,
            ))
            index += 1

            # 计算下一个窗口的起始位置（考虑重叠）
            if end_idx >= len(sentences):
                break

            # 回退 overlap 个 token
            overlap_tokens = 0
            next_start = end_idx
            while next_start > start_idx and overlap_tokens < overlap:
                next_start -= 1
                overlap_tokens += sentence_tokens[next_start]

            # 确保至少前进一个句子
            if next_start <= start_idx:
                next_start = start_idx + 1

            start_idx = next_start

        # 设置 total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_by_chars(
        self,
        text: str,
        source_id: str,
        language: str,
    ) -> List[KnowledgeChunk]:
        """
        按字符分割（回退策略）

        当无法分割句子时使用。
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap

        # 估算字符数
        if language == "zh":
            chars_per_token = 1.5
        elif language == "en":
            chars_per_token = 4.0
        else:
            chars_per_token = 2.5

        target_chars = int(chunk_size * chars_per_token)
        overlap_chars = int(overlap * chars_per_token)
        step = max(1, target_chars - overlap_chars)

        chunks = []
        pos = 0
        index = 0

        while pos < len(text):
            end = min(pos + target_chars, len(text))
            chunk_text = text[pos:end].strip()

            if chunk_text:
                chunk_id = self.generate_chunk_id(chunk_text, source_id, index)
                condition = self.extract_first_sentence(chunk_text)

                chunks.append(KnowledgeChunk(
                    chunk_id=chunk_id,
                    condition=condition,
                    decision=chunk_text,
                    source_id=source_id,
                    chunk_index=index,
                    token_count=self.estimate_tokens(chunk_text, language),
                ))
                index += 1

            pos += step

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks
