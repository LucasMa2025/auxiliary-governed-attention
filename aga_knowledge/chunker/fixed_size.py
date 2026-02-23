"""
固定大小分片器

按固定 token 数分片，不考虑语义边界。
适用于结构化文本或对分片精度要求不高的场景。
"""

import logging
from typing import List

from .base import BaseChunker, ChunkerConfig, KnowledgeChunk

logger = logging.getLogger(__name__)


class FixedSizeChunker(BaseChunker):
    """
    固定大小分片器

    将文本按固定字符数（近似 token 数）分割。
    不保证语义完整性。

    适用场景:
    - 结构化数据（如 JSON、CSV 行）
    - 对分片精度要求不高的场景
    - 快速原型验证
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
        min_size = self.config.min_chunk_size

        # 估算每个 token 对应的字符数
        if language == "auto":
            import re
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            total_chars = len(text)
            chinese_ratio = chinese_chars / max(1, total_chars)
            chars_per_token = 1.5 if chinese_ratio > 0.3 else 4.0
        elif language == "zh":
            chars_per_token = 1.5
        else:
            chars_per_token = 4.0

        # 目标字符数
        target_chars = int(chunk_size * chars_per_token)
        min_chars = int(min_size * chars_per_token)

        # 分片
        chunks = []
        pos = 0
        index = 0

        while pos < len(text):
            end = min(pos + target_chars, len(text))
            chunk_text = text[pos:end].strip()

            if len(chunk_text) < min_chars and chunks:
                # 太短，合并到上一个 chunk
                last = chunks[-1]
                last.decision = last.decision + " " + chunk_text
                last.token_count = self.estimate_tokens(last.decision, language)
                break

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

            pos = end

        # 设置 total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks
