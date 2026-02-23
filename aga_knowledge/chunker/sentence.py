"""
句子边界分片器

按句子边界分片，确保每个 chunk 包含完整的句子。
在语义完整性和大小控制之间取得平衡。
"""

import logging
from typing import List

from .base import BaseChunker, ChunkerConfig, KnowledgeChunk

logger = logging.getLogger(__name__)


class SentenceChunker(BaseChunker):
    """
    句子边界分片器

    将文本按句子边界分割，确保每个 chunk 包含完整句子。
    当累积的句子 token 数达到 chunk_size 时，创建新 chunk。

    适用场景:
    - 自然语言文本（文章、报告、手册）
    - 需要语义完整性的场景
    - 中英文混合文本
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
        max_size = self.config.max_chunk_size

        # 分割句子
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        # 按句子累积分片
        chunks = []
        current_sentences: List[str] = []
        current_tokens = 0
        index = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence, language)

            # 如果单个句子超过 max_size，强制截断
            if sentence_tokens > max_size:
                # 先保存当前累积的
                if current_sentences:
                    chunk_text = " ".join(current_sentences).strip()
                    if self.estimate_tokens(chunk_text, language) >= min_size:
                        chunks.append(self._create_chunk(
                            chunk_text, source_id, index, language
                        ))
                        index += 1
                    current_sentences = []
                    current_tokens = 0

                # 强制截断长句子
                chunks.append(self._create_chunk(
                    sentence, source_id, index, language
                ))
                index += 1
                continue

            # 检查是否超过 chunk_size
            if current_tokens + sentence_tokens > chunk_size and current_sentences:
                # 创建 chunk
                chunk_text = " ".join(current_sentences).strip()
                if self.estimate_tokens(chunk_text, language) >= min_size:
                    chunks.append(self._create_chunk(
                        chunk_text, source_id, index, language
                    ))
                    index += 1
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # 处理剩余的句子
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            token_count = self.estimate_tokens(chunk_text, language)

            if token_count >= min_size or not chunks:
                chunks.append(self._create_chunk(
                    chunk_text, source_id, index, language
                ))
            elif chunks:
                # 太短，合并到上一个 chunk
                last = chunks[-1]
                last.decision = last.decision + " " + chunk_text
                last.token_count = self.estimate_tokens(last.decision, language)

        # 设置 total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _create_chunk(
        self,
        text: str,
        source_id: str,
        index: int,
        language: str,
    ) -> KnowledgeChunk:
        """创建 KnowledgeChunk"""
        chunk_id = self.generate_chunk_id(text, source_id, index)
        condition = self._generate_condition(text)

        return KnowledgeChunk(
            chunk_id=chunk_id,
            condition=condition,
            decision=text,
            source_id=source_id,
            chunk_index=index,
            token_count=self.estimate_tokens(text, language),
        )

    def _generate_condition(self, text: str) -> str:
        """根据配置生成 condition"""
        mode = self.config.condition_mode

        if mode == "first_sentence":
            return self.extract_first_sentence(text)
        elif mode == "full":
            return text
        elif mode == "empty":
            return ""
        else:
            return self.extract_first_sentence(text)
