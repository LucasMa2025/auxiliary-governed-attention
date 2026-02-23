"""
语义段落分片器

按文档结构（标题、段落、列表）分片。
保留文档的层次结构信息。
"""

import re
import logging
from typing import List, Tuple

from .base import BaseChunker, ChunkerConfig, KnowledgeChunk

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    语义段落分片器

    按文档的自然结构分片:
    - Markdown 标题（#, ##, ###）
    - 段落分隔（空行）
    - 列表项

    每个 chunk 保留其所属的标题上下文作为 condition。

    适用场景:
    - Markdown 文档
    - 技术文档、手册
    - 结构化知识库
    """

    # Markdown 标题模式
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def chunk(
        self,
        text: str,
        source_id: str = "",
    ) -> List[KnowledgeChunk]:
        if not text or not text.strip():
            return []

        language = self.config.language
        separator = self.config.separator
        chunk_size = self.config.chunk_size
        min_size = self.config.min_chunk_size

        # 按段落分割
        sections = self._split_sections(text, separator)

        # 合并小段落，拆分大段落
        merged = self._merge_and_split(sections, chunk_size, min_size, language)

        # 生成 chunks
        chunks = []
        for index, (heading, content) in enumerate(merged):
            if not content.strip():
                continue

            token_count = self.estimate_tokens(content, language)
            if token_count < min_size and chunks:
                # 合并到上一个
                last = chunks[-1]
                last.decision = last.decision + "\n\n" + content
                last.token_count = self.estimate_tokens(last.decision, language)
                continue

            chunk_id = self.generate_chunk_id(content, source_id, index)

            # condition: 标题 + 首句
            if heading:
                condition = heading
            else:
                condition = self.extract_first_sentence(content)

            chunks.append(KnowledgeChunk(
                chunk_id=chunk_id,
                condition=condition,
                decision=content,
                source_id=source_id,
                chunk_index=index,
                token_count=token_count,
                metadata={"heading": heading} if heading else None,
            ))

        # 重新编号和设置 total_chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_sections(
        self,
        text: str,
        separator: str,
    ) -> List[Tuple[str, str]]:
        """
        按标题和段落分割文本

        Returns:
            List[(heading, content)] 元组列表
        """
        sections: List[Tuple[str, str]] = []
        current_heading = ""
        current_content: List[str] = []

        lines = text.split("\n")

        for line in lines:
            # 检查是否是标题
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                # 保存当前段落
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        sections.append((current_heading, content))
                    current_content = []

                current_heading = heading_match.group(2).strip()
                continue

            # 检查是否是段落分隔
            if line.strip() == "" and current_content:
                # 检查是否累积了足够的内容
                content = "\n".join(current_content).strip()
                if content and separator == "\n\n":
                    sections.append((current_heading, content))
                    current_content = []
                    continue

            if line.strip():
                current_content.append(line)

        # 处理最后一段
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                sections.append((current_heading, content))

        return sections

    def _merge_and_split(
        self,
        sections: List[Tuple[str, str]],
        chunk_size: int,
        min_size: int,
        language: str,
    ) -> List[Tuple[str, str]]:
        """
        合并小段落，拆分大段落

        Returns:
            调整后的 (heading, content) 列表
        """
        result: List[Tuple[str, str]] = []

        for heading, content in sections:
            token_count = self.estimate_tokens(content, language)

            if token_count <= chunk_size:
                # 大小合适，直接添加
                result.append((heading, content))
            else:
                # 太大，按句子拆分
                sentences = self.split_sentences(content)
                current_text = ""
                current_tokens = 0

                for sentence in sentences:
                    s_tokens = self.estimate_tokens(sentence, language)

                    if current_tokens + s_tokens > chunk_size and current_text:
                        result.append((heading, current_text.strip()))
                        current_text = ""
                        current_tokens = 0

                    current_text += " " + sentence
                    current_tokens += s_tokens

                if current_text.strip():
                    result.append((heading, current_text.strip()))

        # 合并过小的段落
        merged: List[Tuple[str, str]] = []
        for heading, content in result:
            token_count = self.estimate_tokens(content, language)

            if token_count < min_size and merged:
                # 合并到上一个
                prev_heading, prev_content = merged[-1]
                merged[-1] = (prev_heading, prev_content + "\n\n" + content)
            else:
                merged.append((heading, content))

        return merged
