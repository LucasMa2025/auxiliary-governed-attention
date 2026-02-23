"""
文档级分片器

在基础分片策略之上，增加文档结构感知：
  1. 解析 Markdown/HTML 标题层级
  2. 按章节分组
  3. 保留标题上下文到 condition（通过 ConditionGenerator）
  4. 处理图片引用（通过 ImageHandler）

这是面向知识注册的推荐分片入口。

使用:
    from aga_knowledge.chunker import DocumentChunker, ChunkerConfig, create_chunker
    from aga_knowledge.chunker.condition_generator import ConditionGenerator

    config = ChunkerConfig(strategy="sliding_window", chunk_size=300, overlap=50)
    base_chunker = create_chunker(config)
    doc_chunker = DocumentChunker(config, base_chunker)

    chunks = doc_chunker.chunk_document(
        text=markdown_text,
        source_id="doc_001",
        title="心脏解剖学",
    )
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .base import BaseChunker, ChunkerConfig, KnowledgeChunk
from .condition_generator import ConditionGenerator

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """
    文档章节

    解析 Markdown 标题后的结构化表示。

    Attributes:
        title: 章节标题
        level: 标题层级 (1=H1, 2=H2, ...)
        content: 章节正文（不含标题行）
        images: 章节中的图片引用列表
        parent_titles: 父级标题链（用于生成层级路径）
        start_offset: 在原文中的起始位置
        end_offset: 在原文中的结束位置
    """
    title: str
    level: int = 0
    content: str = ""
    images: List[Dict[str, Any]] = field(default_factory=list)
    parent_titles: List[str] = field(default_factory=list)
    start_offset: int = 0
    end_offset: int = 0


class DocumentChunker:
    """
    文档级分片器

    在基础分片策略（如 SlidingWindowChunker）之上，增加文档结构感知。

    功能:
    1. 解析 Markdown 标题层级结构
    2. 按章节分组进行分片
    3. 使用 ConditionGenerator 生成高质量的 condition
    4. 保留文档级和章节级元数据
    5. 可选集成 ImageHandler 处理文档中的图片

    配置:
      chunker:
        strategy: "sliding_window"
        chunk_size: 300
        overlap: 50
        condition_mode: "title_context"

    生命周期:
        1. __init__(config, base_chunker): 初始化
        2. chunk_document(text, source_id, title): 分片文档
    """

    def __init__(
        self,
        config: ChunkerConfig,
        base_chunker: BaseChunker,
        image_handler=None,
    ):
        """
        初始化文档分片器

        Args:
            config: 分片器配置
            base_chunker: 基础分片器（如 SlidingWindowChunker）
            image_handler: 可选的 ImageHandler 实例（处理文档中的图片）
        """
        self.config = config
        self.base_chunker = base_chunker
        self.condition_gen = ConditionGenerator(
            mode=config.condition_mode
        )
        self.image_handler = image_handler

    def chunk_document(
        self,
        text: str,
        source_id: str = "",
        title: str = "",
    ) -> List[KnowledgeChunk]:
        """
        分片文档

        完整的文档分片流程:
        1. 预处理（图片处理，如果启用）
        2. 解析文档结构（Markdown 标题层级）
        3. 按章节分片
        4. 生成 condition（带文档/章节上下文）
        5. 附加元数据
        6. 全局重新编号

        Args:
            text: 文档全文（Markdown 或纯文本）
            source_id: 源文档 ID
            title: 文档标题

        Returns:
            KnowledgeChunk 列表（包含结构化元数据）
        """
        if not text or not text.strip():
            return []

        processed_text = text
        image_assets = []

        # 1. 图片处理（如果启用）
        if self.image_handler:
            try:
                processed_text, image_assets = self.image_handler.process_document(
                    text=text,
                    source_id=source_id,
                )
                if image_assets:
                    logger.info(
                        f"文档 {source_id} 提取了 {len(image_assets)} 张图片"
                    )
            except Exception as e:
                logger.warning(
                    f"图片处理失败，使用原始文本: {e}"
                )
                processed_text = text

        # 2. 解析文档结构
        sections = self._parse_sections(processed_text)

        # 3. 按章节分片 + 生成 condition
        all_chunks = []
        image_index = 0

        for section in sections:
            if not section.content.strip():
                continue

            # 使用基础分片器分片
            section_chunks = self.base_chunker.chunk(
                section.content, source_id
            )

            # 增强每个 chunk
            for chunk in section_chunks:
                # 生成高质量 condition
                chunk.condition = self.condition_gen.generate(
                    text=chunk.decision,
                    title=title,
                    section=section.title,
                    chunk_index=chunk.chunk_index,
                )

                # 附加结构元数据
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata["document_title"] = title
                chunk.metadata["section_title"] = section.title
                chunk.metadata["section_level"] = section.level
                if section.parent_titles:
                    chunk.metadata["section_path"] = " > ".join(
                        section.parent_titles + [section.title]
                    )

                # 附加图片元数据（如果该 chunk 中包含图片 URL）
                if image_assets:
                    chunk_images = self._find_images_in_chunk(
                        chunk.decision, image_assets
                    )
                    if chunk_images:
                        chunk.metadata["images"] = chunk_images

            all_chunks.extend(section_chunks)

        # 4. 全局重新编号
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(all_chunks)

        logger.info(
            f"文档 '{title or source_id}' 分片完成: "
            f"{len(sections)} 个章节 → {len(all_chunks)} 个片段"
        )

        return all_chunks

    def _parse_sections(self, text: str) -> List[DocumentSection]:
        """
        解析 Markdown 标题结构

        支持:
        - ATX 标题: # H1, ## H2, ### H3 ...
        - 无标题文本作为根章节

        保留标题层级关系，构建 parent_titles 链。

        Args:
            text: Markdown 文本

        Returns:
            DocumentSection 列表
        """
        sections: List[DocumentSection] = []

        # 标题栈: [(level, title), ...] 用于追踪层级
        title_stack: List[tuple] = []
        current_title = ""
        current_level = 0
        current_content: List[str] = []
        current_images: List[Dict[str, Any]] = []
        current_start = 0

        lines = text.split("\n")

        for line_num, line in enumerate(lines):
            # 检测 Markdown ATX 标题
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if header_match:
                # 保存前一个章节
                if current_content or current_title:
                    parent_titles = [t for _, t in title_stack]
                    sections.append(DocumentSection(
                        title=current_title,
                        level=current_level,
                        content="\n".join(current_content).strip(),
                        images=current_images,
                        parent_titles=parent_titles,
                        start_offset=current_start,
                        end_offset=line_num,
                    ))

                # 解析新标题
                new_level = len(header_match.group(1))
                new_title = header_match.group(2).strip()

                # 更新标题栈
                while title_stack and title_stack[-1][0] >= new_level:
                    title_stack.pop()
                # 当前标题的父级是栈中所有标题
                # 注意: 不把自己加入 parent_titles
                title_stack.append((new_level, new_title))

                current_title = new_title
                current_level = new_level
                current_content = []
                current_images = []
                current_start = line_num + 1
            else:
                current_content.append(line)
                # 检测图片引用
                img_matches = re.findall(
                    r"!\[([^\]]*)\]\(([^)]+)\)", line
                )
                for alt, src in img_matches:
                    current_images.append({
                        "alt": alt,
                        "src": src,
                        "line": line_num,
                    })

        # 保存最后一个章节
        if current_content or current_title:
            parent_titles = [t for _, t in title_stack[:-1]] if title_stack else []
            sections.append(DocumentSection(
                title=current_title,
                level=current_level,
                content="\n".join(current_content).strip(),
                images=current_images,
                parent_titles=parent_titles,
                start_offset=current_start,
                end_offset=len(lines),
            ))

        # 如果完全没有标题结构，整体作为一个章节
        if not sections:
            sections = [DocumentSection(
                title="",
                level=0,
                content=text.strip(),
            )]

        return sections

    def _find_images_in_chunk(
        self,
        chunk_text: str,
        image_assets: list,
    ) -> List[Dict[str, Any]]:
        """
        查找 chunk 文本中引用的图片资产

        通过检查 chunk 文本是否包含图片的 portal_url 来判断。

        Args:
            chunk_text: chunk 的 decision 文本
            image_assets: 所有的图片资产列表

        Returns:
            匹配的图片信息列表
        """
        found = []
        for asset in image_assets:
            portal_url = getattr(asset, "portal_url", "")
            alt_text = getattr(asset, "alt_text", "")
            if portal_url and portal_url in chunk_text:
                found.append({
                    "asset_id": getattr(asset, "asset_id", ""),
                    "alt": alt_text,
                    "url": portal_url,
                })
            elif alt_text and alt_text in chunk_text:
                found.append({
                    "asset_id": getattr(asset, "asset_id", ""),
                    "alt": alt_text,
                    "url": portal_url,
                })
        return found

    def get_stats(self) -> Dict[str, Any]:
        """获取分片器统计"""
        return {
            "type": "DocumentChunker",
            "base_chunker": self.base_chunker.get_stats(),
            "condition_mode": self.condition_gen.mode,
            "image_handler_enabled": self.image_handler is not None,
        }

    def __repr__(self) -> str:
        return (
            f"DocumentChunker("
            f"base={self.base_chunker!r}, "
            f"condition_mode={self.condition_gen.mode!r})"
        )
