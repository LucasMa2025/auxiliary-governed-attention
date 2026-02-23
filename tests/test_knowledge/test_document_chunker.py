"""
文档级分片器、Condition 生成器、图片处理器单元测试

测试范围:
  - ConditionGenerator: 4 种 condition 生成策略
  - DocumentChunker: 文档结构解析 + 分片 + condition 增强
  - ImageHandler: Base64/URL/本地路径图片处理
  - create_document_chunker: 工厂函数
"""

import base64
import os
import tempfile
import shutil
import pytest

from aga_knowledge.chunker.base import ChunkerConfig, KnowledgeChunk
from aga_knowledge.chunker.condition_generator import ConditionGenerator
from aga_knowledge.chunker.document_chunker import DocumentChunker, DocumentSection
from aga_knowledge.chunker.image_handler import ImageHandler, ImageAsset
from aga_knowledge.chunker import create_chunker, create_document_chunker


# ==================== ConditionGenerator 测试 ====================


class TestConditionGenerator:
    """测试 condition 生成器"""

    def test_first_sentence_mode(self):
        gen = ConditionGenerator(mode="first_sentence")
        result = gen.generate("心脏由四个腔室组成。左心室负责泵血。")
        assert "心脏由四个腔室组成" in result

    def test_first_sentence_english(self):
        gen = ConditionGenerator(mode="first_sentence")
        result = gen.generate("The heart has four chambers. The left ventricle pumps blood.")
        assert "The heart has four chambers." in result

    def test_first_sentence_long_text(self):
        gen = ConditionGenerator(mode="first_sentence")
        # 第一句话很长（超过文本50%）时，应截断
        text = "A" * 200
        result = gen.generate(text)
        assert len(result) <= 85  # 80 + "..."

    def test_title_context_mode(self):
        gen = ConditionGenerator(mode="title_context")
        result = gen.generate(
            text="心脏由四个腔室组成。左心室负责泵血。",
            title="心脏解剖学",
            section="心脏结构",
        )
        assert "心脏解剖学" in result
        assert "心脏结构" in result
        assert "心脏由四个腔室组成" in result
        assert " > " in result

    def test_title_context_no_title(self):
        gen = ConditionGenerator(mode="title_context")
        result = gen.generate(
            text="心脏由四个腔室组成。左心室负责泵血。",
        )
        assert "心脏由四个腔室组成" in result

    def test_keyword_mode(self):
        gen = ConditionGenerator(mode="keyword")
        result = gen.generate(
            text="Transformer 模型在自然语言处理中广泛应用。Transformer 是深度学习的核心架构。",
            title="深度学习",
        )
        assert "深度学习" in result
        assert "transformer" in result.lower()

    def test_keyword_mode_no_title(self):
        gen = ConditionGenerator(mode="keyword")
        result = gen.generate(
            text="Python is a programming language. Python is widely used.",
        )
        assert "python" in result.lower()

    def test_summary_mode_chinese(self):
        gen = ConditionGenerator(mode="summary")
        text = "心脏是人体最重要的器官之一，负责将血液泵送到全身各个部位，维持生命活动的正常进行。"
        result = gen.generate(text)
        assert len(result) <= 35  # 中文取前30字符
        assert "心脏" in result

    def test_summary_mode_english(self):
        gen = ConditionGenerator(mode="summary")
        text = "The heart is one of the most important organs in the human body responsible for pumping blood throughout the body."
        result = gen.generate(text)
        words = result.split()
        assert len(words) <= 15

    def test_empty_text(self):
        gen = ConditionGenerator(mode="first_sentence")
        result = gen.generate("")
        assert result == ""

    def test_empty_text_with_title(self):
        gen = ConditionGenerator(mode="title_context")
        result = gen.generate("", title="标题")
        assert result == "标题"

    def test_unknown_mode_fallback(self):
        gen = ConditionGenerator(mode="nonexistent_mode")
        # 应回退到 first_sentence
        assert gen.mode == "first_sentence"

    def test_get_stats(self):
        gen = ConditionGenerator(mode="keyword")
        stats = gen.get_stats()
        assert stats["type"] == "ConditionGenerator"
        assert stats["mode"] == "keyword"

    def test_repr(self):
        gen = ConditionGenerator(mode="title_context")
        assert "title_context" in repr(gen)


# ==================== DocumentChunker 测试 ====================


class TestDocumentSection:
    """测试文档章节数据类"""

    def test_basic_section(self):
        section = DocumentSection(
            title="测试章节",
            level=2,
            content="这是章节内容。",
        )
        assert section.title == "测试章节"
        assert section.level == 2
        assert section.content == "这是章节内容。"
        assert section.images == []
        assert section.parent_titles == []


class TestDocumentChunker:
    """测试文档级分片器"""

    @pytest.fixture
    def chunker(self):
        config = ChunkerConfig(
            strategy="sliding_window",
            chunk_size=100,
            overlap=20,
            condition_mode="title_context",
        )
        base_chunker = create_chunker(config)
        return DocumentChunker(config, base_chunker)

    def test_basic_document_chunking(self, chunker):
        text = """# 第一章 概述

这是第一章的内容。主要介绍系统的基本概念。

## 1.1 背景

这是背景部分的内容。这里会介绍相关的技术背景。

## 1.2 目标

这是目标部分的内容。这里会说明系统的设计目标。

# 第二章 设计

这是第二章的内容。主要介绍系统的设计方案。
"""
        chunks = chunker.chunk_document(
            text=text,
            source_id="doc_001",
            title="系统设计文档",
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, KnowledgeChunk)
            assert chunk.condition  # condition 不为空
            assert chunk.decision   # decision 不为空
            assert chunk.metadata is not None
            assert "document_title" in chunk.metadata
            assert chunk.metadata["document_title"] == "系统设计文档"

    def test_section_metadata(self, chunker):
        text = """## 心脏结构

心脏由四个腔室组成。左心室负责将富含氧气的血液泵送到全身。右心室将缺氧的血液泵送到肺部。
"""
        chunks = chunker.chunk_document(
            text=text,
            source_id="doc_002",
            title="心脏解剖学",
        )
        assert len(chunks) > 0
        first_chunk = chunks[0]
        assert first_chunk.metadata["section_title"] == "心脏结构"
        assert first_chunk.metadata["section_level"] == 2

    def test_condition_contains_title_context(self, chunker):
        text = """## 心脏结构

心脏由四个腔室组成。左心室负责将富含氧气的血液泵送到全身。
"""
        chunks = chunker.chunk_document(
            text=text,
            source_id="doc_003",
            title="心脏解剖学",
        )
        assert len(chunks) > 0
        # condition 应包含文档标题和章节标题
        cond = chunks[0].condition
        assert "心脏解剖学" in cond
        assert "心脏结构" in cond

    def test_no_headers(self, chunker):
        text = "这是一段没有标题结构的纯文本。内容很简单。"
        chunks = chunker.chunk_document(
            text=text,
            source_id="doc_004",
        )
        assert len(chunks) > 0

    def test_empty_text(self, chunker):
        chunks = chunker.chunk_document(text="", source_id="empty")
        assert len(chunks) == 0

    def test_chunk_numbering(self, chunker):
        text = """# 第一章

第一章内容很长，包含多个句子。这是第一个句子。这是第二个句子。这是第三个句子。这是第四个句子。这是第五个句子。这是第六个句子。这是第七个句子。这是第八个句子。

# 第二章

第二章内容也很长。这是另一组句子。这是第二个句子。这是第三个句子。这是第四个句子。这是第五个句子。
"""
        chunks = chunker.chunk_document(
            text=text,
            source_id="doc_005",
            title="长文档",
        )
        # 检查编号连续
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_get_stats(self, chunker):
        stats = chunker.get_stats()
        assert stats["type"] == "DocumentChunker"
        assert stats["condition_mode"] == "title_context"
        assert stats["image_handler_enabled"] is False

    def test_repr(self, chunker):
        r = repr(chunker)
        assert "DocumentChunker" in r


class TestDocumentChunkerSectionParsing:
    """测试文档结构解析"""

    @pytest.fixture
    def chunker(self):
        config = ChunkerConfig(
            strategy="sliding_window",
            chunk_size=100,
            overlap=20,
        )
        base_chunker = create_chunker(config)
        return DocumentChunker(config, base_chunker)

    def test_parse_markdown_headers(self, chunker):
        text = """# H1 Title

Content under H1.

## H2 Title

Content under H2.

### H3 Title

Content under H3.
"""
        sections = chunker._parse_sections(text)
        assert len(sections) >= 3
        assert sections[0].title == "H1 Title"
        assert sections[0].level == 1
        assert sections[1].title == "H2 Title"
        assert sections[1].level == 2
        assert sections[2].title == "H3 Title"
        assert sections[2].level == 3

    def test_parse_no_headers(self, chunker):
        text = "Just plain text without headers."
        sections = chunker._parse_sections(text)
        assert len(sections) == 1
        assert sections[0].title == ""
        assert sections[0].level == 0

    def test_parse_image_detection(self, chunker):
        text = """## Section with Image

Some text before.

![Architecture Diagram](https://example.com/arch.png)

Some text after.
"""
        sections = chunker._parse_sections(text)
        assert len(sections) >= 1
        # 找到包含图片的章节
        img_section = [s for s in sections if s.images]
        assert len(img_section) > 0
        assert img_section[0].images[0]["alt"] == "Architecture Diagram"
        assert img_section[0].images[0]["src"] == "https://example.com/arch.png"


# ==================== ImageHandler 测试 ====================


class TestImageAsset:
    """测试图片资产数据类"""

    def test_to_dict(self):
        asset = ImageAsset(
            asset_id="doc_001_img_001",
            alt_text="测试图片",
            original_src="data:image/png;base64,...",
            portal_url="http://portal/assets/doc_001/img_001.png",
            mime_type="image/png",
            file_size=1024,
        )
        d = asset.to_dict()
        assert d["asset_id"] == "doc_001_img_001"
        assert d["alt"] == "测试图片"
        assert d["url"] == "http://portal/assets/doc_001/img_001.png"
        assert d["mime_type"] == "image/png"
        assert d["file_size"] == 1024


class TestImageHandler:
    """测试图片处理器"""

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def handler(self, temp_dir):
        return ImageHandler(
            asset_dir=temp_dir,
            base_url="http://portal:8081/assets",
            max_image_size_mb=10,
        )

    def test_external_url_preserved(self, handler):
        text = "Some text ![Diagram](https://example.com/image.png) more text."
        processed, assets = handler.process_document(text, "doc_001")

        assert len(assets) == 1
        assert assets[0].portal_url == "https://example.com/image.png"
        assert "example.com/image.png" in processed

    def test_base64_image_saved(self, handler, temp_dir):
        # 创建一个小的 PNG 图片（1x1 像素）
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(png_data).decode()
        text = f"Before ![Test](data:image/png;base64,{b64}) After"

        processed, assets = handler.process_document(text, "doc_002")

        assert len(assets) == 1
        assert assets[0].mime_type == "image/png"
        assert assets[0].file_size > 0
        assert "portal:8081/assets/doc_002" in assets[0].portal_url

        # 验证文件已保存
        doc_dir = os.path.join(temp_dir, "doc_002")
        assert os.path.exists(doc_dir)
        files = os.listdir(doc_dir)
        assert len(files) == 1

    def test_base64_invalid_format(self, handler):
        text = "![Test](data:application/pdf;base64,AAAA) text"
        processed, assets = handler.process_document(text, "doc_003")
        assert len(assets) == 0
        # 原始引用保留
        assert "data:application/pdf" in processed

    def test_base64_unsupported_image_format(self, handler):
        text = "![Test](data:image/bmp;base64,AAAA) text"
        processed, assets = handler.process_document(text, "doc_003b")
        assert len(assets) == 0

    def test_local_path_image(self, handler, temp_dir):
        # 创建一个临时图片文件
        img_path = os.path.join(temp_dir, "test_image.png")
        with open(img_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        text = f"![Local Image]({img_path}) some text"
        processed, assets = handler.process_document(text, "doc_004")

        assert len(assets) == 1
        assert assets[0].portal_url.startswith("http://portal:8081/assets/doc_004/")

    def test_local_path_not_exists(self, handler):
        text = "![Missing](./nonexistent.png) text"
        processed, assets = handler.process_document(text, "doc_005")
        assert len(assets) == 0
        # 原始引用保留
        assert "nonexistent.png" in processed

    def test_description_template(self, temp_dir):
        handler = ImageHandler(
            asset_dir=temp_dir,
            base_url="http://portal:8081/assets",
            description_template="[IMG: {alt} at {url}]",
        )
        text = "![Test](https://example.com/img.png) text"
        processed, assets = handler.process_document(text, "doc_006")
        assert "[IMG: Test at https://example.com/img.png]" in processed

    def test_multiple_images(self, handler):
        text = """
Text before.
![Image 1](https://example.com/img1.png)
Middle text.
![Image 2](https://example.com/img2.jpg)
End text.
"""
        processed, assets = handler.process_document(text, "doc_007")
        assert len(assets) == 2

    def test_empty_text(self, handler):
        processed, assets = handler.process_document("", "doc_008")
        assert processed == ""
        assert len(assets) == 0

    def test_no_images(self, handler):
        text = "Plain text without any images."
        processed, assets = handler.process_document(text, "doc_009")
        assert processed == text
        assert len(assets) == 0

    def test_cleanup_assets(self, handler, temp_dir):
        # 创建文档资源目录和文件
        doc_dir = os.path.join(temp_dir, "doc_cleanup")
        os.makedirs(doc_dir)
        with open(os.path.join(doc_dir, "img_001.png"), "w") as f:
            f.write("fake image")
        with open(os.path.join(doc_dir, "img_002.png"), "w") as f:
            f.write("fake image 2")

        count = handler.cleanup_assets("doc_cleanup")
        assert count == 2
        assert not os.path.exists(doc_dir)

    def test_cleanup_nonexistent(self, handler):
        count = handler.cleanup_assets("nonexistent_doc")
        assert count == 0

    def test_get_stats(self, handler):
        stats = handler.get_stats()
        assert stats["type"] == "ImageHandler"
        assert "asset_dir" in stats
        assert "base_url" in stats

    def test_repr(self, handler):
        r = repr(handler)
        assert "ImageHandler" in r

    def test_external_url_mime_detection(self, handler):
        text = "![JPEG](https://example.com/photo.jpg)"
        _, assets = handler.process_document(text, "doc_010")
        assert assets[0].mime_type == "image/jpeg"


# ==================== create_document_chunker 测试 ====================


class TestCreateDocumentChunker:
    """测试文档分片器工厂函数"""

    def test_basic_creation(self):
        config = ChunkerConfig(
            strategy="sliding_window",
            chunk_size=200,
            overlap=30,
            condition_mode="title_context",
        )
        chunker = create_document_chunker(config)
        assert isinstance(chunker, DocumentChunker)

    def test_with_image_handler(self):
        config = ChunkerConfig(strategy="sliding_window")
        handler = ImageHandler(
            asset_dir="/tmp/test",
            base_url="http://localhost/assets",
        )
        chunker = create_document_chunker(config, image_handler=handler)
        assert chunker.image_handler is not None
        stats = chunker.get_stats()
        assert stats["image_handler_enabled"] is True

    def test_end_to_end(self):
        """端到端测试: 创建分片器 → 分片文档 → 验证结果"""
        config = ChunkerConfig(
            strategy="sliding_window",
            chunk_size=50,
            overlap=10,
            condition_mode="title_context",
        )
        chunker = create_document_chunker(config)

        text = """# 深度学习概述

深度学习是机器学习的一个分支。它使用多层神经网络来学习数据的表示。近年来取得了巨大的进展。

## Transformer 架构

Transformer 是一种基于自注意力机制的神经网络架构。它在自然语言处理领域取得了突破性的成果。BERT和GPT都是基于Transformer的模型。

## 应用领域

深度学习在计算机视觉、自然语言处理、语音识别等领域广泛应用。
"""
        chunks = chunker.chunk_document(
            text=text,
            source_id="dl_overview",
            title="深度学习教程",
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.condition
            assert chunk.decision
            assert chunk.metadata is not None
            assert chunk.metadata.get("document_title") == "深度学习教程"
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks == len(chunks)
