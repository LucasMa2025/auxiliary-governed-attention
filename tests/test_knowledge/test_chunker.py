"""
aga-knowledge 知识分片器模块测试

注意: SemanticChunker 测试需要安装 sentence-transformers 和 nltk，
在 CI 环境中可能跳过。FixedSizeChunker 和 SlidingWindowChunker
只需要 tiktoken。
"""

import pytest

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

pytestmark = pytest.mark.skipif(
    not _HAS_TIKTOKEN,
    reason="tiktoken not installed",
)

from aga_knowledge.chunker.base import ChunkerConfig, KnowledgeChunk as Chunk
from aga_knowledge.chunker import create_chunker


class TestChunkerConfig:
    """测试分片器配置"""

    def test_default_config(self):
        config = ChunkerConfig()
        assert config.strategy == "fixed_size"
        assert config.chunk_size == 256
        assert config.chunk_overlap == 32

    def test_from_dict(self):
        data = {
            "strategy": "sentence",
            "chunk_size": 128,
            "chunk_overlap": 16,
        }
        config = ChunkerConfig.from_dict(data)
        assert config.strategy == "sentence"
        assert config.chunk_size == 128
        assert config.chunk_overlap == 16

    def test_validate_valid(self):
        config = ChunkerConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_chunk_size(self):
        config = ChunkerConfig(chunk_size=0)
        errors = config.validate()
        assert any("chunk_size" in e for e in errors)

    def test_validate_invalid_overlap(self):
        config = ChunkerConfig(chunk_size=100, chunk_overlap=100)
        errors = config.validate()
        assert any("chunk_overlap" in e for e in errors)

    def test_validate_invalid_strategy(self):
        config = ChunkerConfig(strategy="unknown")
        errors = config.validate()
        assert any("strategy" in e for e in errors)


class TestFixedSizeChunker:
    """测试固定大小分片器"""

    @pytest.fixture
    def chunker(self):
        config = ChunkerConfig(
            strategy="fixed_size",
            chunk_size=10,
            chunk_overlap=2,
        )
        return create_chunker(config)

    def test_basic_chunking(self, chunker):
        text = "This is a test sentence that should be split into multiple chunks by the fixed size chunker."
        chunks = chunker.chunk(text, document_id="doc_001")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.text) > 0
            assert chunk.chunk_id != ""

    def test_short_text(self, chunker):
        text = "Short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text.strip()

    def test_empty_text(self, chunker):
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_metadata_propagation(self, chunker):
        text = "Some text that needs to be chunked into pieces for testing purposes."
        chunks = chunker.chunk(
            text,
            document_id="doc_001",
            metadata={"source": "test"},
        )
        for chunk in chunks:
            assert chunk.metadata.get("document_id") == "doc_001"
            assert chunk.metadata.get("source") == "test"
            assert chunk.metadata.get("chunk_strategy") == "fixed_size"

    def test_chunk_overlap(self):
        config = ChunkerConfig(
            strategy="fixed_size",
            chunk_size=5,
            chunk_overlap=2,
        )
        chunker = create_chunker(config)
        text = "one two three four five six seven eight nine ten eleven twelve"
        chunks = chunker.chunk(text)
        # 有重叠时，应该有更多的 chunks
        assert len(chunks) >= 2

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            config = ChunkerConfig(
                strategy="fixed_size",
                chunk_size=10,
                chunk_overlap=10,
            )
            create_chunker(config)


class TestSlidingWindowChunker:
    """测试滑动窗口分片器"""

    @pytest.fixture
    def chunker(self):
        config = ChunkerConfig(
            strategy="sliding_window",
            chunk_size=10,
            chunk_overlap=3,
        )
        return create_chunker(config)

    def test_basic_chunking(self, chunker):
        text = "This is a longer text that should be split into overlapping chunks using the sliding window approach."
        chunks = chunker.chunk(text, document_id="doc_001")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.metadata.get("chunk_strategy") == "sliding_window"

    def test_short_text(self, chunker):
        text = "Short."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_empty_text(self, chunker):
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_overlap_produces_more_chunks(self):
        # 无重叠
        config_no_overlap = ChunkerConfig(
            strategy="sliding_window",
            chunk_size=10,
            chunk_overlap=0,
        )
        # 有重叠
        config_overlap = ChunkerConfig(
            strategy="sliding_window",
            chunk_size=10,
            chunk_overlap=5,
        )
        chunker_no = create_chunker(config_no_overlap)
        chunker_yes = create_chunker(config_overlap)

        text = "A " * 50  # 足够长的文本
        chunks_no = chunker_no.chunk(text)
        chunks_yes = chunker_yes.chunk(text)

        # 有重叠应该产生更多 chunks
        assert len(chunks_yes) >= len(chunks_no)


class TestSentenceChunker:
    """测试句子分片器"""

    @pytest.fixture
    def chunker(self):
        config = ChunkerConfig(
            strategy="sentence",
            chunk_size=50,
            chunk_overlap=10,
        )
        return create_chunker(config)

    def test_basic_chunking(self, chunker):
        text = (
            "This is the first sentence. This is the second sentence. "
            "This is the third sentence. And this is the fourth one. "
            "Finally, the fifth sentence completes this paragraph."
        )
        chunks = chunker.chunk(text, document_id="doc_001")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.metadata.get("chunk_strategy") == "sentence"

    def test_single_sentence(self, chunker):
        text = "Just one sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_empty_text(self, chunker):
        chunks = chunker.chunk("")
        assert len(chunks) == 0


class TestCreateChunker:
    """测试分片器工厂函数"""

    def test_create_fixed_size(self):
        config = ChunkerConfig(strategy="fixed_size")
        chunker = create_chunker(config)
        from aga_knowledge.chunker.fixed_size import FixedSizeChunker
        assert isinstance(chunker, FixedSizeChunker)

    def test_create_sliding_window(self):
        config = ChunkerConfig(strategy="sliding_window")
        chunker = create_chunker(config)
        from aga_knowledge.chunker.sliding_window import SlidingWindowChunker
        assert isinstance(chunker, SlidingWindowChunker)

    def test_create_sentence(self):
        config = ChunkerConfig(strategy="sentence")
        chunker = create_chunker(config)
        from aga_knowledge.chunker.sentence import SentenceChunker
        assert isinstance(chunker, SentenceChunker)

    def test_create_unknown_raises(self):
        config = ChunkerConfig(strategy="unknown")
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker(config)

    def test_create_semantic(self):
        """语义分片器需要 sentence-transformers"""
        config = ChunkerConfig(strategy="semantic")
        try:
            chunker = create_chunker(config)
            from aga_knowledge.chunker.semantic import SemanticChunker
            assert isinstance(chunker, SemanticChunker)
        except ImportError:
            pytest.skip("sentence-transformers not installed")
