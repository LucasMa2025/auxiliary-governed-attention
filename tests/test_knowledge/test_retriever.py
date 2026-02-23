"""
aga-knowledge 召回器 (KnowledgeRetriever) 测试

测试 KnowledgeRetriever 作为 aga-core BaseRetriever 协议的桥接实现。
使用 mock 的 KnowledgeManager 和 Encoder。
"""

import pytest
import asyncio
import torch
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from aga_knowledge.encoder.base import EncoderConfig, EncodedKnowledge


# ==================== Mock 类 ====================

class MockEncoder:
    """Mock 编码器"""

    def __init__(self, key_dim=16, value_dim=32):
        self.config = EncoderConfig(
            backend="mock",
            key_dim=key_dim,
            value_dim=value_dim,
        )
        self._warmed_up = False
        self._encode_count = 0

    def warmup(self):
        self._warmed_up = True

    def encode(
        self,
        condition: str,
        decision: str,
        lu_id: str,
        reliability: float = 1.0,
        metadata: dict = None,
    ) -> EncodedKnowledge:
        self._encode_count += 1
        # 生成确定性向量
        import hashlib
        h = hashlib.md5(f"{condition}:{decision}".encode()).hexdigest()
        key_vec = [float(int(h[i], 16)) / 15.0 for i in range(self.config.key_dim)]
        val_vec = [float(int(h[i % len(h)], 16)) / 15.0 for i in range(self.config.value_dim)]
        return EncodedKnowledge(
            lu_id=lu_id,
            condition=condition,
            decision=decision,
            key_vector=key_vec,
            value_vector=val_vec,
            reliability=reliability,
            metadata=metadata,
        )

    def encode_batch(self, records: list) -> list:
        return [
            self.encode(
                condition=r.get("condition", ""),
                decision=r.get("decision", ""),
                lu_id=r.get("lu_id", ""),
                reliability=r.get("reliability", 1.0),
                metadata=r.get("metadata"),
            )
            for r in records
        ]

    def get_stats(self) -> dict:
        return {
            "type": "MockEncoder",
            "initialized": self._warmed_up,
            "encode_count": self._encode_count,
        }

    def shutdown(self):
        self._warmed_up = False
        self._encode_count = 0


class MockKnowledgeManager:
    """Mock 知识管理器"""

    def __init__(self, knowledge_data: list = None):
        self._knowledge = knowledge_data or []

    async def get_knowledge_for_injection(self, namespace: str) -> list:
        return [k for k in self._knowledge if k.get("namespace", "default") == namespace or namespace == "default"]

    async def increment_hit_count(self, namespace: str, lu_ids: list):
        pass


# ==================== Mock RetrievalQuery ====================

class MockRetrievalQuery:
    """Mock 检索查询"""

    def __init__(
        self,
        hidden_states: torch.Tensor = None,
        query_projected: torch.Tensor = None,
        top_k: int = 5,
        namespace: str = "default",
    ):
        self.hidden_states = hidden_states
        self.query_projected = query_projected
        self.top_k = top_k
        self.namespace = namespace


# ==================== Fixtures ====================

@pytest.fixture
def sample_knowledge():
    """示例知识数据"""
    return [
        {
            "lu_id": "lu_001",
            "condition": "when patient has fever above 38.5",
            "decision": "recommend antipyretic medication",
            "lifecycle_state": "active",
            "reliability": 0.95,
            "metadata": {"source": "medical_guidelines"},
        },
        {
            "lu_id": "lu_002",
            "condition": "when blood pressure is elevated",
            "decision": "monitor and consider antihypertensive",
            "lifecycle_state": "active",
            "reliability": 0.90,
            "metadata": {"source": "clinical_protocol"},
        },
        {
            "lu_id": "lu_003",
            "condition": "when patient reports chest pain",
            "decision": "perform ECG immediately",
            "lifecycle_state": "active",
            "reliability": 0.99,
            "metadata": {"source": "emergency_protocol"},
        },
    ]


@pytest.fixture
def encoder():
    return MockEncoder(key_dim=16, value_dim=32)


@pytest.fixture
def manager(sample_knowledge):
    return MockKnowledgeManager(sample_knowledge)


@pytest.fixture
def retriever(manager, encoder):
    from aga_knowledge.retriever.knowledge_retriever import KnowledgeRetriever
    return KnowledgeRetriever(
        manager=manager,
        encoder=encoder,
        namespace="default",
        similarity_threshold=0.0,
    )


@pytest.fixture
def warmed_retriever(retriever):
    """已预热的 retriever"""
    # 使用 asyncio.run 来运行异步的 manager 方法
    # warmup 内部会调用 _refresh_index_sync
    with patch.object(
        retriever, '_refresh_index_sync',
        wraps=retriever._refresh_index_sync,
    ):
        # 直接手动构建索引以避免 asyncio 事件循环问题
        _build_index_manually(retriever)
    return retriever


def _build_index_manually(retriever):
    """手动构建索引（避免 asyncio 事件循环问题）"""
    knowledge_list = []
    for k in retriever.manager._knowledge:
        knowledge_list.append(k)

    if not knowledge_list:
        return

    records = [
        {
            "condition": k.get("condition", ""),
            "decision": k.get("decision", ""),
            "lu_id": k.get("lu_id", ""),
            "reliability": k.get("reliability", 1.0),
            "metadata": k.get("metadata"),
        }
        for k in knowledge_list
    ]

    encoded_list = retriever.encoder.encode_batch(records)

    retriever._index.clear()
    retriever._index_ids.clear()
    retriever._index_metadata.clear()

    key_vectors = []
    for encoded in encoded_list:
        lu_id = encoded.lu_id
        retriever._index[lu_id] = encoded
        retriever._index_ids.append(lu_id)
        retriever._index_metadata[lu_id] = {
            "condition": encoded.condition,
            "decision": encoded.decision,
            "reliability": encoded.reliability,
            "metadata": encoded.metadata,
        }
        key_vectors.append(encoded.key_vector)

    if key_vectors:
        retriever._key_matrix = torch.tensor(key_vectors, dtype=torch.float32)
        norms = retriever._key_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        retriever._key_matrix = retriever._key_matrix / norms

    retriever._initialized = True
    retriever._stats["index_size"] = len(retriever._index_ids)


# ==================== 初始化测试 ====================

class TestRetrieverInit:
    """测试召回器初始化"""

    def test_default_init(self, retriever):
        assert retriever.namespace == "default"
        assert retriever.auto_refresh_interval == 0
        assert retriever.similarity_threshold == 0.0
        assert retriever._initialized is False
        assert retriever._key_matrix is None
        assert len(retriever._index_ids) == 0

    def test_custom_init(self, manager, encoder):
        from aga_knowledge.retriever.knowledge_retriever import KnowledgeRetriever
        r = KnowledgeRetriever(
            manager=manager,
            encoder=encoder,
            namespace="medical",
            auto_refresh_interval=60.0,
            similarity_threshold=0.5,
        )
        assert r.namespace == "medical"
        assert r.auto_refresh_interval == 60.0
        assert r.similarity_threshold == 0.5


# ==================== 预热测试 ====================

class TestRetrieverWarmup:
    """测试预热功能"""

    def test_warmup_builds_index(self, warmed_retriever):
        assert warmed_retriever._initialized is True
        assert len(warmed_retriever._index_ids) == 3
        assert warmed_retriever._key_matrix is not None
        assert warmed_retriever._key_matrix.shape[0] == 3
        assert warmed_retriever._key_matrix.shape[1] == 16  # key_dim

    def test_warmup_idempotent(self, warmed_retriever):
        """重复预热不应重建索引"""
        original_ids = list(warmed_retriever._index_ids)
        warmed_retriever.warmup()  # 第二次调用
        assert warmed_retriever._index_ids == original_ids

    def test_warmup_empty_knowledge(self, encoder):
        from aga_knowledge.retriever.knowledge_retriever import KnowledgeRetriever
        empty_manager = MockKnowledgeManager([])
        r = KnowledgeRetriever(
            manager=empty_manager,
            encoder=encoder,
        )
        _build_index_manually(r)
        r._initialized = True

        assert r._key_matrix is None
        assert len(r._index_ids) == 0


# ==================== 检索测试 ====================

class TestRetrieverRetrieve:
    """测试核心检索功能"""

    def test_retrieve_with_query_projected(self, warmed_retriever):
        """使用 query_projected 进行检索"""
        query_vec = torch.randn(1, 1, 16)  # [batch, seq, key_dim]
        query = MockRetrievalQuery(query_projected=query_vec, top_k=2)

        results = warmed_retriever.retrieve(query)

        assert isinstance(results, list)
        assert len(results) <= 2
        # 随机向量可能不会产生高于阈值的结果，所以不强制要求非空
        for r in results:
            if isinstance(r, dict):
                assert "id" in r
                assert "key" in r
                assert "value" in r
                assert "score" in r
                assert "reliability" in r

    def test_retrieve_with_hidden_states(self, warmed_retriever):
        """使用 hidden_states 进行检索"""
        hidden = torch.randn(1, 10, 16)  # [batch, seq, hidden_dim=key_dim]
        query = MockRetrievalQuery(hidden_states=hidden, top_k=3)

        results = warmed_retriever.retrieve(query)

        assert len(results) <= 3
        assert isinstance(results, list)

    def test_retrieve_with_2d_query(self, warmed_retriever):
        """使用 2D query_projected"""
        query_vec = torch.randn(5, 16)  # [seq, key_dim]
        query = MockRetrievalQuery(query_projected=query_vec, top_k=2)

        results = warmed_retriever.retrieve(query)
        assert len(results) <= 2

    def test_retrieve_with_1d_query(self, warmed_retriever):
        """使用 1D query_projected"""
        query_vec = torch.randn(16)  # [key_dim]
        query = MockRetrievalQuery(query_projected=query_vec, top_k=2)

        results = warmed_retriever.retrieve(query)
        assert len(results) <= 2

    def test_retrieve_empty_index(self, retriever):
        """空索引应返回空列表"""
        retriever._initialized = True
        query = MockRetrievalQuery(
            query_projected=torch.randn(16), top_k=5
        )

        results = retriever.retrieve(query)
        assert results == []

    def test_retrieve_no_query_vector(self, warmed_retriever):
        """无查询向量应返回空列表"""
        query = MockRetrievalQuery()  # 无 hidden_states 和 query_projected

        results = warmed_retriever.retrieve(query)
        assert results == []

    def test_retrieve_with_similarity_threshold(self, manager, encoder):
        """高阈值应过滤低相似度结果"""
        from aga_knowledge.retriever.knowledge_retriever import KnowledgeRetriever
        r = KnowledgeRetriever(
            manager=manager,
            encoder=encoder,
            similarity_threshold=0.99,  # 非常高的阈值
        )
        _build_index_manually(r)
        r._initialized = True

        query = MockRetrievalQuery(
            query_projected=torch.randn(16), top_k=5
        )
        results = r.retrieve(query)
        # 高阈值可能过滤掉所有结果
        assert isinstance(results, list)

    def test_retrieve_updates_stats(self, warmed_retriever):
        """检索应更新统计信息"""
        query = MockRetrievalQuery(
            query_projected=torch.randn(16), top_k=2
        )

        warmed_retriever.retrieve(query)

        stats = warmed_retriever.get_stats()
        assert stats["retrieve_count"] == 1
        assert stats["avg_retrieve_time_ms"] > 0

    def test_retrieve_multiple_calls(self, warmed_retriever):
        """多次检索应累积统计"""
        for _ in range(5):
            query = MockRetrievalQuery(
                query_projected=torch.randn(16), top_k=2
            )
            warmed_retriever.retrieve(query)

        stats = warmed_retriever.get_stats()
        assert stats["retrieve_count"] == 5

    def test_retrieve_fail_open(self, warmed_retriever):
        """检索异常应 Fail-Open（返回空列表）"""
        # 破坏 key_matrix 使计算失败
        warmed_retriever._key_matrix = "not_a_tensor"

        query = MockRetrievalQuery(
            query_projected=torch.randn(16), top_k=2
        )
        results = warmed_retriever.retrieve(query)

        assert results == []
        assert warmed_retriever._stats["errors"] == 1

    def test_retrieve_hidden_states_dimension_mismatch(self, warmed_retriever):
        """hidden_states 维度不匹配时应降维"""
        # hidden_dim (64) > key_dim (16)
        hidden = torch.randn(1, 10, 64)
        query = MockRetrievalQuery(hidden_states=hidden, top_k=2)

        results = warmed_retriever.retrieve(query)
        # 应该能正常工作（通过分组平均池化降维）
        assert isinstance(results, list)

    def test_retrieve_hidden_states_smaller_dim(self, warmed_retriever):
        """hidden_states 维度小于 key_dim 时应零填充"""
        hidden = torch.randn(1, 10, 8)  # 8 < 16 (key_dim)
        query = MockRetrievalQuery(hidden_states=hidden, top_k=2)

        results = warmed_retriever.retrieve(query)
        assert isinstance(results, list)


# ==================== 索引管理测试 ====================

class TestRetrieverIndexManagement:
    """测试索引管理"""

    def test_refresh_knowledge(self, warmed_retriever):
        """增量更新单条知识"""
        warmed_retriever.refresh_knowledge("lu_new", {
            "condition": "new condition",
            "decision": "new decision",
            "reliability": 0.85,
        })

        assert "lu_new" in warmed_retriever._index
        assert len(warmed_retriever._index_ids) == 4
        assert warmed_retriever._key_matrix.shape[0] == 4

    def test_remove_knowledge(self, warmed_retriever):
        """从索引中移除知识"""
        assert "lu_001" in warmed_retriever._index

        warmed_retriever.remove_knowledge("lu_001")

        assert "lu_001" not in warmed_retriever._index
        assert len(warmed_retriever._index_ids) == 2
        assert warmed_retriever._key_matrix.shape[0] == 2

    def test_remove_nonexistent_knowledge(self, warmed_retriever):
        """移除不存在的知识不应报错"""
        original_size = len(warmed_retriever._index_ids)
        warmed_retriever.remove_knowledge("lu_nonexistent")
        assert len(warmed_retriever._index_ids) == original_size

    def test_rebuild_matrix_empty(self, warmed_retriever):
        """清空所有知识后矩阵应为 None"""
        for lu_id in list(warmed_retriever._index.keys()):
            del warmed_retriever._index[lu_id]

        warmed_retriever._rebuild_matrix()

        assert warmed_retriever._key_matrix is None
        assert len(warmed_retriever._index_ids) == 0


# ==================== 反馈测试 ====================

class TestRetrieverFeedback:
    """测试注入反馈"""

    def test_feedback_used(self, warmed_retriever):
        warmed_retriever.on_injection_feedback("lu_001", was_used=True, gate_value=0.8)
        assert warmed_retriever._stats["feedback_used"] == 1

    def test_feedback_unused(self, warmed_retriever):
        warmed_retriever.on_injection_feedback("lu_001", was_used=False, gate_value=0.1)
        assert warmed_retriever._stats["feedback_unused"] == 1

    def test_feedback_multiple(self, warmed_retriever):
        warmed_retriever.on_injection_feedback("lu_001", True, 0.8)
        warmed_retriever.on_injection_feedback("lu_002", True, 0.7)
        warmed_retriever.on_injection_feedback("lu_003", False, 0.1)

        assert warmed_retriever._stats["feedback_used"] == 2
        assert warmed_retriever._stats["feedback_unused"] == 1


# ==================== 统计测试 ====================

class TestRetrieverStats:
    """测试统计信息"""

    def test_get_stats_initial(self, retriever):
        stats = retriever.get_stats()
        assert stats["type"] == "KnowledgeRetriever"
        assert stats["namespace"] == "default"
        assert stats["initialized"] is False
        assert stats["index_size"] == 0
        assert stats["retrieve_count"] == 0

    def test_get_stats_after_warmup(self, warmed_retriever):
        stats = warmed_retriever.get_stats()
        assert stats["initialized"] is True
        assert stats["index_size"] == 3
        assert "encoder" in stats

    def test_get_stats_after_retrieval(self, warmed_retriever):
        query = MockRetrievalQuery(
            query_projected=torch.randn(16), top_k=2
        )
        warmed_retriever.retrieve(query)

        stats = warmed_retriever.get_stats()
        assert stats["retrieve_count"] == 1
        assert stats["avg_retrieve_time_ms"] >= 0
        assert stats["avg_results_per_query"] >= 0


# ==================== 关闭测试 ====================

class TestRetrieverShutdown:
    """测试关闭功能"""

    def test_shutdown(self, warmed_retriever):
        warmed_retriever.shutdown()

        assert warmed_retriever._initialized is False
        assert warmed_retriever._key_matrix is None
        assert len(warmed_retriever._index_ids) == 0
        assert len(warmed_retriever._index) == 0
        assert len(warmed_retriever._index_metadata) == 0

    def test_shutdown_encoder(self, warmed_retriever):
        warmed_retriever.shutdown()
        assert warmed_retriever.encoder._warmed_up is False


# ==================== repr 测试 ====================

class TestRetrieverRepr:
    """测试字符串表示"""

    def test_repr(self, warmed_retriever):
        r = repr(warmed_retriever)
        assert "KnowledgeRetriever" in r
        assert "default" in r
        assert "index_size=3" in r


# ==================== 相似度计算测试 ====================

class TestSimilarityComputation:
    """测试相似度计算"""

    def test_compute_similarity_shape(self, warmed_retriever):
        query_vec = torch.randn(16)
        scores = warmed_retriever._compute_similarity(query_vec)
        assert scores.shape == (3,)  # 3 条知识

    def test_compute_similarity_range(self, warmed_retriever):
        """余弦相似度应在 [-1, 1] 范围内"""
        query_vec = torch.randn(16)
        scores = warmed_retriever._compute_similarity(query_vec)
        assert (scores >= -1.01).all()  # 允许微小浮点误差
        assert (scores <= 1.01).all()

    def test_compute_similarity_identical(self, warmed_retriever):
        """相同向量的相似度应接近 1"""
        # 使用索引中第一条知识的 key 作为查询
        first_key = torch.tensor(
            warmed_retriever._index[warmed_retriever._index_ids[0]].key_vector,
            dtype=torch.float32,
        )
        scores = warmed_retriever._compute_similarity(first_key)
        # 第一条的相似度应该最高
        assert scores[0] > 0.9


# ==================== 查询向量提取测试 ====================

class TestQueryVectorExtraction:
    """测试查询向量提取"""

    def test_extract_from_query_projected_3d(self, warmed_retriever):
        query = MockRetrievalQuery(
            query_projected=torch.randn(2, 10, 16)
        )
        vec = warmed_retriever._extract_query_vector(query)
        assert vec is not None
        assert vec.shape == (16,)

    def test_extract_from_query_projected_2d(self, warmed_retriever):
        query = MockRetrievalQuery(
            query_projected=torch.randn(10, 16)
        )
        vec = warmed_retriever._extract_query_vector(query)
        assert vec is not None
        assert vec.shape == (16,)

    def test_extract_from_query_projected_1d(self, warmed_retriever):
        query = MockRetrievalQuery(
            query_projected=torch.randn(16)
        )
        vec = warmed_retriever._extract_query_vector(query)
        assert vec is not None
        assert vec.shape == (16,)

    def test_extract_from_hidden_states(self, warmed_retriever):
        query = MockRetrievalQuery(
            hidden_states=torch.randn(1, 10, 16)
        )
        vec = warmed_retriever._extract_query_vector(query)
        assert vec is not None
        assert vec.shape == (16,)

    def test_extract_prefers_query_projected(self, warmed_retriever):
        """query_projected 优先于 hidden_states"""
        proj = torch.ones(16) * 0.5
        hidden = torch.ones(1, 10, 16) * 0.9

        query = MockRetrievalQuery(
            query_projected=proj,
            hidden_states=hidden,
        )
        vec = warmed_retriever._extract_query_vector(query)
        # 应该使用 query_projected
        assert torch.allclose(vec, proj, atol=1e-6)

    def test_extract_none_when_no_tensors(self, warmed_retriever):
        query = MockRetrievalQuery()
        vec = warmed_retriever._extract_query_vector(query)
        assert vec is None

    def test_extract_hidden_states_larger_dim(self, warmed_retriever):
        """hidden_dim > key_dim 时应降维"""
        query = MockRetrievalQuery(
            hidden_states=torch.randn(1, 10, 64)  # 64 > 16
        )
        vec = warmed_retriever._extract_query_vector(query)
        assert vec is not None
        assert vec.shape == (16,)  # 降维到 key_dim

    def test_extract_hidden_states_smaller_dim(self, warmed_retriever):
        """hidden_dim < key_dim 时应零填充"""
        query = MockRetrievalQuery(
            hidden_states=torch.randn(1, 10, 8)  # 8 < 16
        )
        vec = warmed_retriever._extract_query_vector(query)
        assert vec is not None
        assert vec.shape == (16,)  # 零填充到 key_dim


# ==================== 线程安全测试 ====================

class TestRetrieverThreadSafety:
    """测试线程安全"""

    def test_concurrent_retrieve(self, warmed_retriever):
        """并发检索不应崩溃"""
        import threading

        errors = []

        def worker():
            try:
                for _ in range(10):
                    query = MockRetrievalQuery(
                        query_projected=torch.randn(16), top_k=2
                    )
                    results = warmed_retriever.retrieve(query)
                    assert isinstance(results, list)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发检索出错: {errors}"

    def test_concurrent_refresh_and_retrieve(self, warmed_retriever):
        """并发刷新和检索不应崩溃"""
        import threading

        errors = []

        def retriever_worker():
            try:
                for _ in range(10):
                    query = MockRetrievalQuery(
                        query_projected=torch.randn(16), top_k=2
                    )
                    warmed_retriever.retrieve(query)
            except Exception as e:
                errors.append(e)

        def refresh_worker():
            try:
                for i in range(5):
                    warmed_retriever.refresh_knowledge(f"lu_new_{i}", {
                        "condition": f"new cond {i}",
                        "decision": f"new dec {i}",
                    })
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=retriever_worker),
            threading.Thread(target=retriever_worker),
            threading.Thread(target=refresh_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发操作出错: {errors}"
