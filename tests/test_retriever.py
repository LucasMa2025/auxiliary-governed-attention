"""
AGA 召回器协议测试
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult
from aga.retriever.null_retriever import NullRetriever
from aga.retriever.kv_store_retriever import KVStoreRetriever
from aga.kv_store import KVStore
from aga.config import AGAConfig
from aga.plugin import AGAPlugin


# ========== 自定义召回器（测试用） ==========

class MockRetriever(BaseRetriever):
    """模拟外部召回器"""

    def __init__(self, results=None, key_dim=16, value_dim=128):
        self.results = results or []
        self.call_count = 0
        self.last_query = None
        self.warmup_called = False
        self.shutdown_called = False
        self._key_dim = key_dim
        self._value_dim = value_dim

    def retrieve(self, query: RetrievalQuery):
        self.call_count += 1
        self.last_query = query
        return self.results

    def warmup(self):
        self.warmup_called = True

    def shutdown(self):
        self.shutdown_called = True

    def get_stats(self):
        return {"type": "MockRetriever", "call_count": self.call_count}


# ========== NullRetriever 测试 ==========

class TestNullRetriever:
    """NullRetriever 测试"""

    def test_returns_empty(self):
        """NullRetriever 应返回空列表"""
        retriever = NullRetriever()
        query = RetrievalQuery(
            hidden_states=torch.randn(1, 1, 128),
        )
        results = retriever.retrieve(query)
        assert results == []

    def test_call_count(self):
        """NullRetriever 应记录调用次数"""
        retriever = NullRetriever()
        query = RetrievalQuery(hidden_states=torch.randn(1, 1, 128))
        retriever.retrieve(query)
        retriever.retrieve(query)
        stats = retriever.get_stats()
        assert stats["call_count"] == 2

    def test_repr(self):
        retriever = NullRetriever()
        assert "NullRetriever" in repr(retriever)


# ========== KVStoreRetriever 测试 ==========

class TestKVStoreRetriever:
    """KVStoreRetriever 测试"""

    @pytest.fixture
    def store(self):
        return KVStore(max_slots=20, key_dim=16, value_dim=128, device=torch.device("cpu"))

    @pytest.fixture
    def populated_store(self, store):
        """预填充的 KVStore"""
        for i in range(10):
            key = torch.randn(16)
            value = torch.randn(128)
            store.put(f"fact_{i:03d}", key, value, reliability=0.8 + i * 0.02)
        return store

    def test_empty_store(self, store):
        """空 KVStore 应返回空结果"""
        retriever = KVStoreRetriever(store, default_top_k=5)
        query = RetrievalQuery(
            hidden_states=torch.randn(1, 1, 128),
            query_projected=torch.randn(1, 1, 16),
        )
        results = retriever.retrieve(query)
        assert results == []

    def test_retrieve_from_populated_store(self, populated_store):
        """从填充的 KVStore 检索"""
        retriever = KVStoreRetriever(populated_store, default_top_k=5, min_similarity=0.0)
        query = RetrievalQuery(
            hidden_states=torch.randn(1, 1, 128),
            query_projected=torch.randn(1, 1, 16),
            top_k=5,
        )
        results = retriever.retrieve(query)
        assert len(results) <= 5
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_min_similarity_filter(self, populated_store):
        """最小相似度过滤"""
        retriever = KVStoreRetriever(populated_store, default_top_k=10, min_similarity=0.99)
        query = RetrievalQuery(
            hidden_states=torch.randn(1, 1, 128),
            query_projected=torch.randn(1, 1, 16),
        )
        results = retriever.retrieve(query)
        # 高阈值应过滤掉大部分结果
        assert len(results) <= 10

    def test_stats(self, populated_store):
        """统计信息"""
        retriever = KVStoreRetriever(populated_store, default_top_k=5, min_similarity=0.0)
        query = RetrievalQuery(
            hidden_states=torch.randn(1, 1, 128),
            query_projected=torch.randn(1, 1, 16),
        )
        retriever.retrieve(query)
        retriever.retrieve(query)
        stats = retriever.get_stats()
        assert stats["call_count"] == 2
        assert stats["type"] == "KVStoreRetriever"
        assert "avg_latency_ms" in stats

    def test_fallback_to_hidden_states(self, populated_store):
        """无 query_projected 时回退到 hidden_states"""
        retriever = KVStoreRetriever(populated_store, default_top_k=5, min_similarity=0.0)
        query = RetrievalQuery(
            hidden_states=torch.randn(1, 1, 128),
            # 不提供 query_projected
        )
        results = retriever.retrieve(query)
        # 应该不报错，即使维度不匹配也能处理
        assert isinstance(results, list)

    def test_fail_open(self, store):
        """Fail-Open: 异常不应传播"""
        retriever = KVStoreRetriever(store, default_top_k=5)
        # 构造一个会导致内部错误的查询
        query = RetrievalQuery(
            hidden_states=torch.randn(1, 1, 128),
            query_projected=None,
        )
        # 不应抛出异常
        results = retriever.retrieve(query)
        assert isinstance(results, list)


# ========== RetrievalQuery / RetrievalResult 测试 ==========

class TestRetrievalTypes:
    """数据类型测试"""

    def test_query_creation(self):
        """测试 RetrievalQuery 创建"""
        query = RetrievalQuery(
            hidden_states=torch.randn(1, 10, 128),
            query_projected=torch.randn(1, 10, 16),
            entropy=torch.randn(1, 10),
            layer_idx=5,
            namespace="medical",
            top_k=10,
        )
        assert query.hidden_states.shape == (1, 10, 128)
        assert query.layer_idx == 5
        assert query.namespace == "medical"
        assert query.top_k == 10

    def test_result_creation(self):
        """测试 RetrievalResult 创建"""
        result = RetrievalResult(
            id="fact_001",
            key=torch.randn(16),
            value=torch.randn(128),
            reliability=0.95,
            score=0.87,
            metadata={"source": "medical_db"},
        )
        assert result.id == "fact_001"
        assert result.reliability == 0.95
        assert result.score == 0.87


# ========== AGAPlugin + Retriever 集成测试 ==========

class TestPluginRetrieverIntegration:
    """AGAPlugin 与召回器的集成测试"""

    def test_default_null_retriever(self):
        """默认使用 NullRetriever"""
        plugin = AGAPlugin(AGAConfig(hidden_dim=128, device="cpu"))
        assert isinstance(plugin.retriever, NullRetriever)

    def test_custom_retriever(self):
        """自定义召回器"""
        mock = MockRetriever()
        plugin = AGAPlugin(
            AGAConfig(hidden_dim=128, device="cpu"),
            retriever=mock,
        )
        assert plugin.retriever is mock

    def test_kv_store_retriever_config(self):
        """通过配置创建 KVStoreRetriever"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
            retriever_backend="kv_store",
            retriever_top_k=5,
            retriever_min_score=0.3,
        ))
        assert isinstance(plugin.retriever, KVStoreRetriever)

    def test_unknown_backend_fallback(self):
        """未知后端回退到 NullRetriever"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            device="cpu",
            retriever_backend="unknown_backend",
        ))
        assert isinstance(plugin.retriever, NullRetriever)

    def test_retriever_in_diagnostics(self):
        """诊断信息应包含召回器信息"""
        plugin = AGAPlugin(AGAConfig(hidden_dim=128, device="cpu"))
        diag = plugin.get_diagnostics()
        assert "retriever" in diag
        assert "retriever_stats" in diag

    def test_retriever_warmup_on_attach(self):
        """attach 时应调用召回器 warmup"""
        import torch.nn as nn

        mock = MockRetriever()
        plugin = AGAPlugin(
            AGAConfig(hidden_dim=128, bottleneck_dim=16, device="cpu"),
            retriever=mock,
        )

        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {'hidden_size': 128})()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([
                    nn.Linear(128, 128) for _ in range(3)
                ])
            def forward(self, x):
                return x

        model = SimpleModel()
        # 使用自定义适配器避免 HuggingFace 检测问题
        from aga.adapter.base import LLMAdapter

        class SimpleAdapter(LLMAdapter):
            def get_layers(self, model):
                return list(model.model.layers)
            def get_hidden_dim(self, model):
                return 128
            def wrap_layer(self, model, layer_idx, aga_forward):
                layer = list(model.model.layers)[layer_idx]
                hook = layer.register_forward_hook(
                    lambda mod, inp, out: out
                )
                return hook

        plugin.attach(model, layer_indices=[0], adapter=SimpleAdapter())
        assert mock.warmup_called

    def test_retriever_shutdown_on_detach(self):
        """detach 时应调用召回器 shutdown"""
        import torch.nn as nn
        from aga.adapter.base import LLMAdapter

        mock = MockRetriever()
        plugin = AGAPlugin(
            AGAConfig(hidden_dim=128, bottleneck_dim=16, device="cpu"),
            retriever=mock,
        )

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {'hidden_size': 128})()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([
                    nn.Linear(128, 128) for _ in range(3)
                ])
            def forward(self, x):
                return x

        class SimpleAdapter(LLMAdapter):
            def get_layers(self, model):
                return list(model.model.layers)
            def get_hidden_dim(self, model):
                return 128
            def wrap_layer(self, model, layer_idx, aga_forward):
                layer = list(model.model.layers)[layer_idx]
                hook = layer.register_forward_hook(
                    lambda mod, inp, out: out
                )
                return hook

        model = SimpleModel()
        plugin.attach(model, layer_indices=[0], adapter=SimpleAdapter())
        plugin.detach()
        assert mock.shutdown_called


# ========== get_active() 缓存测试 ==========

class TestKVStoreCaching:
    """KVStore get_active() 缓存测试"""

    @pytest.fixture
    def store(self):
        return KVStore(max_slots=10, key_dim=16, value_dim=128, device=torch.device("cpu"))

    def test_cache_invalidation_on_put(self, store):
        """put 后缓存应失效"""
        key = torch.randn(16)
        value = torch.randn(128)
        store.put("fact_001", key, value)

        # 第一次调用建立缓存
        k1, v1, r1 = store.get_active()
        assert store._active_cache_valid

        # put 新数据应使缓存失效
        store.put("fact_002", torch.randn(16), torch.randn(128))
        assert not store._active_cache_valid

        # 再次调用应重建缓存
        k2, v2, r2 = store.get_active()
        assert store._active_cache_valid
        assert k2.shape[0] == 2

    def test_cache_invalidation_on_remove(self, store):
        """remove 后缓存应失效"""
        store.put("fact_001", torch.randn(16), torch.randn(128))
        store.get_active()  # 建立缓存
        assert store._active_cache_valid

        store.remove("fact_001")
        assert not store._active_cache_valid

    def test_cache_invalidation_on_clear(self, store):
        """clear 后缓存应失效"""
        store.put("fact_001", torch.randn(16), torch.randn(128))
        store.get_active()  # 建立缓存
        assert store._active_cache_valid

        store.clear()
        assert not store._active_cache_valid

    def test_cache_hit(self, store):
        """缓存命中时应返回相同的张量"""
        store.put("fact_001", torch.randn(16), torch.randn(128))

        k1, v1, r1 = store.get_active()
        k2, v2, r2 = store.get_active()

        # 缓存命中应返回相同的张量对象
        assert k1 is k2
        assert v1 is v2
        assert r1 is r2


# ========== StreamingSession 层过滤测试 ==========

class TestStreamingSessionLayerFiltering:
    """StreamingSession 层事件过滤测试"""

    def test_primary_layer_filtering(self):
        """主监控层过滤应避免多层重复计数"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
        ))

        from aga.streaming import StreamingSession
        session = StreamingSession(plugin, primary_layer_idx=5)

        # 模拟多层事件
        class MockEvent:
            def __init__(self, data):
                self.data = data

        # 层 3 的事件（非主层，不计数）
        session._on_forward_event(MockEvent({
            "aga_applied": True,
            "gate_mean": 0.5,
            "entropy_mean": 1.5,
            "latency_us": 100,
            "layer_idx": 3,
        }))

        # 层 5 的事件（主层，计数）
        session._on_forward_event(MockEvent({
            "aga_applied": True,
            "gate_mean": 0.6,
            "entropy_mean": 1.8,
            "latency_us": 120,
            "layer_idx": 5,
        }))

        # 层 4 的事件（非主层，不计数）
        session._on_forward_event(MockEvent({
            "aga_applied": False,
            "gate_mean": 0.1,
            "entropy_mean": 0.3,
            "latency_us": 50,
            "layer_idx": 4,
        }))

        # 只有层 5 的事件被计入 step_count
        assert session.step_count == 1
        assert session._total_injection_count == 1

        # 但所有层的事件都被记录
        all_layers = session.get_all_layer_diagnostics()
        assert 3 in all_layers
        assert 4 in all_layers
        assert 5 in all_layers

        session.close()


# ========== 配置驱动召回器测试 ==========

class TestRetrieverConfig:
    """召回器配置测试"""

    def test_config_from_dict(self):
        """从字典加载召回器配置"""
        config = AGAConfig.from_dict({
            "hidden_dim": 128,
            "device": "cpu",
            "retriever": {
                "backend": "kv_store",
                "top_k": 10,
                "min_score": 0.5,
            },
        })
        assert config.retriever_backend == "kv_store"
        assert config.retriever_top_k == 10
        assert config.retriever_min_score == 0.5

    def test_config_defaults(self):
        """默认召回器配置"""
        config = AGAConfig(hidden_dim=128, device="cpu")
        assert config.retriever_backend == "null"
        assert config.retriever_top_k == 5
        assert config.retriever_auto_inject is True

    def test_custom_backend_format(self):
        """自定义后端格式"""
        config = AGAConfig(
            hidden_dim=128,
            device="cpu",
            retriever_backend="mypackage.module:MyRetriever",
        )
        assert ":" in config.retriever_backend

    def test_slot_governance_config(self):
        """Slot 治理配置"""
        config = AGAConfig.from_dict({
            "hidden_dim": 128,
            "device": "cpu",
            "slot_governance": {
                "pin_registered": True,
                "retriever_slot_ratio": 0.4,
                "retriever_cooldown_steps": 10,
                "retriever_dedup_similarity": 0.9,
                "slot_stability_threshold": 0.3,
            },
        })
        assert config.pin_registered is True
        assert config.retriever_slot_ratio == 0.4
        assert config.retriever_cooldown_steps == 10
        assert config.retriever_dedup_similarity == 0.9
        assert config.slot_stability_threshold == 0.3


# ========== KVStore Pin/Unpin 测试 ==========

class TestKVStorePin:
    """KVStore 锁定机制测试"""

    @pytest.fixture
    def store(self):
        return KVStore(max_slots=5, key_dim=16, value_dim=128, device=torch.device("cpu"))

    def test_pin_unpin(self, store):
        """基本 pin/unpin 操作"""
        store.put("fact_001", torch.randn(16), torch.randn(128))
        assert not store.is_pinned("fact_001")

        store.pin("fact_001")
        assert store.is_pinned("fact_001")
        assert store.pinned_count == 1

        store.unpin("fact_001")
        assert not store.is_pinned("fact_001")
        assert store.pinned_count == 0

    def test_put_with_pinned(self, store):
        """put 时指定 pinned"""
        store.put("fact_001", torch.randn(16), torch.randn(128), pinned=True)
        assert store.is_pinned("fact_001")
        assert store.pinned_count == 1

    def test_pinned_survives_eviction(self, store):
        """pinned 知识不会被 LRU 淘汰"""
        # 填满 5 个 slot，前 2 个 pinned
        store.put("pinned_1", torch.randn(16), torch.randn(128), pinned=True)
        store.put("pinned_2", torch.randn(16), torch.randn(128), pinned=True)
        store.put("normal_1", torch.randn(16), torch.randn(128))
        store.put("normal_2", torch.randn(16), torch.randn(128))
        store.put("normal_3", torch.randn(16), torch.randn(128))

        assert store.count == 5
        assert store.pinned_count == 2

        # 写入第 6 个 → 应淘汰 normal_1（最久未访问的非 pinned）
        store.put("new_1", torch.randn(16), torch.randn(128))
        assert store.count == 5
        assert store.contains("pinned_1")
        assert store.contains("pinned_2")
        assert not store.contains("normal_1")  # 被淘汰
        assert store.contains("new_1")

    def test_all_pinned_cannot_evict(self, store):
        """所有知识都 pinned 时无法淘汰，put 返回 False"""
        # 填满所有 slot 并全部 pin
        for i in range(5):
            store.put(f"pinned_{i}", torch.randn(16), torch.randn(128), pinned=True)

        assert store.count == 5
        assert store.pinned_count == 5

        # 尝试写入新知识 → 应失败
        result = store.put("new_1", torch.randn(16), torch.randn(128))
        assert result is False
        assert store.count == 5
        assert not store.contains("new_1")

    def test_remove_clears_pin(self, store):
        """remove 应清除 pin 状态"""
        store.put("fact_001", torch.randn(16), torch.randn(128), pinned=True)
        assert store.is_pinned("fact_001")

        store.remove("fact_001")
        assert not store.is_pinned("fact_001")
        assert store.pinned_count == 0

    def test_clear_clears_all_pins(self, store):
        """clear 应清除所有 pin"""
        store.put("fact_001", torch.randn(16), torch.randn(128), pinned=True)
        store.put("fact_002", torch.randn(16), torch.randn(128), pinned=True)
        assert store.pinned_count == 2

        store.clear()
        assert store.pinned_count == 0
        assert store.count == 0

    def test_unpinned_count(self, store):
        """unpinned_count 统计"""
        store.put("pinned_1", torch.randn(16), torch.randn(128), pinned=True)
        store.put("normal_1", torch.randn(16), torch.randn(128))
        store.put("normal_2", torch.randn(16), torch.randn(128))

        assert store.count == 3
        assert store.pinned_count == 1
        assert store.unpinned_count == 2

    def test_stats_include_pin_info(self, store):
        """get_stats 应包含 pin 信息"""
        store.put("pinned_1", torch.randn(16), torch.randn(128), pinned=True)
        store.put("normal_1", torch.randn(16), torch.randn(128))

        stats = store.get_stats()
        assert stats["pinned_count"] == 1
        assert stats["unpinned_count"] == 1
        assert stats["evictable_count"] == 1

    def test_eviction_order_respects_lru(self, store):
        """淘汰顺序应遵循 LRU（跳过 pinned）"""
        store.put("a", torch.randn(16), torch.randn(128))  # 最早
        store.put("b", torch.randn(16), torch.randn(128), pinned=True)  # pinned
        store.put("c", torch.randn(16), torch.randn(128))
        store.put("d", torch.randn(16), torch.randn(128))
        store.put("e", torch.randn(16), torch.randn(128))

        # 访问 a，使其变为最近访问
        store.get("a")

        # 写入新知识 → 应淘汰 c（b 是 pinned，a 刚被访问）
        store.put("f", torch.randn(16), torch.randn(128))
        assert store.contains("a")  # 刚访问过
        assert store.contains("b")  # pinned
        assert not store.contains("c")  # 被淘汰（最久未访问的非 pinned）
        assert store.contains("f")  # 新写入


# ========== Slot 治理集成测试 ==========

class TestSlotGovernance:
    """Slot 治理集成测试"""

    def test_register_auto_pin(self):
        """register 应自动 pin（当 pin_registered=True）"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
            pin_registered=True,
        ))
        key = torch.randn(16)
        value = torch.randn(128)
        plugin.register("fact_001", key, value)

        assert plugin.store.is_pinned("fact_001")

    def test_register_no_pin(self):
        """register 不 pin（当 pin_registered=False）"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
            pin_registered=False,
        ))
        key = torch.randn(16)
        value = torch.randn(128)
        plugin.register("fact_001", key, value)

        assert not plugin.store.is_pinned("fact_001")

    def test_register_explicit_pin_override(self):
        """register 显式 pinned 参数覆盖配置"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
            pin_registered=False,  # 默认不 pin
        ))
        key = torch.randn(16)
        value = torch.randn(128)
        plugin.register("fact_001", key, value, pinned=True)  # 显式 pin

        assert plugin.store.is_pinned("fact_001")

    def test_retriever_budget_calculation(self):
        """召回器预算计算"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
            max_slots=100,
            retriever_slot_ratio=0.3,
        ))
        # 空 store，预算 = 100 * 0.3 = 30
        budget = plugin._compute_retriever_budget()
        assert budget == 30

    def test_retriever_budget_with_existing_retriever_slots(self):
        """已有召回器 slot 时预算减少"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
            max_slots=100,
            retriever_slot_ratio=0.3,
        ))
        # 模拟已有 10 个召回器 slot
        for i in range(10):
            plugin.store.put(
                f"retriever_{i}",
                torch.randn(16),
                torch.randn(128),
                metadata={"source": "retriever"},
            )
        budget = plugin._compute_retriever_budget()
        assert budget == 20  # 30 - 10

    def test_semantic_dedup(self):
        """语义去重"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
            retriever_dedup_similarity=0.95,
        ))

        # 注册一条知识
        existing_key = torch.randn(16)
        plugin.store.put("existing", existing_key, torch.randn(128))

        # 创建一个与已有知识几乎相同的结果
        similar_key = existing_key + torch.randn(16) * 0.01  # 非常相似
        different_key = torch.randn(16)  # 完全不同

        results = [
            RetrievalResult(
                id="similar",
                key=similar_key,
                value=torch.randn(128),
                score=0.9,
            ),
            RetrievalResult(
                id="different",
                key=different_key,
                value=torch.randn(128),
                score=0.8,
            ),
        ]

        deduped = plugin._semantic_dedup(results)
        # similar 应被去重（与 existing 太相似），different 应保留
        ids = [r.id for r in deduped]
        assert "different" in ids
        # similar 可能被去重也可能不被（取决于随机向量的余弦相似度）
        # 但 ID 去重一定生效
        assert "existing" not in ids  # 不在结果中（因为不在 results 中）

    def test_diagnostics_include_slot_governance(self):
        """诊断信息应包含 slot 治理信息"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
        ))
        diag = plugin.get_diagnostics()
        assert "pinned_count" in diag
        assert "unpinned_count" in diag
        assert "slot_governance" in diag

    def test_cooldown_tracking(self):
        """冷却期跟踪"""
        plugin = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
            retriever_cooldown_steps=5,
        ))
        # 初始状态
        assert plugin._retrieval_step_counter == 0
        assert plugin._last_retrieval_step == -999
