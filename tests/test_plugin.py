"""
AGAPlugin 集成测试
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from aga.plugin import AGAPlugin
from aga.config import AGAConfig
from aga.exceptions import AttachError


# ========== 模拟 LLM 模型 ==========

class MockSelfAttention(nn.Module):
    """模拟 self_attn 层"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class MockTransformerLayer(nn.Module):
    """模拟 Transformer 层"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = MockSelfAttention(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, **kwargs):
        attn_output = self.self_attn(hidden_states)
        hidden_states = self.norm(hidden_states + attn_output)
        hidden_states = hidden_states + self.mlp(hidden_states)
        return (hidden_states,)


class MockLLMConfig:
    """模拟 HuggingFace 模型 config"""
    def __init__(self, hidden_size=128):
        self.hidden_size = hidden_size


class MockLLM(nn.Module):
    """模拟 HuggingFace LLM"""
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        self.config = MockLLMConfig(hidden_dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, input_ids=None, hidden_states=None, **kwargs):
        if hidden_states is None:
            hidden_states = torch.randn(1, 10, self.config.hidden_size)
        for layer in self.model.layers:
            output = layer(hidden_states)
            hidden_states = output[0] if isinstance(output, tuple) else output
        return hidden_states


# ========== 测试 ==========

class TestAGAPluginInit:
    """AGAPlugin 初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        plugin = AGAPlugin(AGAConfig(hidden_dim=128, device="cpu"))
        assert plugin.config.hidden_dim == 128
        assert plugin.knowledge_count == 0
        assert not plugin.is_attached

    def test_init_with_kwargs(self):
        """测试 kwargs 初始化"""
        plugin = AGAPlugin(hidden_dim=128, device="cpu")
        assert plugin.config.hidden_dim == 128

    def test_from_dict(self):
        """测试从字典创建"""
        plugin = AGAPlugin.from_config({
            "hidden_dim": 128,
            "bottleneck_dim": 16,
            "device": "cpu",
        })
        assert plugin.config.hidden_dim == 128
        assert plugin.config.bottleneck_dim == 16

    def test_repr(self):
        """测试字符串表示"""
        plugin = AGAPlugin(AGAConfig(hidden_dim=128, device="cpu"))
        s = repr(plugin)
        assert "AGAPlugin" in s
        assert "128" in s


class TestAGAPluginKnowledge:
    """AGAPlugin 知识管理测试"""

    @pytest.fixture
    def plugin(self):
        return AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            max_slots=10,
            device="cpu",
        ))

    def test_register(self, plugin):
        """测试注册知识"""
        key = torch.randn(16)
        value = torch.randn(128)
        assert plugin.register("fact_001", key, value, reliability=0.9)
        assert plugin.knowledge_count == 1

    def test_register_multiple(self, plugin):
        """测试注册多条知识"""
        for i in range(5):
            key = torch.randn(16)
            value = torch.randn(128)
            plugin.register(f"fact_{i:03d}", key, value)
        assert plugin.knowledge_count == 5

    def test_unregister(self, plugin):
        """测试移除知识"""
        key = torch.randn(16)
        value = torch.randn(128)
        plugin.register("fact_001", key, value)
        assert plugin.knowledge_count == 1

        assert plugin.unregister("fact_001")
        assert plugin.knowledge_count == 0

    def test_register_batch(self, plugin):
        """测试批量注册"""
        entries = [
            {"id": f"fact_{i:03d}", "key": torch.randn(16), "value": torch.randn(128)}
            for i in range(5)
        ]
        count = plugin.register_batch(entries)
        assert count == 5
        assert plugin.knowledge_count == 5

    def test_clear(self, plugin):
        """测试清空"""
        for i in range(5):
            key = torch.randn(16)
            value = torch.randn(128)
            plugin.register(f"fact_{i:03d}", key, value)
        assert plugin.knowledge_count == 5

        plugin.clear()
        assert plugin.knowledge_count == 0

    def test_norm_clipping(self, plugin):
        """测试范数裁剪"""
        key = torch.randn(16) * 100  # 大范数
        value = torch.randn(128) * 100
        plugin.register("fact_001", key, value)
        assert plugin.knowledge_count == 1


class TestAGAPluginAttach:
    """AGAPlugin 模型挂载测试"""

    @pytest.fixture
    def plugin(self):
        return AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            max_slots=10,
            device="cpu",
        ))

    @pytest.fixture
    def model(self):
        return MockLLM(hidden_dim=128, num_layers=6)

    def test_attach(self, plugin, model):
        """测试挂载"""
        plugin.attach(model, layer_indices=[-1])
        assert plugin.is_attached

    def test_attach_multiple_layers(self, plugin, model):
        """测试挂载多层"""
        plugin.attach(model, layer_indices=[-1, -2, -3])
        assert plugin.is_attached

    def test_attach_default_layers(self, plugin, model):
        """测试默认层挂载"""
        plugin.attach(model)  # 默认 [-1, -2, -3]
        assert plugin.is_attached

    def test_detach(self, plugin, model):
        """测试卸载"""
        plugin.attach(model, layer_indices=[-1])
        assert plugin.is_attached

        plugin.detach()
        assert not plugin.is_attached

    def test_double_attach_raises(self, plugin, model):
        """测试重复挂载抛出异常"""
        plugin.attach(model, layer_indices=[-1])
        with pytest.raises(AttachError):
            plugin.attach(model, layer_indices=[-1])

    def test_reattach_after_detach(self, plugin, model):
        """测试卸载后重新挂载"""
        plugin.attach(model, layer_indices=[-1])
        plugin.detach()
        plugin.attach(model, layer_indices=[-1])
        assert plugin.is_attached


class TestAGAPluginForward:
    """AGAPlugin Forward 测试"""

    @pytest.fixture
    def plugin(self):
        p = AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            max_slots=10,
            device="cpu",
            early_exit_threshold=0.0,  # 禁用 early exit 以确保 AGA 被应用
        ))
        return p

    @pytest.fixture
    def model(self):
        return MockLLM(hidden_dim=128, num_layers=6)

    def test_forward_without_knowledge(self, plugin, model):
        """测试无知识时 forward（应直接旁路）"""
        plugin.attach(model, layer_indices=[-1])
        hidden_states = torch.randn(1, 10, 128)
        output = model(hidden_states=hidden_states)
        assert output.shape == (1, 10, 128)

    def test_forward_with_knowledge(self, plugin, model):
        """测试有知识时 forward"""
        # 注册知识
        for i in range(5):
            key = torch.randn(16)
            value = torch.randn(128)
            plugin.register(f"fact_{i:03d}", key, value, reliability=0.9)

        plugin.attach(model, layer_indices=[-1])
        hidden_states = torch.randn(1, 10, 128)
        output = model(hidden_states=hidden_states)
        assert output.shape == (1, 10, 128)

    def test_forward_fail_open(self, plugin, model):
        """测试 Fail-Open 机制"""
        plugin.attach(model, layer_indices=[-1])

        # 即使内部出错，也应该正常返回
        hidden_states = torch.randn(1, 10, 128)
        output = model(hidden_states=hidden_states)
        assert output.shape == (1, 10, 128)

    def test_forward_multiple_layers(self, plugin, model):
        """测试多层 forward"""
        for i in range(5):
            key = torch.randn(16)
            value = torch.randn(128)
            plugin.register(f"fact_{i:03d}", key, value, reliability=0.9)

        plugin.attach(model, layer_indices=[-1, -2, -3])
        hidden_states = torch.randn(1, 10, 128)
        output = model(hidden_states=hidden_states)
        assert output.shape == (1, 10, 128)


class TestAGAPluginDiagnostics:
    """AGAPlugin 诊断测试"""

    @pytest.fixture
    def plugin(self):
        return AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            max_slots=10,
            device="cpu",
        ))

    def test_get_diagnostics(self, plugin):
        """测试获取诊断信息"""
        diag = plugin.get_diagnostics()
        assert "attached" in diag
        assert "knowledge_count" in diag
        assert "max_slots" in diag
        assert "forward_total" in diag
        assert "activation_rate" in diag

    def test_get_audit_trail(self, plugin):
        """测试获取审计日志"""
        key = torch.randn(16)
        value = torch.randn(128)
        plugin.register("fact_001", key, value)
        plugin.unregister("fact_001")

        trail = plugin.get_audit_trail()
        assert len(trail) == 2
        assert trail[0]["operation"] == "register"
        assert trail[1]["operation"] == "unregister"

    def test_get_store_stats(self, plugin):
        """测试获取存储统计"""
        for i in range(3):
            key = torch.randn(16)
            value = torch.randn(128)
            plugin.register(f"fact_{i:03d}", key, value)

        stats = plugin.get_store_stats()
        assert stats["count"] == 3
        assert stats["max_slots"] == 10
        assert stats["utilization"] == 0.3

    def test_diagnostics_after_forward(self, plugin):
        """测试 forward 后的诊断"""
        model = MockLLM(hidden_dim=128, num_layers=6)
        for i in range(3):
            key = torch.randn(16)
            value = torch.randn(128)
            plugin.register(f"fact_{i:03d}", key, value)

        plugin.attach(model, layer_indices=[-1])
        hidden_states = torch.randn(1, 10, 128)
        model(hidden_states=hidden_states)

        diag = plugin.get_diagnostics()
        assert diag["forward_total"] >= 1
        assert diag["attached"] is True
        assert diag["knowledge_count"] == 3


class TestAGAPluginDecay:
    """AGAPlugin 衰减上下文测试"""

    @pytest.fixture
    def plugin(self):
        return AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            max_slots=10,
            device="cpu",
            decay_enabled=True,
            decay_strategy="exponential",
            decay_gamma=0.9,
        ))

    def test_reset_decay_contexts(self, plugin):
        """测试重置衰减上下文（线程隔离）"""
        # 通过 _get_decay_contexts 获取当前线程的上下文
        contexts = plugin._get_decay_contexts()
        contexts[0] = "dummy"
        contexts[1] = "dummy"
        plugin.reset_decay_contexts()
        assert len(plugin._get_decay_contexts()) == 0

    def test_thread_isolation(self, plugin):
        """测试衰减上下文的线程隔离"""
        import threading

        results = {}

        def thread_fn(thread_id):
            contexts = plugin._get_decay_contexts()
            contexts[0] = f"thread_{thread_id}"
            results[thread_id] = contexts.get(0)

        t1 = threading.Thread(target=thread_fn, args=(1,))
        t2 = threading.Thread(target=thread_fn, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 每个线程应该有自己独立的上下文
        assert results[1] == "thread_1"
        assert results[2] == "thread_2"

        # 主线程的上下文应该是空的（未被其他线程污染）
        main_contexts = plugin._get_decay_contexts()
        assert 0 not in main_contexts


class TestAGAPluginJSONL:
    """AGAPlugin JSONL 加载测试"""

    @pytest.fixture
    def plugin(self):
        return AGAPlugin(AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            max_slots=10,
            device="cpu",
        ))

    def test_load_nonexistent_jsonl(self, plugin):
        """测试加载不存在的 JSONL"""
        count = plugin.load_from("nonexistent.jsonl")
        assert count == 0

    def test_load_from_jsonl(self, plugin, tmp_path):
        """测试从 JSONL 加载"""
        import json

        jsonl_path = tmp_path / "knowledge.jsonl"
        entries = []
        for i in range(5):
            entries.append({
                "id": f"fact_{i:03d}",
                "key": torch.randn(16).tolist(),
                "value": torch.randn(128).tolist(),
                "reliability": 0.9,
            })

        with open(jsonl_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        count = plugin.load_from(str(jsonl_path))
        assert count == 5
        assert plugin.knowledge_count == 5
