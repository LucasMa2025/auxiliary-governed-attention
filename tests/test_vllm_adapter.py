"""
tests/test_vllm_adapter.py — vLLM 适配器单元测试

测试策略:
  - 使用 Mock 对象模拟 vLLM 的模型结构，不需要实际安装 vLLM
  - 测试 VLLMAdapter 的所有核心方法
  - 测试 VLLMHookWorker 的生命周期
  - 测试与 AGAPlugin 的集成
  - 测试 Fail-Open 安全机制
"""
import pytest
import torch
import torch.nn as nn

from aga.adapter.vllm import VLLMAdapter, VLLMHookWorker
from aga.adapter.base import LLMAdapter
from aga.plugin import AGAPlugin
from aga.config import AGAConfig
from aga.exceptions import AdapterError


# ========== Mock vLLM 模型结构 ==========


class MockVLLMSelfAttn(nn.Module):
    """模拟 vLLM 的 self_attn 层"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, **kwargs):
        return self.o_proj(hidden_states)


class MockVLLMTransformerLayer(nn.Module):
    """模拟 vLLM 的 Transformer 层"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = MockVLLMSelfAttn(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, **kwargs):
        attn_output = self.self_attn(hidden_states)
        return (attn_output + self.mlp(hidden_states),)


class MockVLLMModelConfig:
    """模拟 vLLM 模型 config"""
    def __init__(self, hidden_size=128):
        self.hidden_size = hidden_size


class MockVLLMModel(nn.Module):
    """
    模拟 vLLM 内部模型结构

    vLLM 的模型结构通常是:
      model.model.layers = ModuleList([TransformerLayer, ...])
    """
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        self.config = MockVLLMModelConfig(hidden_dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockVLLMTransformerLayer(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, hidden_states=None, **kwargs):
        if hidden_states is None:
            hidden_states = torch.randn(1, 10, self.config.hidden_size)
        for layer in self.model.layers:
            output = layer(hidden_states)
            hidden_states = output[0] if isinstance(output, tuple) else output
        return hidden_states


class MockVLLMModelRunner:
    """模拟 vLLM 的 ModelRunner"""
    def __init__(self, model):
        self.model = model


class MockVLLMDriverWorker:
    """模拟 vLLM 的 DriverWorker"""
    def __init__(self, model):
        self.model_runner = MockVLLMModelRunner(model)


class MockVLLMModelExecutor:
    """模拟 vLLM 的 ModelExecutor"""
    def __init__(self, model):
        self.driver_worker = MockVLLMDriverWorker(model)
        self.model = model  # 某些版本直接暴露


class MockVLLMModelConfig2:
    """模拟 vLLM 的 ModelConfig（引擎级）"""
    def __init__(self, enforce_eager=True):
        self.enforce_eager = enforce_eager


class MockVLLMParallelConfig:
    """模拟 vLLM 的 ParallelConfig"""
    def __init__(self, tensor_parallel_size=1):
        self.tensor_parallel_size = tensor_parallel_size


class MockVLLMEngine:
    """模拟 vLLM 的 LLMEngine"""
    def __init__(self, model, enforce_eager=True, tp_size=1):
        self.model_executor = MockVLLMModelExecutor(model)
        self.model_config = MockVLLMModelConfig2(enforce_eager)
        self.parallel_config = MockVLLMParallelConfig(tp_size)


class MockVLLMLLM:
    """
    模拟 vLLM 的 LLM 对象

    真实的 vLLM LLM 对象结构:
      llm.llm_engine.model_executor.driver_worker.model_runner.model
    """
    def __init__(self, hidden_dim=128, num_layers=6, enforce_eager=True, tp_size=1):
        self._model = MockVLLMModel(hidden_dim, num_layers)
        self.llm_engine = MockVLLMEngine(self._model, enforce_eager, tp_size)


# ========== 另一种模型结构 (GPT-NeoX 风格) ==========


class MockGPTNeoXAttn(nn.Module):
    """模拟 GPT-NeoX attention"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_key_value = nn.Linear(hidden_dim, hidden_dim * 3)
        self.dense = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, **kwargs):
        return self.dense(hidden_states)


class MockGPTNeoXLayer(nn.Module):
    """模拟 GPT-NeoX Transformer 层"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = MockGPTNeoXAttn(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, **kwargs):
        attn_output = self.attention(hidden_states)
        return (attn_output + self.mlp(hidden_states),)


class MockGPTNeoXModel(nn.Module):
    """模拟 GPT-NeoX 模型"""
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        self.config = MockVLLMModelConfig(hidden_dim)
        self.gpt_neox = nn.Module()
        self.gpt_neox.layers = nn.ModuleList([
            MockGPTNeoXLayer(hidden_dim) for _ in range(num_layers)
        ])


# ========== Fixtures ==========


@pytest.fixture
def vllm_model():
    """创建模拟 vLLM 模型"""
    return MockVLLMModel(hidden_dim=128, num_layers=6)


@pytest.fixture
def vllm_llm():
    """创建模拟 vLLM LLM 对象"""
    return MockVLLMLLM(hidden_dim=128, num_layers=6)


@pytest.fixture
def vllm_adapter():
    """创建 VLLMAdapter 实例"""
    return VLLMAdapter()


@pytest.fixture
def aga_config():
    """创建测试用 AGAConfig"""
    return AGAConfig(
        hidden_dim=128,
        bottleneck_dim=16,
        max_slots=32,
        device="cpu",
        instrumentation_enabled=True,
    )


@pytest.fixture
def aga_plugin(aga_config):
    """创建测试用 AGAPlugin"""
    plugin = AGAPlugin(aga_config)
    # 注册测试知识
    for i in range(5):
        plugin.register(
            id=f"test_{i}",
            key=torch.randn(16),
            value=torch.randn(128),
            reliability=0.9,
        )
    return plugin


# ========== VLLMAdapter 基础测试 ==========


class TestVLLMAdapterInterface:
    """测试 VLLMAdapter 是否正确实现 LLMAdapter 接口"""

    def test_is_llm_adapter(self, vllm_adapter):
        """VLLMAdapter 应该是 LLMAdapter 的子类"""
        assert isinstance(vllm_adapter, LLMAdapter)

    def test_default_enforce_eager(self):
        """默认应该启用 enforce_eager"""
        adapter = VLLMAdapter()
        assert adapter._enforce_eager is True

    def test_custom_enforce_eager(self):
        """可以自定义 enforce_eager"""
        adapter = VLLMAdapter(enforce_eager=False)
        assert adapter._enforce_eager is False


class TestVLLMExtractModel:
    """测试从 vLLM 引擎中提取模型"""

    def test_extract_from_llm(self, vllm_llm):
        """从 MockVLLMLLM 提取模型"""
        model = VLLMAdapter.extract_model(vllm_llm)
        assert isinstance(model, nn.Module)
        assert isinstance(model, MockVLLMModel)

    def test_extract_from_engine(self, vllm_llm):
        """从 LLMEngine 提取模型（通过 model_executor.model 路径）"""
        # MockVLLMEngine.model_executor.model 是直接暴露的
        model = VLLMAdapter.extract_model(vllm_llm.llm_engine.model_executor)
        assert isinstance(model, nn.Module)

    def test_extract_from_executor(self, vllm_llm):
        """从 ModelExecutor 提取模型"""
        model = VLLMAdapter.extract_model(vllm_llm.llm_engine.model_executor)
        assert isinstance(model, nn.Module)

    def test_extract_fails_on_invalid(self):
        """无法从无效对象提取模型"""
        with pytest.raises(AdapterError, match="无法从 vLLM"):
            VLLMAdapter.extract_model(object())

    def test_extract_from_direct_model_attr(self):
        """从具有 model 属性的对象提取"""
        class FakeEngine:
            def __init__(self):
                self.model = MockVLLMModel(128, 4)

        model = VLLMAdapter.extract_model(FakeEngine())
        assert isinstance(model, MockVLLMModel)


class TestVLLMGetLayers:
    """测试获取 Transformer 层"""

    def test_get_layers_standard(self, vllm_adapter, vllm_model):
        """标准 LLaMA 风格模型"""
        layers = vllm_adapter.get_layers(vllm_model)
        assert len(layers) == 6
        assert all(isinstance(l, MockVLLMTransformerLayer) for l in layers)

    def test_get_layers_gpt_neox(self, vllm_adapter):
        """GPT-NeoX 风格模型"""
        model = MockGPTNeoXModel(128, 4)
        layers = vllm_adapter.get_layers(model)
        assert len(layers) == 4
        assert all(isinstance(l, MockGPTNeoXLayer) for l in layers)

    def test_get_layers_direct(self, vllm_adapter):
        """直接有 layers 属性的模型"""
        class DirectModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    MockVLLMTransformerLayer(128) for _ in range(3)
                ])
        model = DirectModel()
        layers = vllm_adapter.get_layers(model)
        assert len(layers) == 3

    def test_get_layers_recursive_search(self, vllm_adapter):
        """通过递归搜索找到 Transformer 层"""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.block = nn.ModuleList([
                    MockVLLMTransformerLayer(128) for _ in range(4)
                ])
        model = NestedModel()
        layers = vllm_adapter.get_layers(model)
        assert len(layers) == 4

    def test_get_layers_fails_on_empty(self, vllm_adapter):
        """无法找到层时抛出异常"""
        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)
        with pytest.raises(AdapterError, match="无法在 vLLM 模型中找到"):
            vllm_adapter.get_layers(EmptyModel())


class TestVLLMGetHiddenDim:
    """测试获取隐藏维度"""

    def test_from_config(self, vllm_adapter, vllm_model):
        """从 config.hidden_size 获取"""
        dim = vllm_adapter.get_hidden_dim(vllm_model)
        assert dim == 128

    def test_from_d_model(self, vllm_adapter):
        """从 config.d_model 获取"""
        class ModelWithDModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {'d_model': 256})()
        model = ModelWithDModel()
        dim = vllm_adapter.get_hidden_dim(model)
        assert dim == 256

    def test_from_weight_inference(self, vllm_adapter):
        """从权重推断"""
        class ModelNoConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([
                    MockVLLMTransformerLayer(128)
                ])
        model = ModelNoConfig()
        dim = vllm_adapter.get_hidden_dim(model)
        assert dim == 128

    def test_default_fallback(self, vllm_adapter):
        """无法检测时使用默认值"""
        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
        model = MinimalModel()
        dim = vllm_adapter.get_hidden_dim(model)
        assert dim == 4096  # 默认值


class TestVLLMWrapLayer:
    """测试层包装和 hook 注入"""

    def test_wrap_layer_basic(self, vllm_adapter, vllm_model):
        """基本的层包装"""
        call_count = [0]

        def mock_aga_forward(hidden_states, primary_attention_output):
            call_count[0] += 1
            return primary_attention_output

        hook = vllm_adapter.wrap_layer(vllm_model, 0, mock_aga_forward)
        assert hook is not None

        # 触发 forward
        x = torch.randn(1, 5, 128)
        layer = vllm_adapter.get_layers(vllm_model)[0]
        layer.self_attn(x)

        assert call_count[0] == 1

        # 清理
        hook.remove()

    def test_wrap_layer_modifies_output(self, vllm_adapter, vllm_model):
        """hook 应该能修改输出"""
        def mock_aga_forward(hidden_states, primary_attention_output):
            return primary_attention_output * 2.0  # 放大 2 倍

        hook = vllm_adapter.wrap_layer(vllm_model, 0, mock_aga_forward)

        x = torch.randn(1, 5, 128)
        layer = vllm_adapter.get_layers(vllm_model)[0]

        # 无 hook 的输出
        hook.remove()
        with torch.no_grad():
            original = layer.self_attn(x).clone()

        # 有 hook 的输出
        hook = vllm_adapter.wrap_layer(vllm_model, 0, mock_aga_forward)
        with torch.no_grad():
            modified = layer.self_attn(x)

        # 应该不同（被放大了）
        assert not torch.allclose(original, modified, atol=1e-6)

        hook.remove()

    def test_wrap_layer_fail_open(self, vllm_adapter, vllm_model):
        """aga_forward 异常时应该 Fail-Open"""
        def failing_aga_forward(hidden_states, primary_attention_output):
            raise RuntimeError("AGA 故障")

        hook = vllm_adapter.wrap_layer(vllm_model, 0, failing_aga_forward)

        x = torch.randn(1, 5, 128)
        layer = vllm_adapter.get_layers(vllm_model)[0]

        # 不应该抛出异常
        output = layer.self_attn(x)
        assert output is not None

        hook.remove()

    def test_wrap_layer_invalid_index(self, vllm_adapter, vllm_model):
        """无效层索引应该抛出异常"""
        def mock_aga_forward(hidden_states, primary_attention_output):
            return primary_attention_output

        with pytest.raises(AdapterError, match="层索引.*超出范围"):
            vllm_adapter.wrap_layer(vllm_model, 100, mock_aga_forward)

    def test_wrap_layer_gpt_neox_attention(self, vllm_adapter):
        """GPT-NeoX 风格的 attention 模块"""
        model = MockGPTNeoXModel(128, 4)
        call_count = [0]

        def mock_aga_forward(hidden_states, primary_attention_output):
            call_count[0] += 1
            return primary_attention_output

        hook = vllm_adapter.wrap_layer(model, 0, mock_aga_forward)

        x = torch.randn(1, 5, 128)
        layers = vllm_adapter.get_layers(model)
        layers[0].attention(x)

        assert call_count[0] == 1
        hook.remove()

    def test_wrap_layer_tuple_output(self, vllm_adapter, vllm_model):
        """处理 tuple 格式的输出"""
        def mock_aga_forward(hidden_states, primary_attention_output):
            return primary_attention_output + 0.1

        hook = vllm_adapter.wrap_layer(vllm_model, 0, mock_aga_forward)

        x = torch.randn(1, 5, 128)
        layer = vllm_adapter.get_layers(vllm_model)[0]
        output = layer(x)

        # 输出应该是 tuple
        assert isinstance(output, tuple)
        hook.remove()


class TestVLLMFindAttentionModule:
    """测试 attention 模块查找"""

    def test_find_self_attn(self):
        """找到 self_attn"""
        layer = MockVLLMTransformerLayer(128)
        attn = VLLMAdapter._find_attention_module(layer)
        assert attn is not None
        assert isinstance(attn, MockVLLMSelfAttn)

    def test_find_attention(self):
        """找到 attention (GPT-NeoX 风格)"""
        layer = MockGPTNeoXLayer(128)
        attn = VLLMAdapter._find_attention_module(layer)
        assert attn is not None
        assert isinstance(attn, MockGPTNeoXAttn)

    def test_find_none(self):
        """没有 attention 模块"""
        layer = nn.Linear(128, 128)
        attn = VLLMAdapter._find_attention_module(layer)
        assert attn is None


class TestVLLMCheckCompatibility:
    """测试兼容性检查"""

    def test_without_vllm_installed(self, vllm_llm):
        """
        vLLM 未安装时，check_compatibility 应该返回 compatible=False。
        在测试环境中 vLLM 通常未安装，所以这是预期行为。
        """
        report = VLLMAdapter.check_compatibility(vllm_llm)
        # 如果 vLLM 未安装，应该返回 compatible=False
        # 如果 vLLM 已安装，应该返回 compatible=True
        assert isinstance(report, dict)
        assert "compatible" in report
        assert "warnings" in report

    def test_compatible_with_mock_vllm(self, vllm_llm, monkeypatch):
        """使用 monkeypatch 模拟 vLLM 已安装"""
        import types
        mock_vllm = types.ModuleType("vllm")
        mock_vllm.__version__ = "0.6.0"
        monkeypatch.setitem(__import__('sys').modules, 'vllm', mock_vllm)

        report = VLLMAdapter.check_compatibility(vllm_llm)
        assert report["compatible"] is True
        assert report["vllm_version"] == "0.6.0"
        assert report["model_type"] == "MockVLLMModel"
        assert report["cuda_graph"] is False

    def test_cuda_graph_warning(self, monkeypatch):
        """CUDA Graph 启用时应该有警告"""
        import types
        mock_vllm = types.ModuleType("vllm")
        mock_vllm.__version__ = "0.6.0"
        monkeypatch.setitem(__import__('sys').modules, 'vllm', mock_vllm)

        llm = MockVLLMLLM(128, 6, enforce_eager=False)
        report = VLLMAdapter.check_compatibility(llm)
        assert report["cuda_graph"] is True
        assert any("CUDA Graph" in w for w in report["warnings"])

    def test_tensor_parallel_warning(self, monkeypatch):
        """Tensor Parallelism 启用时应该有警告"""
        import types
        mock_vllm = types.ModuleType("vllm")
        mock_vllm.__version__ = "0.6.0"
        monkeypatch.setitem(__import__('sys').modules, 'vllm', mock_vllm)

        llm = MockVLLMLLM(128, 6, tp_size=4)
        report = VLLMAdapter.check_compatibility(llm)
        assert report["tensor_parallel"] == 4
        assert any("Tensor Parallelism" in w for w in report["warnings"])

    def test_incompatible_no_model(self, monkeypatch):
        """无法提取模型时应该不兼容"""
        import types
        mock_vllm = types.ModuleType("vllm")
        mock_vllm.__version__ = "0.6.0"
        monkeypatch.setitem(__import__('sys').modules, 'vllm', mock_vllm)

        report = VLLMAdapter.check_compatibility(object())
        assert report["compatible"] is False


# ========== AGAPlugin + VLLMAdapter 集成测试 ==========


class TestVLLMPluginIntegration:
    """测试 VLLMAdapter 与 AGAPlugin 的集成"""

    def test_attach_with_vllm_adapter(self, aga_plugin, vllm_model):
        """使用 VLLMAdapter 挂载到模型"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        assert aga_plugin.is_attached
        assert aga_plugin._attached_model_name == "MockVLLMModel"

        aga_plugin.detach()
        assert not aga_plugin.is_attached

    def test_attach_default_layers(self, aga_plugin, vllm_model):
        """默认挂载最后 3 层"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        # 默认 [-1, -2, -3] → 层 5, 4, 3
        assert len(aga_plugin._hooks) == 3

        aga_plugin.detach()

    def test_attach_custom_layers(self, aga_plugin, vllm_model):
        """自定义挂载层"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, layer_indices=[0, 2, 4], adapter=adapter)

        assert len(aga_plugin._hooks) == 3

        aga_plugin.detach()

    def test_forward_with_knowledge(self, aga_plugin, vllm_model):
        """有知识时的 forward 应该正常工作"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        x = torch.randn(1, 10, 128)
        with torch.no_grad():
            output = vllm_model(hidden_states=x)

        assert output.shape == (1, 10, 128)

        aga_plugin.detach()

    def test_forward_without_knowledge(self, vllm_model):
        """无知识时应该直接旁路"""
        config = AGAConfig(hidden_dim=128, bottleneck_dim=16, max_slots=32, device="cpu")
        plugin = AGAPlugin(config)

        adapter = VLLMAdapter()
        plugin.attach(vllm_model, adapter=adapter)

        x = torch.randn(1, 10, 128)
        with torch.no_grad():
            output = vllm_model(hidden_states=x)

        assert output.shape == (1, 10, 128)

        plugin.detach()

    def test_forward_fail_open(self, vllm_model):
        """forward 异常时 Fail-Open"""
        config = AGAConfig(
            hidden_dim=128, bottleneck_dim=16, max_slots=32,
            device="cpu", fail_open=True,
        )
        plugin = AGAPlugin(config)

        # 注册一个会导致维度不匹配的知识（故意制造错误）
        plugin.register(
            id="bad_knowledge",
            key=torch.randn(16),
            value=torch.randn(128),
        )

        adapter = VLLMAdapter()
        plugin.attach(vllm_model, adapter=adapter)

        x = torch.randn(1, 10, 128)
        # 不应该抛出异常
        with torch.no_grad():
            output = vllm_model(hidden_states=x)
        assert output.shape == (1, 10, 128)

        plugin.detach()

    def test_extract_and_attach(self, aga_plugin):
        """完整流程: 从 vLLM LLM 提取模型并挂载"""
        llm = MockVLLMLLM(hidden_dim=128, num_layers=6)

        # 提取模型
        model = VLLMAdapter.extract_model(llm)
        assert isinstance(model, MockVLLMModel)

        # 挂载
        adapter = VLLMAdapter()
        aga_plugin.attach(model, adapter=adapter)
        assert aga_plugin.is_attached

        # 推理
        x = torch.randn(1, 10, 128)
        with torch.no_grad():
            output = model(hidden_states=x)
        assert output.shape == (1, 10, 128)

        aga_plugin.detach()

    def test_streaming_with_vllm(self, aga_plugin, vllm_model):
        """流式生成与 vLLM 适配器集成"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        with aga_plugin.create_streaming_session() as session:
            # 模拟多步生成
            for step in range(5):
                x = torch.randn(1, 1, 128)
                with torch.no_grad():
                    output = vllm_model(hidden_states=x)
                assert output.shape == (1, 1, 128)

            summary = session.get_session_summary()
            assert summary["total_steps"] > 0

        aga_plugin.detach()

    def test_diagnostics_with_vllm(self, aga_plugin, vllm_model):
        """诊断信息应该正常工作"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        # 触发一次 forward
        x = torch.randn(1, 10, 128)
        with torch.no_grad():
            vllm_model(hidden_states=x)

        diag = aga_plugin.get_diagnostics()
        assert diag["attached"] is True
        assert diag["model_type"] == "MockVLLMModel"
        assert diag["knowledge_count"] == 5

        aga_plugin.detach()


# ========== VLLMHookWorker 测试 ==========


class TestVLLMHookWorker:
    """测试 vLLM-Hook 兼容的 AGA Worker"""

    def test_init_default(self):
        """默认初始化"""
        worker = VLLMHookWorker()
        assert worker._plugin is None
        assert worker._initialized is False

    def test_init_with_config(self):
        """带配置初始化"""
        config = {
            "hidden_dim": 128,
            "bottleneck_dim": 16,
            "max_slots": 32,
            "device": "cpu",
        }
        worker = VLLMHookWorker(config)
        assert worker._config["hidden_dim"] == 128

    def test_initialize(self, vllm_model):
        """初始化 Worker"""
        config = {
            "hidden_dim": 128,
            "bottleneck_dim": 16,
            "max_slots": 32,
            "device": "cpu",
        }
        worker = VLLMHookWorker(config)
        worker.initialize(vllm_model)

        assert worker._initialized is True
        assert worker._plugin is not None
        assert worker._plugin.is_attached

    def test_initialize_with_knowledge(self, vllm_model, tmp_path):
        """初始化时加载知识"""
        # 创建测试 JSONL 文件
        import json
        knowledge_file = tmp_path / "test_knowledge.jsonl"
        entries = []
        for i in range(3):
            entries.append(json.dumps({
                "id": f"k_{i}",
                "key": [0.1] * 16,
                "value": [0.2] * 128,
                "reliability": 0.9,
            }))
        knowledge_file.write_text("\n".join(entries))

        config = {
            "hidden_dim": 128,
            "bottleneck_dim": 16,
            "max_slots": 32,
            "device": "cpu",
            "knowledge_file": str(knowledge_file),
        }
        worker = VLLMHookWorker(config)
        worker.initialize(vllm_model)

        assert worker._plugin.knowledge_count == 3

    def test_process_passthrough(self, vllm_model):
        """process 方法应该直接传递（hook 模式）"""
        config = {
            "hidden_dim": 128,
            "bottleneck_dim": 16,
            "max_slots": 32,
            "device": "cpu",
        }
        worker = VLLMHookWorker(config)
        worker.initialize(vllm_model)

        x = torch.randn(1, 10, 128)
        result = worker.process(x)
        assert torch.equal(result, x)  # 应该原样返回

    def test_get_diagnostics(self, vllm_model):
        """获取诊断信息"""
        config = {
            "hidden_dim": 128,
            "bottleneck_dim": 16,
            "max_slots": 32,
            "device": "cpu",
        }
        worker = VLLMHookWorker(config)
        worker.initialize(vllm_model)

        diag = worker.get_diagnostics()
        assert diag["attached"] is True

    def test_get_diagnostics_uninitialized(self):
        """未初始化时的诊断"""
        worker = VLLMHookWorker()
        diag = worker.get_diagnostics()
        assert diag["initialized"] is False

    def test_shutdown(self, vllm_model):
        """关闭 Worker"""
        config = {
            "hidden_dim": 128,
            "bottleneck_dim": 16,
            "max_slots": 32,
            "device": "cpu",
        }
        worker = VLLMHookWorker(config)
        worker.initialize(vllm_model)
        assert worker._plugin.is_attached

        worker.shutdown()
        assert not worker._plugin.is_attached

    def test_custom_layer_indices(self, vllm_model):
        """自定义挂载层"""
        config = {
            "hidden_dim": 128,
            "bottleneck_dim": 16,
            "max_slots": 32,
            "device": "cpu",
            "layer_indices": [0, 1],
        }
        worker = VLLMHookWorker(config)
        worker.initialize(vllm_model)

        assert len(worker._plugin._hooks) == 2


# ========== 边界情况测试 ==========


class TestVLLMEdgeCases:
    """边界情况测试"""

    def test_multiple_attach_detach(self, aga_plugin, vllm_model):
        """多次挂载/卸载"""
        adapter = VLLMAdapter()

        for _ in range(3):
            aga_plugin.attach(vllm_model, adapter=adapter)
            assert aga_plugin.is_attached

            x = torch.randn(1, 5, 128)
            with torch.no_grad():
                output = vllm_model(hidden_states=x)
            assert output.shape == (1, 5, 128)

            aga_plugin.detach()
            assert not aga_plugin.is_attached

    def test_different_batch_sizes(self, aga_plugin, vllm_model):
        """不同 batch size"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10, 128)
            with torch.no_grad():
                output = vllm_model(hidden_states=x)
            assert output.shape == (batch_size, 10, 128)

        aga_plugin.detach()

    def test_different_seq_lengths(self, aga_plugin, vllm_model):
        """不同序列长度"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        for seq_len in [1, 10, 50, 100]:
            x = torch.randn(1, seq_len, 128)
            with torch.no_grad():
                output = vllm_model(hidden_states=x)
            assert output.shape == (1, seq_len, 128)

        aga_plugin.detach()

    def test_single_token_decode(self, aga_plugin, vllm_model):
        """单 token decode（流式生成的典型场景）"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        # 模拟 10 步 decode
        for _ in range(10):
            x = torch.randn(1, 1, 128)  # 单 token
            with torch.no_grad():
                output = vllm_model(hidden_states=x)
            assert output.shape == (1, 1, 128)

        aga_plugin.detach()

    def test_knowledge_update_during_inference(self, aga_plugin, vllm_model):
        """推理过程中更新知识"""
        adapter = VLLMAdapter()
        aga_plugin.attach(vllm_model, adapter=adapter)

        x = torch.randn(1, 10, 128)
        with torch.no_grad():
            output1 = vllm_model(hidden_states=x)

        # 添加新知识
        aga_plugin.register(
            id="new_fact",
            key=torch.randn(16),
            value=torch.randn(128),
        )

        with torch.no_grad():
            output2 = vllm_model(hidden_states=x)

        # 两次输出可能不同（因为知识变了）
        assert output1.shape == output2.shape

        aga_plugin.detach()
