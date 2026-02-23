"""
Operator 单元测试 (BottleneckInjector)
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from aga.operator.bottleneck_injector import BottleneckInjector


class TestBottleneckInjector:
    """BottleneckInjector 测试"""

    @pytest.fixture
    def injector(self):
        return BottleneckInjector(
            hidden_dim=128,
            bottleneck_dim=16,
            value_bottleneck_dim=32,
            top_k=4,
            use_value_projection=True,
        )

    def test_init(self, injector):
        """测试初始化"""
        assert injector.hidden_dim == 128
        assert injector.bottleneck_dim == 16
        assert injector.top_k == 4

    def test_forward_shape(self, injector):
        """测试输出形状"""
        hidden_states = torch.randn(2, 10, 128)
        keys = torch.randn(8, 16)
        values = torch.randn(8, 128)
        reliability = torch.ones(8) * 0.9

        output = injector(hidden_states, keys, values, reliability)
        assert output.shape == (2, 10, 128)

    def test_forward_with_top_k(self, injector):
        """测试 Top-K 路由"""
        hidden_states = torch.randn(2, 10, 128)
        keys = torch.randn(20, 16)  # 多于 top_k
        values = torch.randn(20, 128)
        reliability = torch.ones(20) * 0.9

        output = injector(hidden_states, keys, values, reliability)
        assert output.shape == (2, 10, 128)

    def test_forward_without_top_k(self, injector):
        """测试不需要 Top-K 的情况"""
        hidden_states = torch.randn(2, 10, 128)
        keys = torch.randn(3, 16)  # 少于 top_k
        values = torch.randn(3, 128)
        reliability = torch.ones(3) * 0.9

        output = injector(hidden_states, keys, values, reliability)
        assert output.shape == (2, 10, 128)

    def test_forward_without_value_projection(self):
        """测试不使用 value projection"""
        injector = BottleneckInjector(
            hidden_dim=128,
            bottleneck_dim=16,
            use_value_projection=False,
        )
        hidden_states = torch.randn(2, 10, 128)
        keys = torch.randn(5, 16)
        values = torch.randn(5, 128)
        reliability = torch.ones(5) * 0.9

        output = injector(hidden_states, keys, values, reliability)
        assert output.shape == (2, 10, 128)

    def test_reliability_affects_output(self, injector):
        """测试可靠性分布影响输出"""
        hidden_states = torch.randn(2, 10, 128)
        keys = torch.randn(4, 16)
        values = torch.randn(4, 128)

        # 分布 A: 第一个 slot 可靠性极高，其余极低
        rel_a = torch.tensor([0.99, 0.01, 0.01, 0.01])
        output_a = injector(hidden_states, keys, values, rel_a)

        # 分布 B: 最后一个 slot 可靠性极高，其余极低
        rel_b = torch.tensor([0.01, 0.01, 0.01, 0.99])
        output_b = injector(hidden_states, keys, values, rel_b)

        # 不同的可靠性分布应导致不同的输出（注意力权重偏向不同 slot）
        assert not torch.allclose(output_a, output_b, atol=1e-6)

    def test_single_slot(self, injector):
        """测试单个槽位"""
        hidden_states = torch.randn(1, 5, 128)
        keys = torch.randn(1, 16)
        values = torch.randn(1, 128)
        reliability = torch.ones(1) * 0.9

        output = injector(hidden_states, keys, values, reliability)
        assert output.shape == (1, 5, 128)

    def test_gradient_flow(self, injector):
        """测试梯度流"""
        hidden_states = torch.randn(2, 10, 128, requires_grad=True)
        keys = torch.randn(4, 16)
        values = torch.randn(4, 128)
        reliability = torch.ones(4) * 0.9

        output = injector(hidden_states, keys, values, reliability)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == (2, 10, 128)
