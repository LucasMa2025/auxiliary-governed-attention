"""
AGA 测试配置文件

提供全局 fixtures 和测试配置。
"""
import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, AsyncMock

import pytest
import torch

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==================== 基础配置 ====================

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环（用于异步测试）"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def device():
    """测试设备（CPU）"""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def hidden_dim():
    """隐藏层维度"""
    return 768  # 使用较小的维度加速测试


@pytest.fixture(scope="session")
def bottleneck_dim():
    """瓶颈层维度"""
    return 64


@pytest.fixture(scope="session")
def num_slots():
    """槽位数量"""
    return 32


# ==================== 向量 Fixtures ====================

@pytest.fixture
def random_key_vector(bottleneck_dim, device):
    """随机 key 向量"""
    return torch.randn(bottleneck_dim, device=device)


@pytest.fixture
def random_value_vector(hidden_dim, device):
    """随机 value 向量"""
    return torch.randn(hidden_dim, device=device)


@pytest.fixture
def random_hidden_states(hidden_dim, device):
    """随机隐藏状态 [batch, seq, hidden]"""
    batch_size = 2
    seq_len = 16
    return torch.randn(batch_size, seq_len, hidden_dim, device=device)


@pytest.fixture
def random_attention_mask(device):
    """随机注意力掩码"""
    batch_size = 2
    seq_len = 16
    return torch.ones(batch_size, seq_len, device=device)


# ==================== 配置 Fixtures ====================

@pytest.fixture
def aga_config(hidden_dim, bottleneck_dim, num_slots):
    """AGA 配置（使用 core.py 中的 AGAConfig）"""
    from aga.core import AGAConfig
    return AGAConfig(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_slots=num_slots,
        num_heads=8,
        tau_low=0.5,
        tau_high=2.0,
        top_k_routing=4,
        enable_early_exit=True,
        early_exit_threshold=0.05,
    )


@pytest.fixture
def entropy_gate_config(hidden_dim):
    """熵门控配置"""
    from aga.entropy_gate import EntropyGateConfig
    return EntropyGateConfig(
        tau_low=0.5,
        tau_high=2.0,
        max_gate=0.8,
    )


@pytest.fixture
def decay_config():
    """衰减配置"""
    from aga.decay import DecayConfig, DecayStrategy
    return DecayConfig(
        strategy=DecayStrategy.EXPONENTIAL,
        gamma=0.95,
        enable_hard_reset=True,
        hard_reset_threshold=3.0,
    )


@pytest.fixture
def slot_pool_config(hidden_dim, bottleneck_dim, num_slots):
    """槽位池配置"""
    from aga.production.config import SlotPoolConfig
    return SlotPoolConfig(
        max_slots_per_namespace=num_slots,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
    )


# ==================== 临时文件 Fixtures ====================

@pytest.fixture
def temp_db_path():
    """临时数据库路径"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # 清理
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def temp_dir():
    """临时目录"""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_redis():
    """Mock Redis 客户端"""
    from tests.mocks.redis_mock import MockRedis
    return MockRedis()


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL 客户端"""
    from tests.mocks.postgres_mock import MockPostgres
    return MockPostgres()


@pytest.fixture
def mock_kafka():
    """Mock Kafka 客户端"""
    from tests.mocks.kafka_mock import MockKafka
    return MockKafka()


@pytest.fixture
def mock_http_client():
    """Mock HTTP 客户端"""
    from tests.mocks.http_mock import MockHTTPClient
    return MockHTTPClient()


# ==================== 知识记录 Fixtures ====================

@pytest.fixture
def sample_knowledge_record(random_key_vector, random_value_vector):
    """示例知识记录"""
    from aga.persistence.base import KnowledgeRecord
    from aga import LifecycleState
    return KnowledgeRecord(
        slot_idx=0,
        lu_id="test_lu_001",
        namespace="default",
        condition="测试条件",
        decision="测试决策",
        key_vector=random_key_vector.tolist(),
        value_vector=random_value_vector.tolist(),
        lifecycle_state=LifecycleState.PROBATIONARY.value,
    )


@pytest.fixture
def sample_knowledge_records(bottleneck_dim, hidden_dim):
    """多个示例知识记录"""
    from aga.persistence.base import KnowledgeRecord
    from aga import LifecycleState
    
    records = []
    for i in range(10):
        records.append(KnowledgeRecord(
            slot_idx=i,
            lu_id=f"test_lu_{i:03d}",
            namespace="default",
            condition=f"测试条件 {i}",
            decision=f"测试决策 {i}",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY.value if i % 2 == 0 else LifecycleState.CONFIRMED.value,
        ))
    return records


# ==================== 辅助函数 ====================

def assert_tensor_close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-6):
    """断言两个张量接近"""
    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"Tensors not close: max diff = {(a - b).abs().max()}"


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """断言张量形状"""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


# 导出辅助函数
pytest.assert_tensor_close = assert_tensor_close
pytest.assert_tensor_shape = assert_tensor_shape
