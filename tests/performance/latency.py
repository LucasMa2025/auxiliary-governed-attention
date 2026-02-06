"""
延迟测试

测试各种操作的延迟。
"""
import pytest
import torch
import time
import statistics
from typing import List, Dict
from dataclasses import dataclass

from aga import AGAConfig, LifecycleState
from aga.core import AuxiliaryGovernedAttention as AGA


@dataclass
class LatencyResult:
    """延迟测试结果"""
    operation: str
    samples: List[float]
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.samples)
    
    @property
    def median(self) -> float:
        return statistics.median(self.samples)
    
    @property
    def p95(self) -> float:
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx]
    
    @property
    def p99(self) -> float:
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[idx]
    
    @property
    def min(self) -> float:
        return min(self.samples)
    
    @property
    def max(self) -> float:
        return max(self.samples)
    
    def to_dict(self) -> Dict:
        return {
            "operation": self.operation,
            "mean_ms": self.mean * 1000,
            "median_ms": self.median * 1000,
            "p95_ms": self.p95 * 1000,
            "p99_ms": self.p99 * 1000,
            "min_ms": self.min * 1000,
            "max_ms": self.max * 1000,
            "samples": len(self.samples),
        }


def measure_latency(func, warmup: int = 10, iterations: int = 100) -> LatencyResult:
    """
    测量函数延迟
    
    Args:
        func: 要测量的函数
        warmup: 预热次数
        iterations: 测量次数
        
    Returns:
        延迟结果
    """
    # 预热
    for _ in range(warmup):
        func()
    
    # 测量
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        samples.append(end - start)
    
    return LatencyResult(operation=func.__name__ if hasattr(func, '__name__') else "unknown", samples=samples)


@pytest.mark.performance
class TestInferenceLatency:
    """推理延迟测试"""
    
    @pytest.fixture
    def aga_module(self, aga_config, device):
        module = AGA(aga_config)
        module.to(device)
        module.eval()
        return module
    
    def test_empty_forward_latency(self, aga_module, hidden_dim, device):
        """测试空槽位前向传播延迟"""
        input_tensor = torch.randn(1, 32, hidden_dim, device=device)
        
        def forward():
            with torch.no_grad():
                aga_module(input_tensor)
        
        result = measure_latency(forward, warmup=20, iterations=100)
        
        print(f"\n空槽位前向传播延迟: {result.to_dict()}")
        
        # 断言延迟在合理范围内（CPU 上 < 50ms）
        assert result.p95 < 0.05, f"P95 延迟过高: {result.p95 * 1000:.2f}ms"
    
    def test_loaded_forward_latency(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试有槽位前向传播延迟"""
        # 注入知识
        for i in range(16):
            aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        input_tensor = torch.randn(1, 32, hidden_dim, device=device)
        
        def forward():
            with torch.no_grad():
                aga_module(input_tensor)
        
        result = measure_latency(forward, warmup=20, iterations=100)
        
        print(f"\n有槽位前向传播延迟 (16 slots): {result.to_dict()}")
        
        # 断言延迟在合理范围内
        assert result.p95 < 0.1, f"P95 延迟过高: {result.p95 * 1000:.2f}ms"
    
    def test_batch_forward_latency(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试批量前向传播延迟"""
        # 注入知识
        for i in range(16):
            aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        batch_sizes = [1, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 32, hidden_dim, device=device)
            
            def forward():
                with torch.no_grad():
                    aga_module(input_tensor)
            
            result = measure_latency(forward, warmup=10, iterations=50)
            results[batch_size] = result
            
            print(f"\n批量大小 {batch_size} 延迟: {result.to_dict()}")
        
        # 验证延迟随批量大小增长是亚线性的
        # （由于并行化，批量大小翻倍不应该导致延迟翻倍）
        ratio = results[16].mean / results[1].mean
        assert ratio < 16, f"批量延迟增长过快: {ratio:.2f}x"


@pytest.mark.performance
class TestKnowledgeOperationLatency:
    """知识操作延迟测试"""
    
    @pytest.fixture
    def aga_module(self, aga_config, device):
        module = AGA(aga_config)
        module.to(device)
        module.eval()
        return module
    
    def test_inject_latency(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试注入延迟"""
        key = torch.randn(bottleneck_dim, device=device)
        value = torch.randn(hidden_dim, device=device)
        
        samples = []
        for i in range(100):
            start = time.perf_counter()
            aga_module.inject_knowledge(
                slot_idx=i % 32,
                key_vector=key,
                value_vector=value,
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.PROBATIONARY,
            )
            end = time.perf_counter()
            samples.append(end - start)
        
        result = LatencyResult(operation="inject", samples=samples)
        
        print(f"\n注入延迟: {result.to_dict()}")
        
        # 注入应该很快（< 1ms）
        assert result.p95 < 0.001, f"P95 延迟过高: {result.p95 * 1000:.2f}ms"
    
    def test_lifecycle_update_latency(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试生命周期更新延迟"""
        # 先注入
        for i in range(32):
            aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.PROBATIONARY,
            )
        
        samples = []
        for i in range(100):
            start = time.perf_counter()
            aga_module.update_lifecycle(i % 32, LifecycleState.CONFIRMED)
            end = time.perf_counter()
            samples.append(end - start)
        
        result = LatencyResult(operation="lifecycle_update", samples=samples)
        
        print(f"\n生命周期更新延迟: {result.to_dict()}")
        
        assert result.p95 < 0.001
    
    def test_quarantine_latency(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试隔离延迟"""
        # 先注入
        for i in range(32):
            aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        samples = []
        for i in range(100):
            start = time.perf_counter()
            aga_module.quarantine_slot(i % 32)
            end = time.perf_counter()
            samples.append(end - start)
            
            # 恢复以便下次测试
            aga_module.update_lifecycle(i % 32, LifecycleState.CONFIRMED)
        
        result = LatencyResult(operation="quarantine", samples=samples)
        
        print(f"\n隔离延迟: {result.to_dict()}")
        
        assert result.p95 < 0.001


@pytest.mark.performance
class TestScalingLatency:
    """扩展性延迟测试"""
    
    def test_latency_vs_slot_count(self, aga_config, hidden_dim, bottleneck_dim, device):
        """测试延迟随槽位数量的变化"""
        slot_counts = [8, 16, 32, 64]
        results = {}
        
        for num_slots in slot_counts:
            config = AGAConfig(
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                num_slots=num_slots,
            )
            module = AGA(config)
            module.to(device)
            module.eval()
            
            # 填充槽位
            for i in range(num_slots):
                module.inject_knowledge(
                    slot_idx=i,
                    key_vector=torch.randn(bottleneck_dim, device=device),
                    value_vector=torch.randn(hidden_dim, device=device),
                    lu_id=f"test_{i:03d}",
                    lifecycle_state=LifecycleState.CONFIRMED,
                )
            
            input_tensor = torch.randn(1, 32, hidden_dim, device=device)
            
            def forward():
                with torch.no_grad():
                    module(input_tensor)
            
            result = measure_latency(forward, warmup=10, iterations=50)
            results[num_slots] = result
            
            print(f"\n{num_slots} 槽位延迟: {result.to_dict()}")
        
        # 验证延迟增长是亚线性的
        ratio = results[64].mean / results[8].mean
        assert ratio < 8, f"延迟增长过快: {ratio:.2f}x (8x slots)"
    
    def test_latency_vs_sequence_length(self, aga_config, hidden_dim, bottleneck_dim, device):
        """测试延迟随序列长度的变化"""
        module = AGA(aga_config)
        module.to(device)
        module.eval()
        
        # 填充槽位
        for i in range(16):
            module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        seq_lengths = [16, 32, 64, 128]
        results = {}
        
        for seq_len in seq_lengths:
            input_tensor = torch.randn(1, seq_len, hidden_dim, device=device)
            
            def forward():
                with torch.no_grad():
                    module(input_tensor)
            
            result = measure_latency(forward, warmup=10, iterations=50)
            results[seq_len] = result
            
            print(f"\n序列长度 {seq_len} 延迟: {result.to_dict()}")
        
        # 验证延迟增长是线性或亚线性的
        ratio = results[128].mean / results[16].mean
        assert ratio < 16, f"延迟增长过快: {ratio:.2f}x (8x seq_len)"
