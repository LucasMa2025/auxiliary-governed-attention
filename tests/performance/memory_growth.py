"""
内存增长测试

测试系统的内存使用和泄漏。
"""
import pytest
import torch
import gc
import sys
from typing import List, Dict
from dataclasses import dataclass

from aga import AGAConfig, LifecycleState
from aga.core import AuxiliaryGovernedAttention as AGA


def get_memory_usage() -> Dict[str, float]:
    """获取当前内存使用情况"""
    gc.collect()
    
    result = {
        "python_objects": sys.getsizeof(gc.get_objects()),
    }
    
    if torch.cuda.is_available():
        result["cuda_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        result["cuda_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    
    return result


def get_tensor_memory() -> float:
    """获取所有张量占用的内存（MB）"""
    total = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            total += obj.element_size() * obj.nelement()
    return total / 1024 / 1024


@dataclass
class MemorySnapshot:
    """内存快照"""
    iteration: int
    tensor_memory_mb: float
    python_objects: int
    
    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "tensor_memory_mb": self.tensor_memory_mb,
            "python_objects": self.python_objects,
        }


@pytest.mark.performance
class TestMemoryGrowth:
    """内存增长测试"""
    
    @pytest.fixture
    def aga_module(self, aga_config, device):
        module = AGA(aga_config)
        module.to(device)
        module.eval()
        return module
    
    def test_inference_memory_stability(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试推理内存稳定性"""
        # 注入知识
        for i in range(16):
            aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        snapshots: List[MemorySnapshot] = []
        
        # 多次推理
        for iteration in range(100):
            input_tensor = torch.randn(1, 32, hidden_dim, device=device)
            
            with torch.no_grad():
                output, _ = aga_module(input_tensor)
            
            if iteration % 10 == 0:
                gc.collect()
                snapshots.append(MemorySnapshot(
                    iteration=iteration,
                    tensor_memory_mb=get_tensor_memory(),
                    python_objects=len(gc.get_objects()),
                ))
        
        # 分析内存增长
        initial = snapshots[0].tensor_memory_mb
        final = snapshots[-1].tensor_memory_mb
        growth = final - initial
        
        print(f"\n内存增长: {growth:.2f} MB ({initial:.2f} -> {final:.2f})")
        
        # 内存增长应该很小（< 10MB）
        assert growth < 10, f"内存增长过大: {growth:.2f} MB"
    
    def test_inject_remove_memory_stability(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试注入/移除内存稳定性"""
        snapshots: List[MemorySnapshot] = []
        
        # 多次注入和移除
        for iteration in range(50):
            # 注入
            for i in range(16):
                aga_module.inject_knowledge(
                    slot_idx=i,
                    key_vector=torch.randn(bottleneck_dim, device=device),
                    value_vector=torch.randn(hidden_dim, device=device),
                    lu_id=f"test_{iteration}_{i:03d}",
                    lifecycle_state=LifecycleState.PROBATIONARY,
                )
            
            # 移除
            for i in range(16):
                aga_module.remove_slot(i)
            
            if iteration % 5 == 0:
                gc.collect()
                snapshots.append(MemorySnapshot(
                    iteration=iteration,
                    tensor_memory_mb=get_tensor_memory(),
                    python_objects=len(gc.get_objects()),
                ))
        
        # 分析内存增长
        initial = snapshots[0].tensor_memory_mb
        final = snapshots[-1].tensor_memory_mb
        growth = final - initial
        
        print(f"\n注入/移除内存增长: {growth:.2f} MB")
        
        # 内存应该保持稳定
        assert growth < 5, f"内存泄漏: {growth:.2f} MB"
    
    def test_lifecycle_transition_memory(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试生命周期转换内存"""
        # 注入知识
        for i in range(16):
            aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.PROBATIONARY,
            )
        
        gc.collect()
        initial_memory = get_tensor_memory()
        
        # 多次状态转换
        for _ in range(100):
            for i in range(16):
                aga_module.update_lifecycle(i, LifecycleState.CONFIRMED)
            for i in range(16):
                aga_module.quarantine_slot(i)
            for i in range(16):
                aga_module.update_lifecycle(i, LifecycleState.PROBATIONARY)
        
        gc.collect()
        final_memory = get_tensor_memory()
        growth = final_memory - initial_memory
        
        print(f"\n生命周期转换内存增长: {growth:.2f} MB")
        
        assert growth < 1, f"内存泄漏: {growth:.2f} MB"


@pytest.mark.performance
class TestMemoryScaling:
    """内存扩展性测试"""
    
    def test_memory_vs_slot_count(self, hidden_dim, bottleneck_dim, device):
        """测试内存随槽位数量的变化"""
        slot_counts = [16, 32, 64, 128]
        memory_usage = {}
        
        for num_slots in slot_counts:
            gc.collect()
            
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
            
            gc.collect()
            memory_usage[num_slots] = get_tensor_memory()
            
            print(f"\n{num_slots} 槽位内存: {memory_usage[num_slots]:.2f} MB")
            
            del module
        
        # 验证内存增长是线性的
        ratio = memory_usage[128] / memory_usage[16]
        expected_ratio = 128 / 16  # 8x
        
        # 允许 20% 的误差
        assert ratio < expected_ratio * 1.2, f"内存增长超线性: {ratio:.2f}x (expected ~{expected_ratio}x)"
    
    def test_memory_vs_hidden_dim(self, bottleneck_dim, num_slots, device):
        """测试内存随隐藏维度的变化"""
        hidden_dims = [256, 512, 768, 1024]
        memory_usage = {}
        
        for hidden_dim in hidden_dims:
            gc.collect()
            
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
            
            gc.collect()
            memory_usage[hidden_dim] = get_tensor_memory()
            
            print(f"\nhidden_dim={hidden_dim} 内存: {memory_usage[hidden_dim]:.2f} MB")
            
            del module
        
        # 验证内存增长是线性的
        ratio = memory_usage[1024] / memory_usage[256]
        expected_ratio = 1024 / 256  # 4x
        
        assert ratio < expected_ratio * 1.5, f"内存增长超线性: {ratio:.2f}x"
