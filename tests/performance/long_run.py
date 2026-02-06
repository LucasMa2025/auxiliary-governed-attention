"""
长期运行测试

测试系统的长期稳定性。
"""
import pytest
import torch
import time
import gc
import random
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime

from aga import AGAConfig, LifecycleState
from aga.core import AuxiliaryGovernedAttention as AGA


@dataclass
class RunStatistics:
    """运行统计"""
    total_inferences: int = 0
    total_injects: int = 0
    total_updates: int = 0
    total_quarantines: int = 0
    total_removes: int = 0
    errors: List[str] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    @property
    def duration(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0
    
    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0
    
    def to_dict(self) -> Dict:
        return {
            "duration_seconds": self.duration,
            "total_inferences": self.total_inferences,
            "total_injects": self.total_injects,
            "total_updates": self.total_updates,
            "total_quarantines": self.total_quarantines,
            "total_removes": self.total_removes,
            "error_count": len(self.errors),
            "avg_latency_ms": self.avg_latency * 1000,
            "max_latency_ms": self.max_latency * 1000,
            "memory_growth_mb": (self.memory_samples[-1] - self.memory_samples[0]) if len(self.memory_samples) > 1 else 0,
        }


def get_tensor_memory() -> float:
    """获取张量内存（MB）"""
    total = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            total += obj.element_size() * obj.nelement()
    return total / 1024 / 1024


@pytest.mark.performance
@pytest.mark.slow
class TestLongRunStability:
    """长期运行稳定性测试"""
    
    @pytest.fixture
    def aga_module(self, aga_config, device):
        module = AGA(aga_config)
        module.to(device)
        module.eval()
        return module
    
    def test_continuous_inference(self, aga_module, hidden_dim, bottleneck_dim, device):
        """测试持续推理"""
        # 注入知识
        for i in range(16):
            aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        stats = RunStatistics()
        
        # 运行 1000 次推理
        for i in range(1000):
            try:
                input_tensor = torch.randn(1, 32, hidden_dim, device=device)
                
                start = time.perf_counter()
                with torch.no_grad():
                    output, _ = aga_module(input_tensor)
                end = time.perf_counter()
                
                stats.total_inferences += 1
                stats.latencies.append(end - start)
                
                # 定期采样内存
                if i % 100 == 0:
                    gc.collect()
                    stats.memory_samples.append(get_tensor_memory())
                    
            except Exception as e:
                stats.errors.append(str(e))
        
        print(f"\n持续推理统计: {stats.to_dict()}")
        
        # 验证
        assert len(stats.errors) == 0, f"发生错误: {stats.errors}"
        assert stats.max_latency < 0.5, f"最大延迟过高: {stats.max_latency * 1000:.2f}ms"
    
    def test_mixed_operations(self, aga_module, hidden_dim, bottleneck_dim, device, num_slots):
        """测试混合操作"""
        stats = RunStatistics()
        
        # 运行 500 次混合操作
        for i in range(500):
            try:
                operation = random.choice(["inference", "inject", "update", "quarantine", "remove"])
                
                start = time.perf_counter()
                
                if operation == "inference":
                    input_tensor = torch.randn(1, 32, hidden_dim, device=device)
                    with torch.no_grad():
                        output, _ = aga_module(input_tensor)
                    stats.total_inferences += 1
                    
                elif operation == "inject":
                    slot_idx = random.randint(0, num_slots - 1)
                    aga_module.inject_knowledge(
                        slot_idx=slot_idx,
                        key_vector=torch.randn(bottleneck_dim, device=device),
                        value_vector=torch.randn(hidden_dim, device=device),
                        lu_id=f"test_{i}_{slot_idx}",
                        lifecycle_state=LifecycleState.PROBATIONARY,
                    )
                    stats.total_injects += 1
                    
                elif operation == "update":
                    if aga_module.active_slots > 0:
                        slot_idx = random.randint(0, num_slots - 1)
                        if aga_module.get_slot_info(slot_idx):
                            state = random.choice([LifecycleState.PROBATIONARY, LifecycleState.CONFIRMED])
                            aga_module.update_lifecycle(slot_idx, state)
                            stats.total_updates += 1
                            
                elif operation == "quarantine":
                    if aga_module.active_slots > 0:
                        slot_idx = random.randint(0, num_slots - 1)
                        if aga_module.get_slot_info(slot_idx):
                            aga_module.quarantine_slot(slot_idx)
                            stats.total_quarantines += 1
                            
                elif operation == "remove":
                    if aga_module.active_slots > 0:
                        slot_idx = random.randint(0, num_slots - 1)
                        if aga_module.get_slot_info(slot_idx):
                            aga_module.remove_slot(slot_idx)
                            stats.total_removes += 1
                
                end = time.perf_counter()
                stats.latencies.append(end - start)
                
                # 定期采样内存
                if i % 50 == 0:
                    gc.collect()
                    stats.memory_samples.append(get_tensor_memory())
                    
            except Exception as e:
                stats.errors.append(f"{operation}: {str(e)}")
        
        print(f"\n混合操作统计: {stats.to_dict()}")
        
        # 验证
        assert len(stats.errors) == 0, f"发生错误: {stats.errors[:5]}"
    
    def test_stress_inject_remove(self, aga_module, hidden_dim, bottleneck_dim, device, num_slots):
        """测试压力注入/移除"""
        stats = RunStatistics()
        
        # 运行 100 轮完整的注入/移除周期
        for cycle in range(100):
            try:
                # 填满槽位
                for i in range(num_slots):
                    aga_module.inject_knowledge(
                        slot_idx=i,
                        key_vector=torch.randn(bottleneck_dim, device=device),
                        value_vector=torch.randn(hidden_dim, device=device),
                        lu_id=f"cycle_{cycle}_{i}",
                        lifecycle_state=LifecycleState.PROBATIONARY,
                    )
                    stats.total_injects += 1
                
                # 执行一些推理
                for _ in range(10):
                    input_tensor = torch.randn(1, 32, hidden_dim, device=device)
                    start = time.perf_counter()
                    with torch.no_grad():
                        output, _ = aga_module(input_tensor)
                    end = time.perf_counter()
                    stats.total_inferences += 1
                    stats.latencies.append(end - start)
                
                # 清空槽位
                for i in range(num_slots):
                    if aga_module.get_slot_info(i):
                        aga_module.remove_slot(i)
                        stats.total_removes += 1
                
                # 采样内存
                if cycle % 10 == 0:
                    gc.collect()
                    stats.memory_samples.append(get_tensor_memory())
                    
            except Exception as e:
                stats.errors.append(f"cycle {cycle}: {str(e)}")
        
        print(f"\n压力测试统计: {stats.to_dict()}")
        
        # 验证
        assert len(stats.errors) == 0, f"发生错误: {stats.errors[:5]}"
        
        # 内存增长应该很小
        if len(stats.memory_samples) > 1:
            growth = stats.memory_samples[-1] - stats.memory_samples[0]
            assert growth < 10, f"内存泄漏: {growth:.2f} MB"


@pytest.mark.performance
@pytest.mark.slow
class TestConcurrentStability:
    """并发稳定性测试"""
    
    def test_concurrent_inference(self, aga_config, hidden_dim, bottleneck_dim, device):
        """测试并发推理"""
        import threading
        
        module = AGA(aga_config)
        module.to(device)
        module.eval()
        
        # 注入知识
        for i in range(16):
            module.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(bottleneck_dim, device=device),
                value_vector=torch.randn(hidden_dim, device=device),
                lu_id=f"test_{i:03d}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        errors = []
        inference_count = [0]
        lock = threading.Lock()
        
        def worker(thread_id: int, iterations: int):
            for _ in range(iterations):
                try:
                    input_tensor = torch.randn(1, 32, hidden_dim, device=device)
                    with torch.no_grad():
                        output, _ = module(input_tensor)
                    with lock:
                        inference_count[0] += 1
                except Exception as e:
                    errors.append(f"Thread {thread_id}: {str(e)}")
        
        # 启动多个线程
        threads = []
        num_threads = 4
        iterations_per_thread = 100
        
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i, iterations_per_thread))
            threads.append(t)
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        duration = time.time() - start
        
        print(f"\n并发推理: {inference_count[0]} 次推理, {duration:.2f}s, {inference_count[0]/duration:.1f} ops/s")
        
        assert len(errors) == 0, f"发生错误: {errors[:5]}"
        assert inference_count[0] == num_threads * iterations_per_thread
    
    def test_concurrent_mixed_operations(self, aga_config, hidden_dim, bottleneck_dim, device, num_slots):
        """测试并发混合操作"""
        import threading
        
        module = AGA(aga_config)
        module.to(device)
        module.eval()
        
        errors = []
        operation_count = {"inference": 0, "inject": 0, "update": 0}
        lock = threading.Lock()
        
        def inference_worker(iterations: int):
            for _ in range(iterations):
                try:
                    input_tensor = torch.randn(1, 32, hidden_dim, device=device)
                    with torch.no_grad():
                        output, _ = module(input_tensor)
                    with lock:
                        operation_count["inference"] += 1
                except Exception as e:
                    errors.append(f"Inference: {str(e)}")
        
        def inject_worker(iterations: int):
            for i in range(iterations):
                try:
                    slot_idx = i % num_slots
                    module.inject_knowledge(
                        slot_idx=slot_idx,
                        key_vector=torch.randn(bottleneck_dim, device=device),
                        value_vector=torch.randn(hidden_dim, device=device),
                        lu_id=f"inject_{i}",
                        lifecycle_state=LifecycleState.PROBATIONARY,
                    )
                    with lock:
                        operation_count["inject"] += 1
                except Exception as e:
                    errors.append(f"Inject: {str(e)}")
        
        def update_worker(iterations: int):
            for i in range(iterations):
                try:
                    slot_idx = i % num_slots
                    if module.get_slot_info(slot_idx):
                        module.update_lifecycle(slot_idx, LifecycleState.CONFIRMED)
                        with lock:
                            operation_count["update"] += 1
                except Exception as e:
                    errors.append(f"Update: {str(e)}")
        
        threads = [
            threading.Thread(target=inference_worker, args=(100,)),
            threading.Thread(target=inference_worker, args=(100,)),
            threading.Thread(target=inject_worker, args=(50,)),
            threading.Thread(target=update_worker, args=(50,)),
        ]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        duration = time.time() - start
        
        total_ops = sum(operation_count.values())
        print(f"\n并发混合操作: {operation_count}, 总计 {total_ops} 次, {duration:.2f}s")
        
        # 允许一些错误（由于并发竞争）
        error_rate = len(errors) / total_ops if total_ops > 0 else 0
        assert error_rate < 0.1, f"错误率过高: {error_rate:.2%}"
