"""
多 Runtime 集成测试（使用 Mock）

测试 Portal <-> 多 Runtime 的分布式场景。
"""
import pytest
import asyncio
import torch
import json
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, List, Optional

from aga import AGAConfig, LifecycleState, KnowledgeRecord
from aga.core import AuxiliaryGovernedAttention as AGA
from tests.mocks.redis_mock import MockRedis
from tests.mocks.http_mock import MockHTTPClient, MockHTTPResponse


# ==================== Mock Runtime ====================

@dataclass
class MockRuntime:
    """Mock Runtime 节点"""
    runtime_id: str
    aga: AGA
    device: torch.device
    is_healthy: bool = True
    
    async def inject_knowledge(self, record: KnowledgeRecord, slot_idx: int) -> bool:
        """注入知识"""
        if not self.is_healthy:
            raise ConnectionError(f"Runtime {self.runtime_id} is not healthy")
        
        self.aga.inject_knowledge(
            slot_idx=slot_idx,
            key_vector=torch.tensor(record.key_vector, device=self.device),
            value_vector=torch.tensor(record.value_vector, device=self.device),
            lu_id=record.lu_id,
            lifecycle_state=record.lifecycle_state,
        )
        return True
    
    async def update_lifecycle(self, lu_id: str, slot_idx: int, state: LifecycleState) -> bool:
        """更新生命周期"""
        if not self.is_healthy:
            raise ConnectionError(f"Runtime {self.runtime_id} is not healthy")
        
        self.aga.update_lifecycle(slot_idx, state)
        return True
    
    async def quarantine(self, lu_id: str, slot_idx: int) -> bool:
        """隔离槽位"""
        if not self.is_healthy:
            raise ConnectionError(f"Runtime {self.runtime_id} is not healthy")
        
        self.aga.quarantine_slot(slot_idx)
        return True
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self.aga.get_statistics()


# ==================== Mock Portal ====================

class MockPortal:
    """Mock Portal 服务"""
    
    def __init__(self, redis: MockRedis):
        self._redis = redis
        self._runtimes: Dict[str, MockRuntime] = {}
        self._slot_assignments: Dict[str, Dict[str, int]] = {}  # {namespace: {lu_id: slot_idx}}
        self._next_slot_idx: Dict[str, int] = {}
    
    def register_runtime(self, runtime: MockRuntime):
        """注册 Runtime"""
        self._runtimes[runtime.runtime_id] = runtime
    
    def unregister_runtime(self, runtime_id: str):
        """注销 Runtime"""
        self._runtimes.pop(runtime_id, None)
    
    async def inject_knowledge(self, record: KnowledgeRecord) -> Dict:
        """注入知识到所有 Runtime"""
        namespace = record.namespace
        
        # 分配槽位
        if namespace not in self._slot_assignments:
            self._slot_assignments[namespace] = {}
            self._next_slot_idx[namespace] = 0
        
        if record.lu_id in self._slot_assignments[namespace]:
            slot_idx = self._slot_assignments[namespace][record.lu_id]
        else:
            slot_idx = self._next_slot_idx[namespace]
            self._slot_assignments[namespace][record.lu_id] = slot_idx
            self._next_slot_idx[namespace] += 1
        
        # 广播到所有 Runtime
        results = {}
        for runtime_id, runtime in self._runtimes.items():
            try:
                await runtime.inject_knowledge(record, slot_idx)
                results[runtime_id] = "success"
            except Exception as e:
                results[runtime_id] = f"error: {str(e)}"
        
        # 发布事件
        await self._redis.publish(
            f"aga:events:{namespace}",
            json.dumps({
                "type": "inject",
                "lu_id": record.lu_id,
                "slot_idx": slot_idx,
            }),
        )
        
        return {
            "lu_id": record.lu_id,
            "slot_idx": slot_idx,
            "results": results,
        }
    
    async def update_lifecycle(self, namespace: str, lu_id: str, state: LifecycleState) -> Dict:
        """更新生命周期"""
        if namespace not in self._slot_assignments or lu_id not in self._slot_assignments[namespace]:
            return {"error": "Knowledge not found"}
        
        slot_idx = self._slot_assignments[namespace][lu_id]
        
        results = {}
        for runtime_id, runtime in self._runtimes.items():
            try:
                await runtime.update_lifecycle(lu_id, slot_idx, state)
                results[runtime_id] = "success"
            except Exception as e:
                results[runtime_id] = f"error: {str(e)}"
        
        return {"results": results}
    
    async def quarantine(self, namespace: str, lu_id: str) -> Dict:
        """隔离知识"""
        if namespace not in self._slot_assignments or lu_id not in self._slot_assignments[namespace]:
            return {"error": "Knowledge not found"}
        
        slot_idx = self._slot_assignments[namespace][lu_id]
        
        results = {}
        for runtime_id, runtime in self._runtimes.items():
            try:
                await runtime.quarantine(lu_id, slot_idx)
                results[runtime_id] = "success"
            except Exception as e:
                results[runtime_id] = f"error: {str(e)}"
        
        return {"results": results}
    
    def get_healthy_runtimes(self) -> List[str]:
        """获取健康的 Runtime"""
        return [rid for rid, r in self._runtimes.items() if r.is_healthy]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            "total_runtimes": len(self._runtimes),
            "healthy_runtimes": len(self.get_healthy_runtimes()),
            "runtimes": {},
        }
        
        for runtime_id, runtime in self._runtimes.items():
            stats["runtimes"][runtime_id] = {
                "is_healthy": runtime.is_healthy,
                "stats": runtime.get_statistics(),
            }
        
        return stats


# ==================== 测试 ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestMultiRuntimeIntegration:
    """多 Runtime 集成测试"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis"""
        return MockRedis()
    
    @pytest.fixture
    def aga_config(self, hidden_dim, bottleneck_dim, num_slots):
        """AGA 配置"""
        return AGAConfig(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_slots=num_slots,
        )
    
    @pytest.fixture
    def create_runtime(self, aga_config, device):
        """创建 Runtime 工厂"""
        def _create(runtime_id: str) -> MockRuntime:
            aga = AGA(aga_config)
            aga.to(device)
            aga.eval()
            return MockRuntime(runtime_id=runtime_id, aga=aga, device=device)
        return _create
    
    @pytest.fixture
    async def setup_multi_runtime(self, mock_redis, create_runtime):
        """设置多 Runtime 环境"""
        portal = MockPortal(mock_redis)
        
        # 创建 3 个 Runtime
        runtimes = [create_runtime(f"runtime_{i}") for i in range(3)]
        for runtime in runtimes:
            portal.register_runtime(runtime)
        
        yield {
            "portal": portal,
            "runtimes": runtimes,
            "redis": mock_redis,
        }
    
    async def test_broadcast_inject(self, setup_multi_runtime, bottleneck_dim, hidden_dim):
        """测试广播注入"""
        env = setup_multi_runtime
        
        record = KnowledgeRecord(
            lu_id="test_lu_001",
            namespace="default",
            condition="测试条件",
            decision="测试决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        result = await env["portal"].inject_knowledge(record)
        
        # 验证所有 Runtime 都收到了知识
        assert all(v == "success" for v in result["results"].values())
        
        for runtime in env["runtimes"]:
            assert runtime.aga.active_slots == 1
    
    async def test_broadcast_lifecycle_update(self, setup_multi_runtime, bottleneck_dim, hidden_dim):
        """测试广播生命周期更新"""
        env = setup_multi_runtime
        
        # 先注入
        record = KnowledgeRecord(
            lu_id="test_lu_001",
            namespace="default",
            condition="测试条件",
            decision="测试决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        await env["portal"].inject_knowledge(record)
        
        # 更新生命周期
        result = await env["portal"].update_lifecycle(
            "default",
            "test_lu_001",
            LifecycleState.CONFIRMED,
        )
        
        # 验证所有 Runtime 都更新了
        assert all(v == "success" for v in result["results"].values())
        
        for runtime in env["runtimes"]:
            slot_info = runtime.aga.get_slot_info(0)
            assert slot_info.lifecycle_state == LifecycleState.CONFIRMED
    
    async def test_broadcast_quarantine(self, setup_multi_runtime, bottleneck_dim, hidden_dim):
        """测试广播隔离"""
        env = setup_multi_runtime
        
        # 先注入
        record = KnowledgeRecord(
            lu_id="test_lu_001",
            namespace="default",
            condition="测试条件",
            decision="测试决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        await env["portal"].inject_knowledge(record)
        
        # 隔离
        result = await env["portal"].quarantine("default", "test_lu_001")
        
        # 验证所有 Runtime 都隔离了
        assert all(v == "success" for v in result["results"].values())
        
        for runtime in env["runtimes"]:
            slot_info = runtime.aga.get_slot_info(0)
            assert slot_info.lifecycle_state == LifecycleState.QUARANTINED
    
    async def test_partial_failure(self, setup_multi_runtime, bottleneck_dim, hidden_dim):
        """测试部分失败"""
        env = setup_multi_runtime
        
        # 让一个 Runtime 不健康
        env["runtimes"][1].is_healthy = False
        
        record = KnowledgeRecord(
            lu_id="test_lu_001",
            namespace="default",
            condition="测试条件",
            decision="测试决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        result = await env["portal"].inject_knowledge(record)
        
        # 验证结果
        assert result["results"]["runtime_0"] == "success"
        assert "error" in result["results"]["runtime_1"]
        assert result["results"]["runtime_2"] == "success"
    
    async def test_consistency_check(self, setup_multi_runtime, bottleneck_dim, hidden_dim):
        """测试一致性检查"""
        env = setup_multi_runtime
        
        # 注入多个知识
        for i in range(5):
            record = KnowledgeRecord(
                lu_id=f"test_lu_{i:03d}",
                namespace="default",
                condition=f"条件 {i}",
                decision=f"决策 {i}",
                key_vector=torch.randn(bottleneck_dim).tolist(),
                value_vector=torch.randn(hidden_dim).tolist(),
                lifecycle_state=LifecycleState.PROBATIONARY,
            )
            await env["portal"].inject_knowledge(record)
        
        # 验证所有 Runtime 状态一致
        stats = env["portal"].get_statistics()
        
        slot_counts = [
            stats["runtimes"][f"runtime_{i}"]["stats"]["active_slots"]
            for i in range(3)
        ]
        
        assert all(c == 5 for c in slot_counts)


@pytest.mark.integration
@pytest.mark.asyncio
class TestRuntimeRecovery:
    """Runtime 恢复测试"""
    
    @pytest.fixture
    def mock_redis(self):
        return MockRedis()
    
    @pytest.fixture
    def aga_config(self, hidden_dim, bottleneck_dim, num_slots):
        return AGAConfig(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_slots=num_slots,
        )
    
    @pytest.fixture
    def create_runtime(self, aga_config, device):
        def _create(runtime_id: str) -> MockRuntime:
            aga = AGA(aga_config)
            aga.to(device)
            aga.eval()
            return MockRuntime(runtime_id=runtime_id, aga=aga, device=device)
        return _create
    
    async def test_runtime_rejoin(self, mock_redis, create_runtime, bottleneck_dim, hidden_dim):
        """测试 Runtime 重新加入"""
        portal = MockPortal(mock_redis)
        
        # 初始 2 个 Runtime
        runtime_0 = create_runtime("runtime_0")
        runtime_1 = create_runtime("runtime_1")
        portal.register_runtime(runtime_0)
        portal.register_runtime(runtime_1)
        
        # 注入知识
        record = KnowledgeRecord(
            lu_id="test_lu_001",
            namespace="default",
            condition="测试条件",
            decision="测试决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        await portal.inject_knowledge(record)
        
        # 新 Runtime 加入
        runtime_2 = create_runtime("runtime_2")
        portal.register_runtime(runtime_2)
        
        # 新 Runtime 需要同步现有知识
        # 这里模拟同步过程
        await runtime_2.inject_knowledge(record, 0)
        
        # 验证所有 Runtime 状态一致
        assert runtime_0.aga.active_slots == 1
        assert runtime_1.aga.active_slots == 1
        assert runtime_2.aga.active_slots == 1
    
    async def test_runtime_failure_and_recovery(self, mock_redis, create_runtime, bottleneck_dim, hidden_dim):
        """测试 Runtime 故障和恢复"""
        portal = MockPortal(mock_redis)
        
        runtime_0 = create_runtime("runtime_0")
        runtime_1 = create_runtime("runtime_1")
        portal.register_runtime(runtime_0)
        portal.register_runtime(runtime_1)
        
        # 注入知识
        record = KnowledgeRecord(
            lu_id="test_lu_001",
            namespace="default",
            condition="测试条件",
            decision="测试决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        await portal.inject_knowledge(record)
        
        # Runtime 1 故障
        runtime_1.is_healthy = False
        
        # 更新操作
        await portal.update_lifecycle("default", "test_lu_001", LifecycleState.QUARANTINED)
        
        # Runtime 1 恢复
        runtime_1.is_healthy = True
        
        # 同步状态
        runtime_1.aga.quarantine_slot(0)
        
        # 验证状态一致
        assert runtime_0.aga.get_slot_info(0).lifecycle_state == LifecycleState.QUARANTINED
        assert runtime_1.aga.get_slot_info(0).lifecycle_state == LifecycleState.QUARANTINED
