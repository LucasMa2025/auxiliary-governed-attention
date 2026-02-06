"""
网络分区测试

测试网络分区场景下系统的行为。
"""
import pytest
import asyncio
import torch
from typing import Dict, List, Set
from dataclasses import dataclass, field

from aga import AGAConfig, LifecycleState, KnowledgeRecord
from aga.core import AuxiliaryGovernedAttention as AGA
from tests.mocks.redis_mock import MockRedis


# ==================== 网络模拟器 ====================

class NetworkSimulator:
    """网络模拟器"""
    
    def __init__(self):
        self._partitions: Dict[str, Set[str]] = {}  # {node_id: reachable_nodes}
        self._all_nodes: Set[str] = set()
    
    def add_node(self, node_id: str):
        """添加节点"""
        self._all_nodes.add(node_id)
        self._partitions[node_id] = set(self._all_nodes)
        # 更新其他节点的可达性
        for other in self._all_nodes:
            if other != node_id:
                self._partitions[other].add(node_id)
    
    def remove_node(self, node_id: str):
        """移除节点"""
        self._all_nodes.discard(node_id)
        self._partitions.pop(node_id, None)
        for other in self._all_nodes:
            self._partitions[other].discard(node_id)
    
    def create_partition(self, group_a: Set[str], group_b: Set[str]):
        """创建网络分区"""
        for node_a in group_a:
            for node_b in group_b:
                if node_a in self._partitions:
                    self._partitions[node_a].discard(node_b)
                if node_b in self._partitions:
                    self._partitions[node_b].discard(node_a)
    
    def heal_partition(self):
        """修复分区"""
        for node in self._all_nodes:
            self._partitions[node] = set(self._all_nodes)
    
    def can_reach(self, from_node: str, to_node: str) -> bool:
        """检查是否可达"""
        return to_node in self._partitions.get(from_node, set())


# ==================== 分区感知的 Runtime ====================

@dataclass
class PartitionAwareRuntime:
    """分区感知的 Runtime"""
    runtime_id: str
    aga: AGA
    device: torch.device
    network: NetworkSimulator
    
    async def send_to(self, target_id: str, message: Dict) -> bool:
        """发送消息到目标节点"""
        if not self.network.can_reach(self.runtime_id, target_id):
            raise ConnectionError(f"Cannot reach {target_id} from {self.runtime_id}")
        return True
    
    async def inject_knowledge(self, record: KnowledgeRecord, slot_idx: int) -> bool:
        """注入知识"""
        self.aga.inject_knowledge(
            slot_idx=slot_idx,
            key_vector=torch.tensor(record.key_vector, device=self.device),
            value_vector=torch.tensor(record.value_vector, device=self.device),
            lu_id=record.lu_id,
            lifecycle_state=record.lifecycle_state,
        )
        return True


# ==================== 测试 ====================

@pytest.mark.fault
@pytest.mark.asyncio
class TestNetworkPartition:
    """网络分区测试"""
    
    @pytest.fixture
    def network(self):
        """创建网络模拟器"""
        return NetworkSimulator()
    
    @pytest.fixture
    def aga_config(self, hidden_dim, bottleneck_dim, num_slots):
        return AGAConfig(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_slots=num_slots,
        )
    
    @pytest.fixture
    def create_runtime(self, aga_config, device, network):
        def _create(runtime_id: str) -> PartitionAwareRuntime:
            aga = AGA(aga_config)
            aga.to(device)
            aga.eval()
            network.add_node(runtime_id)
            return PartitionAwareRuntime(
                runtime_id=runtime_id,
                aga=aga,
                device=device,
                network=network,
            )
        return _create
    
    async def test_partition_detection(self, network, create_runtime):
        """测试分区检测"""
        runtime_0 = create_runtime("runtime_0")
        runtime_1 = create_runtime("runtime_1")
        runtime_2 = create_runtime("runtime_2")
        
        # 初始状态：所有节点可达
        assert network.can_reach("runtime_0", "runtime_1")
        assert network.can_reach("runtime_0", "runtime_2")
        
        # 创建分区：{0, 1} 和 {2}
        network.create_partition({"runtime_0", "runtime_1"}, {"runtime_2"})
        
        # 验证分区
        assert network.can_reach("runtime_0", "runtime_1")
        assert not network.can_reach("runtime_0", "runtime_2")
        assert not network.can_reach("runtime_2", "runtime_0")
    
    async def test_communication_during_partition(self, network, create_runtime):
        """测试分区期间的通信"""
        runtime_0 = create_runtime("runtime_0")
        runtime_1 = create_runtime("runtime_1")
        runtime_2 = create_runtime("runtime_2")
        
        # 创建分区
        network.create_partition({"runtime_0", "runtime_1"}, {"runtime_2"})
        
        # 同分区内通信成功
        await runtime_0.send_to("runtime_1", {"type": "test"})
        
        # 跨分区通信失败
        with pytest.raises(ConnectionError):
            await runtime_0.send_to("runtime_2", {"type": "test"})
    
    async def test_split_brain_scenario(self, network, create_runtime, bottleneck_dim, hidden_dim):
        """测试脑裂场景"""
        runtime_0 = create_runtime("runtime_0")
        runtime_1 = create_runtime("runtime_1")
        runtime_2 = create_runtime("runtime_2")
        
        # 初始同步
        record = KnowledgeRecord(
            lu_id="test_001",
            namespace="default",
            condition="条件",
            decision="决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        for runtime in [runtime_0, runtime_1, runtime_2]:
            await runtime.inject_knowledge(record, 0)
        
        # 创建分区
        network.create_partition({"runtime_0", "runtime_1"}, {"runtime_2"})
        
        # 分区 A 更新状态
        runtime_0.aga.update_lifecycle(0, LifecycleState.CONFIRMED)
        runtime_1.aga.update_lifecycle(0, LifecycleState.CONFIRMED)
        
        # 分区 B 保持原状态
        # runtime_2 仍然是 PROBATIONARY
        
        # 验证脑裂
        assert runtime_0.aga.get_slot_info(0).lifecycle_state == LifecycleState.CONFIRMED
        assert runtime_2.aga.get_slot_info(0).lifecycle_state == LifecycleState.PROBATIONARY
    
    async def test_partition_healing(self, network, create_runtime, bottleneck_dim, hidden_dim):
        """测试分区修复"""
        runtime_0 = create_runtime("runtime_0")
        runtime_1 = create_runtime("runtime_1")
        
        # 创建分区
        network.create_partition({"runtime_0"}, {"runtime_1"})
        
        assert not network.can_reach("runtime_0", "runtime_1")
        
        # 修复分区
        network.heal_partition()
        
        assert network.can_reach("runtime_0", "runtime_1")
        assert network.can_reach("runtime_1", "runtime_0")
    
    async def test_state_reconciliation_after_healing(self, network, create_runtime, bottleneck_dim, hidden_dim):
        """测试分区修复后的状态协调"""
        runtime_0 = create_runtime("runtime_0")
        runtime_1 = create_runtime("runtime_1")
        
        # 初始同步
        record = KnowledgeRecord(
            lu_id="test_001",
            namespace="default",
            condition="条件",
            decision="决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        await runtime_0.inject_knowledge(record, 0)
        await runtime_1.inject_knowledge(record, 0)
        
        # 创建分区并独立更新
        network.create_partition({"runtime_0"}, {"runtime_1"})
        
        runtime_0.aga.update_lifecycle(0, LifecycleState.CONFIRMED)
        # runtime_1 保持 PROBATIONARY
        
        # 修复分区
        network.heal_partition()
        
        # 协调状态（使用最新状态）
        # 这里假设 CONFIRMED 优先级更高
        state_0 = runtime_0.aga.get_slot_info(0).lifecycle_state
        state_1 = runtime_1.aga.get_slot_info(0).lifecycle_state
        
        # 选择优先级更高的状态
        if state_0 == LifecycleState.CONFIRMED:
            runtime_1.aga.update_lifecycle(0, LifecycleState.CONFIRMED)
        
        # 验证一致性
        assert runtime_0.aga.get_slot_info(0).lifecycle_state == runtime_1.aga.get_slot_info(0).lifecycle_state


@pytest.mark.fault
@pytest.mark.asyncio
class TestPartialConnectivity:
    """部分连接测试"""
    
    @pytest.fixture
    def network(self):
        return NetworkSimulator()
    
    async def test_asymmetric_partition(self, network):
        """测试非对称分区"""
        network.add_node("A")
        network.add_node("B")
        network.add_node("C")
        
        # 创建非对称分区：A 可以到达 B，但 B 不能到达 A
        network._partitions["A"] = {"A", "B"}
        network._partitions["B"] = {"B", "C"}
        network._partitions["C"] = {"C"}
        
        assert network.can_reach("A", "B")
        assert not network.can_reach("B", "A")
        assert network.can_reach("B", "C")
        assert not network.can_reach("C", "B")
    
    async def test_transitive_unreachability(self, network):
        """测试传递性不可达"""
        network.add_node("A")
        network.add_node("B")
        network.add_node("C")
        
        # A -> B -> C，但 A 不能直接到达 C
        network._partitions["A"] = {"A", "B"}
        network._partitions["B"] = {"A", "B", "C"}
        network._partitions["C"] = {"B", "C"}
        
        assert network.can_reach("A", "B")
        assert network.can_reach("B", "C")
        assert not network.can_reach("A", "C")
