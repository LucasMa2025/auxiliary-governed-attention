"""
AGA 分布式支持模块

提供多实例部署的同步和协调功能：
- 分布式同步器
- 实例协调器
- 分布式锁
- 治理裁决器（v3.1 新增）
- 传播节流器（v3.1 新增）
- 网络分区处理（v3.4.1 新增）
- 一致性保证增强（v3.4.1 新增）

版本: v3.4.1

核心设计原则（基于问题分析）：
1. 治理指令"少数即生效"（quorum 机制）
2. 默认失败 = 拒绝传播
3. 错误传播速度必须 < 隔离速度
4. 语义主权分区（按风险等级分片）
5. CAP 定理下选择 AP（可用性 + 分区容错）
6. 向量时钟实现因果一致性
"""
from .sync import DistributedSynchronizer, SyncMessage, MessageType
from .coordinator import InstanceCoordinator, InstanceInfo
from .lock import DistributedLock, LockManager
from .governance import (
    # 信任层级
    TrustTier,
    PropagationPolicy,
    SlotTrustInfo,
    # 治理裁决
    GovernanceVerdict,
    GovernanceDecision,
    GovernanceArbiter,
    # 传播节流
    PropagationThrottler,
    # 常量
    TIER_PROPAGATION_POLICY,
    TIER_PROPAGATION_RADIUS,
)
from .partition import (
    # 一致性级别
    ConsistencyLevel,
    PartitionState,
    ConflictResolution,
    # 向量时钟
    VectorClock,
    VersionedValue,
    # 分区处理
    PartitionEvent,
    PartitionDetector,
    ConsistencyManager,
    PartitionRecoveryManager,
)

__all__ = [
    # 同步
    "DistributedSynchronizer",
    "SyncMessage",
    "MessageType",
    # 协调
    "InstanceCoordinator",
    "InstanceInfo",
    # 锁
    "DistributedLock",
    "LockManager",
    # 治理（v3.1）
    "TrustTier",
    "PropagationPolicy",
    "SlotTrustInfo",
    "GovernanceVerdict",
    "GovernanceDecision",
    "GovernanceArbiter",
    "PropagationThrottler",
    "TIER_PROPAGATION_POLICY",
    "TIER_PROPAGATION_RADIUS",
    # 分区与一致性（v3.4.1）
    "ConsistencyLevel",
    "PartitionState",
    "ConflictResolution",
    "VectorClock",
    "VersionedValue",
    "PartitionEvent",
    "PartitionDetector",
    "ConsistencyManager",
    "PartitionRecoveryManager",
]
