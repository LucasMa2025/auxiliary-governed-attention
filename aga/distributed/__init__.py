"""
AGA 分布式支持模块

提供多实例部署的同步和协调功能：
- 分布式同步器
- 实例协调器
- 分布式锁
- 治理裁决器（v3.1 新增）
- 传播节流器（v3.1 新增）

版本: v3.1

核心设计原则（基于问题分析）：
1. 治理指令"少数即生效"（quorum 机制）
2. 默认失败 = 拒绝传播
3. 错误传播速度必须 < 隔离速度
4. 语义主权分区（按风险等级分片）
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
]
