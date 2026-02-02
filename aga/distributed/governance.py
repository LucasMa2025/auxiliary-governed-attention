"""
AGA 分布式治理模块（参考实现）

⚠️ 重要说明：
本模块提供治理逻辑的**参考实现**，用于演示和原型验证。

在生产环境中，建议将治理逻辑外置到您自己的系统中：
- 持续自学习系统
- 知识管理平台
- 人工审批系统

AGA 的核心定位是"热插拔知识管理器"，不是治理系统。
治理决策应由外部系统做出，AGA 只执行决策结果。

本模块解决的问题（供参考）：
1. 错误传播速度 > 隔离速度 → PropagationThrottler
2. 缺少治理节流阀 → GovernanceArbiter
3. 治理指令应"少数即生效" → quorum 机制
4. 语义主权分区 → TrustTier

版本: v3.1

使用方式：
- 作为外部治理系统的设计参考
- 用于原型验证和快速测试
- 在正式生产前迁移到外部系统
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Set
import logging

from ..types import LifecycleState

logger = logging.getLogger(__name__)


# ==================== 语义主权分区 ====================

class TrustTier(str, Enum):
    """
    信任层级（语义主权分区）
    
    解决问题：Slot 的价值不是"局部的、短期的、可替换的"
    """
    S0_ACCELERATION = "s0_acceleration"  # 推理加速槽：可丢、可重建
    S1_EXPERIENCE = "s1_experience"      # 经验槽：可回滚
    S2_POLICY = "s2_policy"              # 策略槽：需审批
    S3_IMMUTABLE = "s3_immutable"        # 禁止传播槽：只读


class PropagationPolicy(str, Enum):
    """传播策略"""
    IMMEDIATE = "immediate"      # 立即传播
    DELAYED = "delayed"          # 延迟传播（观察期）
    GATED = "gated"              # 门控传播（需达到阈值）
    BLOCKED = "blocked"          # 禁止传播


# 信任层级到传播策略的默认映射
TIER_PROPAGATION_POLICY: Dict[TrustTier, PropagationPolicy] = {
    TrustTier.S0_ACCELERATION: PropagationPolicy.IMMEDIATE,
    TrustTier.S1_EXPERIENCE: PropagationPolicy.DELAYED,
    TrustTier.S2_POLICY: PropagationPolicy.GATED,
    TrustTier.S3_IMMUTABLE: PropagationPolicy.BLOCKED,
}

# 信任层级到传播半径的默认映射（最大实例数）
TIER_PROPAGATION_RADIUS: Dict[TrustTier, int] = {
    TrustTier.S0_ACCELERATION: -1,  # 无限制
    TrustTier.S1_EXPERIENCE: 10,    # 最多 10 个实例
    TrustTier.S2_POLICY: 3,         # 最多 3 个实例
    TrustTier.S3_IMMUTABLE: 0,      # 不传播
}


@dataclass
class SlotTrustInfo:
    """
    槽位信任信息
    
    用于治理决策。
    """
    lu_id: str
    trust_tier: TrustTier
    propagation_policy: PropagationPolicy
    propagation_radius: int
    
    # 传播状态
    propagated_to: Set[str] = field(default_factory=set)  # 已传播的实例 ID
    propagation_count: int = 0
    
    # 观察期（用于 DELAYED 策略）
    created_at: float = field(default_factory=time.time)
    observation_period_seconds: int = 300  # 5 分钟观察期
    
    # 门控阈值（用于 GATED 策略）
    approval_count: int = 0
    approval_threshold: int = 2  # 需要 2 个审批
    
    # 质量指标
    hit_count: int = 0
    error_count: int = 0
    
    @property
    def can_propagate(self) -> bool:
        """是否可以传播"""
        if self.propagation_policy == PropagationPolicy.BLOCKED:
            return False
        
        if self.propagation_policy == PropagationPolicy.DELAYED:
            elapsed = time.time() - self.created_at
            if elapsed < self.observation_period_seconds:
                return False
        
        if self.propagation_policy == PropagationPolicy.GATED:
            if self.approval_count < self.approval_threshold:
                return False
        
        # 检查传播半径
        if self.propagation_radius >= 0:
            if self.propagation_count >= self.propagation_radius:
                return False
        
        return True
    
    @property
    def quality_score(self) -> float:
        """质量分数"""
        if self.hit_count == 0:
            return 0.0
        return self.hit_count / (self.hit_count + self.error_count + 1)


# ==================== 治理裁决器 ====================

class GovernanceVerdict(str, Enum):
    """治理裁决结果"""
    ALLOW = "allow"              # 允许
    DENY = "deny"                # 拒绝
    DEFER = "defer"              # 延迟决策
    QUARANTINE = "quarantine"    # 立即隔离


@dataclass
class GovernanceDecision:
    """治理决策"""
    verdict: GovernanceVerdict
    reason: str
    timestamp: float = field(default_factory=time.time)
    
    # 决策来源
    source_instance: Optional[str] = None
    is_quorum_decision: bool = False
    quorum_votes: int = 0
    
    # 附加操作
    propagation_blocked: bool = False
    rollback_required: bool = False


class GovernanceArbiter:
    """
    治理裁决器
    
    核心原则：
    1. 治理指令"少数即生效"（quorum 机制）
    2. 默认失败 = 拒绝传播
    3. 风险等级触发强制旁路
    """
    
    def __init__(
        self,
        instance_id: str,
        quorum_size: int = 2,  # 最小 quorum 大小
        risk_threshold: float = 0.3,  # 风险阈值
    ):
        """
        初始化裁决器
        
        Args:
            instance_id: 本实例 ID
            quorum_size: quorum 大小（少数即生效）
            risk_threshold: 风险阈值
        """
        self.instance_id = instance_id
        self.quorum_size = quorum_size
        self.risk_threshold = risk_threshold
        
        # 槽位信任信息
        self._trust_info: Dict[str, SlotTrustInfo] = {}
        
        # 待决投票
        self._pending_votes: Dict[str, List[GovernanceDecision]] = {}
        
        # 黑名单（快速拒绝）
        self._blacklist: Set[str] = set()
        
        # 回调
        self._on_quarantine: Optional[Callable] = None
        self._on_propagation_blocked: Optional[Callable] = None
    
    def register_slot(
        self,
        lu_id: str,
        trust_tier: TrustTier = TrustTier.S1_EXPERIENCE,
        propagation_policy: Optional[PropagationPolicy] = None,
        propagation_radius: Optional[int] = None,
    ) -> SlotTrustInfo:
        """
        注册槽位信任信息
        
        Args:
            lu_id: 知识单元 ID
            trust_tier: 信任层级
            propagation_policy: 传播策略（默认根据层级）
            propagation_radius: 传播半径（默认根据层级）
        """
        policy = propagation_policy or TIER_PROPAGATION_POLICY[trust_tier]
        radius = propagation_radius if propagation_radius is not None else TIER_PROPAGATION_RADIUS[trust_tier]
        
        info = SlotTrustInfo(
            lu_id=lu_id,
            trust_tier=trust_tier,
            propagation_policy=policy,
            propagation_radius=radius,
        )
        
        self._trust_info[lu_id] = info
        return info
    
    def get_trust_info(self, lu_id: str) -> Optional[SlotTrustInfo]:
        """获取槽位信任信息"""
        return self._trust_info.get(lu_id)
    
    async def evaluate_propagation(
        self,
        lu_id: str,
        target_instance: str,
    ) -> GovernanceDecision:
        """
        评估是否允许传播到目标实例
        
        核心原则：默认拒绝传播
        """
        # 检查黑名单
        if lu_id in self._blacklist:
            return GovernanceDecision(
                verdict=GovernanceVerdict.DENY,
                reason="Blacklisted",
                propagation_blocked=True,
            )
        
        # 获取信任信息
        info = self._trust_info.get(lu_id)
        if not info:
            # 未注册的槽位默认拒绝
            return GovernanceDecision(
                verdict=GovernanceVerdict.DENY,
                reason="Unregistered slot - default deny",
                propagation_blocked=True,
            )
        
        # 检查是否已传播到该实例
        if target_instance in info.propagated_to:
            return GovernanceDecision(
                verdict=GovernanceVerdict.ALLOW,
                reason="Already propagated",
            )
        
        # 检查传播策略
        if not info.can_propagate:
            if info.propagation_policy == PropagationPolicy.BLOCKED:
                reason = "Propagation blocked by policy"
            elif info.propagation_policy == PropagationPolicy.DELAYED:
                remaining = info.observation_period_seconds - (time.time() - info.created_at)
                reason = f"In observation period, {remaining:.0f}s remaining"
            elif info.propagation_policy == PropagationPolicy.GATED:
                reason = f"Needs {info.approval_threshold - info.approval_count} more approvals"
            else:
                reason = f"Propagation radius exceeded ({info.propagation_count}/{info.propagation_radius})"
            
            return GovernanceDecision(
                verdict=GovernanceVerdict.DEFER,
                reason=reason,
                propagation_blocked=True,
            )
        
        # 检查质量分数
        if info.quality_score < self.risk_threshold and info.hit_count > 10:
            return GovernanceDecision(
                verdict=GovernanceVerdict.DENY,
                reason=f"Quality score too low: {info.quality_score:.2f}",
                propagation_blocked=True,
            )
        
        # 允许传播
        info.propagated_to.add(target_instance)
        info.propagation_count += 1
        
        return GovernanceDecision(
            verdict=GovernanceVerdict.ALLOW,
            reason="Propagation allowed",
        )
    
    async def evaluate_quarantine(
        self,
        lu_id: str,
        reason: str,
        source_instance: str,
    ) -> GovernanceDecision:
        """
        评估隔离请求
        
        核心原则：少数即生效（quorum）
        """
        # 添加投票
        if lu_id not in self._pending_votes:
            self._pending_votes[lu_id] = []
        
        vote = GovernanceDecision(
            verdict=GovernanceVerdict.QUARANTINE,
            reason=reason,
            source_instance=source_instance,
        )
        self._pending_votes[lu_id].append(vote)
        
        # 检查是否达到 quorum
        votes = self._pending_votes[lu_id]
        if len(votes) >= self.quorum_size:
            # 达到 quorum，立即生效
            self._blacklist.add(lu_id)
            
            decision = GovernanceDecision(
                verdict=GovernanceVerdict.QUARANTINE,
                reason=f"Quorum reached: {len(votes)} votes",
                is_quorum_decision=True,
                quorum_votes=len(votes),
                rollback_required=True,
            )
            
            # 触发回调
            if self._on_quarantine:
                await self._on_quarantine(lu_id, decision)
            
            # 清理投票
            del self._pending_votes[lu_id]
            
            return decision
        
        # 未达到 quorum，但单实例也可以触发（风险优先）
        # 这实现了"少数即生效"的原则
        if source_instance == self.instance_id:
            # 本实例发起的隔离请求，立即生效
            self._blacklist.add(lu_id)
            
            return GovernanceDecision(
                verdict=GovernanceVerdict.QUARANTINE,
                reason=f"Local quarantine: {reason}",
                is_quorum_decision=False,
                quorum_votes=1,
            )
        
        return GovernanceDecision(
            verdict=GovernanceVerdict.DEFER,
            reason=f"Waiting for quorum: {len(votes)}/{self.quorum_size}",
        )
    
    def approve_propagation(self, lu_id: str, approver: str) -> bool:
        """
        审批传播（用于 GATED 策略）
        
        Args:
            lu_id: 知识单元 ID
            approver: 审批者 ID
        
        Returns:
            是否达到审批阈值
        """
        info = self._trust_info.get(lu_id)
        if not info:
            return False
        
        info.approval_count += 1
        logger.info(f"Propagation approved: lu_id={lu_id}, approver={approver}, "
                   f"count={info.approval_count}/{info.approval_threshold}")
        
        return info.approval_count >= info.approval_threshold
    
    def record_hit(self, lu_id: str):
        """记录命中"""
        info = self._trust_info.get(lu_id)
        if info:
            info.hit_count += 1
    
    def record_error(self, lu_id: str):
        """记录错误"""
        info = self._trust_info.get(lu_id)
        if info:
            info.error_count += 1
    
    def on_quarantine(self, callback: Callable):
        """注册隔离回调"""
        self._on_quarantine = callback
    
    def on_propagation_blocked(self, callback: Callable):
        """注册传播阻止回调"""
        self._on_propagation_blocked = callback
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        tier_counts = {}
        for tier in TrustTier:
            tier_counts[tier.value] = sum(
                1 for info in self._trust_info.values()
                if info.trust_tier == tier
            )
        
        return {
            "total_slots": len(self._trust_info),
            "blacklisted": len(self._blacklist),
            "pending_votes": len(self._pending_votes),
            "tier_distribution": tier_counts,
            "quorum_size": self.quorum_size,
            "risk_threshold": self.risk_threshold,
        }


# ==================== 传播节流器 ====================

class PropagationThrottler:
    """
    传播节流器
    
    解决问题：错误传播速度 > 隔离速度
    
    策略：
    1. 新知识默认延迟传播
    2. 传播速率限制
    3. 回流延迟
    """
    
    def __init__(
        self,
        default_delay_seconds: int = 60,  # 默认延迟 1 分钟
        max_propagation_rate: int = 10,   # 每分钟最多传播 10 个
        learning_backflow_delay: int = 300,  # 学习回流延迟 5 分钟
    ):
        """
        初始化节流器
        
        Args:
            default_delay_seconds: 默认传播延迟
            max_propagation_rate: 最大传播速率（每分钟）
            learning_backflow_delay: 学习回流延迟
        """
        self.default_delay = default_delay_seconds
        self.max_rate = max_propagation_rate
        self.backflow_delay = learning_backflow_delay
        
        # 传播队列
        self._pending_propagations: List[Dict[str, Any]] = []
        
        # 速率计数
        self._propagation_count = 0
        self._last_reset_time = time.time()
        
        # 回流队列
        self._backflow_queue: List[Dict[str, Any]] = []
    
    async def schedule_propagation(
        self,
        lu_id: str,
        target_instances: List[str],
        delay_seconds: Optional[int] = None,
    ) -> bool:
        """
        调度传播
        
        Args:
            lu_id: 知识单元 ID
            target_instances: 目标实例列表
            delay_seconds: 延迟秒数
        
        Returns:
            是否成功调度
        """
        delay = delay_seconds if delay_seconds is not None else self.default_delay
        
        propagation = {
            "lu_id": lu_id,
            "targets": target_instances,
            "scheduled_at": time.time(),
            "execute_at": time.time() + delay,
        }
        
        self._pending_propagations.append(propagation)
        logger.info(f"Propagation scheduled: lu_id={lu_id}, delay={delay}s, targets={len(target_instances)}")
        
        return True
    
    async def process_pending(self) -> List[Dict[str, Any]]:
        """
        处理待传播队列
        
        Returns:
            可以传播的项目列表
        """
        now = time.time()
        
        # 重置速率计数
        if now - self._last_reset_time >= 60:
            self._propagation_count = 0
            self._last_reset_time = now
        
        # 检查速率限制
        if self._propagation_count >= self.max_rate:
            return []
        
        # 找出可以执行的传播
        ready = []
        remaining = []
        
        for prop in self._pending_propagations:
            if prop["execute_at"] <= now:
                if self._propagation_count < self.max_rate:
                    ready.append(prop)
                    self._propagation_count += 1
                else:
                    remaining.append(prop)
            else:
                remaining.append(prop)
        
        self._pending_propagations = remaining
        return ready
    
    async def schedule_backflow(
        self,
        lu_id: str,
        learning_data: Dict[str, Any],
    ):
        """
        调度学习回流
        
        学习回流有额外延迟，防止"学错东西"快速扩散
        """
        backflow = {
            "lu_id": lu_id,
            "data": learning_data,
            "scheduled_at": time.time(),
            "execute_at": time.time() + self.backflow_delay,
        }
        
        self._backflow_queue.append(backflow)
        logger.info(f"Backflow scheduled: lu_id={lu_id}, delay={self.backflow_delay}s")
    
    async def process_backflow(self) -> List[Dict[str, Any]]:
        """处理学习回流队列"""
        now = time.time()
        
        ready = []
        remaining = []
        
        for bf in self._backflow_queue:
            if bf["execute_at"] <= now:
                ready.append(bf)
            else:
                remaining.append(bf)
        
        self._backflow_queue = remaining
        return ready
    
    def cancel_propagation(self, lu_id: str) -> int:
        """
        取消待传播
        
        Returns:
            取消的数量
        """
        before = len(self._pending_propagations)
        self._pending_propagations = [
            p for p in self._pending_propagations
            if p["lu_id"] != lu_id
        ]
        cancelled = before - len(self._pending_propagations)
        
        if cancelled > 0:
            logger.info(f"Propagation cancelled: lu_id={lu_id}, count={cancelled}")
        
        return cancelled
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "pending_propagations": len(self._pending_propagations),
            "pending_backflows": len(self._backflow_queue),
            "current_rate": self._propagation_count,
            "max_rate": self.max_rate,
            "default_delay": self.default_delay,
            "backflow_delay": self.backflow_delay,
        }
