"""
分布式治理模块单元测试

测试 TrustTier, GovernanceArbiter, PropagationThrottler 等核心功能。
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from aga.distributed.governance import (
    TrustTier,
    PropagationPolicy,
    SlotTrustInfo,
    GovernanceVerdict,
    GovernanceDecision,
    GovernanceArbiter,
    PropagationThrottler,
    TIER_PROPAGATION_POLICY,
    TIER_PROPAGATION_RADIUS,
)


@pytest.mark.unit
class TestTrustTier:
    """TrustTier 测试"""
    
    def test_all_tiers(self):
        """测试所有信任层级"""
        assert TrustTier.S0_ACCELERATION.value == "s0_acceleration"
        assert TrustTier.S1_EXPERIENCE.value == "s1_experience"
        assert TrustTier.S2_POLICY.value == "s2_policy"
        assert TrustTier.S3_IMMUTABLE.value == "s3_immutable"
    
    def test_tier_propagation_policy_mapping(self):
        """测试层级到传播策略的映射"""
        assert TIER_PROPAGATION_POLICY[TrustTier.S0_ACCELERATION] == PropagationPolicy.IMMEDIATE
        assert TIER_PROPAGATION_POLICY[TrustTier.S1_EXPERIENCE] == PropagationPolicy.DELAYED
        assert TIER_PROPAGATION_POLICY[TrustTier.S2_POLICY] == PropagationPolicy.GATED
        assert TIER_PROPAGATION_POLICY[TrustTier.S3_IMMUTABLE] == PropagationPolicy.BLOCKED
    
    def test_tier_propagation_radius_mapping(self):
        """测试层级到传播半径的映射"""
        assert TIER_PROPAGATION_RADIUS[TrustTier.S0_ACCELERATION] == -1  # 无限制
        assert TIER_PROPAGATION_RADIUS[TrustTier.S1_EXPERIENCE] == 10
        assert TIER_PROPAGATION_RADIUS[TrustTier.S2_POLICY] == 3
        assert TIER_PROPAGATION_RADIUS[TrustTier.S3_IMMUTABLE] == 0


@pytest.mark.unit
class TestSlotTrustInfo:
    """SlotTrustInfo 测试"""
    
    def test_create_trust_info(self):
        """测试创建信任信息"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S1_EXPERIENCE,
            propagation_policy=PropagationPolicy.DELAYED,
            propagation_radius=10,
        )
        
        assert info.lu_id == "LU_001"
        assert info.trust_tier == TrustTier.S1_EXPERIENCE
        assert info.propagation_policy == PropagationPolicy.DELAYED
    
    def test_can_propagate_immediate(self):
        """测试立即传播策略"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S0_ACCELERATION,
            propagation_policy=PropagationPolicy.IMMEDIATE,
            propagation_radius=-1,
        )
        
        assert info.can_propagate is True
    
    def test_can_propagate_blocked(self):
        """测试阻止传播策略"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S3_IMMUTABLE,
            propagation_policy=PropagationPolicy.BLOCKED,
            propagation_radius=0,
        )
        
        assert info.can_propagate is False
    
    def test_can_propagate_delayed_not_ready(self):
        """测试延迟传播策略 - 未到期"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S1_EXPERIENCE,
            propagation_policy=PropagationPolicy.DELAYED,
            propagation_radius=10,
            observation_period_seconds=300,
        )
        
        # 刚创建，还在观察期
        assert info.can_propagate is False
    
    def test_can_propagate_delayed_ready(self):
        """测试延迟传播策略 - 已到期"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S1_EXPERIENCE,
            propagation_policy=PropagationPolicy.DELAYED,
            propagation_radius=10,
            observation_period_seconds=0,  # 无观察期
        )
        info.created_at = time.time() - 1  # 1秒前创建
        
        assert info.can_propagate is True
    
    def test_can_propagate_gated_not_approved(self):
        """测试门控传播策略 - 未审批"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S2_POLICY,
            propagation_policy=PropagationPolicy.GATED,
            propagation_radius=3,
            approval_count=0,
            approval_threshold=2,
        )
        
        assert info.can_propagate is False
    
    def test_can_propagate_gated_approved(self):
        """测试门控传播策略 - 已审批"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S2_POLICY,
            propagation_policy=PropagationPolicy.GATED,
            propagation_radius=3,
            approval_count=2,
            approval_threshold=2,
        )
        
        assert info.can_propagate is True
    
    def test_can_propagate_radius_exceeded(self):
        """测试传播半径超限"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S1_EXPERIENCE,
            propagation_policy=PropagationPolicy.IMMEDIATE,
            propagation_radius=3,
            propagation_count=3,
        )
        
        assert info.can_propagate is False
    
    def test_quality_score(self):
        """测试质量分数"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S0_ACCELERATION,
            propagation_policy=PropagationPolicy.IMMEDIATE,
            propagation_radius=-1,
            hit_count=10,
            error_count=0,
        )
        
        # 10 / (10 + 0 + 1) = 0.909...
        assert info.quality_score > 0.9
    
    def test_quality_score_zero_hits(self):
        """测试零命中的质量分数"""
        info = SlotTrustInfo(
            lu_id="LU_001",
            trust_tier=TrustTier.S0_ACCELERATION,
            propagation_policy=PropagationPolicy.IMMEDIATE,
            propagation_radius=-1,
            hit_count=0,
            error_count=0,
        )
        
        assert info.quality_score == 0.0


@pytest.mark.unit
class TestGovernanceDecision:
    """GovernanceDecision 测试"""
    
    def test_create_decision(self):
        """测试创建决策"""
        decision = GovernanceDecision(
            verdict=GovernanceVerdict.ALLOW,
            reason="通过审批",
        )
        
        assert decision.verdict == GovernanceVerdict.ALLOW
        assert decision.reason == "通过审批"
        assert decision.timestamp > 0
    
    def test_decision_with_quorum(self):
        """测试带 quorum 的决策"""
        decision = GovernanceDecision(
            verdict=GovernanceVerdict.QUARANTINE,
            reason="检测到异常",
            source_instance="instance-001",
            is_quorum_decision=True,
            quorum_votes=3,
        )
        
        assert decision.is_quorum_decision is True
        assert decision.quorum_votes == 3


@pytest.mark.unit
class TestGovernanceArbiter:
    """GovernanceArbiter 测试"""
    
    @pytest.fixture
    def arbiter(self):
        """创建裁决器"""
        return GovernanceArbiter(
            instance_id="instance-001",
            quorum_size=2,
            risk_threshold=0.3,
        )
    
    def test_create_arbiter(self, arbiter):
        """测试创建裁决器"""
        assert arbiter.instance_id == "instance-001"
        assert arbiter.quorum_size == 2
        assert arbiter.risk_threshold == 0.3
    
    def test_register_slot(self, arbiter):
        """测试注册槽位"""
        arbiter.register_slot(
            lu_id="LU_001",
            trust_tier=TrustTier.S1_EXPERIENCE,
        )
        
        assert "LU_001" in arbiter._trust_info
        info = arbiter._trust_info["LU_001"]
        assert info.trust_tier == TrustTier.S1_EXPERIENCE
    
    def test_register_slot_custom_policy(self, arbiter):
        """测试自定义策略注册槽位"""
        arbiter.register_slot(
            lu_id="LU_001",
            trust_tier=TrustTier.S1_EXPERIENCE,
            propagation_policy=PropagationPolicy.IMMEDIATE,
            propagation_radius=5,
        )
        
        info = arbiter._trust_info["LU_001"]
        assert info.propagation_policy == PropagationPolicy.IMMEDIATE
        assert info.propagation_radius == 5
    
    def test_get_trust_info(self, arbiter):
        """测试获取信任信息"""
        arbiter.register_slot(
            lu_id="LU_001",
            trust_tier=TrustTier.S1_EXPERIENCE,
        )
        
        info = arbiter.get_trust_info("LU_001")
        assert info is not None
        assert info.lu_id == "LU_001"
    
    def test_get_trust_info_nonexistent(self, arbiter):
        """测试获取不存在的信任信息"""
        info = arbiter.get_trust_info("NONEXISTENT")
        assert info is None
    
    @pytest.mark.asyncio
    async def test_evaluate_propagation_unknown(self, arbiter):
        """测试评估传播 - 未知槽位"""
        decision = await arbiter.evaluate_propagation("LU_UNKNOWN", "instance-002")
        
        # 未知槽位默认拒绝
        assert decision.verdict == GovernanceVerdict.DENY
    
    @pytest.mark.asyncio
    async def test_evaluate_propagation_allowed(self, arbiter):
        """测试评估传播 - 允许"""
        arbiter.register_slot(
            lu_id="LU_001",
            trust_tier=TrustTier.S0_ACCELERATION,
        )
        
        decision = await arbiter.evaluate_propagation("LU_001", "instance-002")
        
        assert decision.verdict == GovernanceVerdict.ALLOW
    
    @pytest.mark.asyncio
    async def test_evaluate_propagation_blocked_policy(self, arbiter):
        """测试评估传播 - 策略阻止"""
        arbiter.register_slot(
            lu_id="LU_001",
            trust_tier=TrustTier.S3_IMMUTABLE,
        )
        
        decision = await arbiter.evaluate_propagation("LU_001", "instance-002")
        
        # S3_IMMUTABLE 策略是 BLOCKED
        assert decision.verdict == GovernanceVerdict.DEFER
        assert decision.propagation_blocked is True
    
    def test_approve_propagation(self, arbiter):
        """测试审批传播"""
        arbiter.register_slot(
            lu_id="LU_001",
            trust_tier=TrustTier.S2_POLICY,
        )
        
        # 第一次审批
        result1 = arbiter.approve_propagation("LU_001", "approver-1")
        assert result1 is False  # 还未达到阈值
        
        # 第二次审批
        result2 = arbiter.approve_propagation("LU_001", "approver-2")
        assert result2 is True  # 达到阈值
    
    def test_record_hit(self, arbiter):
        """测试记录命中"""
        arbiter.register_slot(
            lu_id="LU_001",
            trust_tier=TrustTier.S0_ACCELERATION,
        )
        
        arbiter.record_hit("LU_001")
        
        info = arbiter._trust_info["LU_001"]
        assert info.hit_count == 1
    
    def test_record_error(self, arbiter):
        """测试记录错误"""
        arbiter.register_slot(
            lu_id="LU_001",
            trust_tier=TrustTier.S0_ACCELERATION,
        )
        
        arbiter.record_error("LU_001")
        
        info = arbiter._trust_info["LU_001"]
        assert info.error_count == 1
    
    def test_get_statistics(self, arbiter):
        """测试获取统计"""
        arbiter.register_slot("LU_001", TrustTier.S0_ACCELERATION)
        arbiter.register_slot("LU_002", TrustTier.S1_EXPERIENCE)
        
        stats = arbiter.get_statistics()
        
        assert stats["total_slots"] == 2
        assert stats["quorum_size"] == 2


@pytest.mark.unit
class TestPropagationThrottler:
    """PropagationThrottler 测试"""
    
    @pytest.fixture
    def throttler(self):
        """创建节流器"""
        return PropagationThrottler(
            default_delay_seconds=1,
            max_propagation_rate=10,
            learning_backflow_delay=5,
        )
    
    def test_create_throttler(self, throttler):
        """测试创建节流器"""
        assert throttler.default_delay == 1
        assert throttler.max_rate == 10
        assert throttler.backflow_delay == 5
    
    @pytest.mark.asyncio
    async def test_schedule_propagation(self, throttler):
        """测试调度传播"""
        result = await throttler.schedule_propagation(
            lu_id="LU_001",
            target_instances=["instance-002", "instance-003"],
            delay_seconds=0,
        )
        
        assert result is True
        assert len(throttler._pending_propagations) == 1
    
    @pytest.mark.asyncio
    async def test_process_pending_empty(self, throttler):
        """测试处理空队列"""
        ready = await throttler.process_pending()
        assert ready == []
    
    @pytest.mark.asyncio
    async def test_process_pending_ready(self, throttler):
        """测试处理就绪的传播"""
        # 调度一个立即执行的传播
        await throttler.schedule_propagation(
            lu_id="LU_001",
            target_instances=["instance-002"],
            delay_seconds=0,
        )
        
        # 等待一小段时间确保到期
        await asyncio.sleep(0.01)
        
        ready = await throttler.process_pending()
        
        assert len(ready) == 1
        assert ready[0]["lu_id"] == "LU_001"
    
    @pytest.mark.asyncio
    async def test_process_pending_delayed(self, throttler):
        """测试处理延迟的传播"""
        # 调度一个延迟执行的传播
        await throttler.schedule_propagation(
            lu_id="LU_001",
            target_instances=["instance-002"],
            delay_seconds=10,  # 10秒后执行
        )
        
        ready = await throttler.process_pending()
        
        # 还没到时间，不应该返回
        assert len(ready) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, throttler):
        """测试速率限制"""
        # 创建一个低速率限制的节流器
        limited_throttler = PropagationThrottler(
            default_delay_seconds=0,
            max_propagation_rate=2,
            learning_backflow_delay=5,
        )
        
        # 调度多个传播
        for i in range(5):
            await limited_throttler.schedule_propagation(
                lu_id=f"LU_{i:03d}",
                target_instances=["instance-002"],
                delay_seconds=0,
            )
        
        await asyncio.sleep(0.01)
        
        # 第一次处理应该只返回 2 个（速率限制）
        ready = await limited_throttler.process_pending()
        assert len(ready) == 2
