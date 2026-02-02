"""
AGA 分布式功能演示

展示分布式同步和内部治理功能。
"""
import asyncio
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def demo_distributed_sync():
    """分布式同步演示"""
    print("\n" + "=" * 60)
    print("Demo 1: 分布式同步器")
    print("=" * 60)
    
    from aga.distributed import (
        DistributedSynchronizer,
        MessageType,
        SyncMessage,
    )
    from aga import LifecycleState
    
    # 创建本地同步器（用于测试）
    sync = DistributedSynchronizer(
        instance_id="demo-instance-1",
        namespace="demo",
        backend="local",  # 本地模式，无需 Redis/Kafka
    )
    
    print("创建本地同步器")
    
    # 注册消息处理器
    @sync.register_handler
    def on_knowledge_inject(message: SyncMessage):
        print(f"  收到知识注入消息: {message.lu_id}")
    
    # 启动
    await sync.start()
    print("同步器已启动")
    
    # 发送同步消息
    print("\n发送同步消息:")
    
    await sync.sync_knowledge_inject(
        lu_id="LU_SYNC_001",
        slot_idx=0,
        key_vector=[0.1, 0.2, 0.3],
        value_vector=[0.4, 0.5, 0.6],
        lifecycle_state=LifecycleState.PROBATIONARY,
        condition="test condition",
        decision="test decision",
    )
    print("  发送: KNOWLEDGE_INJECT")
    
    await sync.sync_lifecycle_update(
        lu_id="LU_SYNC_001",
        new_state=LifecycleState.CONFIRMED,
    )
    print("  发送: LIFECYCLE_UPDATE")
    
    await sync.sync_quarantine("LU_SYNC_001")
    print("  发送: QUARANTINE")
    
    # 停止
    await sync.stop()
    print("\n同步器已停止")


async def demo_governance():
    """内部治理演示"""
    print("\n" + "=" * 60)
    print("Demo 2: 内部治理（参考实现）")
    print("=" * 60)
    
    from aga.distributed import (
        GovernanceArbiter,
        PropagationThrottler,
        TrustTier,
        PropagationPolicy,
        GovernanceVerdict,
    )
    
    # 创建治理裁决器
    arbiter = GovernanceArbiter(
        instance_id="demo-instance",
        quorum_size=2,
        risk_threshold=0.3,
    )
    
    print("创建治理裁决器")
    print(f"  - quorum_size: 2 (少数即生效)")
    print(f"  - risk_threshold: 0.3")
    
    # 注册不同信任层级的槽位
    print("\n注册槽位信任信息:")
    
    slots = [
        ("LU_CACHE_001", TrustTier.S0_ACCELERATION, "加速槽（可丢失）"),
        ("LU_EXP_001", TrustTier.S1_EXPERIENCE, "经验槽（可回滚）"),
        ("LU_POLICY_001", TrustTier.S2_POLICY, "策略槽（需审批）"),
        ("LU_IMMUTABLE_001", TrustTier.S3_IMMUTABLE, "禁止槽（只读）"),
    ]
    
    for lu_id, tier, desc in slots:
        info = arbiter.register_slot(lu_id, tier)
        print(f"  {lu_id}: {tier.value} - {desc}")
    
    # 评估传播
    print("\n传播评估:")
    
    for lu_id, _, _ in slots:
        decision = await arbiter.evaluate_propagation(lu_id, "target-instance")
        status = "✓" if decision.verdict == GovernanceVerdict.ALLOW else "✗"
        print(f"  {status} {lu_id}: {decision.verdict.value}")
        print(f"      原因: {decision.reason}")
    
    # 评估隔离（quorum 机制）
    print("\n隔离投票演示:")
    
    # 第一个实例投票
    decision1 = await arbiter.evaluate_quarantine(
        "LU_EXP_001",
        reason="检测到异常输出",
        source_instance="instance-1",
    )
    print(f"  实例 1 投票: {decision1.verdict.value}")
    print(f"      等待 quorum: {decision1.quorum_votes}/2")
    
    # 第二个实例投票（达到 quorum）
    decision2 = await arbiter.evaluate_quarantine(
        "LU_EXP_001",
        reason="确认异常",
        source_instance="instance-2",
    )
    print(f"  实例 2 投票: {decision2.verdict.value}")
    if decision2.is_quorum_decision:
        print(f"      ✓ 达到 quorum，隔离生效！")
    
    # 统计
    print("\n治理统计:")
    stats = arbiter.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def demo_propagation_throttler():
    """传播节流演示"""
    print("\n" + "=" * 60)
    print("Demo 3: 传播节流器")
    print("=" * 60)
    
    from aga.distributed import PropagationThrottler
    
    throttler = PropagationThrottler(
        default_delay_seconds=5,      # 默认延迟 5 秒
        max_propagation_rate=3,       # 每分钟最多 3 个
        learning_backflow_delay=10,   # 学习回流延迟 10 秒
    )
    
    print("创建传播节流器")
    print(f"  - default_delay: 5s")
    print(f"  - max_rate: 3/min")
    print(f"  - backflow_delay: 10s")
    
    # 调度传播
    print("\n调度传播:")
    
    for i in range(5):
        await throttler.schedule_propagation(
            lu_id=f"LU_THROTTLE_{i}",
            target_instances=["instance-1", "instance-2"],
            delay_seconds=i * 2,  # 不同延迟
        )
        print(f"  调度 LU_THROTTLE_{i}: 延迟 {i*2}s")
    
    # 处理待传播
    print("\n处理待传播队列:")
    ready = await throttler.process_pending()
    print(f"  就绪: {len(ready)} 个")
    
    # 取消传播
    cancelled = throttler.cancel_propagation("LU_THROTTLE_4")
    print(f"  取消 LU_THROTTLE_4: {cancelled} 个")
    
    # 统计
    stats = throttler.get_statistics()
    print("\n节流器统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def main():
    print("=" * 60)
    print("AGA 分布式功能演示")
    print("=" * 60)
    
    await demo_distributed_sync()
    await demo_governance()
    await demo_propagation_throttler()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
