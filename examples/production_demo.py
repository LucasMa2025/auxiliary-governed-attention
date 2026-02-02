"""
AGA Production Runtime 演示

展示 AGA v3.1 生产级功能的使用方式。

功能展示：
1. 基础 AGA 使用（知识注入、前向传播）
2. 多实例管理（命名空间隔离）
3. 持久化（SQLite）
4. 分布式同步（可选）
5. 内部治理（参考实现）
"""
import torch
import time
import asyncio
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入 AGA 模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aga import (
    # 类型定义
    LifecycleState,
    # 核心模块
    AuxiliaryGovernedAttention,
    AGAManager,
    AGAConfig,
)

from aga.persistence import (
    SQLiteAdapter,
    MemoryAdapter,
    PersistenceManager,
)


def demo_basic_usage():
    """基础使用演示"""
    print("\n" + "=" * 60)
    print("Demo 1: 基础 AGA 使用")
    print("=" * 60)
    
    # 1. 创建 AGA 实例
    hidden_dim = 256  # 演示用小维度
    bottleneck_dim = 32
    num_slots = 64
    
    aga = AuxiliaryGovernedAttention(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_slots=num_slots,
        num_heads=8,
        tau_low=0.5,
        tau_high=2.0,
    )
    aga.eval()
    
    print(f"创建 AGA 实例: hidden_dim={hidden_dim}, num_slots={num_slots}")
    
    # 2. 注入知识
    print("\n注入知识...")
    for i in range(5):
        key_vec = torch.randn(bottleneck_dim)
        value_vec = torch.randn(hidden_dim)
        
        slot_idx = aga.find_free_slot()
        if slot_idx is not None:
            aga.inject_knowledge(
                slot_idx=slot_idx,
                key_vector=key_vec,
                value_vector=value_vec,
                lu_id=f"LU_{i:03d}",
                lifecycle_state=LifecycleState.PROBATIONARY,
                condition=f"条件 {i}",
                decision=f"决策 {i}",
            )
            print(f"  注入 LU_{i:03d} -> slot_idx={slot_idx}")
    
    # 3. 前向传播
    print("\n前向传播...")
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    primary_output = torch.randn(batch_size, seq_len, hidden_dim)
    
    with torch.no_grad():
        output, diagnostics = aga(
            hidden_states=hidden_states,
            primary_attention_output=primary_output,
            return_diagnostics=True,
        )
    
    print(f"  输出形状: {output.shape}")
    if diagnostics:
        print(f"  活跃槽位: {diagnostics.active_slots}")
        print(f"  门控均值: {diagnostics.gate_mean:.4f}")
    
    # 4. 更新生命周期
    print("\n更新生命周期...")
    aga.update_lifecycle(0, LifecycleState.CONFIRMED)
    print("  LU_000 -> CONFIRMED")
    
    # 5. 统计信息
    print("\n统计信息:")
    stats = aga.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def demo_persistence():
    """持久化演示"""
    print("\n" + "=" * 60)
    print("Demo 2: 持久化存储")
    print("=" * 60)
    
    # 1. 创建 AGA 实例
    hidden_dim = 128
    bottleneck_dim = 16
    
    aga = AuxiliaryGovernedAttention(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_slots=32,
    )
    aga.eval()
    
    # 2. 注入知识
    for i in range(3):
        slot_idx = aga.find_free_slot()
        if slot_idx is not None:
            aga.inject_knowledge(
                slot_idx=slot_idx,
                key_vector=torch.randn(bottleneck_dim),
                value_vector=torch.randn(hidden_dim),
                lu_id=f"PERSIST_LU_{i}",
                lifecycle_state=LifecycleState.PROBATIONARY,
                condition=f"持久化条件 {i}",
                decision=f"持久化决策 {i}",
            )
    
    print(f"注入 3 条知识")
    
    # 3. 使用 SQLite 持久化（异步）
    async def persist_demo():
        db_path = Path(__file__).parent / "demo_aga.db"
        
        # 创建适配器
        adapter = SQLiteAdapter(str(db_path))
        await adapter.connect()
        
        # 创建持久化管理器
        pm = PersistenceManager(adapter, namespace="demo")
        
        # 保存 AGA 状态
        await pm.save_aga_state(aga)
        print(f"保存到: {db_path}")
        
        # 获取统计
        stats = await adapter.get_statistics()
        print(f"数据库统计: {stats}")
        
        # 关闭连接
        await adapter.disconnect()
        
        # 清理演示文件
        if db_path.exists():
            db_path.unlink()
            print("清理演示数据库")
    
    asyncio.run(persist_demo())


def demo_multi_namespace():
    """多命名空间演示"""
    print("\n" + "=" * 60)
    print("Demo 3: 多命名空间隔离")
    print("=" * 60)
    
    hidden_dim = 128
    bottleneck_dim = 16
    
    # 创建多个 AGA 实例（模拟不同租户）
    namespaces = ["tenant_A", "tenant_B", "tenant_C"]
    aga_instances = {}
    
    for ns in namespaces:
        aga = AuxiliaryGovernedAttention(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_slots=32,
        )
        aga.eval()
        
        # 为每个 namespace 注入不同数量的知识
        count = namespaces.index(ns) + 2
        for i in range(count):
            slot_idx = aga.find_free_slot()
            if slot_idx is not None:
                aga.inject_knowledge(
                    slot_idx=slot_idx,
                    key_vector=torch.randn(bottleneck_dim),
                    value_vector=torch.randn(hidden_dim),
                    lu_id=f"{ns}_LU_{i}",
                )
        
        aga_instances[ns] = aga
        print(f"  {ns}: 注入 {count} 条知识")
    
    # 并发前向传播测试
    print("\n并发前向传播测试...")
    import threading
    
    results = {}
    
    def run_forward(ns):
        aga = aga_instances[ns]
        hidden = torch.randn(1, 4, hidden_dim)
        primary = torch.randn(1, 4, hidden_dim)
        
        start = time.time()
        with torch.no_grad():
            output, _ = aga(hidden, primary)
        elapsed = (time.time() - start) * 1000
        
        results[ns] = {
            "latency_ms": elapsed,
            "output_norm": output.norm().item(),
        }
    
    threads = [threading.Thread(target=run_forward, args=(ns,)) for ns in namespaces]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    for ns, r in results.items():
        print(f"  {ns}: latency={r['latency_ms']:.2f}ms")


def demo_governance():
    """内部治理演示（参考实现）"""
    print("\n" + "=" * 60)
    print("Demo 4: 内部治理（参考实现）")
    print("=" * 60)
    
    try:
        from aga.distributed import (
            GovernanceArbiter,
            TrustTier,
            PropagationThrottler,
            GovernanceVerdict,
        )
        
        # 创建治理裁决器
        arbiter = GovernanceArbiter(
            instance_id="demo-instance",
            quorum_size=2,
            risk_threshold=0.3,
        )
        
        print("创建治理裁决器 (quorum_size=2)")
        
        # 注册不同信任层级的槽位
        trust_levels = [
            ("LU_CACHE_001", TrustTier.S0_ACCELERATION),
            ("LU_EXP_001", TrustTier.S1_EXPERIENCE),
            ("LU_POLICY_001", TrustTier.S2_POLICY),
        ]
        
        for lu_id, tier in trust_levels:
            info = arbiter.register_slot(lu_id, tier)
            print(f"  注册 {lu_id}: {tier.value}")
        
        # 评估传播
        print("\n传播评估:")
        
        async def evaluate():
            for lu_id, _ in trust_levels:
                decision = await arbiter.evaluate_propagation(lu_id, "target-instance")
                print(f"  {lu_id} -> {decision.verdict.value}: {decision.reason}")
        
        asyncio.run(evaluate())
        
        # 统计
        print("\n治理统计:")
        stats = arbiter.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except ImportError as e:
        print(f"跳过治理演示（模块未完全实现）: {e}")


def demo_lifecycle_states():
    """生命周期状态演示"""
    print("\n" + "=" * 60)
    print("Demo 5: 知识生命周期")
    print("=" * 60)
    
    hidden_dim = 128
    bottleneck_dim = 16
    
    aga = AuxiliaryGovernedAttention(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_slots=32,
    )
    aga.eval()
    
    # 注入知识
    slot_idx = aga.find_free_slot()
    aga.inject_knowledge(
        slot_idx=slot_idx,
        key_vector=torch.randn(bottleneck_dim),
        value_vector=torch.randn(hidden_dim),
        lu_id="LIFECYCLE_DEMO",
        lifecycle_state=LifecycleState.PROBATIONARY,
    )
    
    print(f"注入知识: LIFECYCLE_DEMO (slot={slot_idx})")
    
    # 状态转换演示
    transitions = [
        (LifecycleState.CONFIRMED, "确认知识"),
        (LifecycleState.DEPRECATED, "弃用知识"),
        (LifecycleState.QUARANTINED, "隔离知识"),
    ]
    
    print("\n状态转换:")
    for new_state, desc in transitions:
        aga.update_lifecycle(slot_idx, new_state)
        info = aga.get_slot_info(slot_idx)
        print(f"  {desc}: {info.lifecycle_state.value} (reliability={info.reliability:.1f})")
    
    # 可靠性映射说明
    print("\n可靠性映射:")
    print("  PROBATIONARY -> r=0.3 (试用期，低权重)")
    print("  CONFIRMED    -> r=1.0 (已确认，全权重)")
    print("  DEPRECATED   -> r=0.1 (已弃用，极低权重)")
    print("  QUARANTINED  -> r=0.0 (已隔离，不参与推理)")


if __name__ == "__main__":
    print("=" * 60)
    print("AGA v3.1 Production Demo")
    print("=" * 60)
    
    demo_basic_usage()
    demo_persistence()
    demo_multi_namespace()
    demo_governance()
    demo_lifecycle_states()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
