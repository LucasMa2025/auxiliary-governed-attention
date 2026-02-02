"""
AGA Quick Start 示例

最简单的 AGA 使用示例，展示核心功能。
"""
import torch
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from aga import (
    AuxiliaryGovernedAttention,
    LifecycleState,
)


def main():
    print("=" * 50)
    print("AGA Quick Start")
    print("=" * 50)
    
    # 1. 创建 AGA 实例
    hidden_dim = 256
    bottleneck_dim = 32
    
    aga = AuxiliaryGovernedAttention(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_slots=64,
    )
    aga.eval()
    print(f"\n✓ 创建 AGA: hidden_dim={hidden_dim}")
    
    # 2. 注入知识
    slot_idx = aga.find_free_slot()
    
    aga.inject_knowledge(
        slot_idx=slot_idx,
        key_vector=torch.randn(bottleneck_dim),
        value_vector=torch.randn(hidden_dim),
        lu_id="LU_EXAMPLE_001",
        lifecycle_state=LifecycleState.PROBATIONARY,
        condition="capital of France",
        decision="Paris",
    )
    print(f"✓ 注入知识: LU_EXAMPLE_001 -> slot {slot_idx}")
    
    # 3. 确认知识
    aga.update_lifecycle(slot_idx, LifecycleState.CONFIRMED)
    print("✓ 确认知识: PROBATIONARY -> CONFIRMED")
    
    # 4. 前向传播
    batch_size, seq_len = 1, 10
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    primary_output = torch.randn(batch_size, seq_len, hidden_dim)
    
    with torch.no_grad():
        output, diagnostics = aga(
            hidden_states=hidden_states,
            primary_attention_output=primary_output,
            return_diagnostics=True,
        )
    
    print(f"✓ 前向传播完成")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 活跃槽位: {diagnostics.active_slots}")
    print(f"  - 门控均值: {diagnostics.gate_mean:.4f}")
    
    # 5. 查看统计
    stats = aga.get_statistics()
    print(f"\n统计信息:")
    print(f"  - 总槽位: {stats['total_slots']}")
    print(f"  - 活跃槽位: {stats['active_slots']}")
    print(f"  - 状态分布: {stats.get('state_distribution', {})}")
    
    # 6. 隔离知识（演示）
    aga.update_lifecycle(slot_idx, LifecycleState.QUARANTINED)
    print(f"\n✓ 隔离知识: CONFIRMED -> QUARANTINED")
    print("  (隔离后的知识不参与推理，reliability=0.0)")
    
    print("\n" + "=" * 50)
    print("Quick Start 完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
