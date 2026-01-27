# AGA - Auxiliary Governed Attention
"""
AGA (Auxiliary Governed Attention) v2.0 - 热插拔式知识注入系统

v2.0 优化：
- Slot Routing: O(N) → O(k) 复杂度优化
- Delta Subspace: value 通过 bottleneck projection 受控干预
- 熵信号解耦: 支持多种不确定性信号源，兼容 FlashAttention
- 元数据外置: DB 管理 lifecycle/LU，AGA 只是执行层

核心特性：
- 零训练注入：知识直接写入 buffer，无需梯度计算
- 热插拔设计：运行时动态添加/移除知识
- 治理控制：生命周期状态、熵门控、可追溯性
- 即时隔离：问题知识可立即移除影响

使用示例:
    from aga import AGA, AGAConfig, LifecycleState
    
    # v2.0: 使用配置创建 AGA 实例
    config = AGAConfig(
        hidden_dim=4096, 
        num_slots=100,
        top_k_routing=8,  # 路由优化
        use_value_projection=True,  # delta subspace
    )
    aga = AGA(config=config)
    
    # 注入知识
    aga.inject_knowledge(
        slot_idx=0,
        key_vector=key_vec,
        value_vector=val_vec,
        lu_id="LU_001",
    )
    
    # 挂载到模型
    manager = AGAManager()
    manager.attach_to_model(model, layer_indices=[-2, -1])
"""

from .core import (
    AuxiliaryGovernedAttention,
    AGAAugmentedTransformerLayer,
    AGAManager,
    LifecycleState,
    KnowledgeSlotInfo,
    AGADiagnostics,
    # v2.0 新增
    AGAConfig,
    UncertaintySource,
    UncertaintyEstimator,
    SlotRouter,
)

from .persistence import (
    AGAPersistence,
    SQLitePersistence,
    AGAPersistenceManager,
    KnowledgeRecord,
)

__version__ = "2.1.0"
__author__ = "Lucas Ma"

__all__ = [
    # Core
    "AuxiliaryGovernedAttention",
    "AGAAugmentedTransformerLayer", 
    "AGAManager",
    "LifecycleState",
    "KnowledgeSlotInfo",
    "AGADiagnostics",
    # v2.0 Core
    "AGAConfig",
    "UncertaintySource",
    "UncertaintyEstimator",
    "SlotRouter",
    # Persistence
    "AGAPersistence",
    "SQLitePersistence",
    "AGAPersistenceManager",
    "KnowledgeRecord",
    # Alias
    "AGA",
]

# Alias for convenience
AGA = AuxiliaryGovernedAttention

