"""
AGA 算子层

提供统一的 AGA 算子接口，支持：
- 三段式门控
- 持久化衰减
- 多实例管理
- Transformer 集成
- 性能优化（混合精度、CUDA Graph）
- 粒度掩码
- 故障恢复

版本: v3.1
"""
from .aga_operator import AGAOperator
from .manager import AGAManager
from .transformer import AGAAugmentedTransformerLayer
from .optimizations import (
    # 混合精度
    MixedPrecisionConfig,
    MixedPrecisionWrapper,
    # CUDA Graph
    CUDAGraphWrapper,
    # 粒度掩码
    GranularityLevel,
    GranularityConfig,
    GranularityMask,
    # 故障恢复
    ResilienceConfig,
    ResilientAGAWrapper,
    # 健康监控
    HealthConfig,
    HealthMonitor,
)

__all__ = [
    # 核心
    "AGAOperator",
    "AGAManager",
    "AGAAugmentedTransformerLayer",
    # 混合精度
    "MixedPrecisionConfig",
    "MixedPrecisionWrapper",
    # CUDA Graph
    "CUDAGraphWrapper",
    # 粒度掩码
    "GranularityLevel",
    "GranularityConfig",
    "GranularityMask",
    # 故障恢复
    "ResilienceConfig",
    "ResilientAGAWrapper",
    # 健康监控
    "HealthConfig",
    "HealthMonitor",
]


