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
- 多头注意力并行优化（v3.4.1 新增）
- FlashAttention 深度集成（v3.4.1 新增）

版本: v3.4.1
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
from .parallel_attention import (
    # 注意力后端
    AttentionBackend,
    ParallelAttentionConfig,
    MultiHeadParallelAttention,
    # AGA 专用 FlashAttention
    AGAFlashAttention,
    # 分块注意力
    ChunkedAttention,
    # 工具函数
    get_available_backends,
    select_best_backend,
    HAS_FLASH_ATTN,
    HAS_XFORMERS,
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
    # 并行注意力（v3.4.1）
    "AttentionBackend",
    "ParallelAttentionConfig",
    "MultiHeadParallelAttention",
    "AGAFlashAttention",
    "ChunkedAttention",
    "get_available_backends",
    "select_best_backend",
    "HAS_FLASH_ATTN",
    "HAS_XFORMERS",
]
