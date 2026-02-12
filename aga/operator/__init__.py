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
- FlashAttention-3 支持（v3.5.0 新增）

版本: v3.5.0
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
# FlashAttention-3 集成 (v3.5.0)
from .flash_attention3 import (
    # 可用性检测
    FLASH_ATTN_AVAILABLE as FA3_AVAILABLE,
    FLASH_ATTN_VERSION as FA3_VERSION,
    FLASH_ATTN_3_AVAILABLE,
    # 后端
    FA3Backend,
    FlashAttention3Backend,
    # 配置
    FA3Config,
    FA2Config,
    FlashAttention3Config,
    # 函数
    flash_attention3_forward,
    # 模块
    AGAFlashAttention3,
    AGAKnowledgeInjectionOptimizer,
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
    # FlashAttention-3 (v3.5.0)
    "FA3_AVAILABLE",
    "FA3_VERSION",
    "FLASH_ATTN_3_AVAILABLE",
    "FA3Backend",
    "FlashAttention3Backend",
    "FA3Config",
    "FA2Config",
    "FlashAttention3Config",
    "flash_attention3_forward",
    "AGAFlashAttention3",
    "AGAKnowledgeInjectionOptimizer",
]
