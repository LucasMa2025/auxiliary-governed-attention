"""
AGA 算子层

提供统一的 AGA 算子接口，支持：
- 三段式门控
- 持久化衰减
- 多实例管理
- Transformer 集成

版本: v3.0
"""
from .aga_operator import AGAOperator
from .manager import AGAManager
from .transformer import AGAAugmentedTransformerLayer

__all__ = [
    "AGAOperator",
    "AGAManager",
    "AGAAugmentedTransformerLayer",
]
