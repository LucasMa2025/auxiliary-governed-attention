"""
aga/gate/ — 熵门控系统

提供:
  - EntropyGateSystem: 完整三段式门控
  - PersistenceDecay: 跨层衰减
  - DecayContext: 衰减上下文
"""
from .entropy_gate import EntropyGateSystem
from .decay import PersistenceDecay, DecayContext, DecayStrategy

__all__ = [
    "EntropyGateSystem",
    "PersistenceDecay",
    "DecayContext",
    "DecayStrategy",
]
