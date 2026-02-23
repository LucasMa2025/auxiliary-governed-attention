"""
aga/gate/decay.py — 持久化衰减模块

实现论文 Eq. 5: α^{ℓ+1}_effective = γ · α^ℓ_effective

防止辅助注意力跨层累积导致的"推理风格漂移"问题。

源码映射:
  - 来自 decay.py PersistenceDecay (第 100-213 行)
  - 来自 decay.py DecayContext (第 47-97 行)

设计原则:
  - 跨层状态传递：通过 DecayContext 在层间传递累积 gate
  - 可配置衰减策略：支持指数衰减、线性衰减、自适应衰减
  - 硬重置机制：超过阈值时强制重置
"""
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn

from ..config import AGAConfig

logger = logging.getLogger(__name__)


class DecayStrategy(str, Enum):
    """衰减策略"""
    EXPONENTIAL = "exponential"  # 指数衰减: α_{l+1} = γ · α_l
    LINEAR = "linear"  # 线性衰减: α_{l+1} = α_l - δ
    ADAPTIVE = "adaptive"  # 自适应衰减: 基于累积量动态调整
    NONE = "none"  # 不衰减


@dataclass
class DecayContext:
    """
    衰减上下文 - 在层间传递

    用于跟踪跨层的累积 gate 值，实现持久化衰减。
    """
    accumulated_gate: float = 0.0
    current_gate: Optional[torch.Tensor] = None
    effective_gate: Optional[torch.Tensor] = None
    layer_idx: int = 0
    hard_reset_triggered: bool = False
    gate_history: list = field(default_factory=list)

    def record(self, gate_mean: float):
        """记录 gate 历史"""
        self.gate_history.append({
            'layer': self.layer_idx,
            'gate_mean': gate_mean,
            'accumulated': self.accumulated_gate,
        })

    def reset(self):
        """重置上下文"""
        self.accumulated_gate = 0.0
        self.current_gate = None
        self.effective_gate = None
        self.layer_idx = 0
        self.hard_reset_triggered = False
        self.gate_history.clear()

    def clone(self) -> 'DecayContext':
        """克隆上下文"""
        return DecayContext(
            accumulated_gate=self.accumulated_gate,
            current_gate=self.current_gate.clone() if self.current_gate is not None else None,
            effective_gate=self.effective_gate.clone() if self.effective_gate is not None else None,
            layer_idx=self.layer_idx,
            hard_reset_triggered=self.hard_reset_triggered,
            gate_history=self.gate_history.copy(),
        )


class PersistenceDecay(nn.Module):
    """
    持久化衰减模块

    实现论文 Section 3.4.1 的持久化衰减机制。

    使用方式:
        decay = PersistenceDecay(config)
        context = DecayContext()

        for layer_idx, aga_module in enumerate(aga_modules):
            raw_gate = aga_module.compute_gate(hidden_states)
            effective_gate, context = decay(raw_gate, context, layer_idx)
            output = aga_module.apply_gate(effective_gate, aux_output)
    """

    def __init__(self, config: AGAConfig):
        super().__init__()
        self.strategy = DecayStrategy(config.decay_strategy)
        self.gamma = config.decay_gamma
        self.hard_reset_threshold = config.decay_hard_reset_threshold
        self.min_effective_gate = 0.01

        # 可学习的衰减参数（自适应模式）
        if self.strategy == DecayStrategy.ADAPTIVE:
            self.adaptive_weight = nn.Parameter(torch.tensor(0.95))

    def forward(
        self,
        raw_gate: torch.Tensor,
        context: Optional[DecayContext] = None,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, DecayContext]:
        """
        应用持久化衰减

        Args:
            raw_gate: [batch, seq] 原始 gate 值
            context: 衰减上下文（如果为 None，创建新的）
            layer_idx: 当前层索引

        Returns:
            effective_gate: [batch, seq] 衰减后的有效 gate
            updated_context: 更新后的上下文
        """
        if context is None:
            context = DecayContext()

        context.layer_idx = layer_idx
        raw_gate = torch.nan_to_num(raw_gate, nan=0.0, posinf=0.0, neginf=0.0)
        context.current_gate = raw_gate

        # 计算衰减因子
        decay_factor = self._compute_decay_factor(context)

        # 应用衰减
        if layer_idx == 0:
            effective_gate = raw_gate
        else:
            effective_gate = raw_gate * decay_factor

        # 检查硬重置
        gate_mean = effective_gate.mean().item()
        context.accumulated_gate += gate_mean

        if context.accumulated_gate > self.hard_reset_threshold:
            logger.debug(
                f"Hard reset triggered at layer {layer_idx}: "
                f"accumulated_gate={context.accumulated_gate:.4f}"
            )
            effective_gate = torch.zeros_like(effective_gate)
            context.hard_reset_triggered = True
            context.accumulated_gate = 0.0

        # 应用最小阈值
        effective_gate = torch.where(
            effective_gate < self.min_effective_gate,
            torch.zeros_like(effective_gate),
            effective_gate
        )

        context.effective_gate = effective_gate
        context.record(gate_mean)

        return effective_gate, context

    def _compute_decay_factor(self, context: DecayContext) -> float:
        """计算衰减因子"""
        if self.strategy == DecayStrategy.NONE:
            return 1.0
        elif self.strategy == DecayStrategy.EXPONENTIAL:
            return self.gamma ** context.layer_idx
        elif self.strategy == DecayStrategy.LINEAR:
            delta = 1.0 - self.gamma  # 使用 gamma 推导 delta
            return max(0.0, 1.0 - delta * context.layer_idx)
        elif self.strategy == DecayStrategy.ADAPTIVE:
            base = self.adaptive_weight.item() if hasattr(self, 'adaptive_weight') else 0.95
            exponent = 1.0 + 0.5 * context.accumulated_gate
            return base ** exponent
        return 1.0

    def get_diagnostics(self, context: DecayContext) -> Dict[str, Any]:
        """获取诊断信息"""
        return {
            'strategy': self.strategy.value,
            'gamma': self.gamma,
            'accumulated_gate': context.accumulated_gate,
            'hard_reset_triggered': context.hard_reset_triggered,
            'gate_history': context.gate_history,
            'layers_processed': context.layer_idx + 1,
        }
