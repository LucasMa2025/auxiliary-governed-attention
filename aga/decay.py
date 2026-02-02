"""
AGA 持久化衰减模块 (Persistence Decay)

实现论文 Eq. 5: α^{ℓ+1}_effective = γ · α^ℓ_effective

防止辅助注意力跨层累积导致的"推理风格漂移"问题。

设计原则：
- 跨层状态传递：通过 DecayContext 在层间传递累积 gate
- 可配置衰减策略：支持指数衰减、线性衰减、自适应衰减
- 硬重置机制：超过阈值时强制重置
"""
import math
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DecayStrategy(str, Enum):
    """衰减策略"""
    EXPONENTIAL = "exponential"  # 指数衰减: α_{l+1} = γ · α_l
    LINEAR = "linear"            # 线性衰减: α_{l+1} = α_l - δ
    ADAPTIVE = "adaptive"        # 自适应衰减: 基于累积量动态调整
    NONE = "none"                # 不衰减


@dataclass
class DecayConfig:
    """衰减配置"""
    strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    gamma: float = 0.9                    # 指数衰减因子 (论文推荐 0.8-0.95)
    delta: float = 0.1                    # 线性衰减步长
    hard_reset_threshold: float = 3.0    # 累积超过此值时硬重置
    enable_hard_reset: bool = True       # 是否启用硬重置
    min_effective_gate: float = 0.01     # 最小有效 gate（低于此值视为 0）
    adaptive_base: float = 0.95          # 自适应衰减基准
    adaptive_sensitivity: float = 0.5    # 自适应敏感度


@dataclass
class DecayContext:
    """
    衰减上下文 - 在层间传递
    
    用于跟踪跨层的累积 gate 值，实现持久化衰减。
    """
    # 累积的有效 gate（跨层累加）
    accumulated_gate: float = 0.0
    
    # 当前层的原始 gate
    current_gate: Optional[torch.Tensor] = None
    
    # 衰减后的有效 gate
    effective_gate: Optional[torch.Tensor] = None
    
    # 层索引
    layer_idx: int = 0
    
    # 是否触发了硬重置
    hard_reset_triggered: bool = False
    
    # 历史记录（用于诊断）
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
    
    使用方式：
    ```python
    decay = PersistenceDecay(config)
    context = DecayContext()
    
    for layer_idx, aga_module in enumerate(aga_modules):
        # 计算原始 gate
        raw_gate = aga_module.compute_gate(hidden_states)
        
        # 应用衰减
        effective_gate, context = decay(raw_gate, context, layer_idx)
        
        # 使用衰减后的 gate
        output = aga_module.apply_gate(effective_gate, aux_output)
    ```
    """
    
    def __init__(self, config: Optional[DecayConfig] = None):
        super().__init__()
        self.config = config or DecayConfig()
        
        # 可学习的衰减参数（可选）
        if self.config.strategy == DecayStrategy.ADAPTIVE:
            self.adaptive_weight = nn.Parameter(torch.tensor(self.config.adaptive_base))
    
    def forward(
        self,
        raw_gate: torch.Tensor,
        context: DecayContext,
        layer_idx: int,
    ) -> tuple[torch.Tensor, DecayContext]:
        """
        应用持久化衰减
        
        Args:
            raw_gate: [batch, seq] 原始 gate 值
            context: 衰减上下文
            layer_idx: 当前层索引
        
        Returns:
            effective_gate: [batch, seq] 衰减后的有效 gate
            updated_context: 更新后的上下文
        """
        context.layer_idx = layer_idx
        context.current_gate = raw_gate
        
        # 计算衰减因子
        decay_factor = self._compute_decay_factor(context)
        
        # 应用衰减
        if layer_idx == 0:
            # 第一层不衰减
            effective_gate = raw_gate
        else:
            # 后续层应用衰减
            effective_gate = raw_gate * decay_factor
        
        # 检查硬重置
        gate_mean = effective_gate.mean().item()
        context.accumulated_gate += gate_mean
        
        if self.config.enable_hard_reset:
            if context.accumulated_gate > self.config.hard_reset_threshold:
                logger.warning(
                    f"Hard reset triggered at layer {layer_idx}: "
                    f"accumulated_gate={context.accumulated_gate:.4f} > threshold={self.config.hard_reset_threshold}"
                )
                effective_gate = torch.zeros_like(effective_gate)
                context.hard_reset_triggered = True
                context.accumulated_gate = 0.0
        
        # 应用最小阈值
        effective_gate = torch.where(
            effective_gate < self.config.min_effective_gate,
            torch.zeros_like(effective_gate),
            effective_gate
        )
        
        context.effective_gate = effective_gate
        context.record(gate_mean)
        
        return effective_gate, context
    
    def _compute_decay_factor(self, context: DecayContext) -> float:
        """计算衰减因子"""
        if self.config.strategy == DecayStrategy.NONE:
            return 1.0
        
        elif self.config.strategy == DecayStrategy.EXPONENTIAL:
            # γ^layer_idx
            return self.config.gamma ** context.layer_idx
        
        elif self.config.strategy == DecayStrategy.LINEAR:
            # max(0, 1 - δ * layer_idx)
            return max(0.0, 1.0 - self.config.delta * context.layer_idx)
        
        elif self.config.strategy == DecayStrategy.ADAPTIVE:
            # 基于累积量动态调整
            # 累积越多，衰减越快
            base = self.adaptive_weight.item() if hasattr(self, 'adaptive_weight') else self.config.adaptive_base
            sensitivity = self.config.adaptive_sensitivity
            
            # 衰减因子 = base^(1 + sensitivity * accumulated)
            exponent = 1.0 + sensitivity * context.accumulated_gate
            return base ** exponent
        
        return 1.0
    
    def get_diagnostics(self, context: DecayContext) -> Dict[str, Any]:
        """获取诊断信息"""
        return {
            'strategy': self.config.strategy.value,
            'gamma': self.config.gamma,
            'accumulated_gate': context.accumulated_gate,
            'hard_reset_triggered': context.hard_reset_triggered,
            'gate_history': context.gate_history,
            'layers_processed': context.layer_idx + 1,
        }


class DecayAwareAGAManager:
    """
    支持衰减的 AGA 管理器
    
    在多层 AGA 场景下管理衰减上下文。
    """
    
    def __init__(
        self,
        decay_config: Optional[DecayConfig] = None,
    ):
        self.decay = PersistenceDecay(decay_config)
        self._contexts: Dict[str, DecayContext] = {}  # request_id -> context
    
    def get_context(self, request_id: str) -> DecayContext:
        """获取或创建请求的衰减上下文"""
        if request_id not in self._contexts:
            self._contexts[request_id] = DecayContext()
        return self._contexts[request_id]
    
    def apply_decay(
        self,
        request_id: str,
        raw_gate: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """应用衰减"""
        context = self.get_context(request_id)
        effective_gate, updated_context = self.decay(raw_gate, context, layer_idx)
        self._contexts[request_id] = updated_context
        return effective_gate
    
    def finish_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """完成请求，返回诊断信息并清理上下文"""
        if request_id in self._contexts:
            context = self._contexts.pop(request_id)
            return self.decay.get_diagnostics(context)
        return None
    
    def clear_all(self):
        """清理所有上下文"""
        self._contexts.clear()


# 便捷函数：创建默认衰减配置
def create_decay_config(
    strategy: str = "exponential",
    gamma: float = 0.9,
    **kwargs
) -> DecayConfig:
    """创建衰减配置"""
    return DecayConfig(
        strategy=DecayStrategy(strategy),
        gamma=gamma,
        **kwargs
    )

