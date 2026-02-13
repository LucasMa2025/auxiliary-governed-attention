"""
AGA 优化熵门控模块 (Enhanced Entropy Gating)

实现论文 Eq. 2-3 的完整熵门控机制，并增加工程优化。

核心公式：
- Eq. 2: α = σ(w₁·H + w₂·r_s + b)
- Eq. 3: 三段式熵否决机制

优化特性：
- 完整实现论文公式（包含 r_s 项）
- 多头熵计算支持
- 自适应阈值
- 温度缩放
- 梯度裁剪保护
"""
import math
from typing import Optional, Tuple, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .decay import DecayConfig, DecayContext


class EntropySource(str, Enum):
    """熵信号来源"""
    ATTENTION = "attention"           # 注意力权重熵
    LOGITS = "logits"                 # Logits 熵
    HIDDEN_VARIANCE = "hidden_variance"  # 隐藏状态方差
    MULTI_HEAD = "multi_head"         # 多头熵（每头独立计算）
    ENSEMBLE = "ensemble"             # 集成多种信号


@dataclass
class EntropyGateConfig:
    """熵门控配置"""
    # 基础参数
    tau_low: float = 0.5              # 低熵阈值（主模型确信）
    tau_high: float = 2.0             # 高熵阈值（主模型不确定）
    max_gate: float = 0.8             # 最大 gate 值（高熵时的上限）
    
    # 门控公式参数
    w1_init: float = 0.5              # 熵权重初始值
    w2_init: float = 0.3              # 可靠性权重初始值
    bias_init: float = -1.0           # 偏置初始值
    
    # 优化参数
    temperature: float = 1.0          # 温度缩放
    enable_adaptive_threshold: bool = False  # 自适应阈值
    adaptive_momentum: float = 0.99   # 自适应动量
    gradient_clip: float = 1.0        # 梯度裁剪
    
    # 熵源配置
    entropy_source: EntropySource = EntropySource.HIDDEN_VARIANCE
    num_heads: int = 32               # 注意力头数（用于多头熵）
    
    # 数值稳定性
    eps: float = 1e-10
    entropy_clamp_max: float = 10.0


class EntropyCalculator(nn.Module):
    """
    熵计算器
    
    支持多种熵信号源，提供数值稳定的熵计算。
    """
    
    def __init__(self, config: EntropyGateConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        
        # 用于 hidden variance 的投影
        if config.entropy_source in (EntropySource.HIDDEN_VARIANCE, EntropySource.ENSEMBLE):
            self.variance_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            nn.init.zeros_(self.variance_proj[-1].weight)
            nn.init.constant_(self.variance_proj[-1].bias, 0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算熵信号
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            attention_weights: [batch, heads, seq, seq] 可选
            logits: [batch, seq, vocab_size] 可选
        
        Returns:
            entropy: [batch, seq]
        """
        source = self.config.entropy_source
        
        if source == EntropySource.ATTENTION:
            entropy = self._attention_entropy(attention_weights, hidden_states)
        elif source == EntropySource.LOGITS:
            entropy = self._logits_entropy(logits, hidden_states)
        elif source == EntropySource.HIDDEN_VARIANCE:
            entropy = self._hidden_variance_entropy(hidden_states)
        elif source == EntropySource.MULTI_HEAD:
            entropy = self._multi_head_entropy(attention_weights, hidden_states)
        elif source == EntropySource.ENSEMBLE:
            entropy = self._ensemble_entropy(hidden_states, attention_weights, logits)
        else:
            entropy = self._hidden_variance_entropy(hidden_states)
        
        return torch.nan_to_num(entropy, nan=0.0, posinf=self.config.entropy_clamp_max, neginf=0.0)
    
    def _attention_entropy(
        self,
        attention_weights: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算注意力熵 (论文 Eq. 1)
        
        H_i = -Σ_j p_{ij} log p_{ij}
        """
        if attention_weights is None:
            return self._hidden_variance_entropy(hidden_states)
        
        # 平均所有头
        avg_weights = attention_weights.mean(dim=1)  # [batch, seq, seq]
        
        # 数值稳定的熵计算
        # 使用 log_softmax 避免 log(0)
        log_weights = F.log_softmax(
            torch.log(avg_weights.clamp(min=self.config.eps)),
            dim=-1
        )
        
        # H = -Σ p * log(p)
        entropy = -torch.sum(avg_weights * log_weights, dim=-1)
        
        return entropy.clamp(0, self.config.entropy_clamp_max)
    
    def _logits_entropy(
        self,
        logits: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算 logits 熵"""
        if logits is None:
            return self._hidden_variance_entropy(hidden_states)
        
        # 温度缩放
        scaled_logits = logits / self.config.temperature
        
        # 数值稳定的熵计算
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        probs = F.softmax(scaled_logits, dim=-1)
        
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # 归一化到合理范围
        max_entropy = math.log(logits.size(-1))
        normalized = entropy / max_entropy * 3.0
        
        return normalized.clamp(0, self.config.entropy_clamp_max)
    
    def _hidden_variance_entropy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        基于隐藏状态方差的熵估计
        
        兼容 FlashAttention（不需要 attention weights）
        """
        # 计算每个 token 的内部方差
        mean = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean
        variance = (centered ** 2).mean(dim=-1)
        
        # 对数方差（避免极端值）
        log_var = torch.log1p(variance)
        
        # 归一化
        normalized = log_var / (log_var.mean(dim=-1, keepdim=True) + self.config.eps)
        
        # 结合学习的投影
        if hasattr(self, 'variance_proj'):
            learned = self.variance_proj(hidden_states).squeeze(-1)
            combined = normalized * 0.5 + torch.sigmoid(learned) * 2.5
        else:
            combined = normalized * 2.0
        
        return combined.clamp(0, self.config.entropy_clamp_max)
    
    def _multi_head_entropy(
        self,
        attention_weights: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        多头熵计算
        
        分别计算每个头的熵，然后聚合。
        """
        if attention_weights is None:
            return self._hidden_variance_entropy(hidden_states)
        
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 每头独立计算熵
        head_entropies = []
        for h in range(num_heads):
            head_weights = attention_weights[:, h, :, :]  # [batch, seq, seq]
            log_weights = F.log_softmax(
                torch.log(head_weights.clamp(min=self.config.eps)),
                dim=-1
            )
            head_entropy = -torch.sum(head_weights * log_weights, dim=-1)
            head_entropies.append(head_entropy)
        
        # 聚合：使用均值和标准差
        stacked = torch.stack(head_entropies, dim=-1)  # [batch, seq, num_heads]
        mean_entropy = stacked.mean(dim=-1)
        std_entropy = stacked.std(dim=-1)
        
        # 高方差表示头之间不一致，增加不确定性
        combined = mean_entropy + 0.5 * std_entropy
        
        return combined.clamp(0, self.config.entropy_clamp_max)
    
    def _ensemble_entropy(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        集成多种熵信号
        """
        signals = []
        weights = []
        
        # Hidden variance (总是可用)
        signals.append(self._hidden_variance_entropy(hidden_states))
        weights.append(0.4)
        
        # Attention entropy (如果可用)
        if attention_weights is not None:
            signals.append(self._attention_entropy(attention_weights, hidden_states))
            weights.append(0.4)
        
        # Logits entropy (如果可用)
        if logits is not None:
            signals.append(self._logits_entropy(logits, hidden_states))
            weights.append(0.2)
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # 加权平均
        ensemble = sum(s * w for s, w in zip(signals, weights))
        
        return ensemble.clamp(0, self.config.entropy_clamp_max)


class EntropyGate(nn.Module):
    """
    优化的熵门控模块
    
    完整实现论文 Eq. 2-3：
    - Eq. 2: α = σ(w₁·H + w₂·r_s + b)
    - Eq. 3: 三段式熵否决
    """
    
    def __init__(self, config: EntropyGateConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        
        # 熵计算器
        self.entropy_calculator = EntropyCalculator(config, hidden_dim)
        
        # 门控参数 (论文 Eq. 2)
        self.w1 = nn.Parameter(torch.tensor(config.w1_init))  # 熵权重
        self.w2 = nn.Parameter(torch.tensor(config.w2_init))  # 可靠性权重
        self.bias = nn.Parameter(torch.tensor(config.bias_init))
        
        # 自适应阈值（运行时更新）
        if config.enable_adaptive_threshold:
            self.register_buffer('running_tau_low', torch.tensor(config.tau_low))
            self.register_buffer('running_tau_high', torch.tensor(config.tau_high))
            self.register_buffer('entropy_ema', torch.tensor(1.0))
        
        # 梯度裁剪钩子
        if config.gradient_clip > 0:
            for param in [self.w1, self.w2, self.bias]:
                param.register_hook(
                    lambda grad: torch.clamp(grad, -config.gradient_clip, config.gradient_clip)
                )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        reliability: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        计算门控值
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            reliability: [batch, seq] 或标量，可靠性值 r_s
            attention_weights: [batch, heads, seq, seq] 可选
            logits: [batch, seq, vocab_size] 可选
        
        Returns:
            gate: [batch, seq] 门控值
            diagnostics: 诊断信息
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # 1. 计算熵
        entropy = self.entropy_calculator(hidden_states, attention_weights, logits)
        
        # 2. 处理可靠性
        if reliability.dim() == 0:
            reliability = reliability.expand(batch_size, seq_len)
        elif reliability.dim() == 1:
            reliability = reliability.unsqueeze(0).expand(batch_size, -1)
        
        # 3. 计算原始 gate (论文 Eq. 2)
        # α = σ(w₁·H + w₂·r_s + b)
        raw_gate = torch.sigmoid(
            self.w1 * entropy + self.w2 * reliability + self.bias
        )
        
        # 4. 应用熵否决 (论文 Eq. 3)
        tau_low, tau_high = self._get_thresholds()
        gate = self._apply_entropy_veto(raw_gate, entropy, tau_low, tau_high)
        
        # 5. 更新自适应阈值
        if self.config.enable_adaptive_threshold and self.training:
            self._update_adaptive_thresholds(entropy)
        
        # 6. 诊断信息
        diagnostics = {
            'entropy_mean': entropy.mean().item(),
            'entropy_std': entropy.std().item(),
            'raw_gate_mean': raw_gate.mean().item(),
            'final_gate_mean': gate.mean().item(),
            'tau_low': tau_low,
            'tau_high': tau_high,
            'w1': self.w1.item(),
            'w2': self.w2.item(),
            'bias': self.bias.item(),
            'veto_ratio': (gate == 0).float().mean().item(),
        }
        
        return gate, diagnostics
    
    def _get_thresholds(self) -> Tuple[float, float]:
        """获取阈值（支持自适应）"""
        if self.config.enable_adaptive_threshold and hasattr(self, 'running_tau_low'):
            return self.running_tau_low.item(), self.running_tau_high.item()
        return self.config.tau_low, self.config.tau_high
    
    def _apply_entropy_veto(
        self,
        gate: torch.Tensor,
        entropy: torch.Tensor,
        tau_low: float,
        tau_high: float,
    ) -> torch.Tensor:
        """
        应用熵否决机制 (论文 Eq. 3)
        
        - H < τ_low: 主模型确信 → gate = 0
        - H > τ_high: 主模型不确定 → gate ≤ max_gate
        - 否则: 使用原始 gate
        """
        # 低熵：主模型确信，禁止干预
        gate = torch.where(
            entropy < tau_low,
            torch.zeros_like(gate),
            gate
        )
        
        # 高熵：主模型不确定，限制干预强度
        gate = torch.where(
            entropy > tau_high,
            torch.clamp(gate, max=self.config.max_gate),
            gate
        )
        
        return gate
    
    def _update_adaptive_thresholds(self, entropy: torch.Tensor):
        """更新自适应阈值（使用 .data 避免破坏 buffer 注册）"""
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=self.config.entropy_clamp_max, neginf=0.0)
        momentum = self.config.adaptive_momentum
        current_mean = entropy.mean().item()
        current_std = entropy.std().item()
        
        # 更新 EMA（使用 .data 原地修改，保持 buffer 注册）
        self.entropy_ema.data.fill_(momentum * self.entropy_ema.item() + (1 - momentum) * current_mean)
        
        # 更新阈值
        ema = self.entropy_ema.item()
        self.running_tau_low.data.fill_(momentum * self.running_tau_low.item() + (1 - momentum) * (ema - current_std))
        self.running_tau_high.data.fill_(momentum * self.running_tau_high.item() + (1 - momentum) * (ema + current_std))
        
        # 确保合理范围
        self.running_tau_low.data.clamp_(min=0.1, max=self.running_tau_high.item() - 0.1)
        self.running_tau_high.data.clamp_(min=self.running_tau_low.item() + 0.1, max=5.0)


class EntropyGateWithDecay(nn.Module):
    """
    集成衰减的熵门控
    
    结合 EntropyGate 和 PersistenceDecay。
    """
    
    def __init__(
        self,
        gate_config: EntropyGateConfig,
        hidden_dim: int,
        decay_config: Optional['DecayConfig'] = None,
    ):
        super().__init__()
        from .decay import PersistenceDecay, DecayConfig, DecayContext
        
        self.gate = EntropyGate(gate_config, hidden_dim)
        self.decay = PersistenceDecay(decay_config or DecayConfig())
        self.DecayContext = DecayContext
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        reliability: torch.Tensor,
        layer_idx: int,
        decay_context: Optional['DecayContext'] = None,
        attention_weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, 'DecayContext', Dict[str, Any]]:
        """
        计算带衰减的门控值
        """
        # 计算原始 gate
        raw_gate, gate_diagnostics = self.gate(
            hidden_states, reliability, attention_weights, logits
        )
        
        # 应用衰减
        if decay_context is None:
            decay_context = self.DecayContext()
        
        effective_gate, decay_context = self.decay(raw_gate, decay_context, layer_idx)
        
        # 合并诊断
        diagnostics = {
            **gate_diagnostics,
            'decay_accumulated': decay_context.accumulated_gate,
            'decay_hard_reset': decay_context.hard_reset_triggered,
        }
        
        return effective_gate, decay_context, diagnostics


# 便捷函数
def create_entropy_gate(
    hidden_dim: int,
    entropy_source: str = "hidden_variance",
    tau_low: float = 0.5,
    tau_high: float = 2.0,
    **kwargs
) -> EntropyGate:
    """创建熵门控"""
    config = EntropyGateConfig(
        entropy_source=EntropySource(entropy_source),
        tau_low=tau_low,
        tau_high=tau_high,
        **kwargs
    )
    return EntropyGate(config, hidden_dim)

