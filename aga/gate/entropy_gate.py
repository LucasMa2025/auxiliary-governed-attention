"""
aga/gate/entropy_gate.py — 完整三段式门控系统

源码映射:
  - Gate-0: 来自 production/gate.py Gate0 (第 63-100 行)
  - Gate-1: 来自 core.py UncertaintyEstimator (第 132-249 行)
            + core.py forward() 中的 gate 计算 (第 497-528 行)
  - Gate-2: 来自 core.py SlotRouter (第 252-340 行)
  - 熵否决: 来自 core.py _apply_entropy_veto() (第 687-691 行)

  合并 4 个分散的门控实现为 1 个统一系统

核心公式:
  - Eq. 2: α = σ(w₁·H + b)
  - Eq. 3: 三段式熵否决
    - H < τ_low: 模型自信 → gate = 0
    - τ_low ≤ H ≤ τ_high: 正常区间 → gate = α
    - H > τ_high: 模型极度不确定 → gate ≤ max_gate
"""
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AGAConfig
from ..types import GateDiagnostics


class EntropyGateSystem(nn.Module):
    """
    完整三段式熵门控系统

    Gate-0: 先验门控（零成本，基于 namespace/route）
    Gate-1: 置信门控（轻量，基于 hidden state 不确定性）
    Gate-2: Top-K 路由（只在 Gate-1 通过时执行，在 BottleneckInjector 中实现）

    所有阈值从 AGAConfig 外置，支持运行时动态调节。
    """

    def __init__(self, config: AGAConfig):
        super().__init__()
        self.config = config
        self.tau_low = config.tau_low
        self.tau_high = config.tau_high
        self.max_gate = config.max_gate

        # Gate-1: 不确定性估计
        # 使用 hidden_states 的内部方差 + 学习的投影
        if config.gate1_uncertainty_source == "hidden_variance":
            self.uncertainty_proj = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 4, 1),
            )
            # 初始化为输出接近 0 的值
            nn.init.zeros_(self.uncertainty_proj[-1].weight)
            nn.init.constant_(self.uncertainty_proj[-1].bias, 0.0)

        # 门控参数（可学习）
        self.gate_w1 = nn.Parameter(torch.tensor(0.5))
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
        context: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, GateDiagnostics]:
        """
        三段式门控

        Args:
            hidden_states: [batch, seq, hidden_dim]
            layer_idx: 当前层索引
            context: 可选上下文（namespace 等）

        Returns:
            gate: [batch, seq] 门控值
            diagnostics: 诊断信息
        """
        diagnostics = GateDiagnostics()

        # Gate-0: 先验检查
        if self.config.gate0_enabled and context:
            ns = context.get("namespace", "")
            if ns in self.config.gate0_disabled_namespaces:
                diagnostics.gate0_passed = False
                return torch.zeros(
                    hidden_states.shape[0], hidden_states.shape[1],
                    device=hidden_states.device, dtype=hidden_states.dtype
                ), diagnostics

        # Gate-1: 熵计算 + 置信门控
        entropy = self._compute_uncertainty(hidden_states)
        gate = torch.sigmoid(self.gate_w1 * entropy + self.gate_bias)
        gate = self._apply_entropy_veto(gate, entropy)

        diagnostics.entropy_mean = entropy.mean().item()
        diagnostics.gate_mean = gate.mean().item()
        diagnostics.gate_max = gate.max().item()
        diagnostics.veto_ratio = (gate == 0).float().mean().item()

        # Early Exit
        if self.config.early_exit_enabled:
            if diagnostics.gate_mean < self.config.early_exit_threshold:
                diagnostics.early_exit = True
                return torch.zeros_like(gate), diagnostics

        return gate, diagnostics

    def _compute_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算不确定性

        技术依据: core.py 第 220-244 行 _compute_hidden_uncertainty()
        使用 hidden_states 的内部方差作为不确定性信号，
        兼容 FlashAttention（不需要 attention weights）
        """
        # 方法 1: 内部方差
        mean = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean
        variance = (centered ** 2).mean(dim=-1)  # [batch, seq]

        log_var = torch.log1p(variance)
        normalized_var = log_var / (log_var.mean(dim=-1, keepdim=True) + 1e-6)

        # 方法 2: 学习的投影（如果可用）
        if hasattr(self, "uncertainty_proj"):
            learned = self.uncertainty_proj(hidden_states).squeeze(-1)
            combined = normalized_var * 0.5 + torch.sigmoid(learned) * 2.5
        else:
            combined = normalized_var * 2.0

        return torch.clamp(combined, 0, 5.0)

    def _apply_entropy_veto(
        self, gate: torch.Tensor, entropy: torch.Tensor
    ) -> torch.Tensor:
        """
        熵否决机制 (论文 Eq. 3)

        技术依据: core.py 第 687-691 行 _apply_entropy_veto()
        - entropy < tau_low: 模型自信 → gate 强制为 0
        - entropy > tau_high: 模型极度不确定 → gate 限制在 max_gate
        """
        gate = torch.where(entropy < self.tau_low, torch.zeros_like(gate), gate)
        gate = torch.where(
            entropy > self.tau_high, torch.clamp(gate, max=self.max_gate), gate
        )
        return gate

    def update_thresholds(self, **kwargs):
        """运行时更新阈值（供治理系统调用）"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
