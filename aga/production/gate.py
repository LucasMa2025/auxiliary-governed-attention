"""
AGA 三段式门控系统

Gate-0: 先验门控（零成本）
  - 基于 namespace、app_id、route 决定是否启用 AGA
  - 目标：直接挡掉 80-90% 请求

Gate-1: 置信门控（轻量）
  - 基于 hidden state 计算置信度
  - confidence < threshold 时 bypass

Gate-2: Top-k 路由
  - 只在 Gate-1 通过时执行
  - 固定 k，chunked 计算
"""
import math
import logging
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GateConfig, UncertaintySource

logger = logging.getLogger(__name__)


class GateResult(str, Enum):
    """门控结果"""
    DISABLED = "disabled"  # 完全禁用 AGA
    BYPASS = "bypass"      # 跳过 AGA（低置信度）
    POSSIBLE = "possible"  # 可能使用 AGA
    REQUIRED = "required"  # 强制使用 AGA
    PASS = "pass"          # 通过门控


@dataclass
class GateContext:
    """门控上下文"""
    namespace: str
    app_id: Optional[str] = None
    route: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GateDiagnostics:
    """门控诊断信息"""
    gate0_result: GateResult
    gate1_confidence: Optional[float] = None
    gate1_result: Optional[GateResult] = None
    gate2_top_indices: Optional[List[int]] = None
    gate2_scores: Optional[List[float]] = None
    final_gate: Optional[float] = None
    early_exit: bool = False


class Gate0(nn.Module):
    """
    先验门控（零成本）
    
    基于静态规则决定是否启用 AGA，不涉及任何计算。
    """
    
    def __init__(self, config: GateConfig):
        super().__init__()
        self.config = config
        self.enabled = config.gate0_enabled
        self.disabled_namespaces = set(config.gate0_disabled_namespaces)
        self.required_namespaces = set(config.gate0_required_namespaces)
    
    def forward(self, context: GateContext) -> GateResult:
        """
        先验门控判断
        
        Returns:
            DISABLED: 完全禁用
            REQUIRED: 强制启用
            POSSIBLE: 继续到 Gate-1
        """
        if not self.enabled:
            return GateResult.POSSIBLE
        
        # 检查禁用列表
        if context.namespace in self.disabled_namespaces:
            return GateResult.DISABLED
        
        # 检查强制启用列表
        if context.namespace in self.required_namespaces:
            return GateResult.REQUIRED
        
        # 可以根据 app_id、route 等添加更多规则
        # 例如：公共 API 禁用，私域 API 启用
        
        return GateResult.POSSIBLE
    
    def add_disabled_namespace(self, namespace: str):
        """动态添加禁用 namespace"""
        self.disabled_namespaces.add(namespace)
    
    def remove_disabled_namespace(self, namespace: str):
        """动态移除禁用 namespace"""
        self.disabled_namespaces.discard(namespace)


class Gate1(nn.Module):
    """
    置信门控（轻量级）
    
    基于 hidden state 计算不确定性/置信度，
    低置信度时跳过 AGA 计算。
    """
    
    def __init__(self, config: GateConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        self.enabled = config.gate1_enabled
        self.threshold = config.gate1_threshold
        self.source = config.gate1_uncertainty_source
        
        # 轻量投影头（用于计算置信度）
        if self.source in (UncertaintySource.HIDDEN_VARIANCE, UncertaintySource.LEARNED_PROJECTION):
            self.confidence_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 8),
                nn.GELU(),
                nn.Linear(hidden_dim // 8, 1),
            )
            # 初始化为输出接近 0.5 的值
            nn.init.zeros_(self.confidence_proj[-1].weight)
            nn.init.constant_(self.confidence_proj[-1].bias, 0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[GateResult, float]:
        """
        置信门控判断
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            logits: [batch, seq, vocab_size] 可选
        
        Returns:
            (GateResult, confidence)
        """
        if not self.enabled:
            return GateResult.PASS, 1.0
        
        # 计算不确定性
        uncertainty = self._compute_uncertainty(hidden_states, logits)
        
        # 转换为置信度（不确定性高 → 置信度高，需要 AGA）
        # 这里的逻辑是：模型越不确定，越需要外部知识
        confidence = uncertainty.mean().item()
        
        if confidence < self.threshold:
            return GateResult.BYPASS, confidence
        
        return GateResult.PASS, confidence
    
    def _compute_uncertainty(
        self,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算不确定性信号"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if self.source == UncertaintySource.LOGITS_ENTROPY and logits is not None:
            # Logits 熵
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            # 归一化
            return entropy / math.log(logits.size(-1))
        
        elif self.source == UncertaintySource.CONSTANT:
            return torch.ones(batch_size, seq_len, device=hidden_states.device) * 0.5
        
        else:
            # Hidden Variance (默认)
            # 计算每个 token 的内部方差
            mean = hidden_states.mean(dim=-1, keepdim=True)
            variance = ((hidden_states - mean) ** 2).mean(dim=-1)
            
            # 归一化
            log_var = torch.log1p(variance)
            normalized = log_var / (log_var.mean() + 1e-6)
            
            # 如果有学习的投影，结合使用
            if hasattr(self, 'confidence_proj'):
                learned = self.confidence_proj(hidden_states).squeeze(-1)
                combined = normalized * 0.5 + torch.sigmoid(learned) * 0.5
            else:
                combined = normalized
            
            return torch.clamp(combined, 0, 1)


class Gate2(nn.Module):
    """
    Top-k 路由门控
    
    选择最相关的 k 个槽位进行注意力计算。
    使用分块计算避免 OOM。
    """
    
    def __init__(self, config: GateConfig, bottleneck_dim: int, num_slots: int):
        super().__init__()
        self.config = config
        self.top_k = min(config.gate2_top_k, num_slots)
        self.chunk_size = config.gate2_chunk_size
        self.bottleneck_dim = bottleneck_dim
        
        # 路由投影
        self.router_dim = min(bottleneck_dim, 48)
        self.router_proj = nn.Linear(bottleneck_dim, self.router_dim, bias=False)
        nn.init.orthogonal_(self.router_proj.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        slot_keys: torch.Tensor,
        reliability_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-k 路由选择
        
        Args:
            query: [batch, seq, bottleneck_dim]
            slot_keys: [num_slots, bottleneck_dim]
            reliability_mask: [num_slots] log reliability
        
        Returns:
            top_indices: [batch, seq, k]
            router_scores: [batch, seq, k]
        """
        # 投影到路由空间
        router_query = self.router_proj(query)  # [batch, seq, router_dim]
        router_keys = self.router_proj(slot_keys)  # [num_slots, router_dim]
        
        # 路由分数
        router_scores = torch.matmul(router_query, router_keys.T)
        router_scores = router_scores / math.sqrt(self.router_dim)
        
        # 应用可靠性掩码
        if reliability_mask is not None:
            router_scores = router_scores + reliability_mask.unsqueeze(0).unsqueeze(0)
        
        # 选择 top-k
        top_scores, top_indices = torch.topk(router_scores, self.top_k, dim=-1)
        
        return top_indices, top_scores


class GateChain(nn.Module):
    """
    门控链：串联 Gate-0/1/2
    
    实现三段式门控的完整流程。
    """
    
    def __init__(
        self,
        config: GateConfig,
        hidden_dim: int,
        bottleneck_dim: int,
        num_slots: int,
    ):
        super().__init__()
        self.config = config
        
        # 三段门控
        self.gate0 = Gate0(config)
        self.gate1 = Gate1(config, hidden_dim)
        self.gate2 = Gate2(config, bottleneck_dim, num_slots)
        
        # 熵门控参数（用于最终融合）
        self.gate_w1 = nn.Parameter(torch.tensor(0.5))
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        
        # 诊断信息
        self._last_diagnostics: Optional[GateDiagnostics] = None
    
    def forward(
        self,
        context: GateContext,
        hidden_states: torch.Tensor,
        slot_keys: torch.Tensor,
        reliability_mask: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], GateDiagnostics]:
        """
        完整门控流程
        
        Returns:
            top_indices: 选中的槽位索引，None 表示跳过 AGA
            final_gate: 最终门控值
            diagnostics: 诊断信息
        """
        diagnostics = GateDiagnostics(gate0_result=GateResult.POSSIBLE)
        
        # Gate-0: 先验门控
        gate0_result = self.gate0(context)
        diagnostics.gate0_result = gate0_result
        
        if gate0_result == GateResult.DISABLED:
            diagnostics.early_exit = True
            self._last_diagnostics = diagnostics
            return None, None, diagnostics
        
        # Gate-1: 置信门控
        gate1_result, confidence = self.gate1(hidden_states, logits)
        diagnostics.gate1_confidence = confidence
        diagnostics.gate1_result = gate1_result
        
        if gate1_result == GateResult.BYPASS:
            diagnostics.early_exit = True
            self._last_diagnostics = diagnostics
            return None, None, diagnostics
        
        # Gate-2: Top-k 路由
        # 优先使用外部提供的 query（与 AGA q_proj 对齐）
        top_indices, router_scores = self.gate2(
            query if query is not None else self._project_query(hidden_states),
            slot_keys,
            reliability_mask,
        )
        diagnostics.gate2_top_indices = top_indices[0, 0].cpu().tolist()
        diagnostics.gate2_scores = router_scores[0, 0].cpu().tolist()
        
        # 计算最终门控值
        uncertainty = self._compute_uncertainty_for_gate(hidden_states)
        final_gate = torch.sigmoid(self.gate_w1 * uncertainty + self.gate_bias)
        final_gate = self._apply_entropy_veto(final_gate, uncertainty)
        diagnostics.final_gate = final_gate.mean().item()
        
        # Early Exit 检查
        if self.config.early_exit_enabled:
            if final_gate.mean().item() < self.config.early_exit_threshold:
                diagnostics.early_exit = True
                self._last_diagnostics = diagnostics
                return None, None, diagnostics
        
        self._last_diagnostics = diagnostics
        return top_indices, final_gate, diagnostics
    
    def _project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """投影 hidden states 到 bottleneck 空间"""
        # 简单的线性投影（实际使用时应该用 AGA 的 q_proj）
        return hidden_states[..., :self.gate2.bottleneck_dim]
    
    def _compute_uncertainty_for_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算用于门控的不确定性"""
        mean = hidden_states.mean(dim=-1, keepdim=True)
        variance = ((hidden_states - mean) ** 2).mean(dim=-1)
        log_var = torch.log1p(variance)
        return log_var / (log_var.mean() + 1e-6) * 2.0
    
    def _apply_entropy_veto(
        self,
        gate: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """应用熵否决"""
        gate = torch.where(entropy < self.config.tau_low, torch.zeros_like(gate), gate)
        gate = torch.where(entropy > self.config.tau_high, torch.clamp(gate, max=0.8), gate)
        return gate
    
    def get_last_diagnostics(self) -> Optional[GateDiagnostics]:
        """获取最后一次诊断信息"""
        return self._last_diagnostics

