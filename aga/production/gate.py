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
    
    支持自适应阈值：根据 bypass 率动态调整阈值。
    """
    
    def __init__(self, config: GateConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        self.enabled = config.gate1_enabled
        self.threshold = config.gate1_threshold
        self.source = config.gate1_uncertainty_source
        
        # 自适应阈值配置
        self._adaptive_threshold = getattr(config, 'gate1_adaptive_threshold', False)
        self._threshold_ema = config.gate1_threshold  # 指数移动平均
        self._threshold_alpha = 0.01  # 平滑系数
        self._target_bypass_rate = 0.15  # 目标 bypass 率
        self._threshold_min = 0.05
        self._threshold_max = 0.3
        
        # 统计信息
        self._bypass_count = 0
        self._total_count = 0
        
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
        
        # 更新统计
        self._total_count += 1
        
        # 自适应阈值调整
        if self._adaptive_threshold:
            current_bypass_rate = self._bypass_count / max(1, self._total_count)
            
            if current_bypass_rate < self._target_bypass_rate:
                # bypass 率过低，提高阈值
                self._threshold_ema = self._threshold_ema * (1 + self._threshold_alpha)
            elif current_bypass_rate > self._target_bypass_rate:
                # bypass 率过高，降低阈值
                self._threshold_ema = self._threshold_ema * (1 - self._threshold_alpha)
            
            # 限制阈值范围
            self._threshold_ema = max(self._threshold_min, min(self._threshold_max, self._threshold_ema))
            effective_threshold = self._threshold_ema
        else:
            effective_threshold = self.threshold
        
        if confidence < effective_threshold:
            self._bypass_count += 1
            return GateResult.BYPASS, confidence
        
        return GateResult.PASS, confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取 Gate1 统计信息"""
        return {
            "static_threshold": self.threshold,
            "effective_threshold": self._threshold_ema if self._adaptive_threshold else self.threshold,
            "adaptive_enabled": self._adaptive_threshold,
            "bypass_count": self._bypass_count,
            "total_count": self._total_count,
            "bypass_rate": self._bypass_count / max(1, self._total_count),
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self._bypass_count = 0
        self._total_count = 0
    
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
            entropy = -(probs * log_probs).sum(dim=-1)
            # 归一化（防止除零）
            norm = max(1.0, math.log(logits.size(-1)))
            return (entropy / norm).clamp(0.0, 1.0)
        
        elif self.source == UncertaintySource.CONSTANT:
            return torch.full(
                (batch_size, seq_len), 0.5, device=hidden_states.device, dtype=hidden_states.dtype
            )
        
        else:
            # Hidden Variance (默认) - 使用 var() API 更高效
            variance = hidden_states.var(dim=-1, unbiased=False)
            
            # 归一化（显式防护除零）
            log_var = torch.log1p(variance)
            denom = log_var.mean().clamp_min(1e-6)
            normalized = log_var / denom
            
            # 如果有学习的投影，结合使用
            if hasattr(self, 'confidence_proj'):
                learned = self.confidence_proj(hidden_states).squeeze(-1)
                combined = normalized * 0.5 + torch.sigmoid(learned) * 0.5
            else:
                combined = normalized
            
            return combined.clamp_(0.0, 1.0)  # 原地裁剪减少内存分配


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
        Top-k 路由选择（分块计算避免 OOM）
        
        Args:
            query: [batch, seq, bottleneck_dim]
            slot_keys: [num_slots, bottleneck_dim]
            reliability_mask: [num_slots] log reliability
        
        Returns:
            top_indices: [batch, seq, k]
            router_scores: [batch, seq, k]
        """
        batch_size, seq_len, _ = query.shape
        num_slots = slot_keys.size(0)
        
        # 边界检查：有效 k 值
        k = min(self.top_k, max(1, num_slots))
        
        # 投影到路由空间
        router_query = self.router_proj(query)  # [batch, seq, router_dim]
        router_keys = self.router_proj(slot_keys)  # [num_slots, router_dim]
        
        # 预计算缩放因子
        scale = 1.0 / math.sqrt(self.router_dim)
        
        # Flatten query 用于分块处理
        flat_query = router_query.reshape(-1, self.router_dim)  # [batch*seq, router_dim]
        total_queries = flat_query.size(0)
        
        # 分块计算避免 OOM
        top_scores_chunks = []
        top_indices_chunks = []
        
        for start in range(0, total_queries, self.chunk_size):
            end = min(start + self.chunk_size, total_queries)
            q_chunk = flat_query[start:end]  # [chunk, router_dim]
            
            # 计算路由分数
            scores = (q_chunk @ router_keys.T) * scale  # [chunk, num_slots]
            
            # 应用可靠性掩码
            if reliability_mask is not None:
                scores = scores + reliability_mask.unsqueeze(0)
            
            # 选择 top-k
            chunk_scores, chunk_indices = torch.topk(scores, k=k, dim=-1)
            top_scores_chunks.append(chunk_scores)
            top_indices_chunks.append(chunk_indices)
        
        # 拼接结果并重塑
        top_scores = torch.cat(top_scores_chunks, dim=0).view(batch_size, seq_len, k)
        top_indices = torch.cat(top_indices_chunks, dim=0).view(batch_size, seq_len, k)
        
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

