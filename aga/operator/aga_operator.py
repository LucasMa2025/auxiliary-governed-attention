"""
AGA 统一算子

整合三段式门控、持久化衰减、槽位池管理的完整 AGA 实现。

版本: v3.0
"""
import math
import time
import threading
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import Counter
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import (
    LifecycleState, UncertaintySource, GateResult,
    Slot, GateContext, AGADiagnostics, DecayContext, AGAForwardResult,
    LIFECYCLE_RELIABILITY,
)
from ..unified_config import AGAConfig

logger = logging.getLogger(__name__)


class UncertaintyEstimator(nn.Module):
    """
    不确定性估计器
    
    支持多种不确定性信号源。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        source: UncertaintySource = UncertaintySource.HIDDEN_VARIANCE,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.source = source
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # 学习的不确定性投影头
        if source in (UncertaintySource.HIDDEN_VARIANCE, UncertaintySource.LEARNED_PROJECTION):
            self.uncertainty_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            nn.init.zeros_(self.uncertainty_proj[-1].weight)
            nn.init.constant_(self.uncertainty_proj[-1].bias, 0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算不确定性信号
        
        Returns:
            uncertainty: [batch, seq] 范围约 [0, 3]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        if self.source == UncertaintySource.ATTENTION_ENTROPY:
            if attention_weights is None:
                return self._compute_hidden_uncertainty(hidden_states)
            return self._compute_attention_entropy(attention_weights)
        
        elif self.source == UncertaintySource.LOGITS_ENTROPY:
            if logits is None:
                return self._compute_hidden_uncertainty(hidden_states)
            return self._compute_logits_entropy(logits)
        
        elif self.source == UncertaintySource.LEARNED_PROJECTION:
            return self._compute_learned_uncertainty(hidden_states)
        
        elif self.source == UncertaintySource.HIDDEN_VARIANCE:
            return self._compute_hidden_uncertainty(hidden_states)
        
        elif self.source == UncertaintySource.ENSEMBLE:
            return self._compute_ensemble_uncertainty(hidden_states, attention_weights, logits)
        
        else:  # CONSTANT
            return torch.ones(batch_size, seq_len, device=device, dtype=dtype) * 1.0
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        avg_weights = attention_weights.mean(dim=1)
        log_weights = torch.log(avg_weights + 1e-10)
        entropy = -torch.sum(avg_weights * log_weights, dim=-1)
        return entropy
    
    def _compute_logits_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return torch.clamp(entropy / math.log(logits.size(-1)) * 3.0, 0, 5.0)
    
    def _compute_hidden_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean
        variance = (centered ** 2).mean(dim=-1)
        
        log_var = torch.log1p(variance)
        normalized_var = log_var / (log_var.mean() + 1e-6)
        
        if hasattr(self, 'uncertainty_proj'):
            learned = self.uncertainty_proj(hidden_states).squeeze(-1)
            combined = normalized_var * 0.5 + torch.sigmoid(learned) * 2.5
        else:
            combined = normalized_var * 2.0
        
        return torch.clamp(combined, 0, 5.0)
    
    def _compute_learned_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw = self.uncertainty_proj(hidden_states).squeeze(-1)
        return torch.sigmoid(raw) * 3.0
    
    def _compute_ensemble_uncertainty(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        uncertainties = [self._compute_hidden_uncertainty(hidden_states)]
        
        if attention_weights is not None:
            uncertainties.append(self._compute_attention_entropy(attention_weights))
        
        if logits is not None:
            uncertainties.append(self._compute_logits_entropy(logits))
        
        return torch.stack(uncertainties).mean(dim=0)


class SlotRouter(nn.Module):
    """
    槽位路由器
    
    实现 Top-k 路由，O(k) 复杂度。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        num_slots: int,
        top_k: int = 8,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_slots = num_slots
        self.top_k = min(top_k, num_slots)
        self.chunk_size = chunk_size
        
        self.router_dim = min(bottleneck_dim, 48)
        self.router_proj = nn.Linear(bottleneck_dim, self.router_dim, bias=False)
        nn.init.orthogonal_(self.router_proj.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        aux_keys: torch.Tensor,
        reliability_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        路由选择 top-k 槽位
        
        Returns:
            selected_indices: [batch, seq, k]
            router_scores: [batch, seq, k]
        """
        router_query = self.router_proj(query)
        router_keys = self.router_proj(aux_keys)
        
        router_scores = torch.matmul(router_query, router_keys.T)
        router_scores = router_scores / math.sqrt(self.router_dim)
        router_scores = router_scores + reliability_mask.unsqueeze(0).unsqueeze(0)
        
        top_scores, top_indices = torch.topk(router_scores, self.top_k, dim=-1)
        
        return top_indices, top_scores


class AGAOperator(nn.Module):
    """
    AGA 统一算子
    
    整合所有 AGA 功能的生产级实现：
    - 三段式门控 (Gate-0/1/2)
    - 持久化衰减
    - 槽位池管理
    - 诊断和监控
    """
    
    def __init__(self, config: Optional[AGAConfig] = None):
        super().__init__()
        
        self.config = config or AGAConfig()
        
        # 快捷访问
        self.hidden_dim = self.config.slot_pool.hidden_dim
        self.bottleneck_dim = self.config.slot_pool.bottleneck_dim
        self.num_heads = self.config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_slots = self.config.slot_pool.max_slots
        self.top_k = self.config.gate.gate2_top_k
        
        # 查询投影
        self.q_proj = nn.Linear(self.hidden_dim, self.bottleneck_dim, bias=False)
        
        # 辅助键值存储
        self.register_buffer(
            'aux_keys',
            torch.randn(self.num_slots, self.bottleneck_dim) * 0.01
        )
        self.register_buffer(
            'aux_values',
            torch.zeros(self.num_slots, self.hidden_dim)
        )
        
        # 路由器
        self.router = SlotRouter(
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            num_slots=self.num_slots,
            top_k=self.top_k,
            chunk_size=self.config.gate.gate2_chunk_size,
        )
        
        # Value Projection
        if self.config.slot_pool.use_value_projection:
            self.value_down = nn.Linear(
                self.hidden_dim, 
                self.config.slot_pool.value_bottleneck_dim, 
                bias=False
            )
            self.value_up = nn.Linear(
                self.config.slot_pool.value_bottleneck_dim, 
                self.hidden_dim, 
                bias=False
            )
            nn.init.xavier_uniform_(self.value_down.weight, gain=0.1)
            nn.init.xavier_uniform_(self.value_up.weight, gain=0.1)
        
        # 不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator(
            hidden_dim=self.hidden_dim,
            source=self.config.gate.gate1_uncertainty_source,
        )
        
        # 熵门控参数
        self.gate_w1 = nn.Parameter(torch.tensor(0.5))
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        
        # 槽位元数据
        self.slot_lifecycle: List[LifecycleState] = [LifecycleState.QUARANTINED] * self.num_slots
        self.slot_lu_ids: List[Optional[str]] = [None] * self.num_slots
        self.slot_conditions: List[Optional[str]] = [None] * self.num_slots
        self.slot_decisions: List[Optional[str]] = [None] * self.num_slots
        self.slot_hit_counts: List[int] = [0] * self.num_slots
        self.slot_consecutive_misses: List[int] = [0] * self.num_slots
        
        # 缓存
        self._cached_reliability: Optional[torch.Tensor] = None
        self._active_slot_indices: Optional[List[int]] = None
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 日志钩子
        self._logger: Optional[Callable] = None
    
    def register_logging_hook(self, logger_fn: Callable[[AGADiagnostics], None]):
        """注册日志钩子"""
        self._logger = logger_fn
    
    def _invalidate_cache(self):
        """使缓存失效"""
        self._cached_reliability = None
        self._active_slot_indices = None
    
    def _get_reliability_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """获取可靠性张量"""
        if self._cached_reliability is None:
            self._cached_reliability = torch.tensor(
                [LIFECYCLE_RELIABILITY[state] for state in self.slot_lifecycle],
                device=device, dtype=dtype
            )
        return self._cached_reliability.to(device=device, dtype=dtype)
    
    def _get_active_slot_indices(self) -> List[int]:
        """获取活跃槽位索引"""
        if self._active_slot_indices is None:
            self._active_slot_indices = [
                i for i, state in enumerate(self.slot_lifecycle)
                if state != LifecycleState.QUARANTINED
            ]
        return self._active_slot_indices
    
    # ==================== 三段式门控 ====================
    
    def _gate0_check(self, context: Optional[GateContext] = None) -> GateResult:
        """
        Gate-0: 先验门控（零成本）
        
        基于 namespace/app_id/route 决定是否启用 AGA。
        """
        if not self.config.gate.gate0_enabled:
            return GateResult.PASS
        
        if context is None:
            return GateResult.POSSIBLE
        
        # 检查禁用列表
        if context.namespace in self.config.gate.gate0_disabled_namespaces:
            return GateResult.DISABLED
        
        # 检查强制启用列表
        if context.namespace in self.config.gate.gate0_required_namespaces:
            return GateResult.REQUIRED
        
        return GateResult.POSSIBLE
    
    def _gate1_check(
        self, 
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[GateResult, torch.Tensor, float]:
        """
        Gate-1: 置信门控（轻量）
        
        基于不确定性信号决定是否需要 AGA 干预。
        """
        if not self.config.gate.gate1_enabled:
            ones = torch.ones(
                hidden_states.shape[0], hidden_states.shape[1],
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            return GateResult.PASS, ones, 1.0
        
        # 计算不确定性
        uncertainty = self.uncertainty_estimator(hidden_states, attention_weights, logits)
        
        # 计算门控值
        gate = torch.sigmoid(self.gate_w1 * uncertainty + self.gate_bias)
        gate = self._apply_entropy_veto(gate, uncertainty)
        
        gate_mean = gate.mean().item()
        
        # Early Exit 检查
        if self.config.gate.early_exit_enabled:
            if gate_mean < self.config.gate.early_exit_threshold:
                return GateResult.BYPASS, gate, gate_mean
        
        return GateResult.PASS, gate, gate_mean
    
    def _apply_entropy_veto(self, gate: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """应用熵否决"""
        tau_low = self.config.gate.tau_low
        tau_high = self.config.gate.tau_high
        max_gate = self.config.gate.max_gate
        
        gate = torch.where(entropy < tau_low, torch.zeros_like(gate), gate)
        gate = torch.where(entropy > tau_high, torch.clamp(gate, max=max_gate), gate)
        return gate
    
    # ==================== 前向传播 ====================
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        primary_attention_output: torch.Tensor,
        primary_attention_weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        context: Optional[GateContext] = None,
        decay_context: Optional[DecayContext] = None,
        return_diagnostics: bool = False,
    ) -> AGAForwardResult:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            primary_attention_output: [batch, seq, hidden_dim]
            primary_attention_weights: [batch, heads, seq, seq] 可选
            logits: [batch, seq, vocab_size] 可选
            context: 门控上下文
            decay_context: 衰减上下文
            return_diagnostics: 是否返回诊断信息
        
        Returns:
            AGAForwardResult
        """
        start_time = time.time()
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        diagnostics = AGADiagnostics() if return_diagnostics else None
        
        try:
            # Gate-0: 先验门控
            gate0_result = self._gate0_check(context)
            if diagnostics:
                diagnostics.gate0_result = gate0_result
            
            if gate0_result == GateResult.DISABLED:
                return AGAForwardResult(
                    output=primary_attention_output,
                    diagnostics=diagnostics,
                    aga_applied=False,
                )
            
            # 检查活跃槽位
            active_count = self.get_active_slots()
            if active_count == 0:
                return AGAForwardResult(
                    output=primary_attention_output,
                    diagnostics=diagnostics,
                    aga_applied=False,
                )
            
            if diagnostics:
                diagnostics.active_slots = active_count
            
            # Gate-1: 置信门控
            gate1_result, gate, gate_mean = self._gate1_check(
                hidden_states, primary_attention_weights, logits
            )
            
            if diagnostics:
                diagnostics.gate1_result = gate1_result
                diagnostics.gate1_confidence = gate_mean
                diagnostics.gate_mean = gate_mean
                diagnostics.gate_max = gate.max().item()
            
            if gate1_result == GateResult.BYPASS:
                if diagnostics:
                    diagnostics.early_exit = True
                    diagnostics.early_exit_ratio = 1.0
                return AGAForwardResult(
                    output=primary_attention_output,
                    diagnostics=diagnostics,
                    aga_applied=False,
                )
            
            # 应用持久化衰减
            if decay_context is not None and self.config.decay.enabled:
                gate = self._apply_decay(gate, decay_context)
            
            # Gate-2: Top-k 路由
            query = self.q_proj(hidden_states)
            reliability = self._get_reliability_tensor(device, dtype)
            reliability_mask = torch.log(reliability + 1e-10)
            
            if active_count > self.top_k:
                aux_output, routed_slots, router_scores = self._forward_with_routing(
                    query, reliability_mask, batch_size, seq_len, device, dtype
                )
            else:
                aux_output, routed_slots, router_scores = self._forward_full(
                    query, reliability_mask
                )
            
            if diagnostics:
                diagnostics.routed_slots = routed_slots
                diagnostics.router_scores = router_scores
            
            # Value Projection
            if self.config.slot_pool.use_value_projection:
                aux_output = self.value_down(aux_output)
                aux_output = F.gelu(aux_output)
                aux_output = self.value_up(aux_output)
            
            # 融合输出
            fused = primary_attention_output + gate.unsqueeze(-1) * aux_output
            
            # 更新命中计数
            if not self.training and routed_slots:
                self._update_hit_counts(routed_slots)
            
            latency_ms = (time.time() - start_time) * 1000
            
            if diagnostics:
                diagnostics.aga_applied = True
                diagnostics.latency_ms = latency_ms
                diagnostics.final_gate = gate.detach()
            
            # 调用日志钩子
            if self._logger and diagnostics:
                self._logger(diagnostics)
            
            return AGAForwardResult(
                output=fused,
                diagnostics=diagnostics,
                aga_applied=True,
                latency_ms=latency_ms,
            )
        
        except Exception as e:
            logger.error(f"AGA forward error: {e}")
            
            # Fail-open: 返回原始输出
            if self.config.fail_open_enabled:
                return AGAForwardResult(
                    output=primary_attention_output,
                    diagnostics=diagnostics,
                    aga_applied=False,
                    error=str(e),
                )
            raise
    
    def _apply_decay(self, gate: torch.Tensor, decay_context: DecayContext) -> torch.Tensor:
        """应用持久化衰减"""
        strategy = self.config.decay.strategy.value
        gamma = self.config.decay.gamma
        threshold = self.config.decay.hard_reset_threshold
        
        gate_mean = gate.mean().item()
        decay_context.accumulated_gate += gate_mean
        
        # 硬重置检查
        if decay_context.accumulated_gate > threshold:
            decay_context.hard_reset_triggered = True
            decay_context.accumulated_gate = 0.0
            return torch.zeros_like(gate)
        
        # 计算衰减因子
        if strategy == "exponential":
            decay_factor = gamma ** decay_context.layer_idx
        elif strategy == "linear":
            decay_factor = max(0, 1 - decay_context.layer_idx * (1 - gamma))
        else:  # adaptive
            decay_factor = gamma ** (decay_context.accumulated_gate / threshold)
        
        effective_gate = gate * decay_factor
        decay_context.effective_gate = effective_gate
        decay_context.record(gate_mean)
        
        return effective_gate
    
    def _forward_with_routing(
        self,
        query: torch.Tensor,
        reliability_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """带路由的前向传播"""
        chunk_size = self.config.gate.gate2_chunk_size
        
        top_indices, router_scores = self.router(query, self.aux_keys, reliability_mask)
        
        # 分块收集
        selected_keys = torch.zeros(
            batch_size, seq_len, self.top_k, self.bottleneck_dim,
            device=device, dtype=dtype
        )
        selected_values = torch.zeros(
            batch_size, seq_len, self.top_k, self.hidden_dim,
            device=device, dtype=dtype
        )
        
        for b_start in range(0, batch_size, chunk_size):
            b_end = min(b_start + chunk_size, batch_size)
            for s_start in range(0, seq_len, chunk_size):
                s_end = min(s_start + chunk_size, seq_len)
                
                chunk_indices = top_indices[b_start:b_end, s_start:s_end]
                flat_indices = chunk_indices.reshape(-1)
                
                chunk_keys = self.aux_keys[flat_indices]
                chunk_values = self.aux_values[flat_indices]
                
                chunk_b = b_end - b_start
                chunk_s = s_end - s_start
                selected_keys[b_start:b_end, s_start:s_end] = chunk_keys.view(
                    chunk_b, chunk_s, self.top_k, self.bottleneck_dim
                )
                selected_values[b_start:b_end, s_start:s_end] = chunk_values.view(
                    chunk_b, chunk_s, self.top_k, self.hidden_dim
                )
        
        # 计算注意力
        query_expanded = query.unsqueeze(2)
        attn_scores = torch.sum(query_expanded * selected_keys, dim=-1)
        attn_scores = attn_scores / math.sqrt(self.bottleneck_dim)
        attn_scores = attn_scores + router_scores * 0.1
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        aux_output = torch.sum(attn_weights_expanded * selected_values, dim=2)
        
        # 统计路由槽位
        all_indices = top_indices.reshape(-1).cpu().tolist()
        slot_counts = Counter(all_indices)
        routed_slots = [idx for idx, _ in slot_counts.most_common(20)]
        
        return aux_output, routed_slots, attn_weights.mean(dim=(0, 1)).detach()
    
    def _forward_full(
        self,
        query: torch.Tensor,
        reliability_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """完整前向传播"""
        attn_scores = torch.matmul(query, self.aux_keys.T) / math.sqrt(self.bottleneck_dim)
        attn_scores = attn_scores + reliability_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        aux_output = torch.matmul(attn_weights, self.aux_values)
        
        avg_weights = attn_weights.mean(dim=(0, 1))
        _, top_indices = torch.topk(avg_weights, min(20, self.num_slots))
        routed_slots = top_indices.cpu().tolist()
        
        return aux_output, routed_slots, avg_weights.detach()
    
    def _update_hit_counts(self, slot_indices: List[int]):
        """更新命中计数"""
        hit_set = set(slot_indices)
        
        with self._lock:
            for i in range(self.num_slots):
                if self.slot_lifecycle[i] == LifecycleState.QUARANTINED:
                    continue
                
                if i in hit_set:
                    self.slot_hit_counts[i] += 1
                    self.slot_consecutive_misses[i] = 0
                else:
                    self.slot_consecutive_misses[i] += 1
    
    # ==================== 知识注入接口 ====================
    
    def inject_knowledge(
        self,
        slot_idx: int,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lu_id: str,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> bool:
        """注入知识到指定槽位"""
        if self.training:
            raise RuntimeError("Cannot inject knowledge during training mode")
        
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise ValueError(f"slot_idx must be in [0, {self.num_slots})")
        
        with self._lock:
            # 处理维度
            if key_vector.dim() > 1:
                key_vector = key_vector.flatten()
            if value_vector.dim() > 1:
                value_vector = value_vector.flatten()
            
            # 调整维度
            if key_vector.shape[0] < self.bottleneck_dim:
                key_vector = F.pad(key_vector, (0, self.bottleneck_dim - key_vector.shape[0]))
            elif key_vector.shape[0] > self.bottleneck_dim:
                key_vector = key_vector[:self.bottleneck_dim]
            
            if value_vector.shape[0] < self.hidden_dim:
                value_vector = F.pad(value_vector, (0, self.hidden_dim - value_vector.shape[0]))
            elif value_vector.shape[0] > self.hidden_dim:
                value_vector = value_vector[:self.hidden_dim]
            
            with torch.no_grad():
                # 范数裁剪
                if self.config.slot_pool.enable_norm_clipping:
                    key_norm = key_vector.norm()
                    if key_norm > 0:
                        key_vector = key_vector / (key_norm + 1e-8) * self.config.slot_pool.key_norm_target
                    
                    value_norm = value_vector.norm()
                    if value_norm > 0:
                        value_vector = value_vector / (value_norm + 1e-8) * self.config.slot_pool.value_norm_target
                
                self.aux_keys[slot_idx] = key_vector.to(self.aux_keys.device)
                self.aux_values[slot_idx] = value_vector.to(self.aux_values.device)
            
            self.slot_lifecycle[slot_idx] = lifecycle_state
            self.slot_lu_ids[slot_idx] = lu_id
            self.slot_conditions[slot_idx] = condition
            self.slot_decisions[slot_idx] = decision
            self.slot_hit_counts[slot_idx] = 0
            self.slot_consecutive_misses[slot_idx] = 0
            self._invalidate_cache()
        
        return True
    
    def find_free_slot(self) -> Optional[int]:
        """查找空闲槽位"""
        with self._lock:
            for i, state in enumerate(self.slot_lifecycle):
                if state == LifecycleState.QUARANTINED:
                    return i
        return None
    
    def update_lifecycle(self, slot_idx: int, new_state: LifecycleState):
        """更新生命周期状态"""
        with self._lock:
            if slot_idx < 0 or slot_idx >= self.num_slots:
                raise ValueError(f"slot_idx out of range")
            self.slot_lifecycle[slot_idx] = new_state
            self._invalidate_cache()
    
    def quarantine_slot(self, slot_idx: int):
        """隔离槽位"""
        with self._lock:
            self.update_lifecycle(slot_idx, LifecycleState.QUARANTINED)
            with torch.no_grad():
                self.aux_values[slot_idx].zero_()
            self.slot_lu_ids[slot_idx] = None
            self.slot_conditions[slot_idx] = None
            self.slot_decisions[slot_idx] = None
    
    def quarantine_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 隔离"""
        quarantined = []
        with self._lock:
            for i, lid in enumerate(self.slot_lu_ids):
                if lid == lu_id:
                    self.quarantine_slot(i)
                    quarantined.append(i)
        return quarantined
    
    # ==================== 查询接口 ====================
    
    def get_active_slots(self) -> int:
        """获取活跃槽位数"""
        return sum(1 for s in self.slot_lifecycle if s != LifecycleState.QUARANTINED)
    
    def get_slot_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 查找槽位"""
        return [i for i, lid in enumerate(self.slot_lu_ids) if lid == lu_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        state_counts = {}
        for state in LifecycleState:
            state_counts[state.value] = sum(1 for s in self.slot_lifecycle if s == state)
        
        hit_counts = [c for c in self.slot_hit_counts if c > 0]
        
        return {
            'total_slots': self.num_slots,
            'active_slots': self.get_active_slots(),
            'state_distribution': state_counts,
            'total_hits': sum(self.slot_hit_counts),
            'avg_hit_count': sum(hit_counts) / len(hit_counts) if hit_counts else 0,
            'max_hit_count': max(hit_counts) if hit_counts else 0,
            'hidden_dim': self.hidden_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'top_k_routing': self.top_k,
        }
