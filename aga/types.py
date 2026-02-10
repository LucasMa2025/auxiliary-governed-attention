"""
AGA 共享类型定义

统一所有模块使用的枚举、数据类等，消除重复定义。
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

import torch


# ==================== 枚举定义 ====================

class LifecycleState(str, Enum):
    """
    知识槽位生命周期状态
    
    状态转换图：
    PROBATIONARY (0.3) → CONFIRMED (1.0) → DEPRECATED (0.1) → QUARANTINED (0.0)
                     ↘                  ↗
                       QUARANTINED (0.0)
    """
    PROBATIONARY = "probationary"  # 试用期 (r=0.3)
    CONFIRMED = "confirmed"        # 已确认 (r=1.0)
    DEPRECATED = "deprecated"      # 已弃用 (r=0.1)
    QUARANTINED = "quarantined"    # 已隔离 (r=0.0)


class UncertaintySource(str, Enum):
    """不确定性信号来源"""
    ATTENTION_ENTROPY = "attention_entropy"    # 注意力熵（需要 output_attentions）
    LOGITS_ENTROPY = "logits_entropy"          # logits 熵（推荐）
    HIDDEN_VARIANCE = "hidden_variance"        # 隐藏状态方差（FlashAttention 兼容）
    LEARNED_PROJECTION = "learned_projection"  # 学习的投影头
    MULTI_HEAD = "multi_head"                  # 多头熵
    ENSEMBLE = "ensemble"                      # 集成多种信号
    CONSTANT = "constant"                      # 固定值（测试用）


class GateResult(str, Enum):
    """门控结果"""
    DISABLED = "disabled"    # 完全禁用 AGA
    BYPASS = "bypass"        # 跳过 AGA（低置信度）
    POSSIBLE = "possible"    # 可能使用 AGA
    REQUIRED = "required"    # 强制使用 AGA
    PASS = "pass"            # 通过门控


class EvictionPolicy(str, Enum):
    """淘汰策略"""
    LRU = "lru"              # 最近最少使用
    HIT_COUNT = "hit_count"  # 命中次数最低
    HYBRID = "hybrid"        # 混合策略


class DecayStrategy(str, Enum):
    """衰减策略"""
    EXPONENTIAL = "exponential"  # 指数衰减
    LINEAR = "linear"            # 线性衰减
    ADAPTIVE = "adaptive"        # 自适应衰减
    NONE = "none"                # 不衰减


class PersistenceAdapter(str, Enum):
    """持久化适配器类型"""
    SQLITE = "sqlite"
    REDIS = "redis"
    POSTGRES = "postgres"
    COMPOSITE = "composite"
    MEMORY = "memory"


# ==================== 常量映射 ====================

LIFECYCLE_RELIABILITY: Dict[LifecycleState, float] = {
    LifecycleState.PROBATIONARY: 0.3,
    LifecycleState.CONFIRMED: 1.0,
    LifecycleState.DEPRECATED: 0.1,
    LifecycleState.QUARANTINED: 0.0,
}


# ==================== 数据类 ====================

@dataclass
class Slot:
    """
    知识槽位
    
    包含 key/value 向量和元数据。
    """
    slot_idx: int
    lu_id: str
    key_vector: torch.Tensor
    value_vector: torch.Tensor
    lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY
    namespace: str = "default"
    
    # 元数据
    condition: Optional[str] = None
    decision: Optional[str] = None
    
    # 统计信息
    hit_count: int = 0
    consecutive_misses: int = 0
    last_hit_ts: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # 版本控制
    version: int = 1
    
    @property
    def reliability(self) -> float:
        """获取可靠性值"""
        return LIFECYCLE_RELIABILITY.get(self.lifecycle_state, 0.0)
    
    @property
    def age_days(self) -> float:
        """获取槽位年龄（天）"""
        return (time.time() - self.created_at) / 86400
    
    def record_hit(self):
        """记录命中"""
        self.hit_count += 1
        self.consecutive_misses = 0
        self.last_hit_ts = time.time()
    
    def record_miss(self):
        """记录未命中"""
        self.consecutive_misses += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "slot_idx": self.slot_idx,
            "lu_id": self.lu_id,
            "key_vector": self.key_vector.cpu().tolist() if isinstance(self.key_vector, torch.Tensor) else self.key_vector,
            "value_vector": self.value_vector.cpu().tolist() if isinstance(self.value_vector, torch.Tensor) else self.value_vector,
            "lifecycle_state": self.lifecycle_state.value,
            "namespace": self.namespace,
            "condition": self.condition,
            "decision": self.decision,
            "hit_count": self.hit_count,
            "consecutive_misses": self.consecutive_misses,
            "last_hit_ts": self.last_hit_ts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device = None) -> "Slot":
        """从字典创建"""
        device = device or torch.device("cpu")
        return cls(
            slot_idx=data["slot_idx"],
            lu_id=data["lu_id"],
            key_vector=torch.tensor(data["key_vector"], device=device),
            value_vector=torch.tensor(data["value_vector"], device=device),
            lifecycle_state=LifecycleState(data["lifecycle_state"]),
            namespace=data.get("namespace", "default"),
            condition=data.get("condition"),
            decision=data.get("decision"),
            hit_count=data.get("hit_count", 0),
            consecutive_misses=data.get("consecutive_misses", 0),
            last_hit_ts=data.get("last_hit_ts", time.time()),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            version=data.get("version", 1),
        )


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
class KnowledgeSlotInfo:
    """知识槽位信息（只读视图）"""
    slot_idx: int
    lu_id: Optional[str]
    lifecycle_state: LifecycleState
    reliability: float
    key_norm: float
    value_norm: float
    condition: Optional[str] = None
    decision: Optional[str] = None
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    hit_count: int = 0
    consecutive_misses: int = 0


@dataclass
class AGADiagnostics:
    """AGA 诊断信息"""
    # 门控诊断
    gate0_result: Optional[GateResult] = None
    gate1_confidence: Optional[float] = None
    gate1_result: Optional[GateResult] = None
    
    # 路由诊断
    top_indices: Optional[List[int]] = None
    router_scores: Optional[torch.Tensor] = None
    routed_slots: Optional[List[int]] = None
    
    # 熵诊断
    entropy: Optional[torch.Tensor] = None
    final_gate: Optional[torch.Tensor] = None
    gate_mean: float = 0.0
    gate_max: float = 0.0
    
    # 衰减诊断
    decay_accumulated: float = 0.0
    decay_hard_reset: bool = False
    
    # 统计
    active_slots: int = 0
    early_exit: bool = False
    early_exit_ratio: float = 0.0
    aga_applied: bool = False
    latency_ms: float = 0.0
    
    # v3.1: 知识匹配诊断
    max_router_score: float = 0.0
    no_knowledge_match: bool = False
    no_match_triggered: bool = False
    no_match_behavior: Optional[str] = None
    
    # 错误信息
    error: Optional[str] = None


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
    gate_history: List[Dict[str, Any]] = field(default_factory=list)
    
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


@dataclass
class AGAForwardResult:
    """AGA 前向传播结果"""
    output: torch.Tensor
    diagnostics: Optional[AGADiagnostics] = None
    aga_applied: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    
    # v3.1: 知识匹配状态
    no_knowledge_match: bool = False  # 是否未匹配到相关知识

