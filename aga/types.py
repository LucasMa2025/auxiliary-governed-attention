"""
aga/types.py — AGA 数据类型定义

统一所有模块使用的数据类型。
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class KnowledgeEntry:
    """知识条目（用于批量注册）"""
    id: str
    key: Any  # torch.Tensor
    value: Any  # torch.Tensor
    reliability: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GateDiagnostics:
    """门控诊断信息"""
    gate0_passed: bool = True
    entropy_mean: float = 0.0
    gate_mean: float = 0.0
    gate_max: float = 0.0
    early_exit: bool = False
    veto_ratio: float = 0.0


@dataclass
class ForwardResult:
    """单次 forward 结果"""
    aga_applied: bool = False
    gate_mean: float = 0.0
    entropy_mean: float = 0.0
    layer_idx: int = 0
    latency_us: float = 0.0
    error: Optional[str] = None


@dataclass
class PluginDiagnostics:
    """插件诊断信息"""
    attached: bool = False
    knowledge_count: int = 0
    max_slots: int = 0
    hooked_layers: int = 0
    forward_total: int = 0
    forward_applied: int = 0
    forward_bypassed: int = 0
    activation_rate: float = 0.0
    gate_mean_avg: float = 0.0
    entropy_mean_avg: float = 0.0
    layer_stats: Dict[int, Dict] = field(default_factory=dict)
