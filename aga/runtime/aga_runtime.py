"""
AGA Runtime 核心模块

封装 AGA 推理逻辑，与本地缓存集成。
"""

import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

from ..config.runtime import AGAModuleConfig
from .cache import LocalCache


class AGARuntime:
    """
    AGA Runtime 模块
    
    封装 AGA 推理逻辑，从本地缓存读取知识。
    
    与传统 AGA 模块的区别：
    - 不直接管理槽位，使用 LocalCache
    - 支持从 Portal 同步更新
    - 更轻量级，专注推理
    """
    
    def __init__(
        self,
        config: AGAModuleConfig,
        cache: LocalCache,
        namespace: str = "default",
    ):
        """
        初始化 Runtime
        
        Args:
            config: AGA 模块配置
            cache: 本地缓存
            namespace: 命名空间
        """
        if not HAS_TORCH:
            raise ImportError("需要安装 PyTorch")
        
        self.config = config
        self.cache = cache
        self.namespace = namespace
        
        # 设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # 投影层
        self.query_proj = nn.Linear(config.hidden_dim, config.bottleneck_dim).to(self.device)
        self.output_proj = nn.Linear(config.bottleneck_dim, config.hidden_dim).to(self.device)
        
        # 熵门控
        self.entropy_threshold = config.entropy_threshold
        self.high_entropy_threshold = config.high_entropy_threshold
        
        # 统计
        self._stats = {
            "forward_count": 0,
            "aga_applied_count": 0,
            "bypass_count": 0,
        }
    
    def forward(
        self,
        hidden_states: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        return_diagnostics: bool = False,
    ) -> Tuple["torch.Tensor", Optional[Dict[str, Any]]]:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch, seq_len, hidden_dim]
            attention_mask: 注意力掩码
            return_diagnostics: 是否返回诊断信息
        
        Returns:
            (output, diagnostics) 元组
        """
        self._stats["forward_count"] += 1
        
        # 获取缓存向量
        key_matrix, value_matrix, reliability = self.cache.get_vectors(self.namespace)
        
        if key_matrix is None or len(key_matrix) == 0:
            # 无知识，直接返回
            if return_diagnostics:
                return hidden_states, {"aga_applied": False, "reason": "no_knowledge"}
            return hidden_states, None
        
        # 计算熵（简化版）
        entropy = self._compute_entropy(hidden_states)
        
        # 熵门控
        gate_mask = (entropy > self.entropy_threshold).float()
        
        if gate_mask.sum() < 0.1:
            # 熵太低，不需要 AGA
            self._stats["bypass_count"] += 1
            if return_diagnostics:
                return hidden_states, {"aga_applied": False, "reason": "low_entropy", "entropy_mean": entropy.mean().item()}
            return hidden_states, None
        
        # 投影到 bottleneck
        query = self.query_proj(hidden_states)  # [batch, seq, bottleneck]
        
        # 注意力计算
        # key_matrix: [num_slots, bottleneck]
        scores = torch.matmul(query, key_matrix.t()) / (self.config.bottleneck_dim ** 0.5)
        
        # 应用可靠性权重
        scores = scores * reliability.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Top-K 选择（确保 k 不超过实际槽位数）
        actual_k = min(self.config.top_k_slots, key_matrix.size(0))
        if actual_k < key_matrix.size(0):
            topk_values, topk_indices = torch.topk(attn_weights, actual_k, dim=-1)
            # 稀疏化
            sparse_weights = torch.zeros_like(attn_weights)
            sparse_weights.scatter_(-1, topk_indices, topk_values)
            attn_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # 加权求和
        aga_output = torch.matmul(attn_weights, value_matrix)  # [batch, seq, bottleneck]
        
        # 投影回 hidden_dim
        aga_output = self.output_proj(aga_output)
        
        # 应用门控
        gate_expanded = gate_mask.unsqueeze(-1)
        output = hidden_states + gate_expanded * aga_output
        
        self._stats["aga_applied_count"] += 1
        
        if return_diagnostics:
            diagnostics = {
                "aga_applied": True,
                "entropy_mean": entropy.mean().item(),
                "gate_mean": gate_mask.mean().item(),
                "active_slots": len(key_matrix),
                "top_slot_weights": attn_weights.max(dim=-1)[0].mean().item(),
            }
            return output, diagnostics
        
        return output, None
    
    def _compute_entropy(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        """
        计算隐藏状态的熵估计
        
        使用方差作为不确定性的代理指标。
        """
        # 归一化
        normalized = F.normalize(hidden_states, dim=-1)
        
        # 计算方差作为熵代理
        variance = torch.var(normalized, dim=-1)
        
        # 处理 NaN/Inf（数值稳定性）
        variance = torch.nan_to_num(variance, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 缩放到 [0, 1]
        entropy = torch.sigmoid(variance * 10 - 2)
        
        # 确保结果在 [0, 1] 范围内
        entropy = torch.clamp(entropy, 0.0, 1.0)
        
        return entropy
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "cache_stats": self.cache.get_stats(),
            "namespace": self.namespace,
        }
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {
            "forward_count": 0,
            "aga_applied_count": 0,
            "bypass_count": 0,
        }


class AGARuntimeLayer(nn.Module if HAS_TORCH else object):
    """
    AGA Runtime 层（可插入 Transformer）
    
    作为 nn.Module 使用，可以直接替换 Transformer 层。
    """
    
    def __init__(
        self,
        original_layer: "nn.Module",
        aga_runtime: AGARuntime,
    ):
        """
        初始化
        
        Args:
            original_layer: 原始 Transformer 层
            aga_runtime: AGA Runtime 实例
        """
        if not HAS_TORCH:
            raise ImportError("需要安装 PyTorch")
        
        super().__init__()
        self.original_layer = original_layer
        self.aga_runtime = aga_runtime
    
    def forward(
        self,
        hidden_states: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        **kwargs
    ) -> "torch.Tensor":
        """前向传播"""
        # 原始层输出
        original_output = self.original_layer(hidden_states, attention_mask=attention_mask, **kwargs)
        
        # 处理不同的输出格式
        if isinstance(original_output, tuple):
            hidden_out = original_output[0]
        elif isinstance(original_output, torch.Tensor):
            hidden_out = original_output
        else:
            # 不支持的输出类型，直接返回原始输出
            logger.warning(f"Unsupported output type from original layer: {type(original_output)}")
            return original_output
        
        # AGA 增强（带错误保护）
        try:
            enhanced, _ = self.aga_runtime.forward(hidden_out, attention_mask)
        except Exception as e:
            # Fail-open: 出错时返回原始输出
            logger.error(f"AGA runtime forward failed: {e}")
            return original_output
        
        # 返回相同格式
        if isinstance(original_output, tuple):
            return (enhanced,) + original_output[1:]
        return enhanced
