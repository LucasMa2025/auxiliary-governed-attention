"""
AGA 算子优化模块

提供多种性能优化实现：
- 混合精度支持 (AMP)
- CUDA Graph 优化
- 粒度掩码
- 故障恢复

版本: v1.0
"""
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .aga_operator import AGAOperator
    from ..unified_config import AGAConfig

logger = logging.getLogger(__name__)


# ==================== 混合精度支持 ====================

@dataclass
class MixedPrecisionConfig:
    """混合精度配置"""
    enabled: bool = True
    dtype: str = "float16"  # float16, bfloat16
    # 哪些组件使用低精度
    low_precision_keys: bool = True
    low_precision_values: bool = True
    low_precision_routing: bool = True
    # 保持高精度的组件
    high_precision_gate: bool = True
    high_precision_output: bool = True


class MixedPrecisionWrapper(nn.Module):
    """
    混合精度 AGA 包装器
    
    自动管理 AGA 算子的精度转换，优化内存和计算效率。
    
    使用示例：
        ```python
        aga = AGAOperator(config)
        mp_aga = MixedPrecisionWrapper(aga, MixedPrecisionConfig())
        
        # 自动使用混合精度
        output = mp_aga(hidden_states, ...)
        ```
    """
    
    def __init__(
        self,
        aga_operator: 'AGAOperator',
        config: Optional[MixedPrecisionConfig] = None,
    ):
        super().__init__()
        self.aga = aga_operator
        self.config = config or MixedPrecisionConfig()
        
        # 确定目标 dtype
        if self.config.dtype == "bfloat16":
            self.target_dtype = torch.bfloat16
        else:
            self.target_dtype = torch.float16
        
        # 转换存储
        if self.config.enabled:
            self._convert_storage()
    
    def _convert_storage(self):
        """转换存储精度"""
        if self.config.low_precision_keys and hasattr(self.aga, 'aux_keys'):
            self.aga.aux_keys.data = self.aga.aux_keys.data.to(self.target_dtype)
        
        if self.config.low_precision_values and hasattr(self.aga, 'aux_values'):
            self.aga.aux_values.data = self.aga.aux_values.data.to(self.target_dtype)
        
        logger.info(f"Converted AGA storage to {self.target_dtype}")
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """前向传播（自动混合精度）"""
        if not self.config.enabled:
            return self.aga(hidden_states, **kwargs)
        
        # 使用 autocast
        with torch.cuda.amp.autocast(
            enabled=True,
            dtype=self.target_dtype,
        ):
            result = self.aga(hidden_states, **kwargs)
        
        # 输出转回高精度
        if self.config.high_precision_output and hasattr(result, 'output'):
            result.output = result.output.float()
        
        return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        stats = {
            'enabled': self.config.enabled,
            'dtype': str(self.target_dtype),
        }
        
        if hasattr(self.aga, 'aux_keys'):
            keys_size = self.aga.aux_keys.element_size() * self.aga.aux_keys.numel()
            stats['keys_memory_mb'] = keys_size / (1024 * 1024)
        
        if hasattr(self.aga, 'aux_values'):
            values_size = self.aga.aux_values.element_size() * self.aga.aux_values.numel()
            stats['values_memory_mb'] = values_size / (1024 * 1024)
        
        return stats


# ==================== CUDA Graph 优化 ====================

class CUDAGraphWrapper(nn.Module):
    """
    CUDA Graph 优化包装器
    
    通过捕获和重放 CUDA Graph 减少 kernel launch 开销。
    
    注意：
    - 需要固定的输入形状
    - 不支持动态控制流
    - 首次调用会进行预热和捕获
    """
    
    def __init__(
        self,
        aga_operator: 'AGAOperator',
        warmup_iterations: int = 3,
        enabled: bool = True,
    ):
        super().__init__()
        self.aga = aga_operator
        self.warmup_iterations = warmup_iterations
        self.enabled = enabled and torch.cuda.is_available()
        
        # Graph 缓存 (按输入形状)
        self._graphs: Dict[Tuple[int, ...], torch.cuda.CUDAGraph] = {}
        self._static_inputs: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._static_outputs: Dict[Tuple[int, ...], Any] = {}
    
    def _get_shape_key(self, hidden_states: torch.Tensor) -> Tuple[int, ...]:
        """获取形状键"""
        return tuple(hidden_states.shape)
    
    def _build_graph(self, hidden_states: torch.Tensor, **kwargs) -> torch.cuda.CUDAGraph:
        """构建 CUDA Graph"""
        shape_key = self._get_shape_key(hidden_states)
        
        # 预热
        for _ in range(self.warmup_iterations):
            _ = self.aga(hidden_states, **kwargs)
        
        torch.cuda.synchronize()
        
        # 创建静态输入
        static_input = hidden_states.clone()
        self._static_inputs[shape_key] = static_input
        
        # 捕获 Graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = self.aga(static_input, **kwargs)
        
        self._static_outputs[shape_key] = static_output
        self._graphs[shape_key] = g
        
        logger.info(f"Built CUDA Graph for shape {shape_key}")
        return g
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """前向传播"""
        if not self.enabled or not hidden_states.is_cuda:
            return self.aga(hidden_states, **kwargs)
        
        shape_key = self._get_shape_key(hidden_states)
        
        # 检查是否有缓存的 Graph
        if shape_key not in self._graphs:
            self._build_graph(hidden_states, **kwargs)
        
        # 复制输入
        self._static_inputs[shape_key].copy_(hidden_states)
        
        # 重放 Graph
        self._graphs[shape_key].replay()
        
        # 返回输出的副本
        output = self._static_outputs[shape_key]
        if hasattr(output, 'output'):
            return type(output)(
                output=output.output.clone(),
                **{k: v for k, v in output.__dict__.items() if k != 'output'}
            )
        return output
    
    def clear_graphs(self):
        """清除所有缓存的 Graph"""
        self._graphs.clear()
        self._static_inputs.clear()
        self._static_outputs.clear()


# ==================== 粒度掩码 ====================

class GranularityLevel(str, Enum):
    """粒度级别"""
    GLOBAL = "global"      # 全局掩码
    TOKEN = "token"        # Token 级别
    HEAD = "head"          # 注意力头级别
    SUBSPACE = "subspace"  # 子空间级别


@dataclass
class GranularityConfig:
    """粒度掩码配置"""
    level: GranularityLevel = GranularityLevel.GLOBAL
    num_heads: int = 32
    learnable: bool = True


class GranularityMask(nn.Module):
    """
    粒度掩码模块
    
    实现论文 Eq. 4 中的 m，支持 Token/Head/Subspace 级别控制。
    
    使用示例：
        ```python
        mask = GranularityMask(hidden_dim=4096, config=GranularityConfig(
            level=GranularityLevel.HEAD,
            num_heads=32,
        ))
        
        # 应用掩码
        masked_gate = mask(hidden_states, base_gate)
        ```
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[GranularityConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or GranularityConfig()
        self.level = self.config.level
        self.num_heads = self.config.num_heads
        self.head_dim = hidden_dim // self.num_heads
        
        if self.level == GranularityLevel.TOKEN:
            # Token 级别：每个 token 独立的掩码
            self.mask_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )
        elif self.level == GranularityLevel.HEAD:
            # Head 级别：每个头独立的掩码
            self.mask_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, self.num_heads),
            )
        elif self.level == GranularityLevel.SUBSPACE:
            # Subspace 级别：每个子空间独立的掩码
            self.mask_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
        
        # 初始化为接近 1 的值（默认不遮蔽）
        if hasattr(self, 'mask_proj'):
            nn.init.zeros_(self.mask_proj[-1].weight)
            nn.init.constant_(self.mask_proj[-1].bias, 2.0)  # sigmoid(2) ≈ 0.88
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        base_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用粒度掩码
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            base_gate: [batch, seq] 基础门控值
        
        Returns:
            masked_gate: 根据粒度级别调整后的门控
        """
        if self.level == GranularityLevel.GLOBAL:
            return base_gate.unsqueeze(-1)  # [batch, seq, 1]
        
        elif self.level == GranularityLevel.TOKEN:
            # Token 级别掩码
            token_mask = torch.sigmoid(self.mask_proj(hidden_states)).squeeze(-1)
            return (base_gate * token_mask).unsqueeze(-1)
        
        elif self.level == GranularityLevel.HEAD:
            # Head 级别掩码
            head_mask = torch.sigmoid(self.mask_proj(hidden_states))  # [batch, seq, num_heads]
            return base_gate.unsqueeze(-1) * head_mask
        
        elif self.level == GranularityLevel.SUBSPACE:
            # Subspace 级别掩码
            subspace_mask = torch.sigmoid(self.mask_proj(hidden_states))  # [batch, seq, hidden_dim]
            return base_gate.unsqueeze(-1) * subspace_mask
        
        return base_gate.unsqueeze(-1)
    
    def get_mask_stats(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """获取掩码统计"""
        if self.level == GranularityLevel.GLOBAL:
            return {'level': 'global', 'mean_mask': 1.0}
        
        with torch.no_grad():
            if self.level == GranularityLevel.TOKEN:
                mask = torch.sigmoid(self.mask_proj(hidden_states)).squeeze(-1)
            elif self.level == GranularityLevel.HEAD:
                mask = torch.sigmoid(self.mask_proj(hidden_states))
            elif self.level == GranularityLevel.SUBSPACE:
                mask = torch.sigmoid(self.mask_proj(hidden_states))
            else:
                return {'level': self.level.value}
            
            return {
                'level': self.level.value,
                'mean_mask': mask.mean().item(),
                'min_mask': mask.min().item(),
                'max_mask': mask.max().item(),
            }


# ==================== 故障恢复 ====================

@dataclass
class ResilienceConfig:
    """故障恢复配置"""
    checkpoint_interval: int = 100      # 检查点间隔
    max_checkpoints: int = 3            # 最大检查点数
    fail_open_enabled: bool = True      # 失败时是否开放（返回原始输出）
    retry_attempts: int = 2             # 重试次数
    error_threshold: float = 0.01       # 错误率阈值


class ResilientAGAWrapper(nn.Module):
    """
    具有故障恢复能力的 AGA 包装器
    
    提供：
    - 定期检查点
    - 自动恢复
    - Fail-open 模式
    - 错误统计
    """
    
    def __init__(
        self,
        aga_operator: 'AGAOperator',
        config: Optional[ResilienceConfig] = None,
    ):
        super().__init__()
        self.aga = aga_operator
        self.config = config or ResilienceConfig()
        
        # 检查点
        self._checkpoints: deque = deque(maxlen=self.config.max_checkpoints)
        self._operation_count = 0
        
        # 统计
        self._error_count = 0
        self._total_count = 0
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """前向传播（带故障恢复）"""
        self._total_count += 1
        
        try:
            result = self.aga(hidden_states, **kwargs)
            
            # 定期检查点
            if self._operation_count % self.config.checkpoint_interval == 0:
                self._create_checkpoint()
            self._operation_count += 1
            
            return result
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"AGA forward failed: {e}")
            
            # 尝试恢复
            for attempt in range(self.config.retry_attempts):
                try:
                    if self._checkpoints:
                        self._restore_checkpoint()
                    result = self.aga(hidden_states, **kwargs)
                    logger.info(f"Recovery successful on attempt {attempt + 1}")
                    return result
                except Exception as retry_e:
                    logger.warning(f"Retry {attempt + 1} failed: {retry_e}")
            
            # Fail-open
            if self.config.fail_open_enabled:
                logger.warning("Fail-open: returning original hidden states")
                from ..types import AGAForwardResult
                return AGAForwardResult(
                    output=hidden_states,
                    aga_applied=False,
                    gate_mean=0.0,
                    diagnostics={'error': str(e), 'fail_open': True},
                )
            
            raise
    
    def _create_checkpoint(self):
        """创建检查点"""
        checkpoint = {}
        
        if hasattr(self.aga, 'aux_keys'):
            checkpoint['aux_keys'] = self.aga.aux_keys.data.clone()
        if hasattr(self.aga, 'aux_values'):
            checkpoint['aux_values'] = self.aga.aux_values.data.clone()
        if hasattr(self.aga, 'slot_lifecycle'):
            checkpoint['slot_lifecycle'] = self.aga.slot_lifecycle.copy()
        if hasattr(self.aga, 'slot_lu_ids'):
            checkpoint['slot_lu_ids'] = self.aga.slot_lu_ids.copy()
        
        self._checkpoints.append(checkpoint)
    
    def _restore_checkpoint(self):
        """恢复检查点"""
        if not self._checkpoints:
            return
        
        checkpoint = self._checkpoints[-1]
        
        if 'aux_keys' in checkpoint and hasattr(self.aga, 'aux_keys'):
            self.aga.aux_keys.data = checkpoint['aux_keys']
        if 'aux_values' in checkpoint and hasattr(self.aga, 'aux_values'):
            self.aga.aux_values.data = checkpoint['aux_values']
        if 'slot_lifecycle' in checkpoint and hasattr(self.aga, 'slot_lifecycle'):
            self.aga.slot_lifecycle = checkpoint['slot_lifecycle']
        if 'slot_lu_ids' in checkpoint and hasattr(self.aga, 'slot_lu_ids'):
            self.aga.slot_lu_ids = checkpoint['slot_lu_ids']
        
        logger.info("Checkpoint restored")
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        return self._error_count / max(1, self._total_count)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            'total_operations': self._total_count,
            'error_count': self._error_count,
            'error_rate': self.error_rate,
            'checkpoints_count': len(self._checkpoints),
            'fail_open_enabled': self.config.fail_open_enabled,
        }


# ==================== 健康监控 ====================

@dataclass
class HealthConfig:
    """健康监控配置"""
    latency_window_size: int = 100
    error_rate_threshold: float = 0.01
    latency_threshold_ms: float = 50.0
    gate_activation_threshold: float = 0.1
    slot_utilization_threshold: float = 0.9


class HealthMonitor:
    """
    AGA 健康监控
    
    监控 AGA 系统的运行状态，提供健康检查和告警。
    """
    
    def __init__(
        self,
        aga_operator: 'AGAOperator',
        config: Optional[HealthConfig] = None,
    ):
        self.aga = aga_operator
        self.config = config or HealthConfig()
        
        # 指标
        self._forward_count = 0
        self._error_count = 0
        self._latency_window: deque = deque(maxlen=self.config.latency_window_size)
        self._gate_mean_ema = 0.5
        self._last_check_time = time.time()
    
    def record_forward(
        self,
        latency_ms: float,
        success: bool,
        gate_mean: float,
    ):
        """记录前向传播"""
        self._forward_count += 1
        if not success:
            self._error_count += 1
        
        self._latency_window.append(latency_ms)
        
        # EMA 更新
        alpha = 0.1
        self._gate_mean_ema = alpha * gate_mean + (1 - alpha) * self._gate_mean_ema
    
    @property
    def avg_latency_ms(self) -> float:
        """平均延迟"""
        if not self._latency_window:
            return 0.0
        return sum(self._latency_window) / len(self._latency_window)
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        return self._error_count / max(1, self._forward_count)
    
    @property
    def active_slot_ratio(self) -> float:
        """活跃槽位比例"""
        if not hasattr(self.aga, 'get_active_slot_count'):
            return 0.0
        active = self.aga.get_active_slot_count()
        total = getattr(self.aga, 'num_slots', 1)
        return active / total
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        status = "healthy"
        issues = []
        
        # 检查错误率
        if self.error_rate > self.config.error_rate_threshold:
            status = "degraded"
            issues.append(f"High error rate: {self.error_rate:.2%}")
        
        # 检查延迟
        if self.avg_latency_ms > self.config.latency_threshold_ms:
            status = "degraded"
            issues.append(f"High latency: {self.avg_latency_ms:.1f}ms")
        
        # 检查槽位使用率
        if self.active_slot_ratio > self.config.slot_utilization_threshold:
            issues.append(f"Slot pool nearly full: {self.active_slot_ratio:.1%}")
        
        # 检查门控激活
        if self._gate_mean_ema < self.config.gate_activation_threshold:
            issues.append(f"Low gate activation: {self._gate_mean_ema:.2f}")
        
        return {
            'status': status,
            'metrics': {
                'forward_count': self._forward_count,
                'error_count': self._error_count,
                'error_rate': self.error_rate,
                'avg_latency_ms': self.avg_latency_ms,
                'gate_mean_ema': self._gate_mean_ema,
                'active_slot_ratio': self.active_slot_ratio,
            },
            'issues': issues,
            'last_check': self._last_check_time,
        }
    
    def reset(self):
        """重置统计"""
        self._forward_count = 0
        self._error_count = 0
        self._latency_window.clear()
        self._gate_mean_ema = 0.5
        self._last_check_time = time.time()
