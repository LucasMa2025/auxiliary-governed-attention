"""
AGA 多头注意力并行优化与 FlashAttention 集成

实现高性能的注意力计算优化：
1. 多头注意力并行优化 - 利用 GPU 并行性
2. FlashAttention 深度集成 - 内存高效的注意力计算
3. 分块计算优化 - 支持超长序列

版本: v3.4.1
"""
import math
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 检测 FlashAttention 可用性
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    HAS_FLASH_ATTN = True
    FLASH_ATTN_VERSION = "2.x"
except ImportError:
    try:
        # 尝试旧版本 API
        from flash_attn.flash_attention import FlashAttention
        HAS_FLASH_ATTN = True
        FLASH_ATTN_VERSION = "1.x"
    except ImportError:
        HAS_FLASH_ATTN = False
        FLASH_ATTN_VERSION = None

# 检测 xformers 可用性
try:
    import xformers.ops as xops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False


class AttentionBackend(str, Enum):
    """注意力计算后端"""
    STANDARD = "standard"           # PyTorch 标准实现
    FLASH_ATTENTION = "flash_attn"  # FlashAttention
    XFORMERS = "xformers"           # xFormers memory_efficient_attention
    SDPA = "sdpa"                   # PyTorch 2.0+ scaled_dot_product_attention


@dataclass
class ParallelAttentionConfig:
    """并行注意力配置"""
    # 基础配置
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128  # hidden_dim // num_heads
    
    # 后端选择
    backend: AttentionBackend = AttentionBackend.SDPA
    auto_select_backend: bool = True  # 自动选择最优后端
    
    # FlashAttention 配置
    flash_attn_causal: bool = False
    flash_attn_softmax_scale: Optional[float] = None
    flash_attn_dropout: float = 0.0
    
    # 分块配置
    chunk_size: int = 2048  # 序列分块大小
    enable_chunking: bool = True
    
    # 内存优化
    use_memory_efficient: bool = True
    gradient_checkpointing: bool = False
    
    # 多头并行
    parallel_heads: bool = True
    head_groups: int = 4  # 头分组数（用于分组注意力）


def get_available_backends() -> Dict[str, bool]:
    """获取可用的注意力后端"""
    backends = {
        "standard": True,
        "flash_attn": HAS_FLASH_ATTN,
        "xformers": HAS_XFORMERS,
        "sdpa": hasattr(F, "scaled_dot_product_attention"),
    }
    return backends


def select_best_backend(config: ParallelAttentionConfig) -> AttentionBackend:
    """自动选择最优后端"""
    available = get_available_backends()
    
    # 优先级: FlashAttention > SDPA > xFormers > Standard
    if available["flash_attn"] and torch.cuda.is_available():
        return AttentionBackend.FLASH_ATTENTION
    elif available["sdpa"]:
        return AttentionBackend.SDPA
    elif available["xformers"] and torch.cuda.is_available():
        return AttentionBackend.XFORMERS
    else:
        return AttentionBackend.STANDARD


class MultiHeadParallelAttention(nn.Module):
    """
    多头并行注意力模块
    
    特性：
    - 支持多种后端（FlashAttention, SDPA, xFormers）
    - 自动后端选择
    - 分块计算支持超长序列
    - 内存高效实现
    """
    
    def __init__(self, config: ParallelAttentionConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim or (config.hidden_dim // config.num_heads)
        
        # 自动选择后端
        if config.auto_select_backend:
            self.backend = select_best_backend(config)
            logger.info(f"Auto-selected attention backend: {self.backend.value}")
        else:
            self.backend = config.backend
        
        # 投影层
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)
        
        # 缩放因子
        self.scale = config.flash_attn_softmax_scale or (1.0 / math.sqrt(self.head_dim))
        
        # 统计
        self._stats = {
            "forward_count": 0,
            "flash_attn_used": 0,
            "fallback_count": 0,
            "chunked_count": 0,
        }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, 1, 1, seq_len] 或 [batch, seq_len]
            key_value_states: 可选的交叉注意力 KV
            output_attentions: 是否输出注意力权重
        
        Returns:
            output: [batch, seq_len, hidden_dim]
            attention_weights: 可选的注意力权重
        """
        self._stats["forward_count"] += 1
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影
        query = self.q_proj(hidden_states)
        
        if key_value_states is not None:
            key = self.k_proj(key_value_states)
            value = self.v_proj(key_value_states)
        else:
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        
        # 重塑为多头格式
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 根据后端选择计算方式
        if self.backend == AttentionBackend.FLASH_ATTENTION and HAS_FLASH_ATTN:
            output, attn_weights = self._flash_attention(
                query, key, value, attention_mask, output_attentions
            )
        elif self.backend == AttentionBackend.SDPA and hasattr(F, "scaled_dot_product_attention"):
            output, attn_weights = self._sdpa_attention(
                query, key, value, attention_mask, output_attentions
            )
        elif self.backend == AttentionBackend.XFORMERS and HAS_XFORMERS:
            output, attn_weights = self._xformers_attention(
                query, key, value, attention_mask, output_attentions
            )
        else:
            output, attn_weights = self._standard_attention(
                query, key, value, attention_mask, output_attentions
            )
        
        # 输出投影
        output = output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(output)
        
        return output, attn_weights
    
    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FlashAttention 实现"""
        self._stats["flash_attn_used"] += 1
        
        # FlashAttention 需要 [batch, seq, heads, head_dim] 格式
        # 已经是正确格式
        
        try:
            if FLASH_ATTN_VERSION == "2.x":
                # FlashAttention 2.x API
                output = flash_attn_func(
                    query,
                    key,
                    value,
                    dropout_p=self.config.flash_attn_dropout if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=self.config.flash_attn_causal,
                )
            else:
                # FlashAttention 1.x API (需要转置)
                query = query.transpose(1, 2)  # [batch, heads, seq, head_dim]
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                
                flash_attn = FlashAttention(softmax_scale=self.scale)
                output = flash_attn(query, key, value)
                output = output.transpose(1, 2)  # 转回 [batch, seq, heads, head_dim]
            
            # FlashAttention 不返回注意力权重
            return output, None
            
        except Exception as e:
            logger.warning(f"FlashAttention failed, falling back to standard: {e}")
            self._stats["fallback_count"] += 1
            return self._standard_attention(query, key, value, attention_mask, output_attentions)
    
    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch SDPA 实现"""
        # SDPA 需要 [batch, heads, seq, head_dim] 格式
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 处理 attention_mask
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [batch, seq] -> [batch, 1, 1, seq]
                attn_mask = attention_mask[:, None, None, :]
            attn_mask = attn_mask.to(dtype=query.dtype)
            # 转换为加性掩码
            attn_mask = (1.0 - attn_mask) * torch.finfo(query.dtype).min
        
        # 使用 SDPA
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=self.config.use_memory_efficient,
        ):
            output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=self.config.flash_attn_dropout if self.training else 0.0,
                scale=self.scale,
            )
        
        # 转回 [batch, seq, heads, head_dim]
        output = output.transpose(1, 2)
        
        return output, None
    
    def _xformers_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """xFormers memory_efficient_attention 实现"""
        # xFormers 需要 [batch, seq, heads, head_dim] 格式
        # 已经是正确格式
        
        # 处理 attention_mask
        attn_bias = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_bias = attention_mask[:, None, :, None]
            attn_bias = (1.0 - attn_bias.to(dtype=query.dtype)) * torch.finfo(query.dtype).min
        
        output = xops.memory_efficient_attention(
            query, key, value,
            attn_bias=attn_bias,
            scale=self.scale,
        )
        
        return output, None
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """标准注意力实现"""
        batch_size, seq_len, num_heads, head_dim = query.shape
        kv_seq_len = key.shape[1]
        
        # 转置为 [batch, heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores + (1.0 - attention_mask) * torch.finfo(attn_scores.dtype).min
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Dropout
        if self.training and self.config.flash_attn_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.config.flash_attn_dropout)
        
        # 加权求和
        output = torch.matmul(attn_weights, value)
        
        # 转回 [batch, seq, heads, head_dim]
        output = output.transpose(1, 2)
        
        if output_attentions:
            return output, attn_weights
        return output, None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "backend": self.backend.value,
            "flash_attn_available": HAS_FLASH_ATTN,
            "xformers_available": HAS_XFORMERS,
            "sdpa_available": hasattr(F, "scaled_dot_product_attention"),
        }


class AGAFlashAttention(nn.Module):
    """
    AGA 专用 FlashAttention 模块
    
    针对 AGA 的辅助注意力计算优化：
    - 支持稀疏槽位注意力
    - 支持可靠性加权
    - 内存高效的 top-k 路由
    """
    
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        num_slots: int,
        top_k: int = 8,
        use_flash: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_slots = num_slots
        self.top_k = top_k
        self.use_flash = use_flash and HAS_FLASH_ATTN
        
        self.scale = 1.0 / math.sqrt(bottleneck_dim)
        
        # 统计
        self._stats = {
            "forward_count": 0,
            "flash_used": 0,
        }
    
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        reliability: torch.Tensor,
        top_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: [batch, seq, bottleneck_dim]
            keys: [num_slots, bottleneck_dim]
            values: [num_slots, hidden_dim]
            reliability: [num_slots]
            top_indices: 可选的预计算 top-k 索引 [batch, seq, k]
        
        Returns:
            output: [batch, seq, hidden_dim]
            attn_weights: [batch, seq, k]
        """
        self._stats["forward_count"] += 1
        
        batch_size, seq_len, _ = query.shape
        device = query.device
        dtype = query.dtype
        
        # 如果没有预计算索引，计算 top-k
        if top_indices is None:
            # 计算路由分数
            router_scores = torch.matmul(query, keys.T)  # [batch, seq, num_slots]
            router_scores = router_scores + torch.log(reliability + 1e-10)
            
            # 选择 top-k
            _, top_indices = torch.topk(router_scores, self.top_k, dim=-1)
        
        # 收集选中的 keys 和 values
        # [batch, seq, k, bottleneck_dim]
        selected_keys = keys[top_indices]
        # [batch, seq, k, hidden_dim]
        selected_values = values[top_indices]
        
        # 计算注意力
        if self.use_flash and query.is_cuda:
            output, attn_weights = self._flash_sparse_attention(
                query, selected_keys, selected_values
            )
        else:
            output, attn_weights = self._standard_sparse_attention(
                query, selected_keys, selected_values
            )
        
        return output, attn_weights
    
    def _flash_sparse_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 FlashAttention 的稀疏注意力"""
        self._stats["flash_used"] += 1
        
        batch_size, seq_len, k, _ = keys.shape
        
        # 重塑为 FlashAttention 格式
        # query: [batch, seq, 1, bottleneck_dim] -> 作为单头
        query = query.unsqueeze(2)
        
        # 使用 FlashAttention
        try:
            output = flash_attn_func(
                query.to(torch.float16),
                keys.to(torch.float16),
                values.to(torch.float16),
                softmax_scale=self.scale,
            )
            output = output.squeeze(2).to(query.dtype)
            
            # FlashAttention 不返回权重，计算近似权重
            attn_scores = torch.sum(query * keys, dim=-1) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            return output, attn_weights
            
        except Exception as e:
            logger.warning(f"FlashAttention sparse failed: {e}")
            return self._standard_sparse_attention(query.squeeze(2), keys, values)
    
    def _standard_sparse_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """标准稀疏注意力"""
        # query: [batch, seq, bottleneck_dim]
        # keys: [batch, seq, k, bottleneck_dim]
        # values: [batch, seq, k, hidden_dim]
        
        query_expanded = query.unsqueeze(2)  # [batch, seq, 1, bottleneck_dim]
        
        # 计算注意力分数
        attn_scores = torch.sum(query_expanded * keys, dim=-1)  # [batch, seq, k]
        attn_scores = attn_scores * self.scale
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权求和
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch, seq, k, 1]
        output = torch.sum(attn_weights_expanded * values, dim=2)  # [batch, seq, hidden_dim]
        
        return output, attn_weights
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "flash_available": HAS_FLASH_ATTN,
            "using_flash": self.use_flash and HAS_FLASH_ATTN,
        }


class ChunkedAttention(nn.Module):
    """
    分块注意力模块
    
    用于处理超长序列，避免 OOM。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        chunk_size: int = 2048,
        overlap: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        分块注意力前向传播
        
        Args:
            query: [batch, seq, hidden_dim]
            key: [batch, seq, hidden_dim]
            value: [batch, seq, hidden_dim]
            attention_mask: 可选掩码
        
        Returns:
            output: [batch, seq, hidden_dim]
        """
        batch_size, seq_len, _ = query.shape
        
        # 如果序列长度小于分块大小，直接计算
        if seq_len <= self.chunk_size:
            return self._compute_attention(query, key, value, attention_mask)
        
        # 分块计算
        outputs = []
        
        for start in range(0, seq_len, self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, seq_len)
            
            # 扩展 key/value 范围以包含上下文
            kv_start = max(0, start - self.overlap)
            kv_end = min(seq_len, end + self.overlap)
            
            chunk_query = query[:, start:end]
            chunk_key = key[:, kv_start:kv_end]
            chunk_value = value[:, kv_start:kv_end]
            
            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, kv_start:kv_end]
            
            chunk_output = self._compute_attention(
                chunk_query, chunk_key, chunk_value, chunk_mask
            )
            
            outputs.append(chunk_output)
        
        # 合并输出（处理重叠区域）
        return self._merge_chunks(outputs, seq_len)
    
    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算单个块的注意力"""
        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]
        
        # 重塑为多头格式
        query = query.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores + (1.0 - attention_mask) * torch.finfo(attn_scores.dtype).min
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        
        # 转回原始格式
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_dim)
        
        return output
    
    def _merge_chunks(self, chunks: list, total_len: int) -> torch.Tensor:
        """合并分块输出"""
        if len(chunks) == 1:
            return chunks[0]
        
        batch_size = chunks[0].shape[0]
        device = chunks[0].device
        dtype = chunks[0].dtype
        
        output = torch.zeros(batch_size, total_len, self.hidden_dim, device=device, dtype=dtype)
        weights = torch.zeros(batch_size, total_len, 1, device=device, dtype=dtype)
        
        pos = 0
        for i, chunk in enumerate(chunks):
            chunk_len = chunk.shape[1]
            
            # 计算权重（重叠区域使用渐变权重）
            chunk_weight = torch.ones(1, chunk_len, 1, device=device, dtype=dtype)
            
            if i > 0:
                # 开始部分渐变
                fade_len = min(self.overlap, chunk_len)
                chunk_weight[:, :fade_len] = torch.linspace(0, 1, fade_len, device=device, dtype=dtype).view(1, -1, 1)
            
            if i < len(chunks) - 1:
                # 结束部分渐变
                fade_len = min(self.overlap, chunk_len)
                chunk_weight[:, -fade_len:] = torch.linspace(1, 0, fade_len, device=device, dtype=dtype).view(1, -1, 1)
            
            end = min(pos + chunk_len, total_len)
            actual_len = end - pos
            
            output[:, pos:end] += chunk[:, :actual_len] * chunk_weight[:, :actual_len]
            weights[:, pos:end] += chunk_weight[:, :actual_len]
            
            pos += self.chunk_size - self.overlap
        
        # 归一化
        output = output / (weights + 1e-10)
        
        return output


# 导出
__all__ = [
    "AttentionBackend",
    "ParallelAttentionConfig",
    "MultiHeadParallelAttention",
    "AGAFlashAttention",
    "ChunkedAttention",
    "get_available_backends",
    "select_best_backend",
    "HAS_FLASH_ATTN",
    "HAS_XFORMERS",
]
