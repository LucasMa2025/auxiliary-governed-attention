"""
aga/operator/bottleneck_injector.py — Bottleneck 注入器

源码映射:
  - 查询投影: core.py 第 376 行 q_proj
  - 注意力计算: core.py 第 530-545 行 (query @ keys.T → softmax → weighted values)
  - Value Projection: core.py 第 397-402 行 (value_down, value_up)
  - 路由: core.py SlotRouter 第 252-340 行

  这是 AGA 的核心注入路径，延迟 <0.1ms

数学流程:
  query = q_proj(hidden_states)           # [batch, seq, bottleneck_dim]
  scores = query @ keys.T / sqrt(d)       # [batch, seq, num_active_slots]
  weights = softmax(scores + log(reliability))
  aux_output = weights @ values            # [batch, seq, hidden_dim]
  aux_output = value_up(gelu(value_down(aux_output)))  # delta subspace

信息容量:
  - key: [bottleneck_dim=64] — 用于检索匹配
  - value: [hidden_dim=4096] — 实际注入的知识向量
  - 每个 value 可编码 10-50 tokens 的原子事实语义
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckInjector(nn.Module):
    """
    Bottleneck 注入器（核心路径）

    设计要点:
    - 查询投影到低维空间进行匹配
    - Top-K 路由减少计算量
    - Value Projection (delta subspace) 控制注入幅度
    - 可靠性偏置引导知识选择
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 64,
        value_bottleneck_dim: int = 256,
        top_k: int = 8,
        use_value_projection: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.top_k = top_k

        # 查询投影: hidden_dim → bottleneck_dim
        self.q_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)

        # Value Projection (delta subspace)
        # 通过瓶颈层控制注入幅度，防止过度干预
        self.use_value_projection = use_value_projection
        if use_value_projection:
            self.value_down = nn.Linear(hidden_dim, value_bottleneck_dim, bias=False)
            self.value_up = nn.Linear(value_bottleneck_dim, hidden_dim, bias=False)
            # 小初始化确保初始注入幅度小
            nn.init.xavier_uniform_(self.value_down.weight, gain=0.1)
            nn.init.xavier_uniform_(self.value_up.weight, gain=0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 bottleneck 注入

        Args:
            hidden_states: [batch, seq, hidden_dim]
            keys: [num_active, bottleneck_dim]
            values: [num_active, hidden_dim]
            reliability: [num_active]

        Returns:
            aux_output: [batch, seq, hidden_dim]
        """
        num_active = keys.shape[0]

        # 1. 查询投影
        query = self.q_proj(hidden_states)  # [batch, seq, bottleneck_dim]

        # 2. Top-K 路由（如果活跃槽位 > top_k）
        if num_active > self.top_k:
            keys, values, reliability = self._top_k_route(
                query, keys, values, reliability
            )

        # 3. 注意力计算
        scale = math.sqrt(self.bottleneck_dim)
        scores = torch.matmul(query, keys.t()) / scale  # [batch, seq, k]

        # 加入可靠性偏置
        reliability_bias = torch.log(reliability.clamp(min=1e-10))
        scores = scores + reliability_bias.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq, k]

        # 4. 加权求和
        aux_output = torch.matmul(attn_weights, values)  # [batch, seq, hidden_dim]

        # 5. Value Projection (delta subspace)
        if self.use_value_projection:
            aux_output = self.value_down(aux_output)
            aux_output = F.gelu(aux_output)
            aux_output = self.value_up(aux_output)

        return aux_output

    def _top_k_route(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        reliability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Top-K 路由选择最相关的 k 个槽位

        使用 query 均值作为路由信号，选择全局最相关的 top-k 个知识槽位。
        """
        # 使用 query 均值作为路由信号
        avg_query = query.mean(dim=(0, 1))  # [bottleneck_dim]
        scores = torch.matmul(keys, avg_query)  # [num_active]

        # 加入可靠性偏置
        scores = scores + torch.log(reliability.clamp(min=1e-10))

        k = min(self.top_k, keys.shape[0])
        _, top_indices = torch.topk(scores, k)

        return keys[top_indices], values[top_indices], reliability[top_indices]
