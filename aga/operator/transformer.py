"""
AGA Transformer 集成层

提供与 Transformer 模型的无缝集成。

版本: v3.0
"""
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn

from .aga_operator import AGAOperator
from ..types import UncertaintySource, DecayContext, GateContext

logger = logging.getLogger(__name__)


class AGAAugmentedTransformerLayer(nn.Module):
    """
    AGA 增强的 Transformer 层包装器
    
    将 AGA 模块无缝集成到 Transformer 层中。
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        aga_module: AGAOperator,
        require_attention_weights: bool = False,
    ):
        """
        初始化增强层
        
        Args:
            original_layer: 原始 Transformer 层
            aga_module: AGA 算子
            require_attention_weights: 是否需要注意力权重
        """
        super().__init__()
        self.original_layer = original_layer
        self.aga = aga_module
        self.require_attention_weights = require_attention_weights
        
        # 检测层结构
        self.has_input_layernorm = hasattr(original_layer, 'input_layernorm')
        self.has_post_attention_layernorm = hasattr(original_layer, 'post_attention_layernorm')
        
        # 衰减上下文（可选）
        self._decay_context: Optional[DecayContext] = None
        
        # 门控上下文（可选）
        self._gate_context: Optional[GateContext] = None
    
    def set_decay_context(self, context: DecayContext):
        """设置衰减上下文"""
        self._decay_context = context
    
    def set_gate_context(self, context: GateContext):
        """设置门控上下文"""
        self._gate_context = context
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            **kwargs: 其他参数
        
        Returns:
            输出元组
        """
        residual = hidden_states
        
        # 输入层归一化
        if self.has_input_layernorm:
            hidden_states = self.original_layer.input_layernorm(hidden_states)
        
        # 确定是否需要注意力权重
        output_attentions = self.require_attention_weights or \
            self.aga.config.gate.gate1_uncertainty_source == UncertaintySource.ATTENTION_ENTROPY
        
        # 原始自注意力
        attn_outputs = self.original_layer.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            **kwargs
        )
        
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1] if len(attn_outputs) > 1 and output_attentions else None
        
        # AGA 融合
        aga_result = self.aga(
            hidden_states=hidden_states,
            primary_attention_output=attn_output,
            primary_attention_weights=attn_weights,
            context=self._gate_context,
            decay_context=self._decay_context,
            return_diagnostics=self.aga.config.monitoring.enable_diagnostics,
        )
        
        fused_output = aga_result.output
        
        # 残差连接
        hidden_states = residual + fused_output
        
        # 后注意力处理
        residual = hidden_states
        
        if self.has_post_attention_layernorm:
            hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        
        # MLP
        if hasattr(self.original_layer, 'mlp'):
            hidden_states = self.original_layer.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # 更新衰减上下文
        if self._decay_context:
            self._decay_context.layer_idx += 1
        
        return (hidden_states,) + attn_outputs[1:]


class AGAModelWrapper(nn.Module):
    """
    AGA 模型包装器
    
    提供更高级的模型包装，支持：
    - 自动衰减上下文管理
    - 批量门控上下文
    """
    
    def __init__(
        self,
        model: nn.Module,
        aga_layers: dict,  # layer_idx -> AGAAugmentedTransformerLayer
    ):
        """
        初始化包装器
        
        Args:
            model: 原始模型
            aga_layers: AGA 增强层映射
        """
        super().__init__()
        self.model = model
        self.aga_layers = aga_layers
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gate_context: Optional[GateContext] = None,
        enable_decay: bool = True,
        **kwargs
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            gate_context: 门控上下文
            enable_decay: 是否启用衰减
            **kwargs: 其他参数
        """
        # 创建衰减上下文
        decay_context = DecayContext() if enable_decay else None
        
        # 设置上下文
        for layer in self.aga_layers.values():
            if decay_context:
                layer.set_decay_context(decay_context)
            if gate_context:
                layer.set_gate_context(gate_context)
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        gate_context: Optional[GateContext] = None,
        **kwargs
    ):
        """
        生成方法
        
        Args:
            input_ids: 输入 token IDs
            gate_context: 门控上下文
            **kwargs: 生成参数
        """
        # 设置门控上下文
        for layer in self.aga_layers.values():
            if gate_context:
                layer.set_gate_context(gate_context)
        
        return self.model.generate(input_ids=input_ids, **kwargs)
