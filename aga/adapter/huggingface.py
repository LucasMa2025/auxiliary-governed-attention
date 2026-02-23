"""
aga/adapter/huggingface.py — HuggingFace Transformers 适配器

源码映射:
  - 来自 core.py AGAAugmentedTransformerLayer.forward() (第 1131-1176 行)
  - 支持 LLaMA / Qwen / Mistral / GPT-2 / GPT-J / Phi 等架构

设计要点:
  - 通过 register_forward_hook 注入 AGA
  - 在 self_attn 输出后、MLP 之前注入 AGA
  - 自动检测模型架构
"""
from typing import List, Callable, Any

import torch
import torch.nn as nn

from .base import LLMAdapter
from ..exceptions import AdapterError


class HuggingFaceAdapter(LLMAdapter):
    """
    HuggingFace Transformers 适配器

    支持的模型架构:
      - LLaMA / LLaMA-2 / LLaMA-3
      - Qwen / Qwen-2
      - Mistral / Mixtral
      - GPT-2 / GPT-J / GPT-NeoX
      - Phi / Phi-2 / Phi-3
      - Gemma
      - Falcon

    使用方式:
        adapter = HuggingFaceAdapter()
        layers = adapter.get_layers(model)
        hook = adapter.wrap_layer(model, layer_idx=-1, aga_forward=my_forward)
    """

    # 已知的模型架构映射
    _LAYER_PATHS = [
        # LLaMA / Qwen / Mistral / Gemma
        ("model", "layers"),
        # GPT-2 / GPT-J
        ("transformer", "h"),
        # GPT-NeoX
        ("gpt_neox", "layers"),
        # Falcon
        ("transformer", "h"),
        # Bloom
        ("transformer", "h"),
    ]

    def get_layers(self, model: nn.Module) -> List[nn.Module]:
        """获取 Transformer 层列表"""
        for parent_attr, layers_attr in self._LAYER_PATHS:
            parent = getattr(model, parent_attr, None)
            if parent is not None:
                layers = getattr(parent, layers_attr, None)
                if layers is not None:
                    return list(layers)

        # 尝试直接访问
        if hasattr(model, "layers"):
            return list(model.layers)

        raise AdapterError(
            f"不支持的模型架构: {type(model).__name__}。"
            f"请实现自定义 LLMAdapter。"
            f"已尝试的路径: {self._LAYER_PATHS}"
        )

    def get_hidden_dim(self, model: nn.Module) -> int:
        """获取隐藏维度"""
        if hasattr(model, "config"):
            config = model.config
            # 尝试多种属性名
            for attr in ["hidden_size", "d_model", "n_embd", "dim"]:
                if hasattr(config, attr):
                    return getattr(config, attr)
        return 4096  # 默认值

    def wrap_layer(
        self,
        model: nn.Module,
        layer_idx: int,
        aga_forward: Callable,
    ) -> Any:
        """
        通过 register_forward_hook 注入 AGA

        技术依据: core.py AGAAugmentedTransformerLayer.forward()
        在 self_attn 输出后注入 AGA
        """
        layers = self.get_layers(model)

        if layer_idx < 0 or layer_idx >= len(layers):
            raise AdapterError(
                f"层索引 {layer_idx} 超出范围 [0, {len(layers)})"
            )

        layer = layers[layer_idx]

        def hook_fn(module, input, output):
            """
            Forward hook: 在 Transformer 层输出后注入 AGA

            output 通常是 (hidden_states, ...) 的 tuple
            input 通常是 (hidden_states, ...) 的 tuple
            """
            # 提取 output hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # 提取 input hidden_states（用于熵计算）
            if isinstance(input, tuple):
                input_hidden = input[0]
            else:
                input_hidden = input

            # 调用 AGA forward
            try:
                fused = aga_forward(
                    hidden_states=input_hidden,
                    primary_attention_output=hidden_states,
                )
            except Exception:
                # Fail-Open: 出错时返回原始输出
                return output

            # 重新组装输出
            if isinstance(output, tuple):
                return (fused,) + output[1:]
            return fused

        # 优先注册到 self_attn 子模块
        if hasattr(layer, "self_attn"):
            return layer.self_attn.register_forward_hook(hook_fn)
        else:
            # 回退到整个层
            return layer.register_forward_hook(hook_fn)
