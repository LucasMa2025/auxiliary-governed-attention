"""
aga/adapter/base.py — LLM 适配器抽象基类

源码映射:
  - 来自 core.py AGAManager.attach_to_model() (第 1190-1335 行)
  - 来自 core.py AGAAugmentedTransformerLayer (第 1112-1176 行)

设计要点:
  - 抽象基类定义统一接口
  - 具体适配器实现不同模型架构的挂载逻辑
  - 支持 HuggingFace / vLLM / 自定义模型
"""
from abc import ABC, abstractmethod
from typing import List, Callable, Any

import torch.nn as nn


class LLMAdapter(ABC):
    """
    LLM 适配器抽象基类

    实现此接口以支持新的模型架构。

    使用方式:
        class MyAdapter(LLMAdapter):
            def get_layers(self, model):
                return list(model.layers)

            def get_hidden_dim(self, model):
                return model.config.hidden_size

            def wrap_layer(self, model, layer_idx, aga_forward):
                layer = self.get_layers(model)[layer_idx]
                return layer.self_attn.register_forward_hook(...)
    """

    @abstractmethod
    def get_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        获取 Transformer 层列表

        Args:
            model: LLM 模型实例

        Returns:
            Transformer 层列表
        """
        ...

    @abstractmethod
    def get_hidden_dim(self, model: nn.Module) -> int:
        """
        获取隐藏维度

        Args:
            model: LLM 模型实例

        Returns:
            隐藏维度大小
        """
        ...

    @abstractmethod
    def wrap_layer(
        self,
        model: nn.Module,
        layer_idx: int,
        aga_forward: Callable,
    ) -> Any:
        """
        包装 Transformer 层，注入 AGA forward hook

        Args:
            model: LLM 模型实例
            layer_idx: 层索引
            aga_forward: AGA forward 函数

        Returns:
            hook handle (用于 detach 时移除)
        """
        ...
