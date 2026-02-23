"""
aga/retriever/base.py — 标准召回器协议

定义 AGA 的知识检索标准接口。所有外部知识源（Chroma, Milvus,
Elasticsearch, aga-knowledge, 自定义等）都必须实现此协议。

设计思想:
  - AGA 的高熵门控触发后，需要从外部获取推理所需的事实知识
  - hidden_states 是天然的语义查询信号（Transformer 最强的语义表征）
  - 召回器将 hidden_states 语义映射到外部知识库，返回可注入的 KV 对
  - 效果优先于性能：高熵触发频率低（每次推理 1-10 次），可容忍 1-10ms 延迟

使用方式:
    # 1. 使用内置 NullRetriever（默认，不召回外部知识）
    plugin = AGAPlugin(config)

    # 2. 使用 Chroma 召回器（用户实现）
    class ChromaRetriever(BaseRetriever):
        def retrieve(self, query): ...

    plugin = AGAPlugin(config, retriever=ChromaRetriever(...))

    # 3. 配置驱动
    # aga_config.yaml:
    #   retriever:
    #     backend: "chroma"
    #     endpoint: "localhost:8000"
    #     collection: "domain_knowledge"
"""
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQuery:
    """
    召回查询 — 从 AGA forward 路径传递给召回器

    包含高熵位置的语义信息，召回器据此检索相关知识。

    Attributes:
        hidden_states: 当前层的 hidden_states [batch, seq, hidden_dim]
                       这是 Transformer 的语义表征，包含完整上下文信息
        query_projected: q_proj 投影后的查询 [batch, seq, bottleneck_dim]
                         已对齐到 AGA 的知识空间，可直接用于向量检索
        entropy: 当前位置的熵值 [batch, seq]
                 反映模型的不确定性程度
        layer_idx: 当前 Transformer 层索引
        namespace: 可选的命名空间过滤
        metadata: 可选的额外上下文（如 request_id, session_id 等）
        top_k: 期望返回的最大结果数
    """
    hidden_states: torch.Tensor
    query_projected: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    layer_idx: int = 0
    namespace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    top_k: int = 5


@dataclass
class RetrievalResult:
    """
    召回结果 — 召回器返回的知识条目

    每个结果包含可直接注入 KVStore 的 key/value 对。

    Attributes:
        id: 知识唯一标识
        key: 检索键向量 [bottleneck_dim]，用于 AGA 的注意力匹配
        value: 知识值向量 [hidden_dim]，实际注入的语义信息
        reliability: 可靠性分数 (0.0-1.0)，影响注入权重
        score: 检索相关性分数（由召回器计算）
        metadata: 可选元数据（来源、时间戳、标签等）
    """
    id: str
    key: torch.Tensor
    value: torch.Tensor
    reliability: float = 1.0
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class BaseRetriever(ABC):
    """
    AGA 标准召回器协议

    所有外部知识源必须实现此接口。AGA 在高熵触发时调用 retrieve()
    获取相关知识，然后将结果注入 KVStore 供 BottleneckInjector 使用。

    生命周期:
        1. __init__: 初始化连接、索引等资源
        2. warmup(): 可选的预热（预加载索引、建立连接池等）
        3. retrieve(): 核心方法，每次高熵触发时调用
        4. on_injection_feedback(): 可选的反馈回调（知识是否被实际使用）
        5. shutdown(): 释放资源

    实现指南:
        - retrieve() 应在 1-10ms 内返回（高熵触发频率低，可容忍）
        - 返回的 key/value 必须与 AGAConfig 的维度匹配
        - 如果检索失败，返回空列表（Fail-Open）
        - 支持 namespace 过滤（多租户场景）
    """

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        核心检索方法

        在 AGA 高熵触发时被调用。根据 hidden_states 的语义信息，
        从外部知识库检索相关知识。

        Args:
            query: 包含 hidden_states、entropy、layer_idx 等信息的查询

        Returns:
            检索结果列表，每个结果包含可注入的 key/value 对。
            如果没有相关知识，返回空列表。

        Note:
            - 此方法在 GPU forward 路径中被调用，应尽量高效
            - 但效果优先于性能，1-10ms 延迟是可接受的
            - 如果发生异常，应在内部捕获并返回空列表（Fail-Open）
        """
        ...

    def warmup(self) -> None:
        """
        预热（可选）

        在 attach() 之后、推理开始之前调用。
        用于预加载索引、建立连接池、预编译查询等。
        """
        pass

    def on_injection_feedback(
        self,
        result_id: str,
        was_used: bool,
        gate_value: float = 0.0,
    ) -> None:
        """
        注入反馈回调（可选）

        当 AGA 完成一次 forward 后，通知召回器哪些知识被实际使用。
        可用于：
        - 更新知识的热度/优先级
        - 调整检索策略
        - 收集统计信息

        Args:
            result_id: 知识 ID
            was_used: 是否被 BottleneckInjector 实际使用（gate > 0）
            gate_value: 门控值（反映注入强度）
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        获取召回器统计信息（可选）

        Returns:
            包含调用次数、平均延迟、命中率等的字典
        """
        return {}

    def shutdown(self) -> None:
        """
        释放资源（可选）

        在 detach() 或插件销毁时调用。
        用于关闭连接、释放索引等。
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
