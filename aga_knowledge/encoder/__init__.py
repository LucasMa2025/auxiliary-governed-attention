"""
aga-knowledge 编码器模块

提供可配置的文本到向量编码能力，将明文 condition/decision
转换为 aga-core 所需的 key/value 向量。

设计原则:
  1. 配置驱动 — 编码器类型和参数通过配置指定
  2. 与 aga-core 一致 — 输出维度必须匹配 AGAConfig 的 bottleneck_dim 和 hidden_dim
  3. 外置式 — 支持多种编码后端（SentenceTransformer, HuggingFace, OpenAI, 自定义）
  4. 延迟加载 — 编码器在首次使用时才初始化，避免启动时的 GPU 占用
"""

from .base import BaseEncoder, EncoderConfig, EncodedKnowledge
from .sentence_transformer_encoder import SentenceTransformerEncoder
from .simple_encoder import SimpleHashEncoder

__all__ = [
    "BaseEncoder",
    "EncoderConfig",
    "EncodedKnowledge",
    "SentenceTransformerEncoder",
    "SimpleHashEncoder",
    "create_encoder",
]


def create_encoder(config: "EncoderConfig") -> "BaseEncoder":
    """
    根据配置创建编码器

    Args:
        config: 编码器配置

    Returns:
        BaseEncoder 实例
    """
    if config.backend == "sentence_transformer":
        return SentenceTransformerEncoder(config)
    elif config.backend == "simple_hash":
        return SimpleHashEncoder(config)
    elif config.backend == "none":
        # 不编码，要求知识已包含向量数据
        raise ValueError(
            "backend='none' 表示知识已包含向量数据，不需要编码器。"
            "请确保知识记录中包含 key_vector 和 value_vector 字段。"
        )
    else:
        raise ValueError(
            f"未知的编码器后端: {config.backend}。"
            f"支持: sentence_transformer, simple_hash, none"
        )
