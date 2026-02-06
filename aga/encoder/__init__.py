"""
AGA 编码器模块

提供统一的文本编码器接口，用于将文本转换为向量表示。

⚠️ 编码器一致性要求：
    - 注入时的编码器与推理时的编码器必须一致
    - 不同编码器产生的向量空间不同，混用会导致匹配失败
    - 编码器不必与 AGA 绑定的 LLM（生成模型）一致

支持的编码器类型：
    - hash: 哈希编码（测试用，无语义）
    - embedding_layer: 从 LLM 嵌入层提取（需要模型权重访问）
    - openai: OpenAI text-embedding
    - openai_compatible: OpenAI 兼容 API（DeepSeek/Qwen/智谱 等）
    - sentence_transformers: HuggingFace 本地模型
    - ollama: Ollama 本地模型
    - vllm: vLLM 本地部署

使用示例：
    ```python
    # 使用工厂创建编码器
    encoder = EncoderFactory.create("openai_compatible", 
        base_url="https://api.deepseek.com/v1",
        api_key="sk-xxx",
        model="deepseek-embedding",
    )
    
    # 编码文本
    vector = encoder.encode("Hello, world!")
    
    # 编码约束
    key_vec, value_vec = encoder.encode_constraint(
        condition="当用户询问天气时",
        decision="提供当地实时天气信息",
    )
    ```
"""

from .base import (
    BaseEncoder,
    EncoderType,
    EncoderConfig,
    EncoderSignature,
)
from .factory import EncoderFactory
from .adapters import (
    HashEncoder,
    EmbeddingLayerEncoder,
    OpenAIEncoder,
    OpenAICompatibleEncoder,
    SentenceTransformersEncoder,
    OllamaEncoder,
    VLLMEncoder,
)
from .cache import (
    CachedEncoder,
    PersistentCachedEncoder,
    CacheConfig,
    CacheStats,
    LRUCache,
)

__all__ = [
    # 基础类
    "BaseEncoder",
    "EncoderType",
    "EncoderConfig",
    "EncoderSignature",
    # 工厂
    "EncoderFactory",
    # 适配器
    "HashEncoder",
    "EmbeddingLayerEncoder",
    "OpenAIEncoder",
    "OpenAICompatibleEncoder",
    "SentenceTransformersEncoder",
    "OllamaEncoder",
    "VLLMEncoder",
    # 缓存
    "CachedEncoder",
    "PersistentCachedEncoder",
    "CacheConfig",
    "CacheStats",
    "LRUCache",
]
