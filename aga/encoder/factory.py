"""
编码器工厂

提供统一的编码器创建接口。
"""

from typing import Dict, List, Any, Optional, Type
import logging

from .base import BaseEncoder, EncoderType, EncoderConfig, EncoderSignature
from .adapters import (
    HashEncoder,
    EmbeddingLayerEncoder,
    OpenAIEncoder,
    OpenAICompatibleEncoder,
    SentenceTransformersEncoder,
    OllamaEncoder,
    VLLMEncoder,
)

logger = logging.getLogger(__name__)


class EncoderFactory:
    """
    编码器工厂
    
    根据配置创建合适的编码器实例。
    
    使用方式:
    ```python
    # 从配置创建
    config = EncoderConfig(encoder_type=EncoderType.OPENAI, api_key="sk-xxx")
    encoder = EncoderFactory.create(config)
    
    # 从类型和参数创建
    encoder = EncoderFactory.create_from_type(
        EncoderType.OPENAI_COMPATIBLE,
        base_url="https://api.deepseek.com/v1",
        api_key="sk-xxx",
        model="deepseek-embedding",
    )
    
    # 从服务商创建
    encoder = EncoderFactory.create_from_provider("deepseek", "sk-xxx")
    
    # 自动选择最佳可用编码器
    encoder = EncoderFactory.create_best_available()
    ```
    """
    
    # 编码器类映射
    ENCODER_CLASSES: Dict[EncoderType, Type[BaseEncoder]] = {
        EncoderType.HASH: HashEncoder,
        EncoderType.EMBEDDING_LAYER: EmbeddingLayerEncoder,
        EncoderType.OPENAI: OpenAIEncoder,
        EncoderType.OPENAI_COMPATIBLE: OpenAICompatibleEncoder,
        EncoderType.SENTENCE_TRANSFORMERS: SentenceTransformersEncoder,
        EncoderType.OLLAMA: OllamaEncoder,
        EncoderType.VLLM: VLLMEncoder,
    }
    
    # 编码器优先级（用于自动选择）
    ENCODER_PRIORITY = [
        EncoderType.OPENAI,
        EncoderType.OPENAI_COMPATIBLE,
        EncoderType.VLLM,
        EncoderType.SENTENCE_TRANSFORMERS,
        EncoderType.OLLAMA,
        EncoderType.EMBEDDING_LAYER,
        EncoderType.HASH,
    ]
    
    # 自定义编码器注册
    _custom_encoders: Dict[str, Type[BaseEncoder]] = {}
    
    @classmethod
    def create(cls, config: EncoderConfig) -> BaseEncoder:
        """
        从配置创建编码器
        
        Args:
            config: 编码器配置
            
        Returns:
            编码器实例
        """
        encoder_class = cls.ENCODER_CLASSES.get(config.encoder_type)
        
        if encoder_class is None:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")
        
        encoder = encoder_class(config)
        logger.info(f"Created {config.encoder_type.value} encoder")
        return encoder
    
    @classmethod
    def create_from_type(
        cls,
        encoder_type: EncoderType,
        key_dim: int = 64,
        value_dim: int = 4096,
        **kwargs
    ) -> BaseEncoder:
        """
        从类型和参数创建编码器
        
        Args:
            encoder_type: 编码器类型
            key_dim: Key 向量目标维度
            value_dim: Value 向量目标维度
            **kwargs: 其他配置参数
            
        Returns:
            编码器实例
        """
        config = EncoderConfig(
            encoder_type=encoder_type,
            key_dim=key_dim,
            value_dim=value_dim,
            model=kwargs.pop("model", None),
            base_url=kwargs.pop("base_url", None),
            api_key=kwargs.pop("api_key", None),
            provider=kwargs.pop("provider", None),
            device=kwargs.pop("device", "cpu"),
            native_dim=kwargs.pop("native_dim", None),
            extra=kwargs,
        )
        return cls.create(config)
    
    @classmethod
    def create_from_provider(
        cls,
        provider: str,
        api_key: str,
        key_dim: int = 64,
        value_dim: int = 4096,
        **kwargs
    ) -> BaseEncoder:
        """
        从服务商名称创建编码器
        
        简化国产大模型的配置。
        
        支持的服务商:
        - deepseek: DeepSeek
        - qwen: 通义千问
        - zhipu: 智谱 GLM
        - moonshot: Moonshot/Kimi
        - baichuan: 百川
        - yi: 零一万物
        - siliconflow: SiliconFlow
        
        Args:
            provider: 服务商名称
            api_key: API 密钥
            key_dim: Key 向量目标维度
            value_dim: Value 向量目标维度
            **kwargs: 其他参数
            
        Returns:
            编码器实例
        """
        return OpenAICompatibleEncoder.from_provider(
            provider, 
            api_key, 
            key_dim=key_dim,
            value_dim=value_dim,
            **kwargs
        )
    
    @classmethod
    def create_from_env(cls, prefix: str = "AGA_ENCODER_") -> BaseEncoder:
        """
        从环境变量创建编码器
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            编码器实例
        """
        config = EncoderConfig.from_env(prefix)
        return cls.create(config)
    
    @classmethod
    def create_best_available(
        cls,
        key_dim: int = 64,
        value_dim: int = 4096,
        **kwargs
    ) -> BaseEncoder:
        """
        创建最佳可用编码器
        
        按优先级尝试创建编码器，返回第一个可用的。
        
        Returns:
            可用的最佳编码器实例
        """
        for encoder_type in cls.ENCODER_PRIORITY:
            try:
                encoder = cls.create_from_type(
                    encoder_type, 
                    key_dim=key_dim, 
                    value_dim=value_dim,
                    **kwargs
                )
                if encoder.is_available:
                    logger.info(f"Using {encoder_type.value} encoder")
                    return encoder
            except Exception as e:
                logger.debug(f"Encoder {encoder_type.value} not available: {e}")
                continue
        
        # 回退到哈希编码器（始终可用）
        logger.warning("No semantic encoder available, falling back to hash encoder")
        return cls.create_from_type(EncoderType.HASH, key_dim=key_dim, value_dim=value_dim)
    
    @classmethod
    def register_custom(cls, name: str, encoder_class: Type[BaseEncoder]):
        """
        注册自定义编码器
        
        Args:
            name: 编码器名称
            encoder_class: 编码器类
        """
        if not issubclass(encoder_class, BaseEncoder):
            raise TypeError("encoder_class must be a subclass of BaseEncoder")
        
        cls._custom_encoders[name] = encoder_class
        logger.info(f"Registered custom encoder: {name}")
    
    @classmethod
    def list_available(cls) -> List[Dict[str, Any]]:
        """
        列出所有可用的编码器
        
        Returns:
            编码器信息列表
        """
        result = []
        
        for encoder_type in EncoderType:
            if encoder_type == EncoderType.CUSTOM:
                continue
            
            encoder_class = cls.ENCODER_CLASSES.get(encoder_type)
            if encoder_class is None:
                continue
            
            try:
                config = EncoderConfig(encoder_type=encoder_type)
                encoder = encoder_class(config)
                result.append({
                    "type": encoder_type.value,
                    "native_dim": encoder.native_dim,
                    "is_available": encoder.is_available,
                    "requires_api_key": encoder_type in [
                        EncoderType.OPENAI,
                        EncoderType.OPENAI_COMPATIBLE,
                    ],
                    "requires_local_model": encoder_type in [
                        EncoderType.SENTENCE_TRANSFORMERS,
                        EncoderType.OLLAMA,
                        EncoderType.VLLM,
                        EncoderType.EMBEDDING_LAYER,
                    ],
                })
            except Exception as e:
                result.append({
                    "type": encoder_type.value,
                    "is_available": False,
                    "error": str(e),
                })
        
        return result
    
    @classmethod
    def get_signature(cls, encoder: BaseEncoder) -> EncoderSignature:
        """获取编码器签名"""
        return encoder.get_signature()
    
    @classmethod
    def verify_encoder_consistency(
        cls,
        encoder: BaseEncoder,
        expected_signature: Dict[str, Any],
    ) -> tuple:
        """
        验证编码器一致性
        
        Args:
            encoder: 当前编码器
            expected_signature: 期望的编码器签名
            
        Returns:
            (is_consistent, message)
        """
        return encoder.verify_compatibility(expected_signature)
