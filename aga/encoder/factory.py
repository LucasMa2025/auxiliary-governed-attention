"""
编码器工厂

提供统一的编码器创建接口。

版本: v1.1

v1.1 更新:
- 增强错误处理和类型验证
- 改进 create_best_available 的回退逻辑
- 添加自定义编码器的接口验证
- 优化 list_available 的信息安全性
"""

from typing import Dict, List, Any, Optional, Tuple, Type
import logging
import os

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
            
        Raises:
            ValueError: 未知的编码器类型
            RuntimeError: 编码器初始化失败
        """
        encoder_class = cls.ENCODER_CLASSES.get(config.encoder_type)
        
        # 自定义编码器：通过 model 名称在 _custom_encoders 中查找
        if encoder_class is None and config.encoder_type == EncoderType.CUSTOM:
            custom_name = config.model or config.extra.get("custom_name")
            if custom_name and custom_name in cls._custom_encoders:
                encoder_class = cls._custom_encoders[custom_name]
            else:
                registered = list(cls._custom_encoders.keys()) or ["(none)"]
                raise ValueError(
                    f"Custom encoder '{custom_name}' not registered. "
                    f"Registered custom encoders: {registered}"
                )
        
        if encoder_class is None:
            available_types = [t.value for t in cls.ENCODER_CLASSES.keys()]
            if cls._custom_encoders:
                available_types.append(f"custom ({', '.join(cls._custom_encoders.keys())})")
            raise ValueError(
                f"Unknown encoder type: {config.encoder_type}. "
                f"Available types: {available_types}"
            )
        
        try:
            encoder = encoder_class(config)
            logger.info(f"Created {config.encoder_type.value} encoder")
            return encoder
        except Exception as e:
            logger.error(f"Failed to create {config.encoder_type.value} encoder: {e}")
            raise RuntimeError(f"Encoder initialization failed: {e}") from e
    
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
        tried_encoders = []
        
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
                else:
                    tried_encoders.append((encoder_type.value, "not available"))
            except Exception as e:
                tried_encoders.append((encoder_type.value, str(e)))
                logger.debug(f"Encoder {encoder_type.value} not available: {e}")
                continue
        
        # 回退到哈希编码器（始终可用）
        logger.warning(
            f"No semantic encoder available after trying {len(tried_encoders)} types, "
            "falling back to hash encoder. This may affect knowledge retrieval quality."
        )
        
        try:
            return cls.create_from_type(EncoderType.HASH, key_dim=key_dim, value_dim=value_dim)
        except Exception as e:
            # 这不应该发生，但作为最后的保护
            logger.error(f"Even hash encoder failed: {e}")
            raise RuntimeError("No encoder available, including fallback hash encoder")
    
    @classmethod
    def register_custom(cls, name: str, encoder_class: Type[BaseEncoder]):
        """
        注册自定义编码器
        
        Args:
            name: 编码器名称
            encoder_class: 编码器类
            
        Raises:
            TypeError: encoder_class 不是 BaseEncoder 的子类
            ValueError: 编码器类缺少必要的方法
        """
        if not issubclass(encoder_class, BaseEncoder):
            raise TypeError("encoder_class must be a subclass of BaseEncoder")
        
        # 验证必要的方法存在
        required_methods = ['encode', 'encode_batch', 'get_signature']
        missing_methods = []
        for method in required_methods:
            if not hasattr(encoder_class, method) or not callable(getattr(encoder_class, method)):
                missing_methods.append(method)
        
        if missing_methods:
            raise ValueError(
                f"Custom encoder class is missing required methods: {missing_methods}"
            )
        
        cls._custom_encoders[name] = encoder_class
        logger.info(f"Registered custom encoder: {name}")
    
    @classmethod
    def list_available(cls) -> List[Dict[str, Any]]:
        """
        列出所有可用的编码器
        
        Returns:
            编码器信息列表（不包含敏感信息如 API 密钥）
        """
        result = []
        
        for encoder_type in EncoderType:
            if encoder_type == EncoderType.CUSTOM:
                continue
            
            encoder_class = cls.ENCODER_CLASSES.get(encoder_type)
            if encoder_class is None:
                continue
            
            requires_api_key = encoder_type in [
                EncoderType.OPENAI,
                EncoderType.OPENAI_COMPATIBLE,
            ]
            requires_local_model = encoder_type in [
                EncoderType.SENTENCE_TRANSFORMERS,
                EncoderType.OLLAMA,
                EncoderType.VLLM,
                EncoderType.EMBEDDING_LAYER,
            ]
            
            try:
                config = EncoderConfig(encoder_type=encoder_type)
                encoder = encoder_class(config)
                
                # 获取信息但过滤敏感数据
                info = {
                    "type": encoder_type.value,
                    "native_dim": encoder.native_dim,
                    "is_available": encoder.is_available,
                    "requires_api_key": requires_api_key,
                    "requires_local_model": requires_local_model,
                }
                
                # 如果需要 API 密钥，检查环境变量是否已配置（不暴露实际值）
                if requires_api_key:
                    info["api_key_configured"] = bool(os.environ.get("OPENAI_API_KEY"))
                
                result.append(info)
            except Exception as e:
                result.append({
                    "type": encoder_type.value,
                    "is_available": False,
                    "requires_api_key": requires_api_key,
                    "requires_local_model": requires_local_model,
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
    ) -> Tuple[bool, str]:
        """
        验证编码器一致性
        
        Args:
            encoder: 当前编码器
            expected_signature: 期望的编码器签名
            
        Returns:
            (is_consistent, message)
        """
        return encoder.verify_compatibility(expected_signature)
