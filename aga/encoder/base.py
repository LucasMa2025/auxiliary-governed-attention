"""
编码器基础类和类型定义

提供统一的编码器接口，支持：
- 显式维度配置
- 多种编码策略
- 编码器签名用于一致性验证
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import os

logger = logging.getLogger(__name__)


class EncoderType(str, Enum):
    """
    编码器类型枚举
    
    ⚠️ 编码器一致性说明：
    
    AGA 的 Key-Value 匹配机制要求：
    - **注入时的编码器** 与 **推理时的编码器** 必须一致
    - 不同编码器产生的向量空间不同，混用会导致匹配失败
    - 编码器不必与 AGA 绑定的 LLM（生成模型）一致
    
    推荐做法：
    1. 同一命名空间内的知识使用相同编码器
    2. 在命名空间配置中记录使用的编码器签名
    3. 迁移编码器时需要重新编码所有知识
    """
    HASH = "hash"                           # 哈希编码（测试用，无语义）
    EMBEDDING_LAYER = "embedding_layer"     # 从 LLM 嵌入层提取
    OPENAI = "openai"                       # OpenAI text-embedding
    OPENAI_COMPATIBLE = "openai_compatible" # OpenAI 兼容 API（DeepSeek/Qwen/GLM/Moonshot 等）
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # HuggingFace SentenceTransformers
    OLLAMA = "ollama"                       # Ollama 本地模型
    VLLM = "vllm"                           # vLLM 本地部署
    CUSTOM = "custom"                       # 自定义实现


@dataclass
class EncoderConfig:
    """
    编码器配置
    
    支持显式维度配置，确保与 AGA 系统对齐。
    
    维度说明：
    - native_dim: 编码器原生输出维度（如 OpenAI 的 1536）
    - key_dim (bottleneck_dim): Key 向量目标维度，用于条件匹配
    - value_dim (hidden_dim): Value 向量目标维度，用于知识注入
    
    如果原生维度与目标维度不同，编码器会自动进行维度调整：
    - 降维：平均池化
    - 升维：零填充
    """
    # 编码器类型
    encoder_type: EncoderType = EncoderType.HASH
    
    # 目标维度（与 AGA 配置对齐）
    key_dim: int = 64           # bottleneck_dim, 用于 Key 向量
    value_dim: int = 4096       # hidden_dim, 用于 Value 向量
    
    # 原生维度（编码器输出的原始维度）
    # 如果为 None，使用编码器默认值
    native_dim: Optional[int] = None
    
    # 通用配置
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # 服务商预设（用于 OpenAI 兼容 API）
    provider: Optional[str] = None
    
    # 设备配置（用于本地模型）
    device: str = "cpu"  # cpu, cuda, mps
    
    # 额外参数
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, prefix: str = "AGA_ENCODER_") -> "EncoderConfig":
        """从环境变量加载配置"""
        config = cls()
        
        # 编码器类型
        if os.getenv(f"{prefix}TYPE"):
            config.encoder_type = EncoderType(os.getenv(f"{prefix}TYPE"))
        
        # 维度配置
        if os.getenv(f"{prefix}KEY_DIM"):
            config.key_dim = int(os.getenv(f"{prefix}KEY_DIM"))
        if os.getenv(f"{prefix}VALUE_DIM"):
            config.value_dim = int(os.getenv(f"{prefix}VALUE_DIM"))
        if os.getenv(f"{prefix}NATIVE_DIM"):
            config.native_dim = int(os.getenv(f"{prefix}NATIVE_DIM"))
        
        # 通用配置
        if os.getenv(f"{prefix}MODEL"):
            config.model = os.getenv(f"{prefix}MODEL")
        if os.getenv(f"{prefix}BASE_URL"):
            config.base_url = os.getenv(f"{prefix}BASE_URL")
        if os.getenv(f"{prefix}API_KEY"):
            config.api_key = os.getenv(f"{prefix}API_KEY")
        if os.getenv(f"{prefix}PROVIDER"):
            config.provider = os.getenv(f"{prefix}PROVIDER")
        if os.getenv(f"{prefix}DEVICE"):
            config.device = os.getenv(f"{prefix}DEVICE")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "encoder_type": self.encoder_type.value,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "native_dim": self.native_dim,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else None,
            "provider": self.provider,
            "device": self.device,
        }


@dataclass
class EncoderSignature:
    """
    编码器签名
    
    用于验证编码器一致性，确保注入和查询使用相同的编码器。
    
    签名包含：
    - encoder_type: 编码器类型
    - model: 模型名称
    - native_dim: 原生维度
    - key_dim: Key 向量维度
    - value_dim: Value 向量维度
    - provider: 服务商（如适用）
    """
    encoder_type: str
    model: str
    native_dim: int
    key_dim: int
    value_dim: int
    provider: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_type": self.encoder_type,
            "model": self.model,
            "native_dim": self.native_dim,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "provider": self.provider,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncoderSignature":
        return cls(
            encoder_type=data.get("encoder_type", ""),
            model=data.get("model", ""),
            native_dim=data.get("native_dim", 0),
            key_dim=data.get("key_dim", 64),
            value_dim=data.get("value_dim", 4096),
            provider=data.get("provider"),
        )
    
    def is_compatible(self, other: "EncoderSignature") -> bool:
        """
        检查两个编码器签名是否兼容
        
        兼容条件：
        1. 编码器类型相同
        2. 模型名称相同
        3. 原生维度相同
        4. Key/Value 目标维度相同
        """
        return (
            self.encoder_type == other.encoder_type and
            self.model == other.model and
            self.native_dim == other.native_dim and
            self.key_dim == other.key_dim and
            self.value_dim == other.value_dim
        )
    
    def get_mismatch_details(self, other: "EncoderSignature") -> List[str]:
        """获取不匹配的详细信息"""
        mismatches = []
        if self.encoder_type != other.encoder_type:
            mismatches.append(f"type: {self.encoder_type} vs {other.encoder_type}")
        if self.model != other.model:
            mismatches.append(f"model: {self.model} vs {other.model}")
        if self.native_dim != other.native_dim:
            mismatches.append(f"native_dim: {self.native_dim} vs {other.native_dim}")
        if self.key_dim != other.key_dim:
            mismatches.append(f"key_dim: {self.key_dim} vs {other.key_dim}")
        if self.value_dim != other.value_dim:
            mismatches.append(f"value_dim: {self.value_dim} vs {other.value_dim}")
        return mismatches


class BaseEncoder(ABC):
    """
    编码器基础接口
    
    将文本编码为向量表示，用于 AGA 的知识注入和检索。
    
    实现要求:
    - encode(): 编码单个文本到原生维度
    - encode_batch(): 批量编码（可优化）
    - encode_to_key(): 编码为 Key 向量（目标维度）
    - encode_to_value(): 编码为 Value 向量（目标维度）
    - encode_constraint(): 编码约束对
    
    维度处理:
    - 编码器输出原生维度 (native_dim)
    - 自动调整到目标维度 (key_dim / value_dim)
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化编码器
        
        Args:
            config: 编码器配置，包含维度信息
        """
        self.config = config
        self._initialized = False
    
    @property
    @abstractmethod
    def encoder_type(self) -> EncoderType:
        """编码器类型"""
        pass
    
    @property
    @abstractmethod
    def native_dim(self) -> int:
        """原生编码维度（编码器输出的原始维度）"""
        pass
    
    @property
    def key_dim(self) -> int:
        """Key 向量目标维度"""
        return self.config.key_dim
    
    @property
    def value_dim(self) -> int:
        """Value 向量目标维度"""
        return self.config.value_dim
    
    @property
    def model_name(self) -> str:
        """模型名称"""
        return self.config.model or "unknown"
    
    @property
    def is_available(self) -> bool:
        """编码器是否可用"""
        return True
    
    def initialize(self) -> bool:
        """初始化编码器（子类可覆盖）"""
        self._initialized = True
        return True
    
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """
        编码单个文本到原生维度
        
        Args:
            text: 输入文本
            
        Returns:
            编码向量（原生维度）
        """
        pass
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文本（默认逐个编码，子类可优化）
        
        Args:
            texts: 文本列表
            
        Returns:
            编码向量列表（原生维度）
        """
        return [self.encode(t) for t in texts]
    
    def encode_to_key(self, text: str) -> List[float]:
        """
        编码为 Key 向量（调整到 key_dim）
        
        Args:
            text: 输入文本
            
        Returns:
            Key 向量（key_dim 维度）
        """
        vector = self.encode(text)
        return self._adjust_dim(vector, self.key_dim)
    
    def encode_to_value(self, text: str) -> List[float]:
        """
        编码为 Value 向量（调整到 value_dim）
        
        Args:
            text: 输入文本
            
        Returns:
            Value 向量（value_dim 维度）
        """
        vector = self.encode(text)
        return self._adjust_dim(vector, self.value_dim)
    
    def encode_constraint(
        self,
        condition: str,
        decision: str,
    ) -> Tuple[List[float], List[float]]:
        """
        编码约束为 Key-Value 向量对
        
        Args:
            condition: 条件文本
            decision: 决策文本
        
        Returns:
            (key_vector, value_vector)
            - key_vector: 条件编码，维度为 key_dim
            - value_vector: 决策编码，维度为 value_dim
        """
        key_vector = self.encode_to_key(condition)
        value_vector = self.encode_to_value(decision)
        return key_vector, value_vector
    
    def get_signature(self) -> EncoderSignature:
        """
        获取编码器签名
        
        用于记录和验证编码器一致性。
        """
        return EncoderSignature(
            encoder_type=self.encoder_type.value,
            model=self.model_name,
            native_dim=self.native_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            provider=getattr(self, 'provider', None),
        )
    
    def verify_compatibility(self, expected: Union[EncoderSignature, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        验证与期望签名的兼容性
        
        Args:
            expected: 期望的编码器签名
            
        Returns:
            (is_compatible, message)
        """
        current = self.get_signature()
        
        if isinstance(expected, dict):
            expected = EncoderSignature.from_dict(expected)
        
        if current.is_compatible(expected):
            return True, "Encoder is compatible"
        
        mismatches = current.get_mismatch_details(expected)
        return False, f"Encoder mismatch: {', '.join(mismatches)}"
    
    def _adjust_dim(self, vector: List[float], target_dim: int) -> List[float]:
        """
        调整向量维度
        
        Args:
            vector: 原始向量
            target_dim: 目标维度
            
        Returns:
            调整后的向量
        """
        current_dim = len(vector)
        
        if current_dim == target_dim:
            return vector
        
        if current_dim > target_dim:
            # 降维：平均池化
            return self._reduce_dim(vector, target_dim)
        else:
            # 升维：零填充
            return vector + [0.0] * (target_dim - current_dim)
    
    def _reduce_dim(self, vector: List[float], target_dim: int) -> List[float]:
        """降维（平均池化）"""
        current_dim = len(vector)
        chunk_size = current_dim / target_dim
        
        result = []
        for i in range(target_dim):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            chunk = vector[start:end]
            result.append(sum(chunk) / len(chunk) if chunk else 0.0)
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """获取编码器信息"""
        return {
            "encoder_type": self.encoder_type.value,
            "model": self.model_name,
            "native_dim": self.native_dim,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "is_available": self.is_available,
            "initialized": self._initialized,
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_name} native_dim={self.native_dim}>"
