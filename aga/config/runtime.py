"""
AGA Runtime 配置

Runtime 是与 LLM 同部署的 AGA 执行模块，负责：
- 运行时知识融合（推理增强）
- 从 Portal 同步知识
- 上报状态到 Portal
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uuid


@dataclass
class EncoderModuleConfig:
    """
    编码器配置
    
    ⚠️ 编码器一致性要求：
    - 注入时的编码器与推理时的编码器必须一致
    - 不同编码器产生的向量空间不同，混用会导致匹配失败
    
    支持的编码器类型：
    - hash: 哈希编码（测试用）
    - embedding_layer: 从 LLM 嵌入层提取
    - openai: OpenAI text-embedding
    - openai_compatible: OpenAI 兼容 API（DeepSeek/Qwen/智谱 等）
    - sentence_transformers: HuggingFace 本地模型
    - ollama: Ollama 本地模型
    - vllm: vLLM 本地部署
    """
    # 编码器类型
    encoder_type: str = "hash"
    
    # 显式维度配置（如果设置，会覆盖自动检测）
    # 设为 None 表示使用编码器默认值
    native_dim: Optional[int] = None
    
    # 模型配置
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # 服务商预设（用于 OpenAI 兼容 API）
    provider: Optional[str] = None  # deepseek, qwen, zhipu, moonshot 等
    
    # 设备（用于本地模型）
    device: str = "cpu"


@dataclass
class AGAModuleConfig:
    """AGA 模块配置"""
    hidden_dim: int = 4096
    bottleneck_dim: int = 64
    num_slots: int = 100
    num_heads: int = 32
    
    # 熵门控
    entropy_threshold: float = 0.5
    high_entropy_threshold: float = 0.7
    use_adaptive_threshold: bool = True
    
    # 衰减
    decay_enabled: bool = True
    decay_lambda: float = 0.95
    decay_hard_reset_threshold: float = 3.0
    
    # 路由
    top_k_slots: int = 3
    temperature: float = 1.0
    
    # 编码器配置
    encoder: EncoderModuleConfig = field(default_factory=EncoderModuleConfig)


@dataclass
class SyncClientConfig:
    """同步客户端配置"""
    portal_url: str = "http://localhost:8081"
    backend: str = "redis"  # redis, kafka, http_polling
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_channel: str = "aga:sync"
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "aga-sync"
    kafka_group_id: Optional[str] = None  # 默认使用 instance_id
    
    # HTTP Polling (备用)
    polling_interval: int = 5  # 秒
    
    # 启动同步
    sync_on_start: bool = True
    sync_timeout: int = 30  # 秒


@dataclass
class LocalCacheConfig:
    """本地缓存配置"""
    type: str = "memory"  # memory, file
    max_slots: int = 100
    file_path: Optional[str] = None  # 仅 type=file 时使用
    checkpoint_interval: int = 300  # 秒，定期保存检查点


@dataclass
class RuntimeConfig:
    """
    AGA Runtime 完整配置
    
    配置文件示例 (runtime_config.yaml):
    ```yaml
    instance:
      id: "runtime-001"
      namespace: "default"
    
    aga:
      hidden_dim: 4096
      bottleneck_dim: 64
      num_slots: 100
    
    sync:
      portal_url: "http://portal:8081"
      backend: "redis"
      redis_host: "localhost"
      redis_channel: "aga:sync"
    
    cache:
      type: "memory"
      max_slots: 100
    ```
    """
    # 实例标识
    instance_id: str = field(default_factory=lambda: f"runtime-{uuid.uuid4().hex[:8]}")
    namespace: str = "default"
    namespaces: List[str] = field(default_factory=lambda: ["default"])
    
    # 模块配置
    aga: AGAModuleConfig = field(default_factory=AGAModuleConfig)
    
    # 同步配置
    sync: SyncClientConfig = field(default_factory=SyncClientConfig)
    
    # 缓存配置
    cache: LocalCacheConfig = field(default_factory=LocalCacheConfig)
    
    # 设备配置
    device: str = "cuda"  # cuda, cpu
    dtype: str = "float16"  # float16, float32, bfloat16
    
    # 元数据
    version: str = "3.2.0"
    environment: str = "development"
    
    @classmethod
    def for_development(cls, namespace: str = "default") -> "RuntimeConfig":
        """开发环境配置"""
        return cls(
            instance_id=f"dev-{uuid.uuid4().hex[:8]}",
            namespace=namespace,
            namespaces=[namespace],
            aga=AGAModuleConfig(
                hidden_dim=1024,
                bottleneck_dim=32,
                num_slots=20,
            ),
            sync=SyncClientConfig(
                portal_url="http://localhost:8081",
                backend="memory",
                sync_on_start=False,
            ),
            cache=LocalCacheConfig(
                type="memory",
                max_slots=20,
            ),
            device="cpu",
            dtype="float32",
            environment="development",
        )
    
    @classmethod
    def for_production(
        cls,
        instance_id: str,
        portal_url: str,
        redis_host: str,
        namespaces: List[str] = None,
        hidden_dim: int = 4096,
        num_slots: int = 100,
    ) -> "RuntimeConfig":
        """生产环境配置"""
        namespaces = namespaces or ["default"]
        return cls(
            instance_id=instance_id,
            namespace=namespaces[0],
            namespaces=namespaces,
            aga=AGAModuleConfig(
                hidden_dim=hidden_dim,
                bottleneck_dim=hidden_dim // 64,
                num_slots=num_slots,
            ),
            sync=SyncClientConfig(
                portal_url=portal_url,
                backend="redis",
                redis_host=redis_host,
                sync_on_start=True,
            ),
            cache=LocalCacheConfig(
                type="memory",
                max_slots=num_slots,
            ),
            device="cuda",
            dtype="float16",
            environment="production",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [_to_dict(item) for item in obj]
            else:
                return obj
        
        return _to_dict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeConfig":
        """从字典创建"""
        # 处理 AGA 模块配置
        aga_data = data.get("aga", {})
        encoder_data = aga_data.pop("encoder", {}) if "encoder" in aga_data else {}
        aga_config = AGAModuleConfig(
            **aga_data,
            encoder=EncoderModuleConfig(**encoder_data) if encoder_data else EncoderModuleConfig(),
        )
        
        return cls(
            instance_id=data.get("instance_id", f"runtime-{uuid.uuid4().hex[:8]}"),
            namespace=data.get("namespace", "default"),
            namespaces=data.get("namespaces", ["default"]),
            aga=aga_config,
            sync=SyncClientConfig(**data.get("sync", {})),
            cache=LocalCacheConfig(**data.get("cache", {})),
            device=data.get("device", "cuda"),
            dtype=data.get("dtype", "float16"),
            version=data.get("version", "3.2.0"),
            environment=data.get("environment", "development"),
        )
    
    def create_encoder(self):
        """
        根据配置创建编码器
        
        Returns:
            BaseEncoder 实例
        """
        from ..encoder import EncoderFactory, EncoderConfig, EncoderType
        
        enc_cfg = self.aga.encoder
        
        config = EncoderConfig(
            encoder_type=EncoderType(enc_cfg.encoder_type),
            key_dim=self.aga.bottleneck_dim,
            value_dim=self.aga.hidden_dim,
            native_dim=enc_cfg.native_dim,
            model=enc_cfg.model,
            base_url=enc_cfg.base_url,
            api_key=enc_cfg.api_key,
            provider=enc_cfg.provider,
            device=enc_cfg.device,
        )
        
        return EncoderFactory.create(config)
