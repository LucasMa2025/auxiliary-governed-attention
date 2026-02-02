"""
AGA 配置管理模块

提供 Portal 和 Runtime 的独立配置管理。

架构说明：
- PortalConfig: API Portal 服务配置（无 GPU 依赖）
- RuntimeConfig: AGA Runtime 配置（与 LLM 同部署）
- SyncConfig: 同步协议配置
- load_config: 从 YAML 文件加载配置
"""

from .portal import PortalConfig, ServerConfig, PersistenceDBConfig, MessagingConfig
from .runtime import RuntimeConfig, AGAModuleConfig, SyncClientConfig
from .sync import SyncConfig, RedisConfig, KafkaConfig
from .loader import load_config, save_config

__all__ = [
    # Portal 配置
    "PortalConfig",
    "ServerConfig",
    "PersistenceDBConfig",
    "MessagingConfig",
    # Runtime 配置
    "RuntimeConfig",
    "AGAModuleConfig",
    "SyncClientConfig",
    # 同步配置
    "SyncConfig",
    "RedisConfig",
    "KafkaConfig",
    # 加载器
    "load_config",
    "save_config",
]
