"""
AGA Knowledge 配置适配器

支持从多种来源加载和合并配置。

使用示例：
    from aga_knowledge.config_adapter import YAMLConfigAdapter, EnvConfigAdapter

    # 从 YAML 加载
    adapter = YAMLConfigAdapter("configs/portal_config.yaml")
    config = adapter.load()

    # 从环境变量覆盖
    env_adapter = EnvConfigAdapter(prefix="AGA")
    config = env_adapter.apply(config)

    # 链式加载
    config = ConfigAdapter.chain([
        YAMLConfigAdapter("configs/portal_config.yaml"),
        EnvConfigAdapter(prefix="AGA"),
    ])
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..config import PortalConfig
from ..exceptions import ConfigAdapterError

logger = logging.getLogger(__name__)


class ConfigAdapter(ABC):
    """
    配置适配器抽象基类

    定义统一的配置加载接口。
    """

    @abstractmethod
    def load(self) -> PortalConfig:
        """
        加载配置

        Returns:
            PortalConfig 实例
        """
        ...

    def apply(self, config: PortalConfig) -> PortalConfig:
        """
        将此适配器的配置覆盖到已有配置上

        Args:
            config: 已有配置

        Returns:
            合并后的配置
        """
        return self.load()

    @staticmethod
    def chain(adapters: List["ConfigAdapter"]) -> PortalConfig:
        """
        链式加载配置

        按顺序应用多个适配器，后者覆盖前者。

        Args:
            adapters: 适配器列表

        Returns:
            最终合并的 PortalConfig
        """
        if not adapters:
            return PortalConfig.for_development()

        config = adapters[0].load()
        for adapter in adapters[1:]:
            config = adapter.apply(config)
        return config


class YAMLConfigAdapter(ConfigAdapter):
    """
    YAML 文件配置适配器

    从 YAML 文件加载 PortalConfig。

    配置文件格式：
    ```yaml
    server:
      host: "0.0.0.0"
      port: 8081
      workers: 4

    persistence:
      type: "sqlite"
      sqlite_path: "aga_knowledge.db"

    messaging:
      backend: "redis"
      redis_host: "localhost"
      redis_port: 6379

    registry:
      type: "memory"
      timeout: 90

    governance:
      enabled: false

    environment: "development"
    ```
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: YAML 配置文件路径
        """
        self.config_path = config_path

    def load(self) -> PortalConfig:
        """从 YAML 文件加载配置"""
        try:
            import yaml
        except ImportError:
            raise ConfigAdapterError("需要安装 PyYAML: pip install pyyaml")

        if not os.path.exists(self.config_path):
            raise ConfigAdapterError(f"配置文件不存在: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigAdapterError(f"解析 YAML 文件失败: {e}")

        config = PortalConfig.from_dict(data)
        logger.info(f"Config loaded from YAML: {self.config_path}")
        return config

    def apply(self, config: PortalConfig) -> PortalConfig:
        """将 YAML 配置覆盖到已有配置"""
        try:
            import yaml
        except ImportError:
            raise ConfigAdapterError("需要安装 PyYAML: pip install pyyaml")

        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found, skipping: {self.config_path}")
            return config

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 深度合并
        config_dict = config.to_dict()
        merged = self._deep_merge(config_dict, data)
        return PortalConfig.from_dict(merged)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = YAMLConfigAdapter._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


class EnvConfigAdapter(ConfigAdapter):
    """
    环境变量配置适配器

    从环境变量加载配置，覆盖已有配置。

    环境变量映射：
        AGA_PORTAL_HOST          → server.host
        AGA_PORTAL_PORT          → server.port
        AGA_PORTAL_WORKERS       → server.workers
        AGA_PORTAL_LOG_LEVEL     → server.log_level
        AGA_PERSISTENCE_TYPE     → persistence.type
        AGA_PERSISTENCE_SQLITE   → persistence.sqlite_path
        AGA_PERSISTENCE_POSTGRES → persistence.postgres_url
        AGA_MESSAGING_BACKEND    → messaging.backend
        AGA_MESSAGING_REDIS_HOST → messaging.redis_host
        AGA_MESSAGING_REDIS_PORT → messaging.redis_port
        AGA_MESSAGING_CHANNEL    → messaging.redis_channel
        AGA_REGISTRY_TYPE        → registry.type
        AGA_REGISTRY_TIMEOUT     → registry.timeout
        AGA_ENVIRONMENT          → environment
    """

    # 环境变量到配置路径的映射
    ENV_MAP = {
        "AGA_PORTAL_HOST": ("server", "host", str),
        "AGA_PORTAL_PORT": ("server", "port", int),
        "AGA_PORTAL_WORKERS": ("server", "workers", int),
        "AGA_PORTAL_LOG_LEVEL": ("server", "log_level", str),
        "AGA_PORTAL_CORS_ENABLED": ("server", "cors_enabled", lambda v: v.lower() in ("true", "1", "yes")),
        "AGA_PERSISTENCE_TYPE": ("persistence", "type", str),
        "AGA_PERSISTENCE_SQLITE": ("persistence", "sqlite_path", str),
        "AGA_PERSISTENCE_POSTGRES": ("persistence", "postgres_url", str),
        "AGA_PERSISTENCE_POOL_SIZE": ("persistence", "postgres_pool_size", int),
        "AGA_PERSISTENCE_AUDIT": ("persistence", "enable_audit", lambda v: v.lower() in ("true", "1", "yes")),
        "AGA_MESSAGING_BACKEND": ("messaging", "backend", str),
        "AGA_MESSAGING_REDIS_HOST": ("messaging", "redis_host", str),
        "AGA_MESSAGING_REDIS_PORT": ("messaging", "redis_port", int),
        "AGA_MESSAGING_REDIS_DB": ("messaging", "redis_db", int),
        "AGA_MESSAGING_REDIS_PASSWORD": ("messaging", "redis_password", str),
        "AGA_MESSAGING_CHANNEL": ("messaging", "redis_channel", str),
        "AGA_REGISTRY_TYPE": ("registry", "type", str),
        "AGA_REGISTRY_TIMEOUT": ("registry", "timeout", int),
        "AGA_GOVERNANCE_ENABLED": ("governance", "enabled", lambda v: v.lower() in ("true", "1", "yes")),
        "AGA_ENVIRONMENT": ("_root", "environment", str),
    }

    def __init__(self, prefix: str = "AGA"):
        """
        Args:
            prefix: 环境变量前缀（默认 AGA）
        """
        self.prefix = prefix

    def load(self) -> PortalConfig:
        """从环境变量加载配置"""
        return self.apply(PortalConfig.for_development())

    def apply(self, config: PortalConfig) -> PortalConfig:
        """将环境变量覆盖到已有配置"""
        config_dict = config.to_dict()
        applied_count = 0

        for env_key, (section, field, converter) in self.ENV_MAP.items():
            value = os.environ.get(env_key)
            if value is not None:
                try:
                    converted = converter(value)
                    if section == "_root":
                        config_dict[field] = converted
                    else:
                        if section not in config_dict:
                            config_dict[section] = {}
                        config_dict[section][field] = converted
                    applied_count += 1
                    logger.debug(f"Env override: {env_key} = {value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid env value for {env_key}: {value} ({e})")

        if applied_count > 0:
            logger.info(f"Applied {applied_count} environment variable overrides")

        return PortalConfig.from_dict(config_dict)

    def get_env_summary(self) -> Dict[str, Optional[str]]:
        """获取所有 AGA 相关环境变量"""
        result = {}
        for env_key in self.ENV_MAP:
            value = os.environ.get(env_key)
            if value is not None:
                # 隐藏密码
                if "PASSWORD" in env_key or "SECRET" in env_key:
                    result[env_key] = "***"
                else:
                    result[env_key] = value
        return result


class DictConfigAdapter(ConfigAdapter):
    """
    字典配置适配器

    从 Python 字典加载配置。
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def load(self) -> PortalConfig:
        return PortalConfig.from_dict(self.data)

    def apply(self, config: PortalConfig) -> PortalConfig:
        config_dict = config.to_dict()
        merged = YAMLConfigAdapter._deep_merge(config_dict, self.data)
        return PortalConfig.from_dict(merged)
