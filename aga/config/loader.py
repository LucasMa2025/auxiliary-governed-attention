"""
AGA 配置加载器

支持从 YAML 文件加载配置。
"""

import os
from pathlib import Path
from typing import Union, Type, TypeVar

from .portal import PortalConfig
from .runtime import RuntimeConfig
from .sync import SyncConfig

T = TypeVar('T', PortalConfig, RuntimeConfig, SyncConfig)


def load_config(
    path: Union[str, Path],
    config_type: Type[T] = None,
) -> T:
    """
    从 YAML 文件加载配置
    
    Args:
        path: 配置文件路径
        config_type: 配置类型（可选，自动推断）
    
    Returns:
        配置对象
    
    示例:
        >>> portal_config = load_config("portal_config.yaml", PortalConfig)
        >>> runtime_config = load_config("runtime_config.yaml", RuntimeConfig)
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("需要安装 PyYAML: pip install pyyaml")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 自动推断配置类型
    if config_type is None:
        if "server" in data and "messaging" in data:
            config_type = PortalConfig
        elif "aga" in data and "sync" in data:
            config_type = RuntimeConfig
        elif "backend" in data:
            config_type = SyncConfig
        else:
            raise ValueError("无法自动推断配置类型，请指定 config_type")
    
    return config_type.from_dict(data)


def save_config(config: Union[PortalConfig, RuntimeConfig, SyncConfig], path: Union[str, Path]):
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置对象
        path: 目标文件路径
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("需要安装 PyYAML: pip install pyyaml")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)


def load_from_env(config_type: Type[T], prefix: str = "AGA_") -> T:
    """
    从环境变量加载配置
    
    环境变量命名规则：
    - AGA_SERVER_HOST -> server.host
    - AGA_PERSISTENCE_TYPE -> persistence.type
    - AGA_MESSAGING_REDIS_HOST -> messaging.redis_host
    
    Args:
        config_type: 配置类型
        prefix: 环境变量前缀
    
    Returns:
        配置对象
    """
    data = {}
    
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        
        # 解析键路径
        parts = key[len(prefix):].lower().split("_")
        
        # 构建嵌套字典
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # 类型转换
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        elif _is_float(value):
            value = float(value)
        
        current[parts[-1]] = value
    
    return config_type.from_dict(data)


def _is_float(s: str) -> bool:
    """检查字符串是否为浮点数"""
    try:
        float(s)
        return "." in s
    except ValueError:
        return False


def generate_example_configs(output_dir: Union[str, Path] = "."):
    """
    生成示例配置文件
    
    Args:
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Portal 配置
    portal_config = PortalConfig.for_development()
    save_config(portal_config, output_dir / "portal_config.example.yaml")
    
    # Runtime 配置
    runtime_config = RuntimeConfig.for_development()
    save_config(runtime_config, output_dir / "runtime_config.example.yaml")
    
    print(f"示例配置文件已生成到: {output_dir}")
