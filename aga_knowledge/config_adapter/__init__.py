"""
aga-knowledge 配置适配器

支持从多种来源加载配置：
- YAML 文件
- 环境变量
- Python dict
- 远程配置中心（预留）
"""

from .adapter import ConfigAdapter, YAMLConfigAdapter, EnvConfigAdapter

__all__ = [
    "ConfigAdapter",
    "YAMLConfigAdapter",
    "EnvConfigAdapter",
]
