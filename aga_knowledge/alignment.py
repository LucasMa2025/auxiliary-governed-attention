"""
aga-core 对齐配置

从 aga-core 的配置中提取编码器和检索器所需的最小参数集，
确保 aga-knowledge 生成的向量可以直接被 aga-core 使用。

获取方式:
  1. 手动配置（推荐生产环境）— 从 aga-core 的 YAML 复制关键参数
  2. 从 aga-core 的 YAML 配置文件自动提取
  3. 从 AGAConfig 实例直接创建（开发环境）

关键约束:
  - key 向量维度 = bottleneck_dim（默认 64）
  - value 向量维度 = hidden_dim（默认 4096）
  - key 范数目标 = key_norm_target（默认 5.0）
  - value 范数目标 = value_norm_target（默认 3.0）

使用:
    # 方式 1: 手动配置
    alignment = AGACoreAlignment(
        hidden_dim=4096,
        bottleneck_dim=64,
        key_norm_target=5.0,
        value_norm_target=3.0,
    )

    # 方式 2: 从 YAML 文件
    alignment = AGACoreAlignment.from_aga_config_yaml("/path/to/aga_config.yaml")

    # 方式 3: 从 AGAConfig 实例（开发环境）
    alignment = AGACoreAlignment.from_aga_config(aga_config)

    # 创建对齐的编码器
    encoder_config = EncoderConfig.from_alignment(alignment)
"""

from dataclasses import dataclass
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class AGACoreAlignment:
    """
    aga-core 对齐配置

    从 aga-core 的配置中提取编码器和检索器所需的最小参数集。
    确保 aga-knowledge 生成的向量可以直接被 aga-core 使用。

    Attributes:
        hidden_dim: 模型隐藏层维度（aga-core AGAConfig.hidden_dim）
        bottleneck_dim: 瓶颈维度（aga-core AGAConfig.bottleneck_dim）
        num_heads: 注意力头数（aga-core AGAConfig.num_heads）
        value_bottleneck_dim: Value 瓶颈维度（aga-core AGAConfig.value_bottleneck_dim）
        key_norm_target: Key 向量 L2 范数目标（aga-core AGAConfig.key_norm_target）
        value_norm_target: Value 向量 L2 范数目标（aga-core AGAConfig.value_norm_target）
        enable_norm_clipping: 是否启用范数裁剪（aga-core AGAConfig.enable_norm_clipping）
    """

    hidden_dim: int = 4096
    bottleneck_dim: int = 64
    num_heads: int = 32
    value_bottleneck_dim: int = 256
    key_norm_target: float = 5.0
    value_norm_target: float = 3.0
    enable_norm_clipping: bool = True

    @classmethod
    def from_aga_config_yaml(cls, path: str) -> "AGACoreAlignment":
        """
        从 aga-core 的 YAML 配置文件提取对齐参数

        支持两种格式:
          1. 根级别: hidden_dim, bottleneck_dim, ...
          2. aga 命名空间: aga.hidden_dim, aga.bottleneck_dim, ...

        Args:
            path: aga-core 配置文件路径

        Returns:
            AGACoreAlignment 实例
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "从 YAML 加载对齐配置需要 PyYAML。\n"
                "请运行: pip install pyyaml"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 支持 aga.xxx 和根级别 xxx 两种格式
        aga = data.get("aga", data)

        instance = cls(
            hidden_dim=aga.get("hidden_dim", 4096),
            bottleneck_dim=aga.get("bottleneck_dim", 64),
            num_heads=aga.get("num_heads", 32),
            value_bottleneck_dim=aga.get("value_bottleneck_dim", 256),
            key_norm_target=float(aga.get("key_norm_target", 5.0)),
            value_norm_target=float(aga.get("value_norm_target", 3.0)),
            enable_norm_clipping=aga.get("enable_norm_clipping", True),
        )

        errors = instance.validate()
        if errors:
            raise ValueError(
                f"aga-core 对齐配置无效 ({path}):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info(
            f"已从 {path} 加载 aga-core 对齐配置: "
            f"hidden_dim={instance.hidden_dim}, "
            f"bottleneck_dim={instance.bottleneck_dim}, "
            f"key_norm={instance.key_norm_target}, "
            f"value_norm={instance.value_norm_target}"
        )

        return instance

    @classmethod
    def from_aga_config(cls, config) -> "AGACoreAlignment":
        """
        从 AGAConfig 实例直接创建（开发/测试环境）

        Args:
            config: aga-core 的 AGAConfig 实例

        Returns:
            AGACoreAlignment 实例
        """
        return cls(
            hidden_dim=getattr(config, "hidden_dim", 4096),
            bottleneck_dim=getattr(config, "bottleneck_dim", 64),
            num_heads=getattr(config, "num_heads", 32),
            value_bottleneck_dim=getattr(config, "value_bottleneck_dim", 256),
            key_norm_target=float(getattr(config, "key_norm_target", 5.0)),
            value_norm_target=float(getattr(config, "value_norm_target", 3.0)),
            enable_norm_clipping=getattr(config, "enable_norm_clipping", True),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "AGACoreAlignment":
        """
        从字典创建

        Args:
            data: 包含对齐参数的字典

        Returns:
            AGACoreAlignment 实例
        """
        return cls(
            hidden_dim=data.get("hidden_dim", 4096),
            bottleneck_dim=data.get("bottleneck_dim", 64),
            num_heads=data.get("num_heads", 32),
            value_bottleneck_dim=data.get("value_bottleneck_dim", 256),
            key_norm_target=float(data.get("key_norm_target", 5.0)),
            value_norm_target=float(data.get("value_norm_target", 3.0)),
            enable_norm_clipping=data.get("enable_norm_clipping", True),
        )

    def validate(self) -> List[str]:
        """
        验证对齐参数

        Returns:
            错误信息列表（空列表表示验证通过）
        """
        errors = []
        if self.hidden_dim <= 0:
            errors.append("hidden_dim 必须大于 0")
        if self.bottleneck_dim <= 0:
            errors.append("bottleneck_dim 必须大于 0")
        if self.bottleneck_dim >= self.hidden_dim:
            errors.append(
                f"bottleneck_dim ({self.bottleneck_dim}) "
                f"必须小于 hidden_dim ({self.hidden_dim})"
            )
        if self.num_heads <= 0:
            errors.append("num_heads 必须大于 0")
        if self.key_norm_target <= 0:
            errors.append("key_norm_target 必须大于 0")
        if self.value_norm_target <= 0:
            errors.append("value_norm_target 必须大于 0")
        if self.value_bottleneck_dim <= 0:
            errors.append("value_bottleneck_dim 必须大于 0")
        return errors

    def to_encoder_config_overrides(self) -> dict:
        """
        生成编码器配置覆盖项

        用于在已有 EncoderConfig 上覆盖对齐相关参数。

        Returns:
            可直接传入 EncoderConfig 构造函数的字典
        """
        return {
            "key_dim": self.bottleneck_dim,
            "value_dim": self.hidden_dim,
            "options": {
                "key_norm_target": self.key_norm_target,
                "value_norm_target": self.value_norm_target,
            },
        }

    def summary(self) -> str:
        """返回对齐配置的摘要字符串"""
        return (
            f"AGACoreAlignment("
            f"hidden_dim={self.hidden_dim}, "
            f"bottleneck_dim={self.bottleneck_dim}, "
            f"num_heads={self.num_heads}, "
            f"key_norm={self.key_norm_target}, "
            f"value_norm={self.value_norm_target})"
        )

    def __repr__(self) -> str:
        return self.summary()
