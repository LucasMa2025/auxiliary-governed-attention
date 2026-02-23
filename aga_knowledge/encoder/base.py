"""
aga-knowledge 编码器抽象基类

定义文本到向量的编码协议，确保与 aga-core 的维度一致性。

关键约束:
  - key 向量维度 = AGAConfig.bottleneck_dim（默认 64）
  - value 向量维度 = AGAConfig.hidden_dim（默认 4096）
  - 编码器必须保证相同文本的编码结果一致（确定性）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING

import logging

if TYPE_CHECKING:
    from ..alignment import AGACoreAlignment

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """
    编码器配置

    与 aga-core 的 AGAConfig 维度参数保持一致。

    配置示例 (YAML):
    ```yaml
    encoder:
      backend: "sentence_transformer"
      model_name: "all-MiniLM-L6-v2"
      key_dim: 64          # 必须 == AGAConfig.bottleneck_dim
      value_dim: 4096      # 必须 == AGAConfig.hidden_dim
      device: "cpu"        # 编码器运行设备（建议 CPU，避免占用推理 GPU）
      batch_size: 32
      normalize: true
      cache_enabled: true
      cache_max_size: 10000
    ```
    """
    backend: str = "sentence_transformer"
    model_name: str = "all-MiniLM-L6-v2"
    key_dim: int = 64       # 必须 == AGAConfig.bottleneck_dim
    value_dim: int = 4096   # 必须 == AGAConfig.hidden_dim
    device: str = "cpu"     # 编码器设备（建议 CPU）
    batch_size: int = 32
    normalize: bool = True  # 是否 L2 归一化输出
    cache_enabled: bool = True
    cache_max_size: int = 10000
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncoderConfig":
        """从字典创建"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_alignment(
        cls,
        alignment: "AGACoreAlignment",
        backend: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> "EncoderConfig":
        """
        从 AGACoreAlignment 创建编码器配置（推荐方式）

        自动继承维度和范数参数，确保与 aga-core 对齐。
        key_dim 和 value_dim 不再需要手动指定。

        Args:
            alignment: aga-core 对齐配置
            backend: 编码器后端
            model_name: 模型名称
            **kwargs: 其他 EncoderConfig 参数

        Returns:
            与 aga-core 对齐的 EncoderConfig

        Example:
            alignment = AGACoreAlignment(hidden_dim=4096, bottleneck_dim=64)
            config = EncoderConfig.from_alignment(alignment)
            # config.key_dim == 64, config.value_dim == 4096
        """
        # 提取 options 并与对齐范数参数合并
        user_options = kwargs.pop("options", {})
        merged_options = {
            "key_norm_target": alignment.key_norm_target,
            "value_norm_target": alignment.value_norm_target,
            **user_options,
        }

        return cls(
            backend=backend,
            model_name=model_name,
            key_dim=alignment.bottleneck_dim,
            value_dim=alignment.hidden_dim,
            options=merged_options,
            **kwargs,
        )

    def validate_alignment(self, alignment: "AGACoreAlignment") -> List[str]:
        """
        验证编码器配置是否与 aga-core 对齐

        检查维度和范数参数是否与 aga-core 的 AGAConfig 一致。
        任何不一致都会导致编码器生成的向量无法被 aga-core 正确使用。

        Args:
            alignment: aga-core 对齐配置

        Returns:
            不一致项列表（空列表表示对齐正确）

        Example:
            errors = config.validate_alignment(alignment)
            if errors:
                raise ConfigError("编码器与 aga-core 不对齐: " + str(errors))
        """
        errors = []

        # 维度检查
        if self.key_dim != alignment.bottleneck_dim:
            errors.append(
                f"key_dim ({self.key_dim}) != "
                f"AGAConfig.bottleneck_dim ({alignment.bottleneck_dim})"
            )
        if self.value_dim != alignment.hidden_dim:
            errors.append(
                f"value_dim ({self.value_dim}) != "
                f"AGAConfig.hidden_dim ({alignment.hidden_dim})"
            )

        # 范数检查
        key_norm = self.options.get("key_norm_target", 5.0)
        if abs(key_norm - alignment.key_norm_target) > 0.01:
            errors.append(
                f"key_norm_target ({key_norm}) != "
                f"AGAConfig.key_norm_target ({alignment.key_norm_target})"
            )

        value_norm = self.options.get("value_norm_target", 3.0)
        if abs(value_norm - alignment.value_norm_target) > 0.01:
            errors.append(
                f"value_norm_target ({value_norm}) != "
                f"AGAConfig.value_norm_target ({alignment.value_norm_target})"
            )

        return errors

    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        if self.key_dim <= 0:
            errors.append("key_dim 必须大于 0")
        if self.value_dim <= 0:
            errors.append("value_dim 必须大于 0")
        if self.batch_size <= 0:
            errors.append("batch_size 必须大于 0")
        return errors


@dataclass
class EncodedKnowledge:
    """
    编码后的知识

    包含原始文本和编码后的向量，可直接用于 aga-core 的 register()。
    """
    lu_id: str
    condition: str
    decision: str
    key_vector: List[float]     # [key_dim] — 用于 aga-core 的 key
    value_vector: List[float]   # [value_dim] — 用于 aga-core 的 value
    reliability: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class BaseEncoder(ABC):
    """
    编码器抽象基类

    所有编码器必须实现此接口。编码器负责将明文 condition/decision
    转换为 aga-core 所需的 key/value 向量。

    生命周期:
        1. __init__(config): 初始化（延迟加载模型）
        2. warmup(): 预热（加载模型到内存/GPU）
        3. encode() / encode_batch(): 编码文本
        4. shutdown(): 释放资源

    维度约束:
        - key 向量: [key_dim] — 对应 AGAConfig.bottleneck_dim
        - value 向量: [value_dim] — 对应 AGAConfig.hidden_dim
    """

    def __init__(self, config: EncoderConfig):
        self.config = config
        self._initialized = False
        self._encode_count = 0
        self._cache: Dict[str, Tuple[List[float], List[float]]] = {}

    @abstractmethod
    def _initialize(self) -> None:
        """
        延迟初始化（加载模型等资源）

        子类必须实现此方法。在首次 encode() 调用时自动触发。
        """
        ...

    @abstractmethod
    def _encode_texts(
        self,
        conditions: List[str],
        decisions: List[str],
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        批量编码文本

        Args:
            conditions: 条件文本列表
            decisions: 决策文本列表

        Returns:
            (key_vectors, value_vectors) 元组
            key_vectors: List[List[float]]，每个 [key_dim]
            value_vectors: List[List[float]]，每个 [value_dim]
        """
        ...

    def warmup(self) -> None:
        """预热编码器（加载模型）"""
        if not self._initialized:
            self._initialize()
            self._initialized = True
            logger.info(f"编码器已初始化: {self.__class__.__name__}")

    def encode(
        self,
        condition: str,
        decision: str,
        lu_id: str = "",
        reliability: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EncodedKnowledge:
        """
        编码单条知识

        Args:
            condition: 触发条件文本
            decision: 决策文本
            lu_id: 知识 ID
            reliability: 可靠性
            metadata: 元数据

        Returns:
            EncodedKnowledge 实例
        """
        if not self._initialized:
            self.warmup()

        # 检查缓存
        cache_key = f"{condition}|||{decision}"
        if self.config.cache_enabled and cache_key in self._cache:
            key_vec, val_vec = self._cache[cache_key]
            return EncodedKnowledge(
                lu_id=lu_id,
                condition=condition,
                decision=decision,
                key_vector=key_vec,
                value_vector=val_vec,
                reliability=reliability,
                metadata=metadata,
            )

        key_vectors, value_vectors = self._encode_texts([condition], [decision])
        self._encode_count += 1

        key_vec = key_vectors[0]
        val_vec = value_vectors[0]

        # 更新缓存
        if self.config.cache_enabled:
            if len(self._cache) >= self.config.cache_max_size:
                # 简单 FIFO 淘汰
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = (key_vec, val_vec)

        return EncodedKnowledge(
            lu_id=lu_id,
            condition=condition,
            decision=decision,
            key_vector=key_vec,
            value_vector=val_vec,
            reliability=reliability,
            metadata=metadata,
        )

    def encode_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[EncodedKnowledge]:
        """
        批量编码知识

        Args:
            records: 知识记录列表，每项需包含 condition, decision, lu_id

        Returns:
            EncodedKnowledge 列表
        """
        if not self._initialized:
            self.warmup()

        if not records:
            return []

        # 分离已缓存和未缓存
        cached_results: Dict[int, EncodedKnowledge] = {}
        uncached_indices: List[int] = []
        uncached_conditions: List[str] = []
        uncached_decisions: List[str] = []

        for i, record in enumerate(records):
            condition = record.get("condition", "")
            decision = record.get("decision", "")
            cache_key = f"{condition}|||{decision}"

            if self.config.cache_enabled and cache_key in self._cache:
                key_vec, val_vec = self._cache[cache_key]
                cached_results[i] = EncodedKnowledge(
                    lu_id=record.get("lu_id", ""),
                    condition=condition,
                    decision=decision,
                    key_vector=key_vec,
                    value_vector=val_vec,
                    reliability=record.get("reliability", 1.0),
                    metadata=record.get("metadata"),
                )
            else:
                uncached_indices.append(i)
                uncached_conditions.append(condition)
                uncached_decisions.append(decision)

        # 批量编码未缓存的
        if uncached_conditions:
            # 分批处理
            all_key_vecs: List[List[float]] = []
            all_val_vecs: List[List[float]] = []

            for start in range(0, len(uncached_conditions), self.config.batch_size):
                end = start + self.config.batch_size
                batch_conds = uncached_conditions[start:end]
                batch_decs = uncached_decisions[start:end]
                key_vecs, val_vecs = self._encode_texts(batch_conds, batch_decs)
                all_key_vecs.extend(key_vecs)
                all_val_vecs.extend(val_vecs)

            self._encode_count += len(uncached_conditions)

            # 构建结果并更新缓存
            for j, orig_idx in enumerate(uncached_indices):
                record = records[orig_idx]
                condition = record.get("condition", "")
                decision = record.get("decision", "")
                key_vec = all_key_vecs[j]
                val_vec = all_val_vecs[j]

                # 更新缓存
                if self.config.cache_enabled:
                    cache_key = f"{condition}|||{decision}"
                    if len(self._cache) >= self.config.cache_max_size:
                        oldest_key = next(iter(self._cache))
                        del self._cache[oldest_key]
                    self._cache[cache_key] = (key_vec, val_vec)

                cached_results[orig_idx] = EncodedKnowledge(
                    lu_id=record.get("lu_id", ""),
                    condition=condition,
                    decision=decision,
                    key_vector=key_vec,
                    value_vector=val_vec,
                    reliability=record.get("reliability", 1.0),
                    metadata=record.get("metadata"),
                )

        # 按原始顺序返回
        return [cached_results[i] for i in range(len(records))]

    def get_stats(self) -> Dict[str, Any]:
        """获取编码器统计"""
        return {
            "type": self.__class__.__name__,
            "backend": self.config.backend,
            "model_name": self.config.model_name,
            "key_dim": self.config.key_dim,
            "value_dim": self.config.value_dim,
            "device": self.config.device,
            "initialized": self._initialized,
            "encode_count": self._encode_count,
            "cache_size": len(self._cache),
            "cache_max_size": self.config.cache_max_size,
        }

    def shutdown(self) -> None:
        """释放资源"""
        self._cache.clear()
        self._initialized = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"backend={self.config.backend!r}, "
            f"key_dim={self.config.key_dim}, "
            f"value_dim={self.config.value_dim})"
        )
