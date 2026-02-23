"""
简单哈希编码器

基于确定性哈希的轻量级编码器，不依赖任何 ML 模型。
适用于测试、开发和快速原型验证。

特点:
  - 零外部依赖（仅使用标准库）
  - 确定性（相同输入 -> 相同输出）
  - 极快（微秒级）
  - 语义能力有限（基于字符级哈希，非语义嵌入）

注意:
  此编码器不提供真正的语义理解能力。
  在生产环境中应使用 SentenceTransformerEncoder 或自定义编码器。
"""

import hashlib
import math
import logging
from typing import List, Tuple, Dict, Any

from .base import BaseEncoder, EncoderConfig

logger = logging.getLogger(__name__)


class SimpleHashEncoder(BaseEncoder):
    """
    简单哈希编码器

    使用 SHA-256 哈希生成确定性的伪随机向量。
    适用于测试和开发环境。

    编码流程:
        text -> SHA-256 -> 扩展到目标维度 -> 归一化 -> 缩放
    """

    def __init__(self, config: EncoderConfig):
        super().__init__(config)

    def _initialize(self) -> None:
        """无需初始化"""
        logger.info(
            f"SimpleHashEncoder 初始化: "
            f"key_dim={self.config.key_dim}, "
            f"value_dim={self.config.value_dim}"
        )

    def _encode_texts(
        self,
        conditions: List[str],
        decisions: List[str],
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        批量编码文本

        condition -> hash -> key 向量
        decision  -> hash -> value 向量
        """
        key_vectors = []
        value_vectors = []

        for condition, decision in zip(conditions, decisions):
            key_vec = self._hash_to_vector(
                f"key:{condition}",
                self.config.key_dim,
            )
            value_vec = self._hash_to_vector(
                f"value:{decision}",
                self.config.value_dim,
            )

            # 范数缩放
            key_norm_target = self.config.options.get("key_norm_target", 5.0)
            value_norm_target = self.config.options.get("value_norm_target", 3.0)
            key_vec = self._scale_vector(key_vec, key_norm_target)
            value_vec = self._scale_vector(value_vec, value_norm_target)

            key_vectors.append(key_vec)
            value_vectors.append(value_vec)

        return key_vectors, value_vectors

    @staticmethod
    def _hash_to_vector(text: str, dim: int) -> List[float]:
        """
        将文本哈希为指定维度的向量

        使用多轮 SHA-256 哈希扩展到目标维度。
        """
        vector = []
        round_idx = 0

        while len(vector) < dim:
            # 每轮使用不同的种子
            hash_input = f"{text}:round:{round_idx}"
            digest = hashlib.sha256(hash_input.encode("utf-8")).digest()

            # 每 4 字节转换为一个 float（范围 [-1, 1]）
            for i in range(0, len(digest), 4):
                if len(vector) >= dim:
                    break
                # 将 4 字节转换为 uint32，然后映射到 [-1, 1]
                value = int.from_bytes(digest[i:i + 4], "big")
                normalized = (value / (2 ** 32 - 1)) * 2.0 - 1.0
                vector.append(normalized)

            round_idx += 1

        return vector[:dim]

    @staticmethod
    def _scale_vector(vector: List[float], target_norm: float) -> List[float]:
        """缩放向量到目标范数"""
        current_norm = math.sqrt(sum(v * v for v in vector))
        if current_norm < 1e-10:
            return vector
        scale = target_norm / current_norm
        return [v * scale for v in vector]
