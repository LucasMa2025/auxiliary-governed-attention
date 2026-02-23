"""
SentenceTransformer 编码器

使用 sentence-transformers 库将文本编码为向量，
然后通过可训练的投影层映射到 aga-core 所需的维度。

这是推荐的生产编码器，因为：
1. sentence-transformers 提供高质量的语义嵌入
2. 投影层可以通过少量数据微调
3. 支持多语言模型
4. CPU 友好，不占用推理 GPU

依赖:
    pip install sentence-transformers torch
"""

import logging
import hashlib
from typing import List, Tuple, Dict, Any, Optional

from .base import BaseEncoder, EncoderConfig

logger = logging.getLogger(__name__)


class SentenceTransformerEncoder(BaseEncoder):
    """
    基于 SentenceTransformer 的编码器

    编码流程:
        condition -> SentenceTransformer -> embedding [embed_dim]
                                         -> key_proj -> key [key_dim]
        decision  -> SentenceTransformer -> embedding [embed_dim]
                                         -> value_proj -> value [value_dim]

    投影层:
        - key_proj: Linear(embed_dim, key_dim)
        - value_proj: Linear(embed_dim, value_dim)
        - 默认使用 Xavier 初始化
        - 可通过 load_projections() 加载预训练权重

    配置:
        encoder:
          backend: "sentence_transformer"
          model_name: "all-MiniLM-L6-v2"    # 384 维
          key_dim: 64
          value_dim: 4096
          device: "cpu"
          options:
            projection_path: null            # 预训练投影层路径
            condition_prefix: "condition: "  # 条件文本前缀
            decision_prefix: "decision: "    # 决策文本前缀
    """

    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self._model = None
        self._key_proj = None
        self._value_proj = None
        self._embed_dim: Optional[int] = None

    def _initialize(self) -> None:
        """延迟加载 SentenceTransformer 和投影层"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformerEncoder 需要 sentence-transformers 包。\n"
                "请运行: pip install sentence-transformers"
            )

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "SentenceTransformerEncoder 需要 PyTorch。\n"
                "请运行: pip install torch"
            )

        logger.info(f"加载 SentenceTransformer 模型: {self.config.model_name}")
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

        # 获取嵌入维度
        self._embed_dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"嵌入维度: {self._embed_dim}")

        # 创建投影层
        self._key_proj = nn.Linear(
            self._embed_dim, self.config.key_dim, bias=False
        ).to(self.config.device)

        self._value_proj = nn.Linear(
            self._embed_dim, self.config.value_dim, bias=False
        ).to(self.config.device)

        # Xavier 初始化
        nn.init.xavier_uniform_(self._key_proj.weight)
        nn.init.xavier_uniform_(self._value_proj.weight)

        # 加载预训练投影层（如果指定）
        projection_path = self.config.options.get("projection_path")
        if projection_path:
            self._load_projections(projection_path)

        logger.info(
            f"SentenceTransformerEncoder 初始化完成: "
            f"embed_dim={self._embed_dim}, "
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

        condition -> key_proj -> key 向量
        decision  -> value_proj -> value 向量
        """
        import torch

        condition_prefix = self.config.options.get("condition_prefix", "")
        decision_prefix = self.config.options.get("decision_prefix", "")

        # 添加前缀
        prefixed_conditions = [f"{condition_prefix}{c}" for c in conditions]
        prefixed_decisions = [f"{decision_prefix}{d}" for d in decisions]

        # 编码
        with torch.no_grad():
            cond_embeddings = self._model.encode(
                prefixed_conditions,
                convert_to_tensor=True,
                device=self.config.device,
                show_progress_bar=False,
            )
            dec_embeddings = self._model.encode(
                prefixed_decisions,
                convert_to_tensor=True,
                device=self.config.device,
                show_progress_bar=False,
            )

            # 投影
            key_vectors = self._key_proj(cond_embeddings)
            value_vectors = self._value_proj(dec_embeddings)

            # 归一化
            if self.config.normalize:
                key_vectors = torch.nn.functional.normalize(key_vectors, dim=-1)
                value_vectors = torch.nn.functional.normalize(value_vectors, dim=-1)

            # 范数缩放（匹配 aga-core 的 key_norm_target / value_norm_target）
            key_norm_target = self.config.options.get("key_norm_target", 5.0)
            value_norm_target = self.config.options.get("value_norm_target", 3.0)
            key_vectors = key_vectors * key_norm_target
            value_vectors = value_vectors * value_norm_target

        return (
            key_vectors.cpu().tolist(),
            value_vectors.cpu().tolist(),
        )

    def _load_projections(self, path: str) -> None:
        """加载预训练投影层权重"""
        import torch

        try:
            state_dict = torch.load(path, map_location=self.config.device)
            if "key_proj" in state_dict:
                self._key_proj.load_state_dict(state_dict["key_proj"])
            if "value_proj" in state_dict:
                self._value_proj.load_state_dict(state_dict["value_proj"])
            logger.info(f"已加载预训练投影层: {path}")
        except Exception as e:
            logger.warning(f"加载投影层失败，使用随机初始化: {e}")

    def save_projections(self, path: str) -> None:
        """保存投影层权重"""
        import torch

        if self._key_proj is None or self._value_proj is None:
            raise RuntimeError("编码器未初始化")

        torch.save({
            "key_proj": self._key_proj.state_dict(),
            "value_proj": self._value_proj.state_dict(),
            "embed_dim": self._embed_dim,
            "key_dim": self.config.key_dim,
            "value_dim": self.config.value_dim,
            "model_name": self.config.model_name,
        }, path)
        logger.info(f"投影层已保存: {path}")

    def get_stats(self) -> Dict[str, Any]:
        """获取编码器统计"""
        stats = super().get_stats()
        stats["embed_dim"] = self._embed_dim
        return stats

    def shutdown(self) -> None:
        """释放资源"""
        super().shutdown()
        self._model = None
        self._key_proj = None
        self._value_proj = None
        self._embed_dim = None
