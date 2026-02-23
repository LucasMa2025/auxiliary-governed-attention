"""
BM25 稀疏检索索引

对 condition 和 decision 文本建立倒排索引，
提供关键词级别的精确匹配能力。

与稠密检索（HNSW）互补:
  - 稠密检索: 依赖编码器语义理解，适合语义相似度匹配
  - 稀疏检索: 基于关键词精确匹配，不依赖编码器质量

当编码器投影层未充分训练时，BM25 的关键词匹配是必要的补充。
当无法获取文本查询（只有向量查询）时，BM25 自动跳过，系统退化为纯稠密检索。

配置:
  retriever:
    bm25_enabled: true
    bm25_weight: 0.3          # RRF 融合中 BM25 的权重
    bm25_k1: 1.5              # BM25 词频饱和参数
    bm25_b: 0.75              # BM25 文档长度归一化参数

依赖:
    pip install rank-bm25
"""

import logging
import re
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 稀疏检索索引

    对知识的 condition + decision 文本建立倒排索引，
    支持中英文混合分词和增量更新。

    Args:
        k1: BM25 词频饱和参数（默认 1.5）
        b: BM25 文档长度归一化参数（默认 0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._documents: Dict[str, str] = {}  # lu_id → 合并文本
        self._bm25 = None  # BM25Okapi 实例
        self._ids: List[str] = []  # 与 BM25 语料库行对应的 lu_id
        self._k1 = k1
        self._b = b
        self._dirty = False  # 是否需要重建

        logger.debug(f"BM25Index 已初始化: k1={k1}, b={b}")

    def add(self, lu_id: str, condition: str, decision: str) -> None:
        """
        添加文档到索引

        Args:
            lu_id: 知识 ID
            condition: 条件文本
            decision: 决策文本
        """
        text = f"{condition} {decision}".strip()
        if not text:
            return

        self._documents[lu_id] = text
        self._dirty = True

    def add_batch(
        self,
        lu_ids: List[str],
        conditions: List[str],
        decisions: List[str],
    ) -> None:
        """
        批量添加文档

        Args:
            lu_ids: 知识 ID 列表
            conditions: 条件文本列表
            decisions: 决策文本列表
        """
        for lu_id, cond, dec in zip(lu_ids, conditions, decisions):
            text = f"{cond} {dec}".strip()
            if text:
                self._documents[lu_id] = text

        if lu_ids:
            self._dirty = True

    def remove(self, lu_id: str) -> None:
        """
        从索引中移除文档

        Args:
            lu_id: 知识 ID
        """
        if lu_id in self._documents:
            del self._documents[lu_id]
            self._dirty = True

    def search(self, query_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25 搜索

        Args:
            query_text: 查询文本
            k: 返回数量

        Returns:
            [(lu_id, bm25_score)] 列表，按分数降序
        """
        if not query_text or not self._documents:
            return []

        # 按需重建索引
        if self._dirty or self._bm25 is None:
            self._rebuild()

        if self._bm25 is None or not self._ids:
            return []

        tokenized_query = self._tokenize(query_text)
        if not tokenized_query:
            return []

        try:
            scores = self._bm25.get_scores(tokenized_query)
        except Exception as e:
            logger.warning(f"BM25 搜索失败: {e}")
            return []

        # 获取 top-k
        if len(scores) == 0:
            return []

        import numpy as np
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self._ids[idx], score))

        return results

    def _rebuild(self) -> None:
        """重建 BM25 索引"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "BM25Index 需要 rank-bm25 包。\n"
                "请运行: pip install rank-bm25\n"
                "BM25 检索已禁用。"
            )
            self._bm25 = None
            return

        self._ids = list(self._documents.keys())
        if not self._ids:
            self._bm25 = None
            self._dirty = False
            return

        corpus = [self._tokenize(self._documents[lid]) for lid in self._ids]
        self._bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b)
        self._dirty = False

        logger.debug(f"BM25 索引已重建: documents={len(self._ids)}")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        简单分词（中英文混合）

        中文按字分词，英文按词分词，数字保留。
        不依赖外部分词库，足以满足 BM25 的关键词匹配需求。

        Args:
            text: 待分词文本

        Returns:
            token 列表
        """
        if not text:
            return []
        # 中文按字，英文按词，数字保留
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+", text.lower())
        return tokens

    def contains(self, lu_id: str) -> bool:
        """检查 lu_id 是否在索引中"""
        return lu_id in self._documents

    @property
    def count(self) -> int:
        """文档数量"""
        return len(self._documents)

    def clear(self) -> None:
        """清空索引"""
        self._documents.clear()
        self._ids.clear()
        self._bm25 = None
        self._dirty = False

    def get_stats(self) -> Dict:
        """获取索引统计"""
        return {
            "type": "BM25Index",
            "count": self.count,
            "k1": self._k1,
            "b": self._b,
            "dirty": self._dirty,
        }
