"""
HNSW 向量索引

封装 hnswlib，提供增量插入/删除和高效 ANN 搜索。

为什么是 HNSW 而非 FAISS IVF+PQ:
  - HNSW 原生支持增量插入/删除（AGA 知识是动态的）
  - HNSW 对碎片化语义不敏感（AGA 知识 100-500 tokens 片段）
  - HNSW 在 1K-1M 规模下召回率 >95%
  - FAISS IVF+PQ 需要重训练质心，不适合动态数据

性能预期:
  1K:   0.2ms
  10K:  0.5ms
  50K:  0.8ms
  100K: 1.0ms

配置:
  retriever:
    index_backend: "hnsw"
    hnsw_m: 16                 # 每层连接数（影响召回率和内存）
    hnsw_ef_construction: 200  # 构建时搜索宽度
    hnsw_ef_search: 100        # 查询时搜索宽度
    hnsw_max_elements: 100000  # 最大容量

依赖:
    pip install hnswlib
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HNSWIndex:
    """
    HNSW 向量索引

    封装 hnswlib.Index，提供字符串 ID 映射和增量操作。

    Args:
        dim: 向量维度（必须与 AGACoreAlignment.bottleneck_dim 一致）
        max_elements: 最大容量
        m: 每层连接数（越大召回率越高，内存越大）
        ef_construction: 构建时搜索宽度（越大构建越慢但质量越高）
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 100000,
        m: int = 16,
        ef_construction: int = 200,
    ):
        try:
            import hnswlib
        except ImportError:
            raise ImportError(
                "HNSWIndex 需要 hnswlib 包。\n"
                "请运行: pip install hnswlib"
            )

        self._dim = dim
        self._max_elements = max_elements
        self._m = m
        self._ef_construction = ef_construction

        # 初始化 hnswlib 索引
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            M=m,
            ef_construction=ef_construction,
        )

        # ID 映射
        self._id_to_label: Dict[str, int] = {}
        self._label_to_id: Dict[int, str] = {}
        self._next_label: int = 0

        # 标记删除的数量（用于判断是否需要重建）
        self._deleted_count: int = 0

        logger.info(
            f"HNSWIndex 已初始化: dim={dim}, max={max_elements}, "
            f"M={m}, ef_construction={ef_construction}"
        )

    def add(self, lu_id: str, vector: np.ndarray) -> None:
        """
        增量添加向量

        如果 lu_id 已存在，先标记删除旧向量再添加新的。

        Args:
            lu_id: 知识 ID
            vector: 向量 [dim]
        """
        # 如果已存在，先删除
        if lu_id in self._id_to_label:
            self.remove(lu_id)

        # 检查容量
        current = self._index.get_current_count()
        if current >= self._max_elements:
            logger.warning(
                f"HNSW 索引已满 ({current}/{self._max_elements})，"
                f"跳过添加: {lu_id}"
            )
            return

        label = self._next_label
        self._next_label += 1

        vec = vector.reshape(1, -1).astype(np.float32)
        self._index.add_items(vec, np.array([label]))

        self._id_to_label[lu_id] = label
        self._label_to_id[label] = lu_id

    def add_batch(self, lu_ids: List[str], vectors: np.ndarray) -> None:
        """
        批量添加向量

        Args:
            lu_ids: 知识 ID 列表
            vectors: 向量矩阵 [N, dim]
        """
        if len(lu_ids) == 0:
            return

        # 检查容量
        current = self._index.get_current_count()
        available = self._max_elements - current
        if available < len(lu_ids):
            logger.warning(
                f"HNSW 索引容量不足: 需要 {len(lu_ids)}, "
                f"可用 {available}。只添加前 {available} 条。"
            )
            lu_ids = lu_ids[:available]
            vectors = vectors[:available]

        # 移除已存在的
        for lu_id in lu_ids:
            if lu_id in self._id_to_label:
                self.remove(lu_id)

        # 分配标签
        labels = np.arange(
            self._next_label, self._next_label + len(lu_ids)
        )
        self._next_label += len(lu_ids)

        # 批量添加
        vecs = vectors.astype(np.float32)
        self._index.add_items(vecs, labels)

        # 更新映射
        for lu_id, label in zip(lu_ids, labels):
            self._id_to_label[lu_id] = int(label)
            self._label_to_id[int(label)] = lu_id

    def remove(self, lu_id: str) -> None:
        """
        标记删除

        hnswlib 使用标记删除，不会真正释放空间。
        积累大量删除后应调用 rebuild() 重建索引。

        Args:
            lu_id: 知识 ID
        """
        if lu_id in self._id_to_label:
            label = self._id_to_label[lu_id]
            try:
                self._index.mark_deleted(label)
            except RuntimeError:
                # 已被标记删除
                pass
            del self._id_to_label[lu_id]
            del self._label_to_id[label]
            self._deleted_count += 1

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        ANN 搜索

        Args:
            query: 查询向量 [dim]
            k: 返回数量
            ef_search: 搜索时宽度（越大越精确但越慢）

        Returns:
            [(lu_id, similarity)] 列表，按相似度降序
            similarity = 1.0 - cosine_distance
        """
        current_count = self.count
        if current_count == 0:
            return []

        self._index.set_ef(ef_search)

        actual_k = min(k, current_count)
        query_vec = query.reshape(1, -1).astype(np.float32)

        try:
            labels, distances = self._index.knn_query(query_vec, k=actual_k)
        except RuntimeError as e:
            logger.warning(f"HNSW 搜索失败: {e}")
            return []

        results = []
        for label, dist in zip(labels[0], distances[0]):
            lu_id = self._label_to_id.get(int(label))
            if lu_id:
                # hnswlib cosine space: distance = 1 - cosine_similarity
                similarity = 1.0 - float(dist)
                results.append((lu_id, similarity))

        return results

    def contains(self, lu_id: str) -> bool:
        """检查 lu_id 是否在索引中"""
        return lu_id in self._id_to_label

    def rebuild(self, vectors_map: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        重建索引（清除标记删除的碎片）

        当 deleted_count 过大时应调用此方法回收空间。

        Args:
            vectors_map: 如果提供，用此数据重建；否则无法重建
        """
        if vectors_map is None:
            logger.warning("rebuild() 需要提供 vectors_map 参数")
            return

        import hnswlib

        new_index = hnswlib.Index(space="cosine", dim=self._dim)
        new_index.init_index(
            max_elements=self._max_elements,
            M=self._m,
            ef_construction=self._ef_construction,
        )

        self._id_to_label.clear()
        self._label_to_id.clear()
        self._next_label = 0
        self._deleted_count = 0

        if vectors_map:
            lu_ids = list(vectors_map.keys())
            vectors = np.array(
                [vectors_map[lid] for lid in lu_ids], dtype=np.float32
            )
            labels = np.arange(len(lu_ids))
            new_index.add_items(vectors, labels)

            for lu_id, label in zip(lu_ids, labels):
                self._id_to_label[lu_id] = int(label)
                self._label_to_id[int(label)] = lu_id

            self._next_label = len(lu_ids)

        self._index = new_index
        logger.info(f"HNSW 索引已重建: count={self.count}")

    @property
    def count(self) -> int:
        """当前有效元素数量"""
        return len(self._id_to_label)

    @property
    def deleted_count(self) -> int:
        """已标记删除的数量"""
        return self._deleted_count

    @property
    def needs_rebuild(self) -> bool:
        """是否需要重建（删除数量超过有效数量的 50%）"""
        if self.count == 0:
            return self._deleted_count > 0
        return self._deleted_count > self.count * 0.5

    def get_stats(self) -> Dict:
        """获取索引统计"""
        return {
            "type": "HNSWIndex",
            "dim": self._dim,
            "count": self.count,
            "deleted_count": self._deleted_count,
            "max_elements": self._max_elements,
            "M": self._m,
            "ef_construction": self._ef_construction,
            "needs_rebuild": self.needs_rebuild,
        }
