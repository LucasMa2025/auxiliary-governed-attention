"""
倒数排名融合 (Reciprocal Rank Fusion, RRF)

将稠密检索（HNSW）和稀疏检索（BM25）的结果融合为统一排序。

RRF 公式:
  RRF_score(d) = Σ weight_i / (k + rank_i(d))

其中:
  - k: 常数（默认 60），防止排名靠前的结果权重过大
  - rank_i(d): 文档 d 在第 i 个检索系统中的排名（从 1 开始）
  - weight_i: 第 i 个检索系统的权重

参考:
  Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
  and individual Rank Learning Methods", SIGIR 2009
"""

import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> List[Tuple[str, float]]:
    """
    倒数排名融合

    将稠密检索和稀疏检索的结果融合为统一排序。

    Args:
        dense_results: 稠密检索结果 [(lu_id, score), ...]，已按 score 降序
        sparse_results: 稀疏检索结果 [(lu_id, score), ...]，已按 score 降序
        k: RRF 常数（默认 60）
        dense_weight: 稠密检索权重（默认 0.7）
        sparse_weight: 稀疏检索权重（默认 0.3）

    Returns:
        融合后的排序结果 [(lu_id, rrf_score), ...]，按 rrf_score 降序
    """
    scores: Dict[str, float] = {}

    # 稠密检索的 RRF 贡献
    for rank, (lu_id, _) in enumerate(dense_results):
        scores[lu_id] = scores.get(lu_id, 0.0) + dense_weight / (k + rank + 1)

    # 稀疏检索的 RRF 贡献
    for rank, (lu_id, _) in enumerate(sparse_results):
        scores[lu_id] = scores.get(lu_id, 0.0) + sparse_weight / (k + rank + 1)

    # 按 RRF 分数降序排序
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results


def weighted_score_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    normalize: bool = True,
) -> List[Tuple[str, float]]:
    """
    加权分数融合（备选方案）

    直接对分数加权求和，适合两个检索系统分数尺度一致的场景。

    Args:
        dense_results: 稠密检索结果 [(lu_id, score), ...]
        sparse_results: 稀疏检索结果 [(lu_id, score), ...]
        dense_weight: 稠密检索权重
        sparse_weight: 稀疏检索权重
        normalize: 是否先归一化分数到 [0, 1]

    Returns:
        融合后的排序结果 [(lu_id, fused_score), ...]
    """
    # 归一化（可选）
    if normalize:
        dense_results = _normalize_scores(dense_results)
        sparse_results = _normalize_scores(sparse_results)

    scores: Dict[str, float] = {}

    for lu_id, score in dense_results:
        scores[lu_id] = scores.get(lu_id, 0.0) + dense_weight * score

    for lu_id, score in sparse_results:
        scores[lu_id] = scores.get(lu_id, 0.0) + sparse_weight * score

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results


def _normalize_scores(
    results: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    """
    Min-Max 归一化分数到 [0, 1]

    Args:
        results: [(id, score), ...]

    Returns:
        归一化后的 [(id, normalized_score), ...]
    """
    if not results:
        return results

    scores = [s for _, s in results]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range < 1e-10:
        return [(lid, 1.0) for lid, _ in results]

    return [
        (lid, (score - min_score) / score_range)
        for lid, score in results
    ]
