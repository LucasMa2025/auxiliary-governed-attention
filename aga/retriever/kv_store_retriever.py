"""
aga/retriever/kv_store_retriever.py — KVStore 本地召回器

基于 AGA 内部 KVStore 的简单召回器实现。
在 KVStore 中使用 bottleneck 空间的余弦相似度进行检索。

适用场景:
  - KVStore 中已加载大量知识（> top_k）
  - 需要在 forward 路径中动态选择最相关的子集
  - 作为外部召回器的补充（先查本地，再查远程）

注意:
  这不是 BottleneckInjector 内部的 Top-K 路由。
  Top-K 路由是在 get_active() 返回的所有活跃槽位上操作的。
  KVStoreRetriever 是在 forward 路径之前，根据语义相似度
  预筛选候选知识，然后动态加载到 KVStore 中。
"""
import time
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

from .base import BaseRetriever, RetrievalQuery, RetrievalResult

logger = logging.getLogger(__name__)


class KVStoreRetriever(BaseRetriever):
    """
    KVStore 本地召回器

    使用 q_proj 投影后的查询向量与 KVStore 中的 keys 进行
    余弦相似度匹配，返回最相关的 top_k 个知识条目。

    这个召回器的价值在于：当 KVStore 容量很大时（如 5000 slots），
    不需要让 BottleneckInjector 对所有 5000 个 slot 计算注意力，
    而是先用召回器筛选出最相关的 top_k 个。
    """

    def __init__(
        self,
        kv_store: "KVStore",
        default_top_k: int = 10,
        min_similarity: float = 0.3,
    ):
        """
        Args:
            kv_store: AGA 的 KVStore 实例
            default_top_k: 默认返回的最大结果数
            min_similarity: 最小相似度阈值
        """
        self._store = kv_store
        self._default_top_k = default_top_k
        self._min_similarity = min_similarity

        # 统计
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._total_results = 0

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        从 KVStore 中检索最相关的知识

        使用 query_projected（bottleneck 空间）进行余弦相似度匹配。
        如果 query_projected 不可用，使用 hidden_states 的均值。
        """
        start = time.perf_counter()
        self._call_count += 1

        try:
            if self._store.count == 0:
                return []

            keys, values, reliability = self._store.get_active()
            if keys.shape[0] == 0:
                return []

            # 构建查询向量
            if query.query_projected is not None:
                # 使用 bottleneck 空间的投影查询
                # query_projected: [batch, seq, bottleneck_dim]
                q = query.query_projected.mean(dim=(0, 1))  # [bottleneck_dim]
            else:
                # 回退：使用 hidden_states 均值（维度可能不匹配，需要截断/填充）
                q_full = query.hidden_states.mean(dim=(0, 1))  # [hidden_dim]
                # 如果维度不匹配，截断到 key_dim
                if q_full.shape[0] != keys.shape[1]:
                    q = q_full[:keys.shape[1]]
                else:
                    q = q_full

            # 余弦相似度
            q_norm = F.normalize(q.unsqueeze(0), dim=-1)  # [1, dim]
            k_norm = F.normalize(keys.float(), dim=-1)  # [n, dim]
            similarities = torch.mm(q_norm, k_norm.t()).squeeze(0)  # [n]

            # 加入可靠性加权
            weighted_scores = similarities * reliability.float()

            # Top-K 选择
            top_k = min(query.top_k or self._default_top_k, keys.shape[0])
            top_scores, top_indices = torch.topk(weighted_scores, top_k)

            # 过滤低相似度
            results = []
            # 反向映射 slot_idx -> id
            slot_to_id = self._store._slot_to_id
            active_mask = self._store.active
            active_indices = torch.where(active_mask)[0]

            for i in range(top_k):
                score = top_scores[i].item()
                if score < self._min_similarity:
                    continue

                # 获取在 active 子集中的索引对应的原始 slot 索引
                active_local_idx = top_indices[i].item()
                if active_local_idx < len(active_indices):
                    original_slot_idx = active_indices[active_local_idx].item()
                    knowledge_id = slot_to_id.get(original_slot_idx, f"slot_{original_slot_idx}")
                else:
                    knowledge_id = f"unknown_{active_local_idx}"

                results.append(RetrievalResult(
                    id=knowledge_id,
                    key=keys[active_local_idx],
                    value=values[active_local_idx],
                    reliability=reliability[active_local_idx].item(),
                    score=score,
                    metadata=self._store.get_metadata(knowledge_id),
                ))

            self._total_results += len(results)
            return results

        except Exception as e:
            logger.warning(f"KVStoreRetriever 检索失败 (Fail-Open): {e}", exc_info=True)
            return []

        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._total_latency_ms += elapsed_ms

    def get_stats(self) -> Dict[str, Any]:
        avg_latency = self._total_latency_ms / max(self._call_count, 1)
        avg_results = self._total_results / max(self._call_count, 1)
        return {
            "type": "KVStoreRetriever",
            "call_count": self._call_count,
            "avg_latency_ms": round(avg_latency, 3),
            "avg_results_per_call": round(avg_results, 2),
            "total_results": self._total_results,
            "min_similarity": self._min_similarity,
            "default_top_k": self._default_top_k,
        }

    def __repr__(self) -> str:
        return (
            f"KVStoreRetriever("
            f"top_k={self._default_top_k}, "
            f"min_sim={self._min_similarity})"
        )
