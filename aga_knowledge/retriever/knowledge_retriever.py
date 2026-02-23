"""
KnowledgeRetriever — aga-knowledge 到 aga-core 的召回器桥接

实现 aga-core 的 BaseRetriever 协议，将 KnowledgeManager 中的
明文 condition/decision 知识通过 Encoder 转换为向量，
并提供生产级语义检索能力。

核心流程:
    1. 启动时验证编码器与 aga-core 的对齐（AGACoreAlignment）
    2. 从 KnowledgeManager 加载活跃知识
    3. 通过 Encoder 将明文编码为 key/value 向量
    4. 构建 HNSW 索引（稠密检索）和 BM25 索引（稀疏检索）
    5. aga-core 高熵触发时，执行混合检索（HNSW + BM25 + RRF 融合）
    6. 返回 RetrievalResult 列表供 aga-core 注入

检索架构:
    RetrievalQuery
         │
    ┌────▼────┐
    │ 查询路由 │
    └────┬────┘
         │
    ┌────┼────┐
    │         │
    稠密检索   稀疏检索
    (HNSW)    (BM25)
    │         │
    └────┬────┘
         │
    ┌────▼────┐
    │ RRF 融合 │
    └────┬────┘
         │
    List[RetrievalResult]

设计要点:
    - AGACoreAlignment 对齐验证: 构造时强制验证，不对齐则抛出 ConfigError
    - 效果优先于性能: 高熵触发频率低，1-10ms 延迟可接受
    - Fail-Open: 检索失败返回空列表，不影响推理
    - 增量更新: 支持知识变更时的增量索引更新
    - 线程安全: 支持多线程并发检索
    - 优雅降级: HNSW/BM25 不可用时自动回退到暴力搜索/纯稠密检索
"""

import time
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

from ..alignment import AGACoreAlignment
from ..exceptions import KnowledgeError

logger = logging.getLogger(__name__)


class ConfigError(KnowledgeError):
    """配置错误 — 编码器与 aga-core 不对齐"""
    pass


# 延迟导入 aga-core 的类型，避免硬依赖
try:
    from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult
    _HAS_AGA_CORE = True
except ImportError:
    _HAS_AGA_CORE = False

    # 定义占位基类
    class BaseRetriever:  # type: ignore
        def retrieve(self, query): return []
        def warmup(self): pass
        def on_injection_feedback(self, result_id, was_used, gate_value=0.0): pass
        def get_stats(self): return {}
        def shutdown(self): pass

    class RetrievalQuery:  # type: ignore
        pass

    class RetrievalResult:  # type: ignore
        pass


class KnowledgeRetriever(BaseRetriever):
    """
    aga-knowledge 知识召回器（生产版）

    实现 aga-core 的 BaseRetriever 协议，集成:
    - AGACoreAlignment 对齐验证
    - HNSW 稠密检索
    - BM25 稀疏检索
    - RRF 融合

    Args:
        manager: KnowledgeManager 实例（提供明文知识）
        encoder: BaseEncoder 实例（文本 → 向量）
        alignment: AGACoreAlignment 实例（必须提供）
        namespace: 检索的命名空间
        auto_refresh_interval: 自动刷新索引的间隔（秒），0 不自动刷新
        similarity_threshold: 最低相似度阈值
        index_backend: 索引后端（"brute" | "hnsw"）
        bm25_enabled: 是否启用 BM25 稀疏检索
        bm25_weight: RRF 融合中 BM25 的权重
        bm25_k1: BM25 k1 参数
        bm25_b: BM25 b 参数
        hnsw_m: HNSW 每层连接数
        hnsw_ef_construction: HNSW 构建时搜索宽度
        hnsw_ef_search: HNSW 查询时搜索宽度
        hnsw_max_elements: HNSW 最大容量

    Raises:
        ConfigError: 编码器配置与 aga-core 不对齐

    Usage:
        alignment = AGACoreAlignment.from_aga_config_yaml("aga_config.yaml")
        encoder_config = EncoderConfig.from_alignment(alignment)
        encoder = create_encoder(encoder_config)

        retriever = KnowledgeRetriever(
            manager=knowledge_manager,
            encoder=encoder,
            alignment=alignment,
            index_backend="hnsw",
            bm25_enabled=True,
        )
        plugin = AGAPlugin(config, retriever=retriever)
    """

    def __init__(
        self,
        manager,  # KnowledgeManager
        encoder,  # BaseEncoder
        alignment: AGACoreAlignment,
        namespace: str = "default",
        auto_refresh_interval: float = 0,
        similarity_threshold: float = 0.0,
        # 索引配置
        index_backend: str = "brute",      # "brute" | "hnsw"
        # BM25 配置
        bm25_enabled: bool = False,
        bm25_weight: float = 0.3,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        # HNSW 配置
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 100,
        hnsw_max_elements: int = 100000,
    ):
        # ==================== 对齐验证 ====================
        self.alignment = alignment

        # 验证对齐参数
        alignment_errors = alignment.validate()
        if alignment_errors:
            raise ConfigError(
                "AGACoreAlignment 参数无效:\n"
                + "\n".join(f"  - {e}" for e in alignment_errors)
            )

        # 验证编码器与 aga-core 对齐
        mismatches = encoder.config.validate_alignment(alignment)
        if mismatches:
            raise ConfigError(
                "编码器配置与 aga-core 不对齐:\n"
                + "\n".join(f"  - {m}" for m in mismatches)
            )

        logger.info(f"编码器对齐验证通过: {alignment.summary()}")

        # ==================== 基础属性 ====================
        self.manager = manager
        self.encoder = encoder
        self.namespace = namespace
        self.auto_refresh_interval = auto_refresh_interval
        self.similarity_threshold = similarity_threshold
        self._index_backend = index_backend

        # ==================== 向量索引 ====================
        # 编码后的知识: {lu_id: EncodedKnowledge}
        self._index: Dict[str, Any] = {}
        # 暴力搜索用矩阵
        self._key_matrix: Optional[torch.Tensor] = None  # [N, key_dim]
        self._index_ids: List[str] = []  # 与矩阵行对应的 lu_id
        self._index_metadata: Dict[str, Dict[str, Any]] = {}

        # ==================== HNSW 索引 ====================
        self._hnsw_index = None
        self._hnsw_ef_search = hnsw_ef_search

        if index_backend == "hnsw":
            try:
                from .hnsw_index import HNSWIndex
                self._hnsw_index = HNSWIndex(
                    dim=alignment.bottleneck_dim,
                    max_elements=hnsw_max_elements,
                    m=hnsw_m,
                    ef_construction=hnsw_ef_construction,
                )
                logger.info("HNSW 索引已启用")
            except ImportError:
                logger.warning(
                    "hnswlib 未安装，回退到暴力搜索。"
                    "生产环境建议安装: pip install hnswlib"
                )
                self._hnsw_index = None

        # ==================== BM25 索引 ====================
        self._bm25_index = None
        self._bm25_weight = bm25_weight

        if bm25_enabled:
            try:
                from .bm25_index import BM25Index
                self._bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)
                logger.info("BM25 索引已启用")
            except ImportError:
                logger.warning(
                    "rank-bm25 未安装，BM25 检索已禁用。"
                    "安装: pip install rank-bm25"
                )
                self._bm25_index = None

        # ==================== 状态 ====================
        self._initialized = False
        self._lock = threading.RLock()
        self._last_refresh_time: float = 0

        # ==================== 统计 ====================
        self._stats = {
            "retrieve_count": 0,
            "total_retrieve_time_ms": 0.0,
            "total_results_returned": 0,
            "index_size": 0,
            "refresh_count": 0,
            "feedback_used": 0,
            "feedback_unused": 0,
            "errors": 0,
            "hnsw_searches": 0,
            "bm25_searches": 0,
            "brute_searches": 0,
            "fused_searches": 0,
        }

    # ==================== BaseRetriever 协议 ====================

    def warmup(self) -> None:
        """
        预热：加载知识并构建向量索引

        此方法在 AGAPlugin.attach() 后、推理开始前被调用。
        1. 确保 Encoder 已初始化
        2. 从 KnowledgeManager 加载活跃知识
        3. 批量编码为向量
        4. 构建 HNSW + BM25 索引
        """
        if self._initialized:
            return

        try:
            self.encoder.warmup()
            self._refresh_index_sync()
            self._initialized = True

            logger.info(
                f"KnowledgeRetriever 预热完成: "
                f"namespace={self.namespace}, "
                f"index_size={len(self._index_ids)}, "
                f"backend={self._index_backend}, "
                f"hnsw={'启用' if self._hnsw_index else '禁用'}, "
                f"bm25={'启用' if self._bm25_index else '禁用'}"
            )
        except Exception as e:
            logger.error(f"KnowledgeRetriever 预热失败: {e}", exc_info=True)
            # Fail-Open: 预热失败不阻塞推理
            self._initialized = True

    def retrieve(self, query) -> list:
        """
        核心检索方法 — 混合检索（HNSW + BM25 + RRF）

        在 aga-core 高熵触发时被调用。

        流程:
        1. 提取查询向量
        2. 稠密检索（HNSW 或暴力搜索）
        3. 稀疏检索（BM25，如果有文本查询）
        4. RRF 融合（如果有稀疏结果）
        5. 构建 RetrievalResult

        Args:
            query: RetrievalQuery，包含 hidden_states 和 query_projected

        Returns:
            List[RetrievalResult]，可直接注入 KVStore
        """
        start_time = time.perf_counter()

        try:
            # 检查是否需要刷新索引
            if self.auto_refresh_interval > 0:
                now = time.time()
                if now - self._last_refresh_time > self.auto_refresh_interval:
                    self._refresh_index_sync()

            with self._lock:
                if len(self._index_ids) == 0:
                    return []

                # 1. 提取查询向量
                query_vec = self._extract_query_vector(query)
                if query_vec is None:
                    return []

                top_k = getattr(query, 'top_k', 5)

                # 2. 稠密检索
                dense_results = self._dense_search(query_vec, top_k * 2)

                # 3. 稀疏检索（如果可用）
                sparse_results = []
                if self._bm25_index and self._bm25_index.count > 0:
                    query_text = self._extract_query_text(query)
                    if query_text:
                        sparse_results = self._bm25_index.search(
                            query_text, k=top_k * 2
                        )
                        self._stats["bm25_searches"] += 1

                # 4. 融合
                if sparse_results:
                    from .fusion import reciprocal_rank_fusion
                    fused = reciprocal_rank_fusion(
                        dense_results,
                        sparse_results,
                        dense_weight=1.0 - self._bm25_weight,
                        sparse_weight=self._bm25_weight,
                    )
                    self._stats["fused_searches"] += 1
                else:
                    fused = dense_results

                # 5. 构建 RetrievalResult
                results = self._build_results(fused[:top_k], query)

            # 更新统计
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._stats["retrieve_count"] += 1
            self._stats["total_retrieve_time_ms"] += elapsed_ms
            self._stats["total_results_returned"] += len(results)

            return results

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(
                f"KnowledgeRetriever 检索失败 (Fail-Open): {e}",
                exc_info=True,
            )
            return []

    def on_injection_feedback(
        self,
        result_id: str,
        was_used: bool,
        gate_value: float = 0.0,
    ) -> None:
        """注入反馈回调"""
        if was_used:
            self._stats["feedback_used"] += 1
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(
                        self.manager.increment_hit_count(
                            self.namespace, [result_id]
                        )
                    )
                else:
                    asyncio.run(
                        self.manager.increment_hit_count(
                            self.namespace, [result_id]
                        )
                    )
            except Exception:
                pass
        else:
            self._stats["feedback_unused"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取召回器统计"""
        avg_time = (
            self._stats["total_retrieve_time_ms"]
            / max(1, self._stats["retrieve_count"])
        )
        avg_results = (
            self._stats["total_results_returned"]
            / max(1, self._stats["retrieve_count"])
        )

        stats = {
            "type": "KnowledgeRetriever",
            "namespace": self.namespace,
            "initialized": self._initialized,
            "index_backend": self._index_backend,
            "index_size": len(self._index_ids),
            "alignment": self.alignment.summary(),
            "retrieve_count": self._stats["retrieve_count"],
            "avg_retrieve_time_ms": round(avg_time, 3),
            "avg_results_per_query": round(avg_results, 2),
            "refresh_count": self._stats["refresh_count"],
            "feedback_used": self._stats["feedback_used"],
            "feedback_unused": self._stats["feedback_unused"],
            "errors": self._stats["errors"],
            "hnsw_searches": self._stats["hnsw_searches"],
            "bm25_searches": self._stats["bm25_searches"],
            "brute_searches": self._stats["brute_searches"],
            "fused_searches": self._stats["fused_searches"],
            "encoder": self.encoder.get_stats(),
        }

        if self._hnsw_index:
            stats["hnsw"] = self._hnsw_index.get_stats()
        if self._bm25_index:
            stats["bm25"] = self._bm25_index.get_stats()

        return stats

    def shutdown(self) -> None:
        """释放资源"""
        with self._lock:
            self._index.clear()
            self._key_matrix = None
            self._index_ids.clear()
            self._index_metadata.clear()
            self._hnsw_index = None
            if self._bm25_index:
                self._bm25_index.clear()
                self._bm25_index = None
            self._initialized = False

        self.encoder.shutdown()
        logger.info("KnowledgeRetriever 已关闭")

    # ==================== 索引管理 ====================

    def _refresh_index_sync(self) -> None:
        """
        同步刷新向量索引

        从 KnowledgeManager 加载活跃知识，编码为向量，
        并重建 HNSW + BM25 索引。
        """
        try:
            import asyncio

            # 获取活跃知识
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    future = asyncio.run_coroutine_threadsafe(
                        self.manager.get_knowledge_for_injection(self.namespace),
                        loop,
                    )
                    knowledge_list = future.result(timeout=30)
                else:
                    knowledge_list = asyncio.run(
                        self.manager.get_knowledge_for_injection(self.namespace)
                    )
            except RuntimeError:
                knowledge_list = asyncio.run(
                    self.manager.get_knowledge_for_injection(self.namespace)
                )

            if not knowledge_list:
                logger.info(f"命名空间 {self.namespace} 无活跃知识")
                return

            # 批量编码
            records = [
                {
                    "condition": k.get("condition", ""),
                    "decision": k.get("decision", ""),
                    "lu_id": k.get("lu_id", ""),
                    "reliability": k.get("reliability", 1.0),
                    "metadata": k.get("metadata"),
                }
                for k in knowledge_list
            ]

            encoded_list = self.encoder.encode_batch(records)

            # 重建索引
            with self._lock:
                self._index.clear()
                self._index_ids.clear()
                self._index_metadata.clear()

                key_vectors = []
                value_vectors = []
                conditions = []
                decisions = []

                for encoded in encoded_list:
                    lu_id = encoded.lu_id
                    self._index[lu_id] = encoded
                    self._index_ids.append(lu_id)
                    self._index_metadata[lu_id] = {
                        "condition": encoded.condition,
                        "decision": encoded.decision,
                        "reliability": encoded.reliability,
                        "metadata": encoded.metadata,
                    }
                    key_vectors.append(encoded.key_vector)
                    value_vectors.append(encoded.value_vector)
                    conditions.append(encoded.condition)
                    decisions.append(encoded.decision)

                # 构建暴力搜索矩阵 [N, key_dim]
                if key_vectors:
                    self._key_matrix = torch.tensor(
                        key_vectors, dtype=torch.float32
                    )
                    norms = self._key_matrix.norm(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-10)
                    self._key_matrix = self._key_matrix / norms
                else:
                    self._key_matrix = None

                # 构建 HNSW 索引
                if self._hnsw_index and key_vectors:
                    key_np = np.array(key_vectors, dtype=np.float32)
                    self._hnsw_index.add_batch(self._index_ids.copy(), key_np)

                # 构建 BM25 索引
                if self._bm25_index:
                    self._bm25_index.clear()
                    self._bm25_index.add_batch(
                        self._index_ids.copy(), conditions, decisions
                    )

            self._last_refresh_time = time.time()
            self._stats["refresh_count"] += 1
            self._stats["index_size"] = len(self._index_ids)

            logger.info(
                f"索引已刷新: namespace={self.namespace}, "
                f"size={len(self._index_ids)}, "
                f"hnsw={self._hnsw_index.count if self._hnsw_index else 0}, "
                f"bm25={self._bm25_index.count if self._bm25_index else 0}"
            )

        except Exception as e:
            logger.error(f"刷新向量索引失败: {e}", exc_info=True)

    def refresh_knowledge(self, lu_id: str, record: Dict[str, Any]) -> None:
        """
        增量更新单条知识

        Args:
            lu_id: 知识 ID
            record: 知识记录字典 {condition, decision, reliability, metadata}
        """
        try:
            encoded = self.encoder.encode(
                condition=record.get("condition", ""),
                decision=record.get("decision", ""),
                lu_id=lu_id,
                reliability=record.get("reliability", 1.0),
                metadata=record.get("metadata"),
            )

            with self._lock:
                self._index[lu_id] = encoded

                # 更新 HNSW
                if self._hnsw_index:
                    key_np = np.array(encoded.key_vector, dtype=np.float32)
                    self._hnsw_index.add(lu_id, key_np)

                # 更新 BM25
                if self._bm25_index:
                    self._bm25_index.add(
                        lu_id, encoded.condition, encoded.decision
                    )

                # 重建暴力搜索矩阵
                self._rebuild_matrix()

            logger.debug(f"知识已增量更新: {lu_id}")

        except Exception as e:
            logger.warning(f"增量更新知识失败: {lu_id}: {e}")

    def remove_knowledge(self, lu_id: str) -> None:
        """从索引中移除知识"""
        with self._lock:
            if lu_id in self._index:
                del self._index[lu_id]
                self._index_metadata.pop(lu_id, None)

                # 移除 HNSW
                if self._hnsw_index:
                    self._hnsw_index.remove(lu_id)

                # 移除 BM25
                if self._bm25_index:
                    self._bm25_index.remove(lu_id)

                self._rebuild_matrix()
                logger.debug(f"知识已从索引移除: {lu_id}")

    def _rebuild_matrix(self) -> None:
        """重建暴力搜索矩阵（需在锁内调用）"""
        self._index_ids.clear()
        key_vectors = []

        for lu_id, encoded in self._index.items():
            self._index_ids.append(lu_id)
            key_vectors.append(encoded.key_vector)

        if key_vectors:
            self._key_matrix = torch.tensor(key_vectors, dtype=torch.float32)
            norms = self._key_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            self._key_matrix = self._key_matrix / norms
        else:
            self._key_matrix = None

        self._stats["index_size"] = len(self._index_ids)

    # ==================== 检索核心 ====================

    def _dense_search(
        self,
        query_vec: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        稠密检索 — HNSW 或暴力搜索

        Args:
            query_vec: [key_dim] 查询向量
            top_k: 返回数量

        Returns:
            [(lu_id, similarity)] 列表
        """
        # 优先使用 HNSW
        if self._hnsw_index and self._hnsw_index.count > 0:
            query_np = query_vec.numpy().astype(np.float32)
            results = self._hnsw_index.search(
                query_np, k=top_k, ef_search=self._hnsw_ef_search
            )
            self._stats["hnsw_searches"] += 1
            return results

        # 回退到暴力搜索
        return self._brute_force_search(query_vec, top_k)

    def _brute_force_search(
        self,
        query_vec: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        暴力余弦相似度搜索

        Args:
            query_vec: [key_dim] 查询向量
            top_k: 返回数量

        Returns:
            [(lu_id, similarity)] 列表
        """
        if self._key_matrix is None or len(self._index_ids) == 0:
            return []

        self._stats["brute_searches"] += 1

        # L2 归一化查询向量
        query_norm = query_vec.norm().clamp(min=1e-10)
        query_normalized = query_vec / query_norm

        # 余弦相似度
        scores = torch.mv(self._key_matrix, query_normalized)

        # top-k
        k = min(top_k, len(self._index_ids))
        top_scores, top_indices = scores.topk(k)

        results = []
        for i in range(k):
            idx = top_indices[i].item()
            score = top_scores[i].item()
            if score >= self.similarity_threshold:
                results.append((self._index_ids[idx], score))

        return results

    def _extract_query_vector(self, query) -> Optional[torch.Tensor]:
        """
        从 RetrievalQuery 中提取查询向量

        优先级:
        1. query_projected（已投影到 bottleneck_dim，与 key 空间对齐）
        2. hidden_states（需要降维到 key_dim）
        """
        # 优先使用 query_projected
        query_projected = getattr(query, 'query_projected', None)
        if query_projected is not None and isinstance(query_projected, torch.Tensor):
            if query_projected.dim() == 3:
                vec = query_projected[:, -1, :].mean(dim=0)
            elif query_projected.dim() == 2:
                vec = query_projected[-1, :]
            else:
                vec = query_projected
            return vec.detach().float().cpu()

        # 回退到 hidden_states
        hidden_states = getattr(query, 'hidden_states', None)
        if hidden_states is not None and isinstance(hidden_states, torch.Tensor):
            if hidden_states.dim() == 3:
                vec = hidden_states[:, -1, :].mean(dim=0)
            elif hidden_states.dim() == 2:
                vec = hidden_states[-1, :]
            else:
                vec = hidden_states

            vec = vec.detach().float().cpu()

            # 维度适配
            key_dim = self.encoder.config.key_dim
            if vec.shape[-1] != key_dim:
                if vec.shape[-1] > key_dim:
                    group_size = vec.shape[-1] // key_dim
                    vec = vec[:group_size * key_dim].reshape(
                        key_dim, group_size
                    ).mean(dim=-1)
                else:
                    padded = torch.zeros(key_dim)
                    padded[:vec.shape[-1]] = vec
                    vec = padded
            return vec

        return None

    def _extract_query_text(self, query) -> Optional[str]:
        """
        从 RetrievalQuery 的 metadata 中提取查询文本

        aga-core 在构建 RetrievalQuery 时可以在 metadata 中传递
        当前正在生成的 token 文本（如果可用）。

        回退策略: 如果无文本，BM25 跳过，仅使用稠密检索。
        """
        metadata = getattr(query, 'metadata', None)
        if metadata:
            # 优先使用显式查询文本
            text = metadata.get("query_text")
            if text:
                return text
            # 回退: 使用最近的 token 文本
            text = metadata.get("recent_tokens_text")
            if text:
                return text
        return None

    def _build_results(
        self,
        ranked_ids: List[Tuple[str, float]],
        query,
    ) -> list:
        """
        将排序后的 ID 列表构建为 RetrievalResult

        Args:
            ranked_ids: [(lu_id, score)] 列表
            query: 原始查询

        Returns:
            List[RetrievalResult]
        """
        results = []

        for lu_id, score in ranked_ids:
            if score < self.similarity_threshold:
                continue

            encoded = self._index.get(lu_id)
            if encoded is None:
                continue

            key_tensor = torch.tensor(encoded.key_vector, dtype=torch.float32)
            value_tensor = torch.tensor(
                encoded.value_vector, dtype=torch.float32
            )

            metadata = {
                "condition": encoded.condition,
                "decision": encoded.decision,
                "source": "aga-knowledge",
                "retrieval_score": score,
                **(encoded.metadata or {}),
            }

            if _HAS_AGA_CORE:
                result = RetrievalResult(
                    id=lu_id,
                    key=key_tensor,
                    value=value_tensor,
                    reliability=encoded.reliability,
                    score=score,
                    metadata=metadata,
                )
            else:
                result = {
                    "id": lu_id,
                    "key": key_tensor,
                    "value": value_tensor,
                    "reliability": encoded.reliability,
                    "score": score,
                    "metadata": metadata,
                }

            results.append(result)

        return results

    def __repr__(self) -> str:
        return (
            f"KnowledgeRetriever("
            f"namespace={self.namespace!r}, "
            f"index_size={len(self._index_ids)}, "
            f"backend={self._index_backend!r}, "
            f"alignment={self.alignment.summary()}, "
            f"encoder={self.encoder!r})"
        )
