"""
AGA ANN 索引模块

提供近似最近邻 (ANN) 索引实现，支持大规模知识库的高效检索。

支持的索引类型:
- FAISS: Facebook AI Similarity Search
- HNSW: Hierarchical Navigable Small World

版本: v1.2
更新: 
- 添加配置校验 (__post_init__)
- 增强异常处理和 Fail-Open 机制
- 添加异步重建支持
- 向量存储用于 rebuild
- HNSW 实际重建支持
- 集成统一监控指标 (v1.2)
"""

import time
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set, Callable
from enum import Enum

import numpy as np

# 监控模块导入
try:
    from ..monitoring import (
        get_metrics_registry,
        track_ann_search,
        record_ann_index_metrics,
    )
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    get_metrics_registry = None
    track_ann_search = None
    record_ann_index_metrics = None

logger = logging.getLogger(__name__)


class IndexBackend(str, Enum):
    """索引后端类型"""
    FAISS = "faiss"
    HNSW = "hnsw"
    FLAT = "flat"  # 精确搜索，用于小规模或测试


@dataclass
class ANNIndexConfig:
    """ANN 索引配置"""
    # 基础配置
    enabled: bool = False  # 默认关闭，向后兼容
    backend: IndexBackend = IndexBackend.FAISS
    
    # 索引类型 (FAISS 专用)
    index_type: str = "IVF4096,PQ64"  # IVF + PQ 组合
    
    # 容量配置
    index_capacity: int = 1_000_000  # 索引容量上限
    
    # 检索配置
    retrieval_top_k: int = 200  # ANN 返回候选数
    nprobe: int = 64  # IVF 探测数（精度 vs 延迟）
    ef_search: int = 128  # HNSW 搜索参数
    
    # 性能配置
    use_gpu: bool = True  # 使用 GPU 加速
    gpu_device_id: int = 0  # GPU 设备 ID
    search_timeout_ms: float = 10.0  # 检索超时
    
    # 更新配置
    incremental_update: bool = True  # 增量更新
    rebuild_interval_hours: int = 24  # 全量重建间隔
    rebuild_threshold: float = 0.3  # 删除比例超过此值触发重建
    async_rebuild: bool = True  # 异步重建（避免阻塞）
    
    # HNSW 专用配置
    hnsw_m: int = 32  # 连接数
    hnsw_ef_construction: int = 200  # 构建时的 ef 参数
    
    # 训练配置 (IVF 需要)
    train_sample_size: int = 100_000  # 训练样本数
    auto_train: bool = True  # 自动训练
    
    # 向量存储配置
    store_vectors: bool = True  # 存储向量副本用于 rebuild
    max_stored_vectors: int = 500_000  # 最大存储向量数（防止 OOM）
    
    def __post_init__(self):
        """配置校验"""
        # 容量校验
        if self.index_capacity <= 0:
            raise ValueError("index_capacity must be positive")
        
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        
        if self.nprobe <= 0:
            raise ValueError("nprobe must be positive")
        
        # FAISS 默认索引类型
        if self.backend == IndexBackend.FAISS and not self.index_type:
            self.index_type = "IVF4096,PQ64"
        
        # GPU 仅支持 FAISS
        if self.use_gpu and self.backend != IndexBackend.FAISS:
            logger.warning("GPU only supported for FAISS backend, disabling GPU")
            self.use_gpu = False
        
        # 训练样本数校验
        if self.train_sample_size <= 0:
            self.train_sample_size = 100_000
        
        # 重建阈值校验
        if not 0 < self.rebuild_threshold < 1:
            self.rebuild_threshold = 0.3


class BaseANNIndex(ABC):
    """ANN 索引抽象基类"""
    
    @abstractmethod
    def add(self, lu_id: str, key_vector: np.ndarray) -> bool:
        """添加向量到索引"""
        pass
    
    @abstractmethod
    def add_batch(self, lu_ids: List[str], key_vectors: np.ndarray) -> int:
        """批量添加向量"""
        pass
    
    @abstractmethod
    def remove(self, lu_id: str) -> bool:
        """从索引移除向量（标记删除）"""
        pass
    
    @abstractmethod
    def search(
        self, 
        query: np.ndarray, 
        top_k: int,
        filter_lu_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[str], List[float]]:
        """检索 Top-k 候选"""
        pass
    
    @abstractmethod
    def rebuild(self) -> bool:
        """重建索引（清理已删除向量）"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """关闭索引，清理资源"""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """当前索引大小"""
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """索引是否已训练"""
        pass


class FAISSIndex(BaseANNIndex):
    """
    FAISS 索引实现
    
    支持多种索引类型:
    - Flat: 精确搜索，适合小规模
    - IVF: 倒排索引，适合中等规模
    - IVF+PQ: 倒排 + 量化，适合大规模
    - HNSW: 图索引，高召回率
    
    特性:
    - 线程安全
    - Fail-Open 机制
    - 异步重建支持
    - 向量存储用于 rebuild
    """
    
    def __init__(self, config: ANNIndexConfig, dim: int):
        self.config = config
        self.dim = dim
        
        # 延迟导入 FAISS
        self._faiss = None
        self._index = None
        self._gpu_resources = None
        
        # ID 映射
        self._id_to_faiss: Dict[str, int] = {}
        self._faiss_to_id: Dict[int, str] = {}
        self._deleted_ids: Set[str] = set()  # 标记删除的 ID
        self._next_id = 0
        
        # 向量存储（用于 rebuild）
        self._vectors: Dict[str, np.ndarray] = {} if config.store_vectors else None
        
        # 训练数据缓存
        self._training_vectors: List[np.ndarray] = []
        self._is_trained = False
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 异步重建
        self._rebuild_thread: Optional[threading.Thread] = None
        self._stop_rebuild = threading.Event()
        self._rebuild_scheduled = False
        
        # 统计
        self._stats = {
            "total_vectors": 0,
            "deleted_vectors": 0,
            "total_searches": 0,
            "total_search_time_ms": 0.0,
            "avg_search_time_ms": 0.0,
            "last_rebuild_time": 0.0,
            "search_errors": 0,
            "search_timeouts": 0,
        }
        
        # 初始化索引
        self._initialize_index()
        
        # 启动异步重建监控
        if config.async_rebuild:
            self._start_rebuild_monitor()
    
    def _initialize_index(self):
        """初始化 FAISS 索引"""
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            logger.warning("FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
            return
        
        # 创建索引
        index_type = self.config.index_type
        
        if index_type.upper() == "FLAT" or self.config.backend == IndexBackend.FLAT:
            # 精确搜索
            self._index = faiss.IndexFlatIP(self.dim)  # 内积
            self._is_trained = True
        else:
            # 需要训练的索引
            self._index = faiss.index_factory(self.dim, index_type, faiss.METRIC_INNER_PRODUCT)
        
        # GPU 加速
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            try:
                self._gpu_resources = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(
                    self._gpu_resources, 
                    self.config.gpu_device_id, 
                    self._index
                )
                logger.info(f"FAISS index moved to GPU {self.config.gpu_device_id}")
            except Exception as e:
                logger.warning(f"Failed to move FAISS index to GPU: {e}")
        
        logger.info(f"Initialized FAISS index: type={index_type}, dim={self.dim}")
    
    def _start_rebuild_monitor(self):
        """启动异步重建监控线程"""
        def rebuild_loop():
            interval = self.config.rebuild_interval_hours * 3600
            while not self._stop_rebuild.wait(interval):
                if self._rebuild_scheduled or self._should_rebuild():
                    try:
                        logger.info("Starting scheduled index rebuild...")
                        self._do_rebuild()
                        self._rebuild_scheduled = False
                    except Exception as e:
                        logger.error(f"Scheduled rebuild failed: {e}")
        
        self._rebuild_thread = threading.Thread(
            target=rebuild_loop,
            daemon=True,
            name="FAISSIndex-RebuildMonitor"
        )
        self._rebuild_thread.start()
        logger.debug("FAISS rebuild monitor started")
    
    def _should_rebuild(self) -> bool:
        """检查是否需要重建"""
        if self._stats["total_vectors"] == 0:
            return False
        delete_ratio = self._stats["deleted_vectors"] / self._stats["total_vectors"]
        return delete_ratio >= self.config.rebuild_threshold
    
    def _ensure_trained(self, vectors: Optional[np.ndarray] = None):
        """确保索引已训练"""
        if self._is_trained or self._index is None:
            return
        
        if not self.config.auto_train:
            return
        
        # 收集训练数据
        if vectors is not None:
            self._training_vectors.append(vectors)
        
        # 检查是否有足够的训练数据
        total_vectors = sum(v.shape[0] for v in self._training_vectors)
        if total_vectors < self.config.train_sample_size:
            return
        
        # 合并训练数据
        train_data = np.vstack(self._training_vectors)[:self.config.train_sample_size]
        train_data = train_data.astype(np.float32)
        
        # 训练索引
        logger.info(f"Training FAISS index with {train_data.shape[0]} vectors...")
        self._index.train(train_data)
        self._is_trained = True
        
        # 清理训练数据
        self._training_vectors.clear()
        logger.info("FAISS index training completed")
    
    def add(self, lu_id: str, key_vector: np.ndarray) -> bool:
        """添加向量"""
        if self._index is None:
            return False
        
        # 向量维度校验
        if key_vector.shape[-1] != self.dim:
            logger.warning(f"Vector dimension mismatch: expected {self.dim}, got {key_vector.shape[-1]}")
            return False
        
        with self._lock:
            try:
                # 如果已存在，先标记删除
                if lu_id in self._id_to_faiss:
                    self._deleted_ids.add(lu_id)
                    self._stats["deleted_vectors"] += 1
                
                # 准备向量
                vector = key_vector.reshape(1, -1).astype(np.float32)
                
                # 存储向量副本（用于 rebuild）
                if self._vectors is not None:
                    # 限制存储数量防止 OOM
                    if len(self._vectors) < self.config.max_stored_vectors:
                        self._vectors[lu_id] = key_vector.copy()
                    elif lu_id in self._vectors:
                        # 更新已存在的
                        self._vectors[lu_id] = key_vector.copy()
                
                # 确保已训练（_ensure_trained 内部已将 vector 加入训练数据）
                self._ensure_trained(vector)
                
                if not self._is_trained:
                    # 训练数据不足，向量已在 _ensure_trained 中缓存，等待训练
                    return False
                
                # 分配 ID
                faiss_id = self._next_id
                self._next_id += 1
                
                # 添加到索引
                self._index.add(vector)
                
                # 更新映射
                self._id_to_faiss[lu_id] = faiss_id
                self._faiss_to_id[faiss_id] = lu_id
                self._stats["total_vectors"] += 1
                
                # 检查是否需要重建
                self._check_rebuild_needed()
                
                return True
            except Exception as e:
                logger.error(f"Failed to add vector {lu_id}: {e}")
                return False
    
    def add_batch(self, lu_ids: List[str], key_vectors: np.ndarray) -> int:
        """批量添加向量"""
        if self._index is None or len(lu_ids) == 0:
            return 0
        
        with self._lock:
            try:
                # 准备向量
                vectors = key_vectors.astype(np.float32)
                if vectors.ndim == 1:
                    vectors = vectors.reshape(1, -1)
                
                # 维度校验
                if vectors.shape[-1] != self.dim:
                    logger.warning(f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[-1]}")
                    return 0
                
                # 存储向量副本
                if self._vectors is not None:
                    for i, lu_id in enumerate(lu_ids):
                        if len(self._vectors) < self.config.max_stored_vectors or lu_id in self._vectors:
                            self._vectors[lu_id] = key_vectors[i].copy()
                
                # 确保已训练（_ensure_trained 内部已将 vectors 加入训练数据）
                self._ensure_trained(vectors)
                
                if not self._is_trained:
                    # 训练数据不足，向量已在 _ensure_trained 中缓存，等待训练
                    return 0
                
                # 批量添加
                added = 0
                for i, lu_id in enumerate(lu_ids):
                    # 标记删除旧的
                    if lu_id in self._id_to_faiss:
                        self._deleted_ids.add(lu_id)
                        self._stats["deleted_vectors"] += 1
                    
                    faiss_id = self._next_id
                    self._next_id += 1
                    
                    self._id_to_faiss[lu_id] = faiss_id
                    self._faiss_to_id[faiss_id] = lu_id
                    added += 1
                
                # 批量添加到索引
                self._index.add(vectors)
                self._stats["total_vectors"] += added
                
                self._check_rebuild_needed()
                return added
            except Exception as e:
                logger.error(f"Failed to add batch: {e}")
                return 0
    
    def remove(self, lu_id: str) -> bool:
        """标记删除向量"""
        with self._lock:
            if lu_id not in self._id_to_faiss:
                return False
            
            self._deleted_ids.add(lu_id)
            self._stats["deleted_vectors"] += 1
            
            # 从向量存储中移除
            if self._vectors is not None and lu_id in self._vectors:
                del self._vectors[lu_id]
            
            self._check_rebuild_needed()
            return True
    
    def search(
        self, 
        query: np.ndarray, 
        top_k: int,
        filter_lu_ids: Optional[Set[str]] = None,
        namespace: str = "default",
    ) -> Tuple[List[str], List[float]]:
        """
        检索 Top-k
        
        Fail-Open: 出错时返回空结果而非抛出异常
        
        Args:
            query: 查询向量
            top_k: 返回候选数
            filter_lu_ids: 可选的 lu_id 过滤集合
            namespace: 命名空间（用于指标）
        """
        if self._index is None or not self._is_trained:
            return [], []
        
        start = time.perf_counter()
        status = "success"
        candidates_count = 0
        
        try:
            with self._lock:
                # 超时检查
                timeout_ms = self.config.search_timeout_ms
                
                # 设置检索参数
                if hasattr(self._index, 'nprobe'):
                    self._index.nprobe = self.config.nprobe
                
                # 准备查询向量
                query_vec = query.reshape(1, -1).astype(np.float32)
                
                # 维度校验
                if query_vec.shape[-1] != self.dim:
                    logger.warning(f"Query dimension mismatch: expected {self.dim}, got {query_vec.shape[-1]}")
                    return [], []
                
                # 检索更多候选以补偿已删除的
                search_k = min(top_k * 2, self._index.ntotal)
                if search_k == 0:
                    return [], []
                
                # 执行检索
                distances, indices = self._index.search(query_vec, search_k)
                
                # 检查是否超时
                elapsed = (time.perf_counter() - start) * 1000
                if elapsed > timeout_ms:
                    self._stats["search_timeouts"] += 1
                    logger.warning(f"Search exceeded timeout: {elapsed:.1f}ms > {timeout_ms}ms")
                    # 继续返回已有结果（Fail-Open）
                
                # 过滤结果
                lu_ids = []
                scores = []
                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx not in self._faiss_to_id:
                        continue
                    
                    lu_id = self._faiss_to_id[idx]
                    
                    # 跳过已删除的
                    if lu_id in self._deleted_ids:
                        continue
                    
                    # 应用过滤器
                    if filter_lu_ids is not None and lu_id not in filter_lu_ids:
                        continue
                    
                    lu_ids.append(lu_id)
                    scores.append(float(distances[0][i]))
                    
                    if len(lu_ids) >= top_k:
                        break
                
                candidates_count = len(lu_ids)
                
                # 更新统计
                elapsed = (time.perf_counter() - start) * 1000
                self._stats["total_searches"] += 1
                self._stats["total_search_time_ms"] += elapsed
                self._stats["avg_search_time_ms"] = (
                    self._stats["total_search_time_ms"] / self._stats["total_searches"]
                )
                
                return lu_ids, scores
                
        except Exception as e:
            # Fail-Open: 记录错误但返回空结果
            status = "error"
            self._stats["search_errors"] += 1
            logger.error(f"Search error (Fail-Open): {e}")
            return [], []
        
        finally:
            # 记录监控指标
            if HAS_MONITORING:
                elapsed_sec = time.perf_counter() - start
                registry = get_metrics_registry()
                if registry and registry.enabled:
                    registry.get_metric("ann_search_total").labels(
                        namespace=namespace,
                        status=status
                    ).inc()
                    registry.get_metric("ann_search_duration_seconds").labels(
                        namespace=namespace
                    ).observe(elapsed_sec)
                    if candidates_count > 0:
                        registry.get_metric("ann_candidates_returned").labels(
                            namespace=namespace
                        ).observe(candidates_count)
    
    def _check_rebuild_needed(self):
        """检查是否需要重建"""
        if self._stats["total_vectors"] == 0:
            return
        
        delete_ratio = self._stats["deleted_vectors"] / self._stats["total_vectors"]
        if delete_ratio >= self.config.rebuild_threshold:
            logger.info(f"Delete ratio {delete_ratio:.2%} exceeds threshold, scheduling rebuild")
            self._rebuild_scheduled = True
    
    def rebuild(self) -> bool:
        """
        重建索引
        
        如果配置了 store_vectors，将使用存储的向量自动重建。
        否则需要调用 rebuild_from_vectors 提供向量数据。
        """
        if self._index is None:
            return False
        
        # 如果有存储的向量，使用它们重建
        if self._vectors is not None and len(self._vectors) > 0:
            valid_lu_ids = [
                lu_id for lu_id in self._vectors.keys()
                if lu_id not in self._deleted_ids
            ]
            if valid_lu_ids:
                valid_vectors = np.array([self._vectors[lu_id] for lu_id in valid_lu_ids])
                return self.rebuild_from_vectors(valid_lu_ids, valid_vectors)
        
        # 没有存储向量，只清理状态
        return self._do_rebuild()
    
    def _do_rebuild(self) -> bool:
        """
        执行重建（内部方法）
        
        仅在没有存储向量副本时调用。
        由于无法重建实际索引，只记录警告。
        不能简单清除 _deleted_ids，否则已删除向量会在搜索中"复活"。
        """
        with self._lock:
            if self._deleted_ids:
                logger.warning(
                    f"Cannot rebuild FAISS index: no stored vectors available "
                    f"(store_vectors=False). {len(self._deleted_ids)} deleted "
                    f"vectors remain as soft-deleted. Enable store_vectors for "
                    f"full rebuild support."
                )
            
            self._stats["last_rebuild_time"] = time.time()
            return False  # 无法实际重建
    
    def rebuild_from_vectors(
        self, 
        lu_ids: List[str], 
        vectors: np.ndarray,
        namespace: str = "default",
        trigger: str = "manual",
    ) -> bool:
        """从向量数据重建索引"""
        if self._index is None:
            return False
        
        start = time.perf_counter()
        
        with self._lock:
            logger.info(f"Rebuilding FAISS index from {len(lu_ids)} vectors...")
            
            # 重置索引
            self._index.reset()
            self._id_to_faiss.clear()
            self._faiss_to_id.clear()
            self._deleted_ids.clear()
            self._next_id = 0
            
            # 重置统计计数器（add_batch 会重新累加）
            self._stats["total_vectors"] = 0
            self._stats["deleted_vectors"] = 0
            
            # 重新训练（如果需要）
            if not isinstance(self._index, self._faiss.IndexFlat):
                vectors_f32 = vectors.astype(np.float32)
                self._index.train(vectors_f32)
            
            # 批量添加
            self.add_batch(lu_ids, vectors)
            
            self._stats["last_rebuild_time"] = time.time()
            
            # 记录监控指标
            if HAS_MONITORING:
                elapsed_sec = time.perf_counter() - start
                registry = get_metrics_registry()
                if registry and registry.enabled:
                    registry.get_metric("ann_rebuild_total").labels(
                        namespace=namespace,
                        trigger=trigger
                    ).inc()
                    registry.get_metric("ann_rebuild_duration_seconds").labels(
                        namespace=namespace
                    ).observe(elapsed_sec)
            
            logger.info("FAISS index rebuild completed")
            return True
    
    def get_statistics(self, namespace: str = "default") -> Dict[str, Any]:
        """获取统计信息并更新监控指标"""
        with self._lock:
            total = self._stats["total_vectors"]
            deleted = self._stats["deleted_vectors"]
            active = total - deleted
            deleted_ratio = deleted / total if total > 0 else 0.0
            
            # 更新监控指标
            if HAS_MONITORING:
                record_ann_index_metrics(
                    namespace=namespace,
                    backend=self.config.backend.value,
                    total_vectors=total,
                    active_vectors=active,
                    deleted_ratio=deleted_ratio,
                )
            
            return {
                **self._stats,
                "index_type": self.config.index_type,
                "backend": self.config.backend.value,
                "use_gpu": self.config.use_gpu,
                "dim": self.dim,
                "is_trained": self._is_trained,
                "active_vectors": active,
                "deleted_ratio": deleted_ratio,
                "stored_vectors": len(self._vectors) if self._vectors else 0,
                "rebuild_scheduled": self._rebuild_scheduled,
            }
    
    def shutdown(self):
        """关闭索引，清理资源"""
        logger.info("Shutting down FAISS index...")
        
        # 停止重建线程
        self._stop_rebuild.set()
        if self._rebuild_thread and self._rebuild_thread.is_alive():
            self._rebuild_thread.join(timeout=5.0)
        
        # 清理 GPU 资源
        if self._gpu_resources is not None:
            self._gpu_resources = None
        
        logger.info("FAISS index shutdown complete")
    
    @property
    def size(self) -> int:
        """当前索引大小（不含已删除）"""
        return self._stats["total_vectors"] - self._stats["deleted_vectors"]
    
    @property
    def is_trained(self) -> bool:
        """索引是否已训练"""
        return self._is_trained


class HNSWIndex(BaseANNIndex):
    """
    HNSW 索引实现
    
    基于 hnswlib 库，提供高召回率的近似最近邻搜索。
    
    特性:
    - 线程安全
    - Fail-Open 机制
    - 实际重建支持（HNSW 不支持高效移除）
    """
    
    def __init__(self, config: ANNIndexConfig, dim: int):
        self.config = config
        self.dim = dim
        
        # 延迟导入
        self._hnswlib = None
        self._index = None
        
        # ID 映射
        self._id_to_hnsw: Dict[str, int] = {}
        self._hnsw_to_id: Dict[int, str] = {}
        self._deleted_ids: Set[str] = set()
        self._next_id = 0
        
        # 向量存储（HNSW 重建必需）
        self._vectors: Dict[str, np.ndarray] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 统计
        self._stats = {
            "total_vectors": 0,
            "deleted_vectors": 0,
            "total_searches": 0,
            "avg_search_time_ms": 0.0,
            "search_errors": 0,
            "rebuilds": 0,
        }
        
        self._initialize_index()
    
    def _initialize_index(self):
        """初始化 HNSW 索引"""
        try:
            import hnswlib
            self._hnswlib = hnswlib
        except ImportError:
            logger.warning("hnswlib not installed. Install with: pip install hnswlib")
            return
        
        # 创建索引
        self._index = hnswlib.Index(space='ip', dim=self.dim)  # 内积
        self._index.init_index(
            max_elements=self.config.index_capacity,
            ef_construction=self.config.hnsw_ef_construction,
            M=self.config.hnsw_m,
        )
        self._index.set_ef(self.config.ef_search)
        
        logger.info(f"Initialized HNSW index: dim={self.dim}, M={self.config.hnsw_m}")
    
    def add(self, lu_id: str, key_vector: np.ndarray) -> bool:
        """添加向量"""
        if self._index is None:
            return False
        
        # 维度校验
        if key_vector.shape[-1] != self.dim:
            logger.warning(f"Vector dimension mismatch: expected {self.dim}, got {key_vector.shape[-1]}")
            return False
        
        with self._lock:
            try:
                # 标记删除旧的
                if lu_id in self._id_to_hnsw:
                    self._deleted_ids.add(lu_id)
                    self._stats["deleted_vectors"] += 1
                
                # 存储向量（HNSW 重建必需）
                self._vectors[lu_id] = key_vector.copy()
                
                # 分配 ID
                hnsw_id = self._next_id
                self._next_id += 1
                
                # 添加到索引
                vector = key_vector.reshape(1, -1).astype(np.float32)
                self._index.add_items(vector, np.array([hnsw_id]))
                
                # 更新映射
                self._id_to_hnsw[lu_id] = hnsw_id
                self._hnsw_to_id[hnsw_id] = lu_id
                self._stats["total_vectors"] += 1
                
                return True
            except Exception as e:
                logger.error(f"Failed to add vector {lu_id}: {e}")
                return False
    
    def add_batch(self, lu_ids: List[str], key_vectors: np.ndarray) -> int:
        """批量添加向量"""
        if self._index is None or len(lu_ids) == 0:
            return 0
        
        with self._lock:
            try:
                vectors = key_vectors.astype(np.float32)
                
                # 维度校验
                if vectors.shape[-1] != self.dim:
                    logger.warning(f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[-1]}")
                    return 0
                
                hnsw_ids = []
                
                for i, lu_id in enumerate(lu_ids):
                    if lu_id in self._id_to_hnsw:
                        self._deleted_ids.add(lu_id)
                        self._stats["deleted_vectors"] += 1
                    
                    # 存储向量
                    self._vectors[lu_id] = key_vectors[i].copy()
                    
                    hnsw_id = self._next_id
                    self._next_id += 1
                    
                    self._id_to_hnsw[lu_id] = hnsw_id
                    self._hnsw_to_id[hnsw_id] = lu_id
                    hnsw_ids.append(hnsw_id)
                
                self._index.add_items(vectors, np.array(hnsw_ids))
                self._stats["total_vectors"] += len(lu_ids)
                
                return len(lu_ids)
            except Exception as e:
                logger.error(f"Failed to add batch: {e}")
                return 0
    
    def remove(self, lu_id: str) -> bool:
        """标记删除"""
        with self._lock:
            if lu_id not in self._id_to_hnsw:
                return False
            
            self._deleted_ids.add(lu_id)
            self._stats["deleted_vectors"] += 1
            
            # 从向量存储中移除
            if lu_id in self._vectors:
                del self._vectors[lu_id]
            
            return True
    
    def search(
        self, 
        query: np.ndarray, 
        top_k: int,
        filter_lu_ids: Optional[Set[str]] = None,
        namespace: str = "default",
    ) -> Tuple[List[str], List[float]]:
        """
        检索 Top-k
        
        Fail-Open: 出错时返回空结果而非抛出异常
        
        Args:
            query: 查询向量
            top_k: 返回候选数
            filter_lu_ids: 可选的 lu_id 过滤集合
            namespace: 命名空间（用于指标）
        """
        if self._index is None:
            return [], []
        
        start = time.perf_counter()
        status = "success"
        candidates_count = 0
        
        try:
            with self._lock:
                query_vec = query.reshape(1, -1).astype(np.float32)
                
                # 维度校验
                if query_vec.shape[-1] != self.dim:
                    logger.warning(f"Query dimension mismatch: expected {self.dim}, got {query_vec.shape[-1]}")
                    return [], []
                
                # 检索更多以补偿删除
                search_k = min(top_k * 2, self._index.get_current_count())
                if search_k == 0:
                    return [], []
                
                labels, distances = self._index.knn_query(query_vec, k=search_k)
                
                # 过滤结果
                lu_ids = []
                scores = []
                for i, hnsw_id in enumerate(labels[0]):
                    if hnsw_id not in self._hnsw_to_id:
                        continue
                    
                    lu_id = self._hnsw_to_id[hnsw_id]
                    
                    if lu_id in self._deleted_ids:
                        continue
                    
                    if filter_lu_ids is not None and lu_id not in filter_lu_ids:
                        continue
                    
                    lu_ids.append(lu_id)
                    scores.append(float(distances[0][i]))
                    
                    if len(lu_ids) >= top_k:
                        break
                
                candidates_count = len(lu_ids)
                
                # 更新统计
                elapsed = (time.perf_counter() - start) * 1000
                self._stats["total_searches"] += 1
                self._stats["avg_search_time_ms"] = (
                    self._stats["avg_search_time_ms"] * 0.9 + elapsed * 0.1
                )
                
                return lu_ids, scores
                
        except Exception as e:
            # Fail-Open: 记录错误但返回空结果
            status = "error"
            self._stats["search_errors"] += 1
            logger.error(f"HNSW search error (Fail-Open): {e}")
            return [], []
        
        finally:
            # 记录监控指标
            if HAS_MONITORING:
                elapsed_sec = time.perf_counter() - start
                registry = get_metrics_registry()
                if registry and registry.enabled:
                    registry.get_metric("ann_search_total").labels(
                        namespace=namespace,
                        status=status
                    ).inc()
                    registry.get_metric("ann_search_duration_seconds").labels(
                        namespace=namespace
                    ).observe(elapsed_sec)
                    if candidates_count > 0:
                        registry.get_metric("ann_candidates_returned").labels(
                            namespace=namespace
                        ).observe(candidates_count)
    
    def rebuild(self) -> bool:
        """
        重建索引
        
        HNSW 不支持高效移除，需要完全重建索引。
        使用存储的向量数据重新构建。
        """
        with self._lock:
            try:
                logger.info("Rebuilding HNSW index...")
                
                # 收集有效向量
                valid_lu_ids = [
                    lu_id for lu_id in self._vectors.keys()
                    if lu_id not in self._deleted_ids
                ]
                
                if not valid_lu_ids:
                    # 没有有效向量，只清理状态
                    self._deleted_ids.clear()
                    self._stats["deleted_vectors"] = 0
                    return True
                
                valid_vectors = np.array([self._vectors[lu_id] for lu_id in valid_lu_ids])
                
                # 重新初始化索引
                self._initialize_index()
                
                # 重置映射
                self._id_to_hnsw.clear()
                self._hnsw_to_id.clear()
                self._deleted_ids.clear()
                self._next_id = 0
                
                # 重置统计计数器（add_batch 会重新累加）
                self._stats["total_vectors"] = 0
                self._stats["deleted_vectors"] = 0
                
                # 重新添加有效向量
                self.add_batch(valid_lu_ids, valid_vectors)
                
                self._stats["rebuilds"] += 1
                
                logger.info(f"HNSW index rebuilt with {len(valid_lu_ids)} vectors")
                return True
                
            except Exception as e:
                logger.error(f"HNSW rebuild failed: {e}")
                return False
    
    def get_statistics(self, namespace: str = "default") -> Dict[str, Any]:
        """获取统计信息并更新监控指标"""
        with self._lock:
            total = self._stats["total_vectors"]
            deleted = self._stats["deleted_vectors"]
            active = total - deleted
            deleted_ratio = deleted / total if total > 0 else 0.0
            
            # 更新监控指标
            if HAS_MONITORING:
                record_ann_index_metrics(
                    namespace=namespace,
                    backend="hnsw",
                    total_vectors=total,
                    active_vectors=active,
                    deleted_ratio=deleted_ratio,
                )
            
            return {
                **self._stats,
                "backend": "hnsw",
                "dim": self.dim,
                "M": self.config.hnsw_m,
                "ef_search": self.config.ef_search,
                "active_vectors": active,
                "deleted_ratio": deleted_ratio,
                "stored_vectors": len(self._vectors),
            }
    
    def shutdown(self):
        """关闭索引，清理资源"""
        logger.info("Shutting down HNSW index...")
        with self._lock:
            self._vectors.clear()
            self._id_to_hnsw.clear()
            self._hnsw_to_id.clear()
        logger.info("HNSW index shutdown complete")
    
    @property
    def size(self) -> int:
        return self._stats["total_vectors"] - self._stats["deleted_vectors"]
    
    @property
    def is_trained(self) -> bool:
        return True  # HNSW 不需要训练


def create_ann_index(config: ANNIndexConfig, dim: int) -> Optional[BaseANNIndex]:
    """
    工厂函数：创建 ANN 索引
    
    Args:
        config: ANN 索引配置
        dim: 向量维度
        
    Returns:
        ANN 索引实例，如果创建失败返回 None
    """
    try:
        if config.backend == IndexBackend.HNSW:
            return HNSWIndex(config, dim)
        else:
            return FAISSIndex(config, dim)
    except ImportError as e:
        logger.error(f"Failed to create ANN index: {e}. "
                    f"Install required library: pip install faiss-cpu or pip install hnswlib")
        return None
    except Exception as e:
        logger.error(f"Failed to create ANN index: {e}")
        return None
