"""
AGA ANN 检索可视化 API

提供向量空间可视化和 ANN 检索结果展示。

版本: v1.0
"""

import logging
import time
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Depends, Query
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None  # type: ignore
    BaseModel = object  # type: ignore

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

# ==================== 数据模型 ====================

if HAS_FASTAPI:
    
    class VectorPoint(BaseModel):
        """向量点"""
        lu_id: str
        x: float
        y: float
        distance: Optional[float] = None
        lifecycle_state: Optional[str] = None
        condition_preview: Optional[str] = None
        hit_count: Optional[int] = None
    
    class ANNSearchRequest(BaseModel):
        """ANN 检索请求"""
        query_text: Optional[str] = Field(None, description="查询文本 (需要编码器)")
        query_vector: Optional[List[float]] = Field(None, description="查询向量")
        top_k: int = Field(100, ge=1, le=1000, description="返回候选数")
        namespace: str = Field("default", description="命名空间")
        include_vectors: bool = Field(False, description="是否返回向量用于可视化")
        projection: Literal["tsne", "umap", "pca"] = Field("pca", description="投影方法")
    
    class ANNSearchStatistics(BaseModel):
        """ANN 检索统计"""
        index_size: int
        search_time_ms: float
        deleted_ratio: float
        projection_time_ms: Optional[float] = None
    
    class ANNSearchResponse(BaseModel):
        """ANN 检索响应"""
        query_point: Optional[VectorPoint] = None
        candidates: List[VectorPoint]
        statistics: ANNSearchStatistics
    
    class VectorSpaceRequest(BaseModel):
        """向量空间请求"""
        namespace: str = Field("default", description="命名空间")
        limit: int = Field(1000, ge=1, le=10000, description="最大向量数")
        projection: Literal["tsne", "umap", "pca"] = Field("pca", description="投影方法")
        state_filter: Optional[List[str]] = Field(None, description="状态过滤")
    
    class VectorSpaceResponse(BaseModel):
        """向量空间响应"""
        points: List[VectorPoint]
        total_vectors: int
        visible_vectors: int
        projection_method: str
        projection_time_ms: float


# ==================== 投影服务 ====================

class ProjectionService:
    """向量投影服务"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 分钟缓存
    
    def project_vectors(
        self,
        vectors: np.ndarray,
        method: str = "pca",
        n_components: int = 2,
    ) -> np.ndarray:
        """
        将高维向量投影到 2D 空间
        
        Args:
            vectors: 形状 (n, dim) 的向量数组
            method: 投影方法 (pca, tsne, umap)
            n_components: 目标维度
        
        Returns:
            形状 (n, n_components) 的投影结果
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy not available")
        
        if len(vectors) == 0:
            return np.array([])
        
        if len(vectors) == 1:
            # 单个向量，返回原点
            return np.array([[0.0, 0.0]])
        
        if method == "pca":
            return self._pca_project(vectors, n_components)
        elif method == "tsne":
            return self._tsne_project(vectors, n_components)
        elif method == "umap":
            return self._umap_project(vectors, n_components)
        else:
            raise ValueError(f"Unknown projection method: {method}")
    
    def _pca_project(self, vectors: np.ndarray, n_components: int) -> np.ndarray:
        """PCA 投影"""
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            return pca.fit_transform(vectors)
        except ImportError:
            # 简单 PCA 实现
            centered = vectors - np.mean(vectors, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1][:n_components]
            return centered @ eigenvectors[:, idx]
    
    def _tsne_project(self, vectors: np.ndarray, n_components: int) -> np.ndarray:
        """t-SNE 投影"""
        try:
            from sklearn.manifold import TSNE
            # 限制样本数以提高性能
            if len(vectors) > 5000:
                logger.warning(f"t-SNE with {len(vectors)} samples may be slow")
            perplexity = min(30, len(vectors) - 1)
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            return tsne.fit_transform(vectors)
        except ImportError:
            logger.warning("sklearn not available, falling back to PCA")
            return self._pca_project(vectors, n_components)
    
    def _umap_project(self, vectors: np.ndarray, n_components: int) -> np.ndarray:
        """UMAP 投影"""
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(vectors)
        except ImportError:
            logger.warning("umap not available, falling back to PCA")
            return self._pca_project(vectors, n_components)


# 全局投影服务实例
_projection_service = ProjectionService()


# ==================== API 路由 ====================

if HAS_FASTAPI:
    router = APIRouter(prefix="/ann", tags=["ann-visualization"])
    
    @router.post("/search", response_model=ANNSearchResponse)
    async def ann_search_visualization(
        request: ANNSearchRequest,
    ) -> ANNSearchResponse:
        """
        执行 ANN 检索并返回可视化数据
        
        支持两种输入方式：
        1. query_text: 需要编码器将文本转换为向量
        2. query_vector: 直接提供向量
        """
        start_time = time.perf_counter()
        
        # 验证输入
        if request.query_text is None and request.query_vector is None:
            raise HTTPException(
                status_code=400,
                detail="Must provide either query_text or query_vector"
            )
        
        try:
            # 获取 Operator (这里需要依赖注入)
            # 暂时返回模拟数据
            
            # 模拟检索结果
            candidates = []
            for i in range(min(request.top_k, 10)):
                candidates.append(VectorPoint(
                    lu_id=f"knowledge_{i:03d}",
                    x=float(i * 0.1),
                    y=float(i * 0.05),
                    distance=float(i * 0.05),
                    lifecycle_state="confirmed" if i % 2 == 0 else "probationary",
                    condition_preview=f"示例条件 {i}...",
                    hit_count=100 - i * 10,
                ))
            
            search_time = (time.perf_counter() - start_time) * 1000
            
            return ANNSearchResponse(
                query_point=VectorPoint(
                    lu_id="query",
                    x=0.0,
                    y=0.0,
                ),
                candidates=candidates,
                statistics=ANNSearchStatistics(
                    index_size=10000,
                    search_time_ms=search_time,
                    deleted_ratio=0.05,
                    projection_time_ms=0.0,
                ),
            )
        
        except Exception as e:
            logger.error(f"ANN search visualization failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/vectors", response_model=VectorSpaceResponse)
    async def get_vector_space(
        request: VectorSpaceRequest,
    ) -> VectorSpaceResponse:
        """
        获取向量空间数据用于可视化
        
        返回指定命名空间的所有向量的 2D 投影。
        """
        start_time = time.perf_counter()
        
        try:
            # 模拟向量空间数据
            points = []
            for i in range(min(request.limit, 100)):
                state = ["confirmed", "probationary", "deprecated"][i % 3]
                if request.state_filter and state not in request.state_filter:
                    continue
                
                points.append(VectorPoint(
                    lu_id=f"knowledge_{i:03d}",
                    x=float(np.random.randn()) if HAS_NUMPY else float(i * 0.1),
                    y=float(np.random.randn()) if HAS_NUMPY else float(i * 0.05),
                    lifecycle_state=state,
                    condition_preview=f"示例条件 {i}...",
                    hit_count=100 - i,
                ))
            
            projection_time = (time.perf_counter() - start_time) * 1000
            
            return VectorSpaceResponse(
                points=points,
                total_vectors=10000,
                visible_vectors=len(points),
                projection_method=request.projection,
                projection_time_ms=projection_time,
            )
        
        except Exception as e:
            logger.error(f"Get vector space failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/statistics")
    async def get_ann_statistics(
        namespace: str = Query("default", description="命名空间"),
    ) -> Dict[str, Any]:
        """
        获取 ANN 索引统计信息
        """
        # 模拟统计数据
        return {
            "namespace": namespace,
            "index_type": "faiss",
            "total_vectors": 10000,
            "deleted_vectors": 500,
            "active_vectors": 9500,
            "dimension": 64,
            "last_rebuild": "2026-02-09T10:00:00Z",
            "rebuild_count": 5,
            "average_search_time_ms": 2.3,
        }

else:
    router = None  # type: ignore


__all__ = [
    "router",
    "ProjectionService",
]
