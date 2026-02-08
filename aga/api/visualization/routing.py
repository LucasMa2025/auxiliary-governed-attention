"""
AGA 路由决策可视化 API

提供三段式门控决策流程的可视化接口。

版本: v1.0
"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None  # type: ignore
    BaseModel = object  # type: ignore


# ==================== 数据模型 ====================

if HAS_FASTAPI:
    
    class Gate0Result(BaseModel):
        """Gate-0 先验门控结果"""
        decision: str = Field(..., description="决策: PASS, BYPASS, DISABLED")
        namespace: str
        app_id: Optional[str] = None
        reason: str
    
    class Gate1Result(BaseModel):
        """Gate-1 置信门控结果"""
        decision: str = Field(..., description="决策: PASS, BYPASS")
        confidence: float = Field(..., ge=0, le=1)
        threshold: float = Field(..., ge=0, le=1)
        entropy: float
        bypass_rate: float = Field(..., ge=0, le=1)
    
    class SelectedSlot(BaseModel):
        """选中的槽位"""
        slot_idx: int
        score: float
        lu_id: str
        condition_preview: Optional[str] = None
    
    class Gate2Result(BaseModel):
        """Gate-2 Top-k 路由结果"""
        selected_slots: List[SelectedSlot]
        total_candidates: int
        top_k: int
        ann_candidates: Optional[int] = None
    
    class FinalResult(BaseModel):
        """最终结果"""
        aga_applied: bool
        fusion_weight: float
        latency_ms: float
        hot_hits: Optional[int] = None
        warm_hits: Optional[int] = None
        cold_loads: Optional[int] = None
    
    class RouteTraceRequest(BaseModel):
        """路由追踪请求"""
        query_text: Optional[str] = Field(None, description="查询文本")
        hidden_states: Optional[List[List[float]]] = Field(None, description="隐藏状态")
        namespace: str = Field("default", description="命名空间")
        return_details: bool = Field(True, description="返回详细信息")
    
    class RouteTraceResponse(BaseModel):
        """路由追踪响应"""
        trace_id: str
        timestamp: str
        gate0: Gate0Result
        gate1: Gate1Result
        gate2: Gate2Result
        final: FinalResult
    
    class RouteStatistics(BaseModel):
        """路由统计"""
        total_requests: int
        aga_applied_count: int
        aga_applied_rate: float
        gate1_bypass_rate: float
        average_latency_ms: float
        p50_latency_ms: float
        p95_latency_ms: float
        p99_latency_ms: float


# ==================== API 路由 ====================

if HAS_FASTAPI:
    router = APIRouter(prefix="/route", tags=["routing-visualization"])
    
    @router.post("/trace", response_model=RouteTraceResponse)
    async def trace_route_decision(
        request: RouteTraceRequest,
    ) -> RouteTraceResponse:
        """
        追踪单次路由决策
        
        返回 Gate-0/1/2 的完整决策链。
        """
        import uuid
        from datetime import datetime
        
        start_time = time.perf_counter()
        
        try:
            # 模拟路由追踪
            trace_id = str(uuid.uuid4())[:8]
            
            # Gate-0: 先验门控
            gate0 = Gate0Result(
                decision="PASS",
                namespace=request.namespace,
                app_id=None,
                reason="namespace enabled",
            )
            
            # Gate-1: 置信门控
            gate1 = Gate1Result(
                decision="PASS",
                confidence=0.32,
                threshold=0.5,
                entropy=2.1,
                bypass_rate=0.235,
            )
            
            # Gate-2: Top-k 路由
            selected_slots = []
            for i in range(8):
                selected_slots.append(SelectedSlot(
                    slot_idx=i * 2,
                    score=0.95 - i * 0.05,
                    lu_id=f"knowledge_{i:03d}",
                    condition_preview=f"当用户询问... {i}",
                ))
            
            gate2 = Gate2Result(
                selected_slots=selected_slots,
                total_candidates=200,
                top_k=8,
                ann_candidates=200,
            )
            
            latency = (time.perf_counter() - start_time) * 1000
            
            final = FinalResult(
                aga_applied=True,
                fusion_weight=0.45,
                latency_ms=latency,
                hot_hits=6,
                warm_hits=2,
                cold_loads=0,
            )
            
            return RouteTraceResponse(
                trace_id=trace_id,
                timestamp=datetime.utcnow().isoformat() + "Z",
                gate0=gate0,
                gate1=gate1,
                gate2=gate2,
                final=final,
            )
        
        except Exception as e:
            logger.error(f"Route trace failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/statistics", response_model=RouteStatistics)
    async def get_route_statistics(
        namespace: str = Query("default", description="命名空间"),
        time_window: str = Query("1h", description="时间窗口"),
    ) -> RouteStatistics:
        """
        获取路由统计信息
        """
        # 模拟统计数据
        return RouteStatistics(
            total_requests=10000,
            aga_applied_count=7650,
            aga_applied_rate=0.765,
            gate1_bypass_rate=0.235,
            average_latency_ms=1.5,
            p50_latency_ms=1.2,
            p95_latency_ms=3.5,
            p99_latency_ms=8.2,
        )
    
    @router.get("/gate1/threshold-history")
    async def get_gate1_threshold_history(
        namespace: str = Query("default", description="命名空间"),
        limit: int = Query(100, ge=1, le=1000, description="数据点数"),
    ) -> Dict[str, Any]:
        """
        获取 Gate-1 自适应阈值历史
        """
        import random
        from datetime import datetime, timedelta
        
        # 模拟历史数据
        now = datetime.utcnow()
        data_points = []
        
        threshold = 0.5
        for i in range(limit):
            timestamp = now - timedelta(minutes=limit - i)
            # 模拟阈值变化
            threshold += random.uniform(-0.02, 0.02)
            threshold = max(0.3, min(0.7, threshold))
            
            data_points.append({
                "timestamp": timestamp.isoformat() + "Z",
                "threshold": round(threshold, 4),
                "bypass_rate": round(random.uniform(0.15, 0.35), 4),
            })
        
        return {
            "namespace": namespace,
            "current_threshold": round(threshold, 4),
            "history": data_points,
        }
    
    @router.get("/gate2/top-k-distribution")
    async def get_gate2_topk_distribution(
        namespace: str = Query("default", description="命名空间"),
        time_window: str = Query("1h", description="时间窗口"),
    ) -> Dict[str, Any]:
        """
        获取 Gate-2 Top-k 选择分布
        """
        import random
        
        # 模拟分布数据
        distribution = {}
        for i in range(1, 9):
            distribution[f"top_{i}"] = random.randint(100, 1000)
        
        return {
            "namespace": namespace,
            "time_window": time_window,
            "top_k": 8,
            "distribution": distribution,
            "most_selected_slots": [
                {"lu_id": "knowledge_042", "count": 1234},
                {"lu_id": "knowledge_108", "count": 987},
                {"lu_id": "knowledge_023", "count": 876},
            ],
        }

else:
    router = None  # type: ignore


__all__ = [
    "router",
]
