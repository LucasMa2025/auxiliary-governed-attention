"""
AGA 知识状态可视化 API

提供知识生命周期管理和状态变更可视化。

版本: v1.0
"""

import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query, Path
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None  # type: ignore
    BaseModel = object  # type: ignore


# ==================== 数据模型 ====================

if HAS_FASTAPI:
    
    class StateTransitionEvent(BaseModel):
        """状态变更事件"""
        timestamp: str
        lu_id: str
        namespace: str
        from_state: Optional[str] = Field(None, description="原状态 (null 表示新建)")
        to_state: Optional[str] = Field(None, description="目标状态 (null 表示删除)")
        reason: str
        operator: str
        metadata: Optional[Dict[str, Any]] = None
    
    class TimelineRequest(BaseModel):
        """时间线请求"""
        namespace: str = Field("default", description="命名空间")
        start_time: Optional[str] = Field(None, description="开始时间 (ISO 8601)")
        end_time: Optional[str] = Field(None, description="结束时间 (ISO 8601)")
        limit: int = Field(100, ge=1, le=1000, description="最大事件数")
        state_filter: Optional[List[str]] = Field(None, description="状态过滤")
        lu_id_filter: Optional[str] = Field(None, description="LU ID 过滤")
    
    class TransitionStatistics(BaseModel):
        """状态变更统计"""
        total_events: int
        by_transition: Dict[str, int]
        by_operator: Dict[str, int]
    
    class TimelineResponse(BaseModel):
        """时间线响应"""
        events: List[StateTransitionEvent]
        statistics: TransitionStatistics
    
    class KnowledgeDetail(BaseModel):
        """知识详情"""
        lu_id: str
        namespace: str
        lifecycle_state: str
        reliability: float
        hit_count: int
        created_at: str
        last_hit_at: Optional[str] = None
        condition: str
        decision: str
        trust_tier: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None
    
    class StateTransitionRequest(BaseModel):
        """状态变更请求"""
        target_state: str = Field(..., description="目标状态")
        reason: str = Field(..., description="变更原因")
        operator: str = Field("admin", description="操作者")
    
    class StateTransitionResponse(BaseModel):
        """状态变更响应"""
        success: bool
        lu_id: str
        from_state: str
        to_state: str
        timestamp: str
    
    class StateDistribution(BaseModel):
        """状态分布"""
        namespace: str
        total: int
        distribution: Dict[str, int]
        percentages: Dict[str, float]


# ==================== API 路由 ====================

if HAS_FASTAPI:
    router = APIRouter(prefix="/knowledge", tags=["knowledge-visualization"])
    
    @router.post("/timeline", response_model=TimelineResponse)
    async def get_state_timeline(
        request: TimelineRequest,
    ) -> TimelineResponse:
        """
        获取知识状态变更时间线
        """
        # 模拟时间线数据
        now = datetime.utcnow()
        events = []
        
        transitions = [
            ("PROBATIONARY", "CONFIRMED", "验证通过, 命中率 > 80%"),
            (None, "PROBATIONARY", "自动学习系统注入"),
            ("CONFIRMED", "QUARANTINED", "检测到错误输出"),
            ("DEPRECATED", None, "保留期过期"),
            ("PROBATIONARY", "DEPRECATED", "长期未命中"),
        ]
        
        for i in range(min(request.limit, 20)):
            from_state, to_state, reason = transitions[i % len(transitions)]
            
            # 应用状态过滤
            if request.state_filter:
                if from_state and from_state not in request.state_filter:
                    if to_state and to_state not in request.state_filter:
                        continue
            
            events.append(StateTransitionEvent(
                timestamp=(now - timedelta(minutes=i * 5)).isoformat() + "Z",
                lu_id=f"knowledge_{i:03d}",
                namespace=request.namespace,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                operator="governance_system" if i % 2 == 0 else "admin",
            ))
        
        # 统计
        by_transition = {}
        by_operator = {}
        for event in events:
            key = f"{event.from_state or 'NEW'}→{event.to_state or 'REMOVED'}"
            by_transition[key] = by_transition.get(key, 0) + 1
            by_operator[event.operator] = by_operator.get(event.operator, 0) + 1
        
        return TimelineResponse(
            events=events,
            statistics=TransitionStatistics(
                total_events=len(events),
                by_transition=by_transition,
                by_operator=by_operator,
            ),
        )
    
    @router.get("/distribution", response_model=StateDistribution)
    async def get_state_distribution(
        namespace: str = Query("default", description="命名空间"),
    ) -> StateDistribution:
        """
        获取知识状态分布
        """
        # 模拟分布数据
        distribution = {
            "PROBATIONARY": 45,
            "CONFIRMED": 120,
            "DEPRECATED": 15,
            "QUARANTINED": 5,
        }
        
        total = sum(distribution.values())
        percentages = {k: round(v / total * 100, 1) for k, v in distribution.items()}
        
        return StateDistribution(
            namespace=namespace,
            total=total,
            distribution=distribution,
            percentages=percentages,
        )
    
    @router.get("/{lu_id}", response_model=KnowledgeDetail)
    async def get_knowledge_detail(
        lu_id: str = Path(..., description="知识 LU ID"),
        namespace: str = Query("default", description="命名空间"),
    ) -> KnowledgeDetail:
        """
        获取知识详情
        """
        now = datetime.utcnow()
        
        # 模拟知识详情
        return KnowledgeDetail(
            lu_id=lu_id,
            namespace=namespace,
            lifecycle_state="CONFIRMED",
            reliability=1.0,
            hit_count=1234,
            created_at=(now - timedelta(days=7)).isoformat() + "Z",
            last_hit_at=(now - timedelta(minutes=5)).isoformat() + "Z",
            condition="当用户询问法国的首都时",
            decision="回答'巴黎是法国的首都'",
            trust_tier="verified",
            metadata={
                "source": "manual_injection",
                "version": "1.0",
            },
        )
    
    @router.post("/{lu_id}/transition", response_model=StateTransitionResponse)
    async def transition_knowledge_state(
        lu_id: str = Path(..., description="知识 LU ID"),
        request: StateTransitionRequest,
        namespace: str = Query("default", description="命名空间"),
    ) -> StateTransitionResponse:
        """
        执行知识状态变更
        """
        now = datetime.utcnow()
        
        # 验证目标状态
        valid_states = ["PROBATIONARY", "CONFIRMED", "DEPRECATED", "QUARANTINED"]
        if request.target_state not in valid_states:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target state. Must be one of: {valid_states}"
            )
        
        # 模拟状态变更
        return StateTransitionResponse(
            success=True,
            lu_id=lu_id,
            from_state="CONFIRMED",
            to_state=request.target_state,
            timestamp=now.isoformat() + "Z",
        )
    
    @router.get("/{lu_id}/history")
    async def get_knowledge_history(
        lu_id: str = Path(..., description="知识 LU ID"),
        namespace: str = Query("default", description="命名空间"),
        limit: int = Query(50, ge=1, le=500, description="最大事件数"),
    ) -> Dict[str, Any]:
        """
        获取单个知识的历史记录
        """
        now = datetime.utcnow()
        
        # 模拟历史记录
        history = [
            {
                "timestamp": (now - timedelta(days=7)).isoformat() + "Z",
                "event": "CREATED",
                "from_state": None,
                "to_state": "PROBATIONARY",
                "operator": "admin",
            },
            {
                "timestamp": (now - timedelta(days=5)).isoformat() + "Z",
                "event": "CONFIRMED",
                "from_state": "PROBATIONARY",
                "to_state": "CONFIRMED",
                "operator": "governance_system",
            },
            {
                "timestamp": (now - timedelta(hours=2)).isoformat() + "Z",
                "event": "HIT",
                "hit_count": 1234,
            },
        ]
        
        return {
            "lu_id": lu_id,
            "namespace": namespace,
            "history": history[:limit],
            "total_events": len(history),
        }
    
    @router.get("/{lu_id}/hits")
    async def get_knowledge_hit_statistics(
        lu_id: str = Path(..., description="知识 LU ID"),
        namespace: str = Query("default", description="命名空间"),
        time_window: str = Query("24h", description="时间窗口"),
    ) -> Dict[str, Any]:
        """
        获取知识命中统计
        """
        import random
        
        now = datetime.utcnow()
        
        # 模拟命中统计
        hourly_hits = []
        for i in range(24):
            hourly_hits.append({
                "hour": (now - timedelta(hours=23 - i)).strftime("%H:00"),
                "hits": random.randint(10, 100),
            })
        
        return {
            "lu_id": lu_id,
            "namespace": namespace,
            "time_window": time_window,
            "total_hits": 1234,
            "average_hits_per_hour": 51.4,
            "peak_hour": "14:00",
            "hourly_distribution": hourly_hits,
        }

else:
    router = None  # type: ignore


__all__ = [
    "router",
]
