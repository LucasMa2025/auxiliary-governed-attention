"""
AGA API 数据模型

使用 Pydantic v2 定义请求/响应模型。
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

try:
    from pydantic import BaseModel, Field, ConfigDict
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback to dataclass
    from dataclasses import dataclass, field
    
    class BaseModel:
        """Fallback BaseModel"""
        pass
    
    def Field(*args, **kwargs):
        return field(default=kwargs.get('default'))


# ============================================================
# 枚举类型
# ============================================================

class LifecycleStateEnum(str, Enum):
    """生命周期状态枚举"""
    PROBATIONARY = "probationary"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"
    QUARANTINED = "quarantined"


class TrustTierEnum(str, Enum):
    """信任层级枚举"""
    S0_ACCELERATION = "s0_acceleration"
    S1_EXPERIENCE = "s1_experience"
    S2_POLICY = "s2_policy"
    S3_IMMUTABLE = "s3_immutable"


# ============================================================
# 请求模型
# ============================================================

if HAS_PYDANTIC:
    class InjectKnowledgeRequest(BaseModel):
        """知识注入请求"""
        model_config = ConfigDict(extra="forbid")
        
        lu_id: str = Field(..., description="Learning Unit ID (唯一标识)")
        namespace: str = Field(default="default", description="命名空间")
        condition: str = Field(..., description="触发条件描述")
        decision: str = Field(..., description="决策/动作描述")
        key_vector: List[float] = Field(..., description="条件编码向量")
        value_vector: List[float] = Field(..., description="决策编码向量")
        lifecycle_state: LifecycleStateEnum = Field(
            default=LifecycleStateEnum.PROBATIONARY,
            description="初始生命周期状态"
        )
        trust_tier: Optional[TrustTierEnum] = Field(
            default=None,
            description="信任层级（用于治理）"
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="扩展元数据"
        )
    
    class BatchInjectRequest(BaseModel):
        """批量注入请求"""
        model_config = ConfigDict(extra="forbid")
        
        items: List[InjectKnowledgeRequest] = Field(..., description="知识列表")
        namespace: str = Field(default="default", description="默认命名空间")
        skip_duplicates: bool = Field(default=True, description="跳过重复 LU ID")
    
    class UpdateLifecycleRequest(BaseModel):
        """更新生命周期请求"""
        model_config = ConfigDict(extra="forbid")
        
        lu_id: str = Field(..., description="Learning Unit ID")
        namespace: str = Field(default="default", description="命名空间")
        new_state: LifecycleStateEnum = Field(..., description="新状态")
        reason: Optional[str] = Field(default=None, description="变更原因")
    
    class BatchUpdateLifecycleRequest(BaseModel):
        """批量更新生命周期请求"""
        model_config = ConfigDict(extra="forbid")
        
        updates: List[UpdateLifecycleRequest] = Field(..., description="更新列表")
    
    class QuarantineRequest(BaseModel):
        """隔离请求"""
        model_config = ConfigDict(extra="forbid")
        
        lu_id: str = Field(..., description="Learning Unit ID")
        namespace: str = Field(default="default", description="命名空间")
        reason: str = Field(..., description="隔离原因")
        source_instance: Optional[str] = Field(default=None, description="来源实例")
    
    class QueryKnowledgeRequest(BaseModel):
        """查询知识请求"""
        model_config = ConfigDict(extra="forbid")
        
        namespace: str = Field(default="default", description="命名空间")
        lifecycle_states: Optional[List[LifecycleStateEnum]] = Field(
            default=None,
            description="筛选生命周期状态"
        )
        trust_tiers: Optional[List[TrustTierEnum]] = Field(
            default=None,
            description="筛选信任层级"
        )
        limit: int = Field(default=100, ge=1, le=1000, description="返回数量限制")
        offset: int = Field(default=0, ge=0, description="偏移量")
        include_vectors: bool = Field(default=False, description="是否包含向量数据")
    
    class SimilaritySearchRequest(BaseModel):
        """相似性搜索请求"""
        model_config = ConfigDict(extra="forbid")
        
        namespace: str = Field(default="default", description="命名空间")
        query_vector: List[float] = Field(..., description="查询向量")
        top_k: int = Field(default=10, ge=1, le=100, description="返回数量")
        threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="相似度阈值")


# ============================================================
# 响应模型
# ============================================================

if HAS_PYDANTIC:
    class APIResponse(BaseModel):
        """通用 API 响应"""
        success: bool = Field(..., description="是否成功")
        message: Optional[str] = Field(default=None, description="消息")
        data: Optional[Any] = Field(default=None, description="数据")
        timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    
    class KnowledgeResponse(BaseModel):
        """知识响应"""
        lu_id: str
        namespace: str
        slot_idx: int
        condition: str
        decision: str
        lifecycle_state: str
        trust_tier: Optional[str] = None
        reliability: float
        hit_count: int
        key_vector: Optional[List[float]] = None
        value_vector: Optional[List[float]] = None
        created_at: Optional[datetime] = None
        updated_at: Optional[datetime] = None
        metadata: Optional[Dict[str, Any]] = None
    
    class SlotInfoResponse(BaseModel):
        """槽位信息响应"""
        slot_idx: int
        lu_id: Optional[str]
        namespace: str
        lifecycle_state: str
        reliability: float
        key_norm: float
        value_norm: float
        condition: Optional[str]
        decision: Optional[str]
        hit_count: int
        is_active: bool
    
    class StatisticsResponse(BaseModel):
        """统计信息响应"""
        namespace: str
        total_slots: int
        active_slots: int
        free_slots: int
        state_distribution: Dict[str, int]
        trust_tier_distribution: Optional[Dict[str, int]] = None
        total_hits: int
        avg_reliability: float
        avg_key_norm: float
        avg_value_norm: float
    
    class AuditLogEntry(BaseModel):
        """审计日志条目"""
        id: int
        lu_id: Optional[str]
        namespace: str
        action: str
        old_state: Optional[str]
        new_state: Optional[str]
        reason: Optional[str]
        source_instance: Optional[str]
        timestamp: datetime
        details: Optional[Dict[str, Any]] = None
    
    class AuditLogResponse(BaseModel):
        """审计日志响应"""
        entries: List[AuditLogEntry]
        total: int
        limit: int
        offset: int
    
    class BatchResultResponse(BaseModel):
        """批量操作结果响应"""
        total: int
        success_count: int
        failed_count: int
        results: List[Dict[str, Any]]
    
    class HealthResponse(BaseModel):
        """健康检查响应"""
        status: str
        version: str
        aga_initialized: bool
        namespaces: List[str]
        total_knowledge: int
        uptime_seconds: float
        timestamp: datetime
    
    class NamespaceStatsResponse(BaseModel):
        """命名空间统计响应"""
        namespaces: Dict[str, StatisticsResponse]
        total_namespaces: int
        total_knowledge: int

else:
    # Fallback definitions without Pydantic
    from dataclasses import dataclass
    
    @dataclass
    class InjectKnowledgeRequest:
        lu_id: str
        condition: str
        decision: str
        key_vector: List[float]
        value_vector: List[float]
        namespace: str = "default"
        lifecycle_state: str = "probationary"
        trust_tier: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None
    
    @dataclass
    class APIResponse:
        success: bool
        message: Optional[str] = None
        data: Any = None
        timestamp: datetime = None
    
    # ... 其他模型的简化定义
    BatchInjectRequest = None
    UpdateLifecycleRequest = None
    QuarantineRequest = None
    QueryKnowledgeRequest = None
    KnowledgeResponse = None
    SlotInfoResponse = None
    StatisticsResponse = None
    AuditLogResponse = None
