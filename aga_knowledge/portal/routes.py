"""
AGA Knowledge Portal 路由层（明文 KV 版本）

定义 FastAPI 路由，委托给 PortalService。
保持与原 AGA Portal API 结构兼容，移除向量化注入端点。

API 端点一览：
    POST   /knowledge/inject-text     — 注入知识（明文，推荐）
    POST   /knowledge/batch-text      — 批量注入（明文，推荐）
    GET    /knowledge/{ns}/{lu_id}    — 获取单个知识
    GET    /knowledge/{ns}            — 查询知识列表
    DELETE /knowledge/{ns}/{lu_id}    — 删除知识
    PUT    /lifecycle/update          — 更新生命周期
    POST   /lifecycle/quarantine      — 隔离知识
    GET    /statistics                — 全局统计
    GET    /statistics/{ns}           — 命名空间统计
    GET    /audit                     — 审计日志
    GET    /namespaces                — 命名空间列表
    GET    /health                    — 健康检查
    GET    /health/ready              — 就绪检查
"""

import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None

from .service import PortalService


# ==================== 请求/响应模型 ====================

if HAS_FASTAPI:

    class InjectTextRequest(BaseModel):
        """
        文本知识注入请求（推荐）

        Portal 接收明文 condition/decision，存储并同步到 Runtime。
        Runtime 在推理时按需将文本编码为 KV 向量。

        架构原则：
        - 治理系统 = 规则与授权的"主权层"，只传递语义描述
        - AGA Portal = 规则被执行的"计算层"，负责存储和同步
        - AGA Runtime = 推理时按需编码和注入
        """
        lu_id: str = Field(..., description="Learning Unit ID", min_length=1, max_length=256)
        condition: str = Field(..., description="触发条件描述（文本）", min_length=1)
        decision: str = Field(..., description="决策描述（文本）", min_length=1)
        namespace: str = Field(default="default", description="命名空间", max_length=128)
        lifecycle_state: str = Field(
            default="probationary",
            description="初始状态: probationary | confirmed | deprecated | quarantined",
        )
        trust_tier: Optional[str] = Field(
            default=None,
            description="信任层级: system | verified | standard | experimental | untrusted",
        )
        metadata: Optional[dict] = Field(default=None, description="扩展元数据")

    class BatchInjectTextRequest(BaseModel):
        """批量文本注入请求"""
        items: List[InjectTextRequest] = Field(..., description="知识列表", min_length=1)
        namespace: str = Field(default="default", description="默认命名空间")
        skip_duplicates: bool = Field(default=True, description="跳过重复")

    class UpdateLifecycleRequest(BaseModel):
        """更新生命周期请求"""
        lu_id: str = Field(..., description="Learning Unit ID")
        new_state: str = Field(
            ...,
            description="新状态: probationary | confirmed | deprecated | quarantined",
        )
        namespace: str = Field(default="default", description="命名空间")
        reason: Optional[str] = Field(default=None, description="变更原因")

    class QuarantineRequest(BaseModel):
        """隔离请求"""
        lu_id: str = Field(..., description="Learning Unit ID")
        reason: str = Field(..., description="隔离原因")
        namespace: str = Field(default="default", description="命名空间")

    class DeleteRequest(BaseModel):
        """删除请求"""
        lu_id: str = Field(..., description="Learning Unit ID")
        namespace: str = Field(default="default", description="命名空间")
        reason: Optional[str] = Field(default=None, description="删除原因")

    class APIResponse(BaseModel):
        """通用响应"""
        success: bool
        message: Optional[str] = None
        data: Optional[dict] = None
        timestamp: datetime = Field(default_factory=datetime.utcnow)


def create_portal_routers(service: PortalService):
    """
    创建 Portal 路由

    Args:
        service: PortalService 实例

    Returns:
        路由字典 {name: APIRouter}
    """
    if not HAS_FASTAPI:
        raise ImportError("需要安装 FastAPI: pip install fastapi")

    # ==================== 健康检查路由 ====================

    health_router = APIRouter(prefix="/health", tags=["Health"])

    @health_router.get("")
    async def health():
        """健康检查"""
        return await service.health_check()

    @health_router.get("/ready")
    async def readiness():
        """就绪检查"""
        health_data = await service.health_check()
        if health_data["status"] != "healthy":
            raise HTTPException(status_code=503, detail="Service not ready")
        return {"status": "ready"}

    # ==================== 知识管理路由 ====================

    knowledge_router = APIRouter(prefix="/knowledge", tags=["Knowledge"])

    @knowledge_router.post("/inject-text")
    async def inject_knowledge_text(request: InjectTextRequest):
        """
        注入知识（明文 condition/decision）

        ✅ 推荐 API：外部系统应使用此端点。

        Portal 存储明文 condition/decision 并同步到 Runtime。
        Runtime 在推理时按需将文本编码为 KV 向量，确保：
        1. 编码器一致性（与推理时使用相同编码器）
        2. 治理系统与模型解耦
        3. 规则文本可审计
        """
        result = await service.inject_knowledge_text(
            lu_id=request.lu_id,
            condition=request.condition,
            decision=request.decision,
            namespace=request.namespace,
            lifecycle_state=request.lifecycle_state,
            trust_tier=request.trust_tier,
            metadata=request.metadata,
        )
        return APIResponse(success=result["success"], data=result)

    @knowledge_router.post("/batch-text")
    async def batch_inject_text(request: BatchInjectTextRequest):
        """
        批量注入知识（明文）

        ✅ 推荐 API：外部系统应使用此端点。
        """
        items = [item.dict() for item in request.items]
        result = await service.batch_inject_text(
            items=items,
            namespace=request.namespace,
            skip_duplicates=request.skip_duplicates,
        )
        return APIResponse(success=True, data=result)

    # 分页限制常量
    MAX_LIMIT = 500
    DEFAULT_LIMIT = 50

    @knowledge_router.get("/{namespace}/{lu_id}")
    async def get_knowledge(
        namespace: str,
        lu_id: str,
    ):
        """获取单个知识"""
        record = await service.get_knowledge(
            lu_id=lu_id,
            namespace=namespace,
        )
        if not record:
            raise HTTPException(status_code=404, detail="Knowledge not found")
        return APIResponse(success=True, data=record)

    @knowledge_router.get("/{namespace}")
    async def query_knowledge(
        namespace: str,
        lifecycle_states: Optional[str] = Query(None, description="状态过滤，逗号分隔"),
        trust_tiers: Optional[str] = Query(None, description="层级过滤，逗号分隔"),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description=f"最大 {MAX_LIMIT}"),
        offset: int = Query(0, ge=0),
    ):
        """查询知识列表"""
        actual_limit = min(limit, MAX_LIMIT)

        states = lifecycle_states.split(",") if lifecycle_states else None
        tiers = trust_tiers.split(",") if trust_tiers else None

        result = await service.query_knowledge(
            namespace=namespace,
            lifecycle_states=states,
            trust_tiers=tiers,
            limit=actual_limit,
            offset=offset,
        )
        return APIResponse(success=True, data=result)

    @knowledge_router.delete("/{namespace}/{lu_id}")
    async def delete_knowledge(
        namespace: str,
        lu_id: str,
        reason: Optional[str] = Query(None),
    ):
        """删除知识"""
        result = await service.delete_knowledge(
            lu_id=lu_id,
            namespace=namespace,
            reason=reason,
        )
        return APIResponse(success=result["success"], data=result)

    # ==================== 生命周期路由 ====================

    lifecycle_router = APIRouter(prefix="/lifecycle", tags=["Lifecycle"])

    @lifecycle_router.put("/update")
    async def update_lifecycle(request: UpdateLifecycleRequest):
        """更新生命周期状态"""
        result = await service.update_lifecycle(
            lu_id=request.lu_id,
            new_state=request.new_state,
            namespace=request.namespace,
            reason=request.reason,
        )
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error"))
        return APIResponse(success=True, data=result)

    @lifecycle_router.post("/quarantine")
    async def quarantine(request: QuarantineRequest):
        """隔离知识"""
        result = await service.quarantine(
            lu_id=request.lu_id,
            reason=request.reason,
            namespace=request.namespace,
        )
        return APIResponse(success=result["success"], data=result)

    # ==================== 统计路由 ====================

    stats_router = APIRouter(prefix="/statistics", tags=["Statistics"])

    @stats_router.get("")
    async def get_all_statistics():
        """获取所有统计"""
        result = await service.get_statistics()
        return APIResponse(success=True, data=result)

    @stats_router.get("/{namespace}")
    async def get_namespace_statistics(namespace: str):
        """获取命名空间统计"""
        result = await service.get_statistics(namespace=namespace)
        return APIResponse(success=True, data=result)

    # ==================== 审计路由 ====================

    audit_router = APIRouter(prefix="/audit", tags=["Audit"])

    @audit_router.get("")
    async def get_audit_log(
        namespace: Optional[str] = Query(None),
        lu_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        """获取审计日志"""
        result = await service.get_audit_log(
            namespace=namespace,
            lu_id=lu_id,
            limit=min(limit, 500),
            offset=offset,
        )
        return APIResponse(success=True, data=result)

    # ==================== 命名空间路由 ====================

    namespace_router = APIRouter(prefix="/namespaces", tags=["Namespaces"])

    @namespace_router.get("")
    async def list_namespaces():
        """列出所有命名空间"""
        namespaces = await service.get_namespaces()
        return APIResponse(success=True, data={"namespaces": namespaces})

    return {
        "health": health_router,
        "knowledge": knowledge_router,
        "lifecycle": lifecycle_router,
        "statistics": stats_router,
        "audit": audit_router,
        "namespaces": namespace_router,
    }
