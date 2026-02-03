"""
AGA API 路由层

只负责 HTTP 协议转换，所有业务逻辑委托给服务层。
"""
from typing import List, Optional
from datetime import datetime
import logging

try:
    from fastapi import APIRouter, HTTPException, Query, Path, Depends
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None

from .models import (
    InjectKnowledgeRequest,
    InjectKnowledgeTextRequest,
    BatchInjectRequest,
    BatchInjectTextRequest,
    UpdateLifecycleRequest,
    BatchUpdateLifecycleRequest,
    QuarantineRequest,
    QueryKnowledgeRequest,
    HealthResponse,
    StatisticsResponse,
    SlotInfoResponse,
    AuditLogResponse,
    BatchResultResponse,
    NamespaceStatsResponse,
)
from .service import AGAService

logger = logging.getLogger(__name__)


def get_service() -> AGAService:
    """依赖注入：获取服务实例"""
    return AGAService.get_instance()


def create_routers(service: AGAService = None):
    """
    创建所有路由器
    
    Returns:
        包含所有路由器的字典
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required")
    
    # ============================================================
    # 健康检查路由
    # ============================================================
    
    health_router = APIRouter(prefix="", tags=["Health"])
    
    @health_router.get("/health", response_model=HealthResponse)
    async def health_check(svc: AGAService = Depends(get_service)):
        """
        健康检查
        
        返回服务状态、版本、运行时间等信息。
        """
        return await svc.health_check()
    
    @health_router.get("/config")
    async def get_config(svc: AGAService = Depends(get_service)):
        """获取服务配置"""
        return svc.get_config()
    
    # ============================================================
    # 命名空间路由
    # ============================================================
    
    namespace_router = APIRouter(prefix="/namespaces", tags=["Namespace"])
    
    @namespace_router.get("")
    async def list_namespaces(svc: AGAService = Depends(get_service)):
        """列出所有命名空间"""
        return {"namespaces": svc.get_namespaces()}
    
    @namespace_router.delete("/{namespace}")
    async def delete_namespace(
        namespace: str = Path(..., description="命名空间"),
        svc: AGAService = Depends(get_service),
    ):
        """删除命名空间"""
        return await svc.delete_namespace(namespace)
    
    # ============================================================
    # 知识管理路由
    # ============================================================
    
    knowledge_router = APIRouter(prefix="/knowledge", tags=["Knowledge"])
    
    @knowledge_router.post("/inject")
    async def inject_knowledge(
        request: InjectKnowledgeRequest,
        svc: AGAService = Depends(get_service),
    ):
        """
        注入单条知识（包含向量）
        
        ⚠️ 内部使用：此 API 要求调用方提供预编码的向量。
        治理系统应使用 /inject-text API。
        
        - **lu_id**: Learning Unit 唯一标识
        - **namespace**: 命名空间（用于租户隔离）
        - **condition**: 触发条件描述
        - **decision**: 决策/动作描述
        - **key_vector**: 条件编码向量（维度=bottleneck_dim）
        - **value_vector**: 决策编码向量（维度=hidden_dim）
        - **lifecycle_state**: 初始生命周期状态
        """
        try:
            return await svc.inject_knowledge(
                namespace=request.namespace,
                lu_id=request.lu_id,
                condition=request.condition,
                decision=request.decision,
                key_vector=request.key_vector,
                value_vector=request.value_vector,
                lifecycle_state=request.lifecycle_state.value,
                trust_tier=request.trust_tier.value if request.trust_tier else None,
                metadata=request.metadata,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Inject knowledge failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @knowledge_router.post("/inject-text")
    async def inject_knowledge_text(
        request: InjectKnowledgeTextRequest,
        svc: AGAService = Depends(get_service),
    ):
        """
        注入单条知识（文本，服务负责编码）
        
        ✅ 推荐 API：治理系统应使用此端点。
        
        服务会使用配置的编码器将文本转换为向量，确保：
        1. 编码器一致性（与推理时使用相同编码器）
        2. 治理系统与模型解耦
        3. 规则文本可审计
        
        - **lu_id**: Learning Unit 唯一标识
        - **namespace**: 命名空间（用于租户隔离）
        - **condition**: 触发条件描述（文本）
        - **decision**: 决策/动作描述（文本）
        - **lifecycle_state**: 初始生命周期状态
        """
        try:
            return await svc.inject_knowledge_text(
                namespace=request.namespace,
                lu_id=request.lu_id,
                condition=request.condition,
                decision=request.decision,
                lifecycle_state=request.lifecycle_state.value,
                trust_tier=request.trust_tier.value if request.trust_tier else None,
                metadata=request.metadata,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Inject knowledge text failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @knowledge_router.post("/inject/batch", response_model=BatchResultResponse)
    async def batch_inject(
        request: BatchInjectRequest,
        svc: AGAService = Depends(get_service),
    ):
        """
        批量注入知识（包含向量）
        
        ⚠️ 内部使用：此 API 要求调用方提供预编码的向量。
        治理系统应使用 /inject-text/batch API。
        """
        try:
            items = [
                {
                    "namespace": item.namespace,
                    "lu_id": item.lu_id,
                    "condition": item.condition,
                    "decision": item.decision,
                    "key_vector": item.key_vector,
                    "value_vector": item.value_vector,
                    "lifecycle_state": item.lifecycle_state.value,
                    "trust_tier": item.trust_tier.value if item.trust_tier else None,
                    "metadata": item.metadata,
                }
                for item in request.items
            ]
            return await svc.batch_inject_knowledge(
                items=items,
                namespace=request.namespace,
                skip_duplicates=request.skip_duplicates,
            )
        except Exception as e:
            logger.error(f"Batch inject failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @knowledge_router.post("/inject-text/batch", response_model=BatchResultResponse)
    async def batch_inject_text(
        request: BatchInjectTextRequest,
        svc: AGAService = Depends(get_service),
    ):
        """
        批量注入知识（文本，服务负责编码）
        
        ✅ 推荐 API：治理系统应使用此端点。
        """
        try:
            items = [
                {
                    "namespace": item.namespace,
                    "lu_id": item.lu_id,
                    "condition": item.condition,
                    "decision": item.decision,
                    "lifecycle_state": item.lifecycle_state.value,
                    "trust_tier": item.trust_tier.value if item.trust_tier else None,
                    "metadata": item.metadata,
                }
                for item in request.items
            ]
            return await svc.batch_inject_knowledge_text(
                items=items,
                namespace=request.namespace,
                skip_duplicates=request.skip_duplicates,
            )
        except Exception as e:
            logger.error(f"Batch inject text failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @knowledge_router.get("/{namespace}/{lu_id}")
    async def get_knowledge(
        namespace: str = Path(..., description="命名空间"),
        lu_id: str = Path(..., description="Learning Unit ID"),
        include_vectors: bool = Query(False, description="是否包含向量数据"),
        svc: AGAService = Depends(get_service),
    ):
        """
        获取单条知识
        
        根据 namespace 和 lu_id 获取知识详情。
        """
        result = await svc.get_knowledge(namespace, lu_id, include_vectors)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge {lu_id} not found in namespace {namespace}"
            )
        return result
    
    @knowledge_router.post("/query")
    async def query_knowledge(
        request: QueryKnowledgeRequest,
        svc: AGAService = Depends(get_service),
    ):
        """
        查询知识列表
        
        支持按状态、信任层级筛选，支持分页。
        """
        lifecycle_states = None
        if request.lifecycle_states:
            lifecycle_states = [s.value for s in request.lifecycle_states]
        
        return await svc.query_knowledge(
            namespace=request.namespace,
            lifecycle_states=lifecycle_states,
            trust_tiers=[t.value for t in request.trust_tiers] if request.trust_tiers else None,
            limit=request.limit,
            offset=request.offset,
            include_vectors=request.include_vectors,
        )
    
    @knowledge_router.delete("/{namespace}/{lu_id}")
    async def delete_knowledge(
        namespace: str = Path(..., description="命名空间"),
        lu_id: str = Path(..., description="Learning Unit ID"),
        svc: AGAService = Depends(get_service),
    ):
        """
        删除知识
        
        从 AGA 和持久化层中删除指定知识。
        """
        try:
            return await svc.delete_knowledge(namespace, lu_id)
        except Exception as e:
            logger.error(f"Delete knowledge failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============================================================
    # 生命周期路由
    # ============================================================
    
    lifecycle_router = APIRouter(prefix="/lifecycle", tags=["Lifecycle"])
    
    @lifecycle_router.post("/update")
    async def update_lifecycle(
        request: UpdateLifecycleRequest,
        svc: AGAService = Depends(get_service),
    ):
        """
        更新生命周期状态
        
        状态转换路径：
        - probationary → confirmed (确认)
        - probationary → quarantined (直接隔离)
        - confirmed → deprecated (弃用)
        - confirmed → quarantined (紧急隔离)
        - deprecated → quarantined (完全隔离)
        """
        try:
            return await svc.update_lifecycle(
                namespace=request.namespace,
                lu_id=request.lu_id,
                new_state=request.new_state.value,
                reason=request.reason,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Update lifecycle failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @lifecycle_router.post("/update/batch", response_model=BatchResultResponse)
    async def batch_update_lifecycle(
        request: BatchUpdateLifecycleRequest,
        svc: AGAService = Depends(get_service),
    ):
        """批量更新生命周期状态"""
        results = []
        success_count = 0
        
        for update in request.updates:
            try:
                result = await svc.update_lifecycle(
                    namespace=update.namespace,
                    lu_id=update.lu_id,
                    new_state=update.new_state.value,
                    reason=update.reason,
                )
                results.append(result)
                if result.get("success"):
                    success_count += 1
            except Exception as e:
                results.append({
                    "success": False,
                    "lu_id": update.lu_id,
                    "error": str(e),
                })
        
        return {
            "total": len(request.updates),
            "success_count": success_count,
            "failed_count": len(request.updates) - success_count,
            "results": results,
        }
    
    @lifecycle_router.post("/quarantine")
    async def quarantine_knowledge(
        request: QuarantineRequest,
        svc: AGAService = Depends(get_service),
    ):
        """
        隔离知识
        
        立即将知识标记为 QUARANTINED 状态，不再参与推理。
        适用于紧急情况，如检测到错误输出。
        """
        try:
            return await svc.quarantine_knowledge(
                namespace=request.namespace,
                lu_id=request.lu_id,
                reason=request.reason,
                source_instance=request.source_instance,
            )
        except Exception as e:
            logger.error(f"Quarantine failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============================================================
    # 槽位路由
    # ============================================================
    
    slots_router = APIRouter(prefix="/slots", tags=["Slots"])
    
    @slots_router.get("/{namespace}/free")
    async def find_free_slot(
        namespace: str = Path(..., description="命名空间"),
        svc: AGAService = Depends(get_service),
    ):
        """查找空闲槽位"""
        slot_idx = await svc.find_free_slot(namespace)
        return {"namespace": namespace, "free_slot": slot_idx}
    
    @slots_router.get("/{namespace}/{slot_idx}", response_model=SlotInfoResponse)
    async def get_slot_info(
        namespace: str = Path(..., description="命名空间"),
        slot_idx: int = Path(..., description="槽位索引"),
        svc: AGAService = Depends(get_service),
    ):
        """获取槽位详细信息"""
        try:
            return await svc.get_slot_info(namespace, slot_idx)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # ============================================================
    # 统计路由
    # ============================================================
    
    statistics_router = APIRouter(prefix="/statistics", tags=["Statistics"])
    
    @statistics_router.get("/{namespace}", response_model=StatisticsResponse)
    async def get_namespace_statistics(
        namespace: str = Path(..., description="命名空间"),
        svc: AGAService = Depends(get_service),
    ):
        """获取命名空间统计"""
        return await svc.get_namespace_statistics(namespace)
    
    @statistics_router.get("", response_model=NamespaceStatsResponse)
    async def get_all_statistics(svc: AGAService = Depends(get_service)):
        """获取所有命名空间统计"""
        return await svc.get_all_statistics()
    
    @statistics_router.get("/writer")
    async def get_writer_statistics(svc: AGAService = Depends(get_service)):
        """获取写入器统计"""
        return await svc.get_writer_statistics()
    
    # ============================================================
    # 审计路由
    # ============================================================
    
    audit_router = APIRouter(prefix="/audit", tags=["Audit"])
    
    @audit_router.get("/{namespace}", response_model=AuditLogResponse)
    async def get_audit_log(
        namespace: str = Path(..., description="命名空间"),
        limit: int = Query(100, ge=1, le=1000, description="返回数量"),
        offset: int = Query(0, ge=0, description="偏移量"),
        svc: AGAService = Depends(get_service),
    ):
        """获取审计日志"""
        return await svc.get_audit_log(namespace, limit, offset)
    
    # 返回所有路由器
    return {
        "health": health_router,
        "namespace": namespace_router,
        "knowledge": knowledge_router,
        "lifecycle": lifecycle_router,
        "slots": slots_router,
        "statistics": statistics_router,
        "audit": audit_router,
    }
