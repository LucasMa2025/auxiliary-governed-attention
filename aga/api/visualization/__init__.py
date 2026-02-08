"""
AGA 可视化 API 模块

提供 ANN 检索、路由决策、知识状态的可视化接口。

版本: v1.0
"""

from typing import Optional

try:
    from fastapi import APIRouter
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None  # type: ignore

# 创建路由器
if HAS_FASTAPI:
    router = APIRouter(prefix="/api/v1/visualization", tags=["visualization"])
    
    # 延迟导入子模块路由
    def include_routers():
        """包含子模块路由"""
        from . import ann, routing, knowledge, events
        router.include_router(ann.router)
        router.include_router(routing.router)
        router.include_router(knowledge.router)
        router.include_router(events.router)
else:
    router = None  # type: ignore
    
    def include_routers():
        pass

__all__ = [
    "router",
    "include_routers",
]
