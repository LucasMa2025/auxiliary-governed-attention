"""
AGA REST API 模块

为外部治理系统提供知识管理 API 接口。

架构：
=====

    ┌─────────────────────────────────────────────┐
    │              AGA API 模块                    │
    ├─────────────────────────────────────────────┤
    │  app.py      - 应用入口和配置               │
    │  routes.py   - 路由层（HTTP 协议转换）      │
    │  service.py  - 服务层（业务逻辑）           │
    │  models.py   - 数据模型（Pydantic）         │
    │  client.py   - 客户端库（供外部系统使用）   │
    └─────────────────────────────────────────────┘

主要功能：
=========
- 知识注入/更新/删除
- 生命周期管理
- 隔离/恢复
- 统计和监控
- 审计日志

使用方式：
=========

启动 API 服务：
    python -m aga.api --port 8081

在代码中创建应用：
    from aga.api import create_api_app
    app = create_api_app(hidden_dim=4096, bottleneck_dim=64)

外部系统调用（使用客户端库）：
    from aga.api import AGAClient
    
    client = AGAClient("http://localhost:8081")
    client.inject_knowledge(lu_id="LU_001", ...)

版本: v3.1
"""

from .app import create_api_app
from .service import AGAService, ServiceConfig

from .models import (
    # 请求模型
    InjectKnowledgeRequest,
    BatchInjectRequest,
    UpdateLifecycleRequest,
    QuarantineRequest,
    QueryKnowledgeRequest,
    # 响应模型
    KnowledgeResponse,
    SlotInfoResponse,
    StatisticsResponse,
    AuditLogResponse,
    APIResponse,
)

from .client import AGAClient, AsyncAGAClient

__all__ = [
    # App
    "create_api_app",
    # Service
    "AGAService",
    "ServiceConfig",
    # Models
    "InjectKnowledgeRequest",
    "BatchInjectRequest",
    "UpdateLifecycleRequest",
    "QuarantineRequest",
    "QueryKnowledgeRequest",
    "KnowledgeResponse",
    "SlotInfoResponse",
    "StatisticsResponse",
    "AuditLogResponse",
    "APIResponse",
    # Client (for external governance systems)
    "AGAClient",
    "AsyncAGAClient",
]
