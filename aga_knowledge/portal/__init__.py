"""
aga-knowledge Portal — 独立部署的知识管理 API 服务

Portal 是 AGA 系统的知识管理入口，无需 GPU。

主要职责：
- 知识元数据管理（CRUD）— 明文 condition/decision
- 生命周期状态管理
- 审计日志记录
- 同步消息发布到 Runtime

使用示例：
    from aga_knowledge.portal import create_portal_app
    from aga_knowledge.config import PortalConfig

    config = PortalConfig.for_development()
    app = create_portal_app(config)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
"""

from .app import create_portal_app
from .service import PortalService
from .routes import create_portal_routers
from .registry import RuntimeRegistry

__all__ = [
    "create_portal_app",
    "PortalService",
    "create_portal_routers",
    "RuntimeRegistry",
]
