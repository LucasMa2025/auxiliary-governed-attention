"""
AGA Portal - 独立部署的 API 服务

Portal 是 AGA 系统的知识管理入口，无需 GPU。

主要职责：
- 知识元数据管理（CRUD）
- 生命周期状态管理
- 审计日志记录
- 同步消息发布到 Runtime

架构特点：
- 无 AGA 推理实例（不需要 GPU）
- 持久化存储（PostgreSQL/SQLite）
- 消息队列发布（Redis/Kafka）
- 水平可扩展

使用示例：

```python
from aga.portal import create_portal_app, PortalService
from aga.config import PortalConfig

# 创建 Portal
config = PortalConfig.for_production(
    postgres_url="postgresql://...",
    redis_host="localhost",
)
app = create_portal_app(config)

# 或直接使用服务
service = PortalService(config)
await service.initialize()
result = await service.inject_knowledge(...)
```
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
