"""
AGA Knowledge Portal FastAPI 应用（明文 KV 版本）

创建独立部署的 Portal API 服务。
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, APIRouter
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None
    StaticFiles = None

from ..config import PortalConfig
from .service import PortalService
from .routes import create_portal_routers
from .registry import RuntimeRegistry


def create_portal_app(
    config: PortalConfig = None,
    title: str = "AGA Knowledge Portal",
    description: str = None,
) -> "FastAPI":
    """
    创建 Portal FastAPI 应用

    Args:
        config: Portal 配置
        title: API 标题
        description: API 描述

    Returns:
        FastAPI 应用实例

    示例:
        from aga_knowledge.portal import create_portal_app
        from aga_knowledge.config import PortalConfig

        config = PortalConfig.for_production(
            postgres_url="postgresql://...",
            redis_host="localhost",
        )
        app = create_portal_app(config)

        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8081)
    """
    if not HAS_FASTAPI:
        raise ImportError("需要安装 FastAPI: pip install fastapi uvicorn")

    config = config or PortalConfig.for_development()

    if description is None:
        description = """
## AGA Knowledge Portal — 独立部署的知识管理 API（明文 KV 版本）

AGA Knowledge Portal 是 AGA 系统的知识管理入口，无需 GPU 资源。

### 架构特点

- **无 GPU 依赖**: Portal 只管理明文 condition/decision 元数据
- **分离部署**: Portal 与 AGA Runtime 独立部署
- **消息同步**: 通过 Redis 将变更同步到 Runtime
- **水平扩展**: 支持多 Portal 实例负载均衡
- **明文 KV**: 不涉及向量化编码，Runtime 推理时按需处理

### 数据流

```
外部治理系统 → Portal API → 持久化存储 (SQLite/PostgreSQL)
                    ↓
              消息队列 (Redis/Memory)
                    ↓
         AGA Runtime #1, #2, ..., #N
         (推理时按需编码 condition/decision → KV 向量)
```

### 主要功能

- **知识管理**: 注入、查询、删除知识（明文 condition/decision）
- **生命周期管理**: 状态转换、隔离
- **审计日志**: 完整的操作记录
- **统计信息**: 知识分布、命中统计

### 生命周期状态

| 状态 | 说明 | 可靠性 |
|------|------|--------|
| probationary | 试用期 | 0.3 |
| confirmed | 已确认 | 1.0 |
| deprecated | 已弃用 | 0.1 |
| quarantined | 已隔离 | 0.0 |

### 信任层级

| 层级 | 说明 | 优先级 |
|------|------|--------|
| system | 系统级 | 100 |
| verified | 已验证 | 80 |
| standard | 标准级 | 50 |
| experimental | 实验性 | 30 |
| untrusted | 不可信 | 10 |
"""

    # 创建服务和注册表
    service = PortalService(config)
    registry = RuntimeRegistry(
        heartbeat_timeout=config.registry.timeout,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        # 启动
        await service.initialize()
        await registry.start()
        logger.info("Portal application started (plaintext KV mode)")

        yield

        # 关闭
        await registry.stop()
        await service.shutdown()
        logger.info("Portal application shutdown")

    # 创建 FastAPI 应用
    app = FastAPI(
        title=title,
        description=description,
        version=config.version,
        lifespan=lifespan,
    )

    # CORS
    if config.server.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 注册路由
    routers = create_portal_routers(service)
    for router in routers.values():
        app.include_router(router)

    # 添加 Runtime 注册路由
    runtime_router = APIRouter(prefix="/runtimes", tags=["Runtimes"])

    @runtime_router.get("")
    async def list_runtimes():
        """列出所有 Runtime"""
        return registry.get_stats()

    @runtime_router.post("/register")
    async def register_runtime(
        instance_id: str,
        namespaces: str,  # 逗号分隔
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """注册 Runtime"""
        ns_list = namespaces.split(",")
        registry.register(instance_id, ns_list, host, port)
        return {"success": True, "instance_id": instance_id}

    @runtime_router.post("/heartbeat/{instance_id}")
    async def runtime_heartbeat(instance_id: str):
        """Runtime 心跳"""
        registry.heartbeat(instance_id)
        return {"success": True}

    @runtime_router.delete("/{instance_id}")
    async def deregister_runtime(instance_id: str):
        """注销 Runtime"""
        registry.deregister(instance_id)
        return {"success": True}

    app.include_router(runtime_router)

    # 挂载静态资源目录（图片等）
    if config.image_handling.enabled and config.image_handling.asset_dir:
        asset_path = Path(config.image_handling.asset_dir)
        if asset_path.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(asset_path)),
                name="knowledge_assets",
            )
            logger.info(
                f"静态资源目录已挂载: /assets → {asset_path}"
            )
        else:
            logger.warning(
                f"静态资源目录不存在: {asset_path}，"
                f"图片访问将不可用。"
                f"请创建目录或修改 image_handling.asset_dir 配置"
            )

    # 存储服务引用
    app.state.service = service
    app.state.registry = registry
    app.state.config = config

    return app


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="AGA Knowledge Portal Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8081, help="监听端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--config", help="配置文件路径")

    # 持久化
    parser.add_argument("--persistence", default="sqlite", choices=["sqlite", "postgres", "memory"])
    parser.add_argument("--sqlite-path", default="aga_knowledge.db")
    parser.add_argument("--postgres-url")

    # 消息队列
    parser.add_argument("--messaging", default="memory", choices=["memory", "redis"])
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)

    # 环境
    parser.add_argument("--env", default="development", choices=["development", "staging", "production"])

    args = parser.parse_args()

    # 加载或创建配置
    if args.config:
        from ..config import load_config
        config = load_config(args.config)
    else:
        from ..config import (
            PortalConfig, ServerConfig, PersistenceDBConfig, MessagingConfig
        )

        config = PortalConfig(
            server=ServerConfig(
                host=args.host,
                port=args.port,
                workers=args.workers,
                reload=args.reload,
            ),
            persistence=PersistenceDBConfig(
                type=args.persistence,
                sqlite_path=args.sqlite_path,
                postgres_url=args.postgres_url,
            ),
            messaging=MessagingConfig(
                backend=args.messaging,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
            ),
            environment=args.env,
        )

    # 创建应用
    app = create_portal_app(config)

    # 启动服务器
    try:
        import uvicorn
    except ImportError:
        raise ImportError("需要安装 uvicorn: pip install uvicorn")

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers if not config.server.reload else 1,
        reload=config.server.reload,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()
