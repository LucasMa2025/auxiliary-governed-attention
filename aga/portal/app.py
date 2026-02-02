"""
AGA Portal FastAPI 应用

创建独立部署的 Portal API 服务。
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None

from ..config.portal import PortalConfig
from .service import PortalService
from .routes import create_portal_routers
from .registry import RuntimeRegistry, RedisRuntimeRegistry


def create_portal_app(
    config: PortalConfig = None,
    title: str = "AGA Portal",
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
        from aga.portal import create_portal_app
        from aga.config import PortalConfig
        
        config = PortalConfig.for_production(
            postgres_url="postgresql://...",
            redis_host="localhost",
        )
        app = create_portal_app(config)
        
        # 启动
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8081)
    """
    if not HAS_FASTAPI:
        raise ImportError("需要安装 FastAPI: pip install fastapi uvicorn")
    
    config = config or PortalConfig.for_development()
    
    if description is None:
        description = """
## AGA Portal - 独立部署的知识管理 API

AGA Portal 是 AGA 系统的知识管理入口，无需 GPU 资源。

### 架构特点

- **无 GPU 依赖**: Portal 只管理元数据，不执行推理
- **分离部署**: Portal 与 AGA Runtime 独立部署
- **消息同步**: 通过 Redis/Kafka 将变更同步到 Runtime
- **水平扩展**: 支持多 Portal 实例负载均衡

### 数据流

```
外部治理系统 → Portal API → 持久化存储
                    ↓
              消息队列 (Redis/Kafka)
                    ↓
         AGA Runtime #1, #2, ..., #N
```

### 主要功能

- **知识管理**: 注入、查询、删除知识
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

| 层级 | 说明 | 风险 |
|------|------|------|
| s0_acceleration | 加速型 | 低 |
| s1_experience | 经验型 | 中低 |
| s2_policy | 策略型 | 中高 |
| s3_immutable | 不可变 | 高 |
"""
    
    # 创建服务和注册表
    service = PortalService(config)
    
    if config.registry.type == "redis":
        registry = RedisRuntimeRegistry(
            redis_host=config.messaging.redis_host,
            redis_port=config.messaging.redis_port,
            heartbeat_timeout=config.registry.timeout,
        )
    else:
        registry = RuntimeRegistry(
            heartbeat_timeout=config.registry.timeout,
        )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        # 启动
        await service.initialize()
        await registry.start()
        logger.info("Portal application started")
        
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
    from fastapi import APIRouter
    
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
    
    # 存储服务引用
    app.state.service = service
    app.state.registry = registry
    app.state.config = config
    
    return app


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGA Portal Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8081, help="监听端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--config", help="配置文件路径")
    
    # 持久化
    parser.add_argument("--persistence", default="sqlite", choices=["sqlite", "postgres", "memory"])
    parser.add_argument("--sqlite-path", default="aga_portal.db")
    parser.add_argument("--postgres-url")
    
    # 消息队列
    parser.add_argument("--messaging", default="memory", choices=["memory", "redis", "kafka"])
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    
    # 环境
    parser.add_argument("--env", default="development", choices=["development", "staging", "production"])
    
    args = parser.parse_args()
    
    # 加载或创建配置
    if args.config:
        from ..config import load_config
        config = load_config(args.config, PortalConfig)
    else:
        from ..config.portal import (
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
