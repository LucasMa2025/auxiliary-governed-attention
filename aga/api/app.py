"""
AGA REST API 应用

基于 FastAPI 的 RESTful API，为外部治理系统提供集成接口。

架构：
- app.py: 应用入口和配置
- routes.py: 路由层（HTTP 协议转换）
- service.py: 服务层（业务逻辑）
- models.py: 数据模型

使用：
    python -m aga.api --port 8081
"""
from typing import Optional, List
from contextlib import asynccontextmanager
import logging

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None

from .service import AGAService, ServiceConfig
from .routes import create_routers

logger = logging.getLogger(__name__)


def create_api_app(
    hidden_dim: int = 4096,
    bottleneck_dim: int = 64,
    num_slots: int = 100,
    num_heads: int = 32,
    persistence_enabled: bool = True,
    persistence_type: str = "sqlite",
    persistence_path: str = "./aga_api_data.db",
    writer_enabled: bool = True,
    enable_quality_assessment: bool = True,
    enable_cors: bool = True,
    cors_origins: List[str] = None,
) -> "FastAPI":
    """
    创建 AGA API 应用
    
    Args:
        hidden_dim: 隐藏层维度
        bottleneck_dim: 瓶颈层维度
        num_slots: 每个命名空间的槽位数
        num_heads: 注意力头数
        persistence_enabled: 是否启用持久化
        persistence_type: 持久化类型 (sqlite, postgres)
        persistence_path: 持久化路径
        writer_enabled: 是否启用写入器
        enable_quality_assessment: 是否启用质量评估
        enable_cors: 是否启用 CORS
        cors_origins: 允许的 CORS 来源
    
    Returns:
        FastAPI 应用实例
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn pydantic"
        )
    
    # 创建服务配置
    config = ServiceConfig(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_slots=num_slots,
        num_heads=num_heads,
        persistence_enabled=persistence_enabled,
        persistence_type=persistence_type,
        persistence_path=persistence_path,
        writer_enabled=writer_enabled,
        enable_quality_assessment=enable_quality_assessment,
    )
    
    # 创建服务实例
    service = AGAService.get_instance(config)
    
    # 生命周期管理
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动
        logger.info("AGA API starting...")
        await service.initialize()
        yield
        # 关闭
        logger.info("AGA API shutting down...")
        await service.shutdown()
    
    # 创建应用
    app = FastAPI(
        title="AGA (Auxiliary Governed Attention) API",
        description="""
## AGA 知识管理 REST API

为外部治理系统提供知识注入、生命周期管理、查询和监控接口。

### 核心功能

- **知识管理**: 注入、更新、删除知识
- **生命周期治理**: 状态转换（probationary → confirmed → deprecated/quarantined）
- **多命名空间**: 支持租户隔离
- **审计日志**: 完整的操作记录
- **健康监控**: 系统状态和统计

### 生命周期状态

| 状态 | 可靠性 | 说明 |
|------|--------|------|
| `probationary` | 0.3 | 试用期，新注入的知识 |
| `confirmed` | 1.0 | 已确认，验证通过 |
| `deprecated` | 0.1 | 已弃用，准备下线 |
| `quarantined` | 0.0 | 已隔离，不参与推理 |

### 信任层级

| 层级 | 说明 |
|------|------|
| `s0_acceleration` | 可丢失的缓存知识 |
| `s1_experience` | 可回滚的经验知识 |
| `s2_policy` | 需审批的策略知识 |
| `s3_immutable` | 只读的核心知识 |

### 架构说明

本 API 采用三层架构：
- **路由层** (routes.py): HTTP 协议转换
- **服务层** (service.py): 业务逻辑
- **持久化层**: 数据存储

服务层集成了 AGA 的 KnowledgeWriter 和 PersistenceManager。
        """,
        version="3.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # CORS 配置
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # 注册路由
    routers = create_routers(service)
    for router in routers.values():
        app.include_router(router)
    
    return app


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Start AGA REST API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 启动默认配置
    python -m aga.api
    
    # 指定端口和维度
    python -m aga.api --port 8082 --hidden-dim 2048 --bottleneck-dim 32
    
    # 禁用持久化
    python -m aga.api --no-persistence
    
    # 使用 PostgreSQL
    python -m aga.api --persistence-type postgres --db-path "postgresql://user:pass@localhost/aga"
        """
    )
    
    # 服务器配置
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind (default: 8081)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    # AGA 配置
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension (default: 4096)")
    parser.add_argument("--bottleneck-dim", type=int, default=64, help="Bottleneck dimension (default: 64)")
    parser.add_argument("--num-slots", type=int, default=100, help="Number of slots per namespace (default: 100)")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads (default: 32)")
    
    # 持久化配置
    parser.add_argument("--no-persistence", action="store_true", help="Disable persistence")
    parser.add_argument("--persistence-type", default="sqlite", choices=["sqlite", "postgres"], help="Persistence type")
    parser.add_argument("--db-path", default="./aga_api_data.db", help="Database path or connection string")
    
    # 写入器配置
    parser.add_argument("--no-writer", action="store_true", help="Disable knowledge writer")
    parser.add_argument("--no-quality-check", action="store_true", help="Disable quality assessment")
    
    # CORS 配置
    parser.add_argument("--no-cors", action="store_true", help="Disable CORS")
    parser.add_argument("--cors-origins", nargs="+", help="Allowed CORS origins")
    
    args = parser.parse_args()
    
    # 创建应用
    app = create_api_app(
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_slots=args.num_slots,
        num_heads=args.num_heads,
        persistence_enabled=not args.no_persistence,
        persistence_type=args.persistence_type,
        persistence_path=args.db_path,
        writer_enabled=not args.no_writer,
        enable_quality_assessment=not args.no_quality_check,
        enable_cors=not args.no_cors,
        cors_origins=args.cors_origins,
    )
    
    # 启动服务器
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
