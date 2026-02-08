"""
AGA HTTP 指标端点

提供 Prometheus 指标导出的 HTTP 端点。

支持：
1. FastAPI 集成
2. 独立指标服务
3. 健康检查端点

版本: v1.0
"""
import logging
from typing import Optional

from .metrics import get_metrics_registry, MetricsConfig

logger = logging.getLogger(__name__)

# 检查 FastAPI 可用性
try:
    from fastapi import FastAPI, Response, APIRouter
    from fastapi.responses import PlainTextResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None
    Response = None
    APIRouter = None


def add_metrics_endpoint(
    app: "FastAPI",
    path: str = "/metrics",
    config: Optional[MetricsConfig] = None,
    include_health: bool = True,
) -> None:
    """
    添加 Prometheus 指标端点到 FastAPI 应用
    
    Args:
        app: FastAPI 应用实例
        path: 指标端点路径
        config: 指标配置
        include_health: 是否包含健康检查端点
    
    Usage:
        ```python
        from fastapi import FastAPI
        from aga.monitoring.http import add_metrics_endpoint
        
        app = FastAPI()
        add_metrics_endpoint(app)
        
        # 访问 http://localhost:8000/metrics
        ```
    """
    if not HAS_FASTAPI:
        logger.warning(
            "FastAPI not available, metrics endpoint not added. "
            "Install with: pip install fastapi"
        )
        return
    
    registry = get_metrics_registry(config)
    
    @app.get(path, include_in_schema=False, tags=["monitoring"])
    async def prometheus_metrics():
        """
        Prometheus 指标端点
        
        返回 Prometheus 格式的指标数据。
        """
        return Response(
            content=registry.export_text(),
            media_type=registry.content_type,
        )
    
    if include_health:
        @app.get("/health", tags=["monitoring"])
        async def health_check():
            """健康检查端点"""
            return {
                "status": "healthy",
                "metrics_enabled": registry.enabled,
            }
    
    logger.info(f"Metrics endpoint added at {path}")


def create_metrics_router(
    config: Optional[MetricsConfig] = None,
    prefix: str = "",
) -> "APIRouter":
    """
    创建指标路由器
    
    用于模块化的 FastAPI 应用。
    
    Args:
        config: 指标配置
        prefix: 路由前缀
    
    Returns:
        FastAPI APIRouter
    
    Usage:
        ```python
        from fastapi import FastAPI
        from aga.monitoring.http import create_metrics_router
        
        app = FastAPI()
        metrics_router = create_metrics_router()
        app.include_router(metrics_router, prefix="/monitoring")
        
        # 访问 http://localhost:8000/monitoring/metrics
        ```
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI required for metrics router. "
            "Install with: pip install fastapi"
        )
    
    router = APIRouter(prefix=prefix, tags=["monitoring"])
    registry = get_metrics_registry(config)
    
    @router.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus 指标端点"""
        return Response(
            content=registry.export_text(),
            media_type=registry.content_type,
        )
    
    @router.get("/health")
    async def health_check():
        """健康检查端点"""
        return {
            "status": "healthy",
            "metrics_enabled": registry.enabled,
            "metrics_count": len(registry.get_all_metric_names()),
        }
    
    @router.get("/metrics/list")
    async def list_metrics():
        """列出所有已注册的指标"""
        return {
            "metrics": registry.get_all_metric_names(),
            "count": len(registry.get_all_metric_names()),
        }
    
    return router


def create_metrics_app(
    config: Optional[MetricsConfig] = None,
    title: str = "AGA Metrics",
    version: str = "1.0.0",
) -> "FastAPI":
    """
    创建独立的指标服务应用
    
    用于需要独立指标端口的场景。
    
    Args:
        config: 指标配置
        title: 应用标题
        version: 应用版本
    
    Returns:
        FastAPI 应用实例
    
    Usage:
        ```python
        from aga.monitoring.http import create_metrics_app
        import uvicorn
        
        app = create_metrics_app()
        uvicorn.run(app, host="0.0.0.0", port=9090)
        ```
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI required for metrics app. "
            "Install with: pip install fastapi uvicorn"
        )
    
    app = FastAPI(
        title=title,
        version=version,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    
    registry = get_metrics_registry(config)
    
    @app.get("/")
    async def root():
        """根路径"""
        return {
            "service": "AGA Metrics",
            "endpoints": ["/metrics", "/health"],
        }
    
    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus 指标端点"""
        return Response(
            content=registry.export_text(),
            media_type=registry.content_type,
        )
    
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "healthy"}
    
    logger.info("Standalone metrics app created")
    return app


# ==================== WSGI/ASGI 适配 ====================

class MetricsMiddleware:
    """
    ASGI 中间件，用于请求级别的指标采集
    
    Usage:
        ```python
        from fastapi import FastAPI
        from aga.monitoring.http import MetricsMiddleware
        
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)
        ```
    """
    
    def __init__(
        self,
        app,
        namespace: str = "default",
        exclude_paths: list = None,
    ):
        self.app = app
        self.namespace = namespace
        self.exclude_paths = exclude_paths or ["/metrics", "/health"]
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        path = scope.get("path", "")
        
        # 排除指定路径
        if any(path.startswith(p) for p in self.exclude_paths):
            await self.app(scope, receive, send)
            return
        
        import time
        from .metrics import get_metrics_registry
        
        registry = get_metrics_registry()
        start_time = time.perf_counter()
        status_code = 500
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if registry.enabled:
                duration = time.perf_counter() - start_time
                method = scope.get("method", "UNKNOWN")
                
                # 简化路径（移除动态部分）
                simplified_path = self._simplify_path(path)
                
                registry.get_metric("requests_total").labels(
                    namespace=self.namespace,
                    operation=f"{method}:{simplified_path}",
                    status=str(status_code)
                ).inc()
                
                registry.get_metric("request_duration_seconds").labels(
                    namespace=self.namespace,
                    operation=f"{method}:{simplified_path}"
                ).observe(duration)
    
    def _simplify_path(self, path: str) -> str:
        """简化路径，移除动态部分"""
        import re
        # 替换 UUID
        path = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}',
            path
        )
        # 替换数字 ID
        path = re.sub(r'/\d+', '/{id}', path)
        return path


# ==================== 命令行入口 ====================

def run_metrics_server(
    host: str = "0.0.0.0",
    port: int = 9090,
    config: Optional[MetricsConfig] = None,
):
    """
    运行独立的指标服务器
    
    Args:
        host: 监听地址
        port: 监听端口
        config: 指标配置
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn required for metrics server. "
            "Install with: pip install uvicorn"
        )
    
    app = create_metrics_app(config)
    logger.info(f"Starting metrics server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ==================== 导出 ====================

__all__ = [
    "add_metrics_endpoint",
    "create_metrics_router",
    "create_metrics_app",
    "MetricsMiddleware",
    "run_metrics_server",
    "HAS_FASTAPI",
]
