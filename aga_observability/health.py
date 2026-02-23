"""
aga_observability/health.py — 健康检查

提供 AGA 系统的健康状态检查，支持:
  - 组件级健康检查（KVStore / GateSystem / Retriever / EventBus）
  - 聚合健康状态
  - HTTP 端点（可选，用于 K8s liveness/readiness probe）
"""
import time
import logging
import threading
from enum import Enum
from typing import Dict, Any, Optional, List
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # 部分功能受限
    UNHEALTHY = "unhealthy"


class HealthChecker:
    """
    健康检查器

    使用方式:
        checker = HealthChecker()
        checker.bind_plugin(plugin)

        # 获取健康状态
        status = checker.check()
        # {
        #     "status": "healthy",
        #     "components": {
        #         "kv_store": {"status": "healthy", "details": {...}},
        #         "gate_system": {"status": "healthy"},
        #         "retriever": {"status": "healthy"},
        #         "event_bus": {"status": "healthy"},
        #     },
        #     "timestamp": 1234567890.0,
        # }

        # 启动 HTTP 端点
        checker.start_server(port=8080, path="/health")
    """

    def __init__(self):
        self._plugin = None
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._custom_checks: Dict[str, callable] = {}

    def bind_plugin(self, plugin) -> None:
        """绑定 AGAPlugin 实例"""
        self._plugin = plugin

    def add_check(self, name: str, check_fn: callable) -> None:
        """添加自定义健康检查"""
        self._custom_checks[name] = check_fn

    def check(self) -> Dict[str, Any]:
        """
        执行健康检查

        Returns:
            健康状态字典
        """
        components = {}
        overall = HealthStatus.HEALTHY

        if self._plugin:
            # KVStore 检查
            components["kv_store"] = self._check_kv_store()

            # GateSystem 检查
            components["gate_system"] = self._check_gate_system()

            # Retriever 检查
            components["retriever"] = self._check_retriever()

            # EventBus 检查
            components["event_bus"] = self._check_event_bus()

            # Attachment 检查
            components["attachment"] = self._check_attachment()

        # 自定义检查
        for name, check_fn in self._custom_checks.items():
            try:
                components[name] = check_fn()
            except Exception as e:
                components[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(e),
                }

        # 聚合状态
        for comp_name, comp_status in components.items():
            status_str = comp_status.get("status", "healthy")
            if status_str == HealthStatus.UNHEALTHY.value:
                overall = HealthStatus.UNHEALTHY
                break
            elif status_str == HealthStatus.DEGRADED.value:
                if overall != HealthStatus.UNHEALTHY:
                    overall = HealthStatus.DEGRADED

        return {
            "status": overall.value,
            "components": components,
            "timestamp": time.time(),
        }

    def _check_kv_store(self) -> Dict[str, Any]:
        """检查 KVStore"""
        try:
            stats = self._plugin.get_store_stats()
            utilization = stats.get("utilization", 0.0)

            status = HealthStatus.HEALTHY
            if utilization > 0.95:
                status = HealthStatus.DEGRADED

            return {
                "status": status.value,
                "count": stats.get("count", 0),
                "max_slots": stats.get("max_slots", 0),
                "utilization": utilization,
                "pinned": stats.get("pinned_count", 0),
            }
        except Exception as e:
            return {"status": HealthStatus.UNHEALTHY.value, "error": str(e)}

    def _check_gate_system(self) -> Dict[str, Any]:
        """检查 GateSystem"""
        try:
            # 验证 gate_system 存在且参数正常
            gs = self._plugin.gate_system
            param_count = sum(p.numel() for p in gs.parameters())
            has_nan = any(p.isnan().any().item() for p in gs.parameters())

            if has_nan:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": "GateSystem 参数包含 NaN",
                    "param_count": param_count,
                }

            return {
                "status": HealthStatus.HEALTHY.value,
                "param_count": param_count,
            }
        except Exception as e:
            return {"status": HealthStatus.UNHEALTHY.value, "error": str(e)}

    def _check_retriever(self) -> Dict[str, Any]:
        """检查 Retriever"""
        try:
            retriever = self._plugin.retriever
            stats = retriever.get_stats()
            return {
                "status": HealthStatus.HEALTHY.value,
                "type": type(retriever).__name__,
                **stats,
            }
        except Exception as e:
            return {"status": HealthStatus.DEGRADED.value, "error": str(e)}

    def _check_event_bus(self) -> Dict[str, Any]:
        """检查 EventBus"""
        try:
            stats = self._plugin.event_bus.get_stats()
            buffer_usage = stats["buffer_size"] / max(stats["buffer_capacity"], 1)

            status = HealthStatus.HEALTHY
            if buffer_usage > 0.9:
                status = HealthStatus.DEGRADED

            return {
                "status": status.value,
                "enabled": self._plugin.event_bus.enabled,
                "buffer_usage": buffer_usage,
                **stats,
            }
        except Exception as e:
            return {"status": HealthStatus.UNHEALTHY.value, "error": str(e)}

    def _check_attachment(self) -> Dict[str, Any]:
        """检查模型挂载状态"""
        try:
            attached = self._plugin.is_attached
            return {
                "status": HealthStatus.HEALTHY.value if attached else HealthStatus.DEGRADED.value,
                "attached": attached,
                "model_type": self._plugin._attached_model_name,
                "hooks_count": len(self._plugin._hooks),
            }
        except Exception as e:
            return {"status": HealthStatus.UNHEALTHY.value, "error": str(e)}

    def start_server(self, port: int = 8080, path: str = "/health") -> None:
        """
        启动 HTTP 健康检查端点

        Args:
            port: 端口号
            path: 路径
        """
        if self._server:
            logger.warning("健康检查 HTTP 端点已启动")
            return

        checker = self

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == path:
                    result = checker.check()
                    status_code = 200 if result["status"] != "unhealthy" else 503

                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(result, indent=2, default=str).encode("utf-8")
                    )
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # 静默 HTTP 日志

        try:
            self._server = HTTPServer(("0.0.0.0", port), HealthHandler)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
            )
            self._server_thread.start()
            logger.info(f"健康检查 HTTP 端点已启动: http://0.0.0.0:{port}{path}")
        except Exception as e:
            logger.error(f"健康检查 HTTP 端点启动失败: {e}")

    def shutdown(self) -> None:
        """关闭"""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._server_thread = None
        logger.info("HealthChecker 已关闭")
