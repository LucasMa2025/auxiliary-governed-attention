"""
aga_observability/stack.py — 可观测性组件编排

ObservabilityStack 是所有可观测性组件的编排器:
  - PrometheusExporter
  - LogExporter
  - AuditStorage (File / SQLite)
  - AlertManager
  - HealthChecker
  - GaugeUpdater (定时更新 Gauge 指标)

它根据配置自动创建、连接和管理所有组件的生命周期。
"""
import time
import logging
import threading
from typing import Optional, Dict, Any

from .config import ObservabilityConfig
from .prometheus_exporter import PrometheusExporter, _PROMETHEUS_AVAILABLE
from .grafana_dashboard import GrafanaDashboardGenerator
from .alert_manager import AlertManager, AlertRule, AlertSeverity
from .log_exporter import LogExporter
from .audit_storage import (
    AuditStorageBackend,
    FileAuditStorage,
    SQLiteAuditStorage,
)
from .health import HealthChecker

logger = logging.getLogger(__name__)


class ObservabilityStack:
    """
    可观测性组件编排器

    使用方式:
        # 自动模式（推荐，由 setup_observability 调用）
        stack = ObservabilityStack(event_bus=plugin.event_bus, config=obs_config)
        stack.start()
        stack.bind_plugin(plugin)

        # 手动模式
        stack = ObservabilityStack(event_bus=event_bus)
        stack.prometheus.start_server()
        stack.health.start_server()

        # 关闭
        stack.shutdown()
    """

    def __init__(
        self,
        event_bus,
        config: Optional[ObservabilityConfig] = None,
    ):
        self._event_bus = event_bus
        self.config = config or ObservabilityConfig()
        self._plugin = None
        self._gauge_timer: Optional[threading.Timer] = None
        self._running = False

        # 组件实例（按需创建）
        self.prometheus: Optional[PrometheusExporter] = None
        self.log_exporter: Optional[LogExporter] = None
        self.audit_storage: Optional[AuditStorageBackend] = None
        self.alert_manager: Optional[AlertManager] = None
        self.health: Optional[HealthChecker] = None
        self.dashboard_generator: Optional[GrafanaDashboardGenerator] = None

        # 根据配置创建组件
        self._create_components()

    def _create_components(self) -> None:
        """根据配置创建组件"""
        cfg = self.config

        # 1. Prometheus
        if cfg.prometheus_enabled and _PROMETHEUS_AVAILABLE:
            try:
                self.prometheus = PrometheusExporter(
                    prefix=cfg.prometheus_prefix,
                    port=cfg.prometheus_port,
                    labels=cfg.prometheus_labels,
                )
            except Exception as e:
                logger.warning(f"PrometheusExporter 创建失败: {e}")

        # 2. LogExporter
        if cfg.log_enabled:
            try:
                self.log_exporter = LogExporter(
                    format=cfg.log_format,
                    level=cfg.log_level,
                    file=cfg.log_file,
                    max_bytes=cfg.log_max_bytes,
                    backup_count=cfg.log_backup_count,
                )
            except Exception as e:
                logger.warning(f"LogExporter 创建失败: {e}")

        # 3. AuditStorage
        if cfg.audit_storage_backend != "memory":
            try:
                if cfg.audit_storage_backend == "file":
                    self.audit_storage = FileAuditStorage(
                        path=cfg.audit_storage_path.replace(".db", ".jsonl"),
                        flush_interval=cfg.audit_flush_interval,
                        batch_size=cfg.audit_batch_size,
                    )
                elif cfg.audit_storage_backend == "sqlite":
                    self.audit_storage = SQLiteAuditStorage(
                        path=cfg.audit_storage_path,
                        flush_interval=cfg.audit_flush_interval,
                        batch_size=cfg.audit_batch_size,
                    )
            except Exception as e:
                logger.warning(f"AuditStorage 创建失败: {e}")

        # 4. AlertManager
        if cfg.alert_enabled:
            try:
                rules = []
                for rule_cfg in cfg.alert_rules:
                    rules.append(AlertRule(
                        name=rule_cfg.name,
                        metric=rule_cfg.metric,
                        operator=rule_cfg.operator,
                        threshold=rule_cfg.threshold,
                        window_seconds=rule_cfg.window_seconds,
                        severity=AlertSeverity(rule_cfg.severity),
                        message=rule_cfg.message,
                        cooldown_seconds=rule_cfg.cooldown_seconds,
                    ))

                self.alert_manager = AlertManager(
                    rules=rules,
                    use_defaults=True,
                    webhook_url=cfg.alert_webhook_url,
                )
            except Exception as e:
                logger.warning(f"AlertManager 创建失败: {e}")

        # 5. HealthChecker
        if cfg.health_enabled:
            self.health = HealthChecker()

        # 6. Dashboard Generator（始终可用）
        self.dashboard_generator = GrafanaDashboardGenerator(
            prefix=cfg.prometheus_prefix if cfg.prometheus_enabled else "aga",
        )

    def start(self) -> None:
        """启动所有组件"""
        if self._running:
            logger.warning("ObservabilityStack 已在运行")
            return

        self._running = True

        # 订阅 EventBus
        if self.prometheus:
            self.prometheus.subscribe(self._event_bus)
        if self.log_exporter:
            self.log_exporter.subscribe(self._event_bus)
        if self.audit_storage:
            self.audit_storage.subscribe(self._event_bus)
        if self.alert_manager:
            self.alert_manager.subscribe(self._event_bus)

        # 启动 Prometheus HTTP 端点
        if self.prometheus and self.config.prometheus_enabled:
            try:
                self.prometheus.start_server()
            except Exception as e:
                logger.warning(f"Prometheus HTTP 端点启动失败: {e}")

        # 启动健康检查 HTTP 端点
        if self.health and self.config.health_enabled:
            try:
                self.health.start_server(
                    port=self.config.health_port,
                    path=self.config.health_path,
                )
            except Exception as e:
                logger.warning(f"健康检查 HTTP 端点启动失败: {e}")

        # 启动 Gauge 定时更新
        self._start_gauge_updater()

        logger.info(
            f"ObservabilityStack 已启动: "
            f"prometheus={'✓' if self.prometheus else '✗'}, "
            f"log={'✓' if self.log_exporter else '✗'}, "
            f"audit={'✓' if self.audit_storage else '✗'}, "
            f"alert={'✓' if self.alert_manager else '✗'}, "
            f"health={'✓' if self.health else '✗'}"
        )

    def bind_plugin(self, plugin) -> None:
        """
        绑定 AGAPlugin 实例

        绑定后可以:
        - 定时更新 Gauge 指标
        - 健康检查访问 plugin 状态
        - AlertManager 获取 plugin 指标

        Args:
            plugin: AGAPlugin 实例
        """
        self._plugin = plugin

        if self.health:
            self.health.bind_plugin(plugin)

        if self.prometheus:
            try:
                from aga import __version__ as aga_version
            except ImportError:
                aga_version = "unknown"

            self.prometheus.set_build_info({
                "version": aga_version,
                "hidden_dim": str(plugin.config.hidden_dim),
                "bottleneck_dim": str(plugin.config.bottleneck_dim),
                "max_slots": str(plugin.config.max_slots),
                "device": str(plugin.device),
            })

        logger.info("ObservabilityStack 已绑定 AGAPlugin")

    def _start_gauge_updater(self) -> None:
        """启动 Gauge 定时更新"""
        if not self._running:
            return

        if self._plugin:
            # 更新 Prometheus Gauge
            if self.prometheus:
                self.prometheus.update_gauges(self._plugin)

            # 更新 AlertManager 指标
            if self.alert_manager:
                self.alert_manager.update_plugin_metrics(self._plugin)

        # 每 5 秒更新一次
        self._gauge_timer = threading.Timer(5.0, self._start_gauge_updater)
        self._gauge_timer.daemon = True
        self._gauge_timer.start()

    def shutdown(self) -> None:
        """关闭所有组件"""
        self._running = False

        if self._gauge_timer:
            self._gauge_timer.cancel()

        if self.prometheus:
            self.prometheus.unsubscribe(self._event_bus)
            self.prometheus.shutdown()
        if self.log_exporter:
            self.log_exporter.unsubscribe(self._event_bus)
            self.log_exporter.shutdown()
        if self.audit_storage:
            self.audit_storage.unsubscribe(self._event_bus)
            self.audit_storage.shutdown()
        if self.alert_manager:
            self.alert_manager.unsubscribe(self._event_bus)
            self.alert_manager.shutdown()
        if self.health:
            self.health.shutdown()

        logger.info("ObservabilityStack 已关闭")

    def get_stats(self) -> Dict[str, Any]:
        """获取所有组件统计"""
        stats = {"running": self._running}

        if self.prometheus:
            stats["prometheus"] = self.prometheus.get_stats()
        if self.log_exporter:
            stats["log_exporter"] = self.log_exporter.get_stats()
        if self.audit_storage:
            stats["audit_storage"] = self.audit_storage.get_stats()
        if self.alert_manager:
            stats["alert_manager"] = self.alert_manager.get_stats()

        return stats

    def generate_dashboard(self, path: Optional[str] = None) -> str:
        """
        生成 Grafana Dashboard

        Args:
            path: 保存路径（None=仅返回 JSON 字符串）

        Returns:
            Dashboard JSON 字符串
        """
        if path:
            self.dashboard_generator.save(path)
        return self.dashboard_generator.generate()
