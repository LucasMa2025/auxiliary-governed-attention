"""
aga-observability — AGA 可观测性附加包

为 aga-core 提供生产级可观测性能力:
  - Prometheus 指标导出（Counter / Histogram / Gauge）
  - Grafana Dashboard 自动生成
  - SLO/SLI 告警管理
  - 结构化日志导出（JSON / Text）
  - 审计日志持久化（File / SQLite）
  - 健康检查端点

设计原则:
  1. 零侵入: 通过 EventBus 订阅事件，不修改 aga-core 代码
  2. 自动集成: pip install aga-observability 后 AGAPlugin 自动检测并启用
  3. Fail-Open: 可观测性组件故障不影响推理
  4. 配置驱动: 所有行为通过 AGAConfig 中的 observability_* 字段控制

使用方式:
    # 自动集成（推荐）
    # pip install aga-observability 后，AGAPlugin 初始化时自动调用:
    # from aga_observability import setup_observability
    # setup_observability(event_bus, config)

    # 手动集成
    from aga_observability import ObservabilityStack
    stack = ObservabilityStack(event_bus=plugin.event_bus, config=plugin.config)
    stack.start()
    # ... 推理 ...
    stack.shutdown()
"""

__version__ = "1.0.0"

from .config import ObservabilityConfig
from .stack import ObservabilityStack
from .integration import setup_observability
from .prometheus_exporter import PrometheusExporter
from .grafana_dashboard import GrafanaDashboardGenerator
from .alert_manager import AlertManager, AlertRule, AlertSeverity
from .log_exporter import LogExporter
from .audit_storage import AuditStorageBackend, FileAuditStorage, SQLiteAuditStorage
from .health import HealthChecker, HealthStatus

__all__ = [
    # 核心入口
    "setup_observability",
    "ObservabilityStack",
    "ObservabilityConfig",
    # Prometheus
    "PrometheusExporter",
    # Grafana
    "GrafanaDashboardGenerator",
    # 告警
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    # 日志
    "LogExporter",
    # 审计持久化
    "AuditStorageBackend",
    "FileAuditStorage",
    "SQLiteAuditStorage",
    # 健康检查
    "HealthChecker",
    "HealthStatus",
]
