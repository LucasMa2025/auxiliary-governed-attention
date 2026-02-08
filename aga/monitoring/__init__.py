"""
AGA 监控模块

提供生产环境所需的完整监控能力：

1. **指标采集** (metrics.py)
   - 统一指标注册中心 MetricsRegistry
   - Prometheus Counter/Gauge/Histogram 支持
   - 线程安全，优雅降级

2. **采集装饰器** (decorators.py)
   - @track_latency - 延迟追踪
   - @track_errors - 错误追踪
   - track_ann_search - ANN 搜索追踪
   - track_loader - 动态加载器追踪

3. **HTTP 端点** (http.py)
   - /metrics - Prometheus 指标端点
   - FastAPI 集成
   - 独立指标服务

4. **告警规则** (alerts.py)
   - Prometheus 告警规则生成
   - Grafana 仪表盘配置
   - AlertManager 本地告警

5. **结构化日志** (logging_utils.py)
   - JSON 格式输出
   - 请求上下文
   - ELK/Loki 兼容

6. **SLO/SLI** (slo.py)
   - 服务级别目标定义
   - 错误预算告警
   - SLO 仪表盘

7. **简单 UI** (simple_ui.py)
   - Flask 轻量级监控界面
   - 知识注入管理

版本: v2.0
"""

# ==================== 指标采集 ====================

from .metrics import (
    MetricsConfig,
    MetricType,
    MetricsRegistry,
    NoOpMetric,
    get_metrics_registry,
    reset_metrics_registry,
    inc_counter,
    set_gauge,
    observe_histogram,
)

# ==================== 采集装饰器 ====================

from .decorators import (
    # 装饰器
    track_latency,
    track_errors,
    # 上下文管理器
    track_operation,
    track_ann_search,
    track_loader,
    track_gate,
    track_persistence,
    async_track_operation,
    # 辅助函数
    record_slot_metrics,
    record_ann_index_metrics,
    record_cache_metrics,
)

# ==================== HTTP 端点 ====================

from .http import (
    add_metrics_endpoint,
    create_metrics_router,
    create_metrics_app,
    MetricsMiddleware,
    run_metrics_server,
    HAS_FASTAPI,
)

# ==================== 告警规则 ====================

from .alerts import (
    AlertSeverity,
    AlertState,
    AlertRule,
    AGA_ALERT_RULES,
    generate_prometheus_rules,
    generate_prometheus_rules_yaml,
    GrafanaPanel,
    create_aga_dashboard,
    export_dashboard_json,
    AlertManager,
)

# ==================== 结构化日志 ====================

from .logging_utils import (
    LoggingConfig,
    setup_logging,
    get_logger,
    LogContext,
    log_context,
    generate_trace_id,
    StructuredFormatter,
    TextFormatter,
    log_with_context,
    log_request,
    log_operation,
)

# ==================== SLO/SLI ====================

from .slo import (
    SLOType,
    SLOWindow,
    SLO,
    AGA_SLOS,
    generate_slo_recording_rules,
    generate_slo_alerts,
    generate_slo_rules_yaml,
    create_slo_dashboard_panels,
    export_slo_dashboard,
)

# ==================== 简单 UI ====================

try:
    from .simple_ui import SimpleUIConfig, SimpleMonitorUI
    HAS_SIMPLE_UI = True
except ImportError:
    HAS_SIMPLE_UI = False
    SimpleUIConfig = None
    SimpleMonitorUI = None


# ==================== 导出 ====================

__all__ = [
    # 指标采集
    "MetricsConfig",
    "MetricType",
    "MetricsRegistry",
    "NoOpMetric",
    "get_metrics_registry",
    "reset_metrics_registry",
    "inc_counter",
    "set_gauge",
    "observe_histogram",
    
    # 采集装饰器
    "track_latency",
    "track_errors",
    "track_operation",
    "track_ann_search",
    "track_loader",
    "track_gate",
    "track_persistence",
    "async_track_operation",
    "record_slot_metrics",
    "record_ann_index_metrics",
    "record_cache_metrics",
    
    # HTTP 端点
    "add_metrics_endpoint",
    "create_metrics_router",
    "create_metrics_app",
    "MetricsMiddleware",
    "run_metrics_server",
    "HAS_FASTAPI",
    
    # 告警规则
    "AlertSeverity",
    "AlertState",
    "AlertRule",
    "AGA_ALERT_RULES",
    "generate_prometheus_rules",
    "generate_prometheus_rules_yaml",
    "AlertManager",
    
    # Grafana
    "GrafanaPanel",
    "create_aga_dashboard",
    "export_dashboard_json",
    
    # 结构化日志
    "LoggingConfig",
    "setup_logging",
    "get_logger",
    "LogContext",
    "log_context",
    "generate_trace_id",
    "StructuredFormatter",
    "TextFormatter",
    "log_with_context",
    "log_request",
    "log_operation",
    
    # SLO/SLI
    "SLOType",
    "SLOWindow",
    "SLO",
    "AGA_SLOS",
    "generate_slo_recording_rules",
    "generate_slo_alerts",
    "generate_slo_rules_yaml",
    "create_slo_dashboard_panels",
    "export_slo_dashboard",
    
    # 简单 UI
    "SimpleUIConfig",
    "SimpleMonitorUI",
    "HAS_SIMPLE_UI",
]
