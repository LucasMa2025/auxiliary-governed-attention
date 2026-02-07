"""
AGA 监控模块

提供生产环境所需的监控、告警和可视化功能：
1. Prometheus 告警规则生成
2. Grafana 仪表盘配置
3. 简单的 Web UI（监控和知识注入）
"""

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

# 简单 UI（可选，需要 Flask）
try:
    from .simple_ui import SimpleUIConfig, SimpleMonitorUI
    HAS_SIMPLE_UI = True
except ImportError:
    HAS_SIMPLE_UI = False
    SimpleUIConfig = None
    SimpleMonitorUI = None

__all__ = [
    # 告警
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
    # 简单 UI
    "SimpleUIConfig",
    "SimpleMonitorUI",
    "HAS_SIMPLE_UI",
]
