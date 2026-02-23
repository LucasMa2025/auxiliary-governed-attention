"""
aga/instrumentation/ — 内置埋点层（零外部依赖）

提供:
  - EventBus: 可插拔事件总线
  - ForwardMetrics: Forward 指标收集器
  - AuditLog: 审计日志
"""
from .event_bus import EventBus, Event
from .forward_metrics import ForwardMetrics
from .audit_log import AuditLog, AuditEntry

__all__ = [
    "EventBus",
    "Event",
    "ForwardMetrics",
    "AuditLog",
    "AuditEntry",
]
