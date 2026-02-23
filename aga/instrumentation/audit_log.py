"""
aga/instrumentation/audit_log.py — 审计日志

记录所有知识管理操作的完整审计轨迹。
默认使用 Python logging 输出，aga-observability 可接管存储到数据库。

审计事件类型:
  - register: 注册知识
  - unregister: 移除知识
  - register_batch: 批量注册
  - attach: 挂载模型
  - detach: 卸载模型
  - clear: 清空知识
  - load_from: 从外部加载
  - config_change: 配置变更
"""
import time
import logging
import threading
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass

from .event_bus import EventBus

logger = logging.getLogger("aga.audit")


@dataclass
class AuditEntry:
    """审计条目"""
    timestamp: float
    operation: str
    details: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class AuditLog:
    """
    审计日志

    特性:
    - 始终记录到 Python logging（零依赖）
    - 内存环形缓冲区保留最近 N 条（可查询）
    - 通过 EventBus 发射审计事件（供 aga-observability 消费）
    - 支持按操作类型过滤

    使用方式:
        # 自动（AGAPlugin 内部调用）
        audit.record("register", {"id": "fact_001", "reliability": 0.9})

        # 查询
        trail = audit.query(limit=50, operation="register")

        # aga-observability 消费
        event_bus.subscribe("audit", audit_storage_handler)
    """

    def __init__(
        self,
        event_bus: EventBus,
        buffer_size: int = 5000,
        log_level: str = "INFO",
    ):
        self._bus = event_bus
        self._buffer: deque = deque(maxlen=buffer_size)
        self._log_level = getattr(logging, log_level.upper(), logging.INFO)
        self._lock = threading.Lock()

        # 统计
        self._counts: Dict[str, int] = {}

    def record(
        self,
        operation: str,
        details: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
    ):
        """
        记录审计事件

        Args:
            operation: 操作类型
            details: 操作详情
            success: 是否成功
            error: 错误信息（如果失败）
        """
        entry = AuditEntry(
            timestamp=time.time(),
            operation=operation,
            details=details,
            success=success,
            error=error,
        )

        # 1. 写入内存缓冲区
        with self._lock:
            self._buffer.append(entry)
            self._counts[operation] = self._counts.get(operation, 0) + 1

        # 2. 写入 Python logging（始终可用）
        log_data = {
            "audit": True,
            "operation": operation,
            "success": success,
            **{k: v for k, v in details.items() if not isinstance(v, (bytes, memoryview))},
        }
        if error:
            log_data["error"] = error

        logger.log(
            self._log_level if success else logging.WARNING,
            f"[AUDIT] {operation}: {details}",
            extra=log_data,
        )

        # 3. 发射到 EventBus（供 aga-observability 消费）
        self._bus.emit("audit", {
            "operation": operation,
            "details": details,
            "success": success,
            "error": error,
            "timestamp": entry.timestamp,
        })

    def query(
        self,
        limit: int = 100,
        operation: Optional[str] = None,
        since: Optional[float] = None,
        success_only: bool = False,
    ) -> List[Dict]:
        """
        查询审计日志（始终可用）

        Args:
            limit: 返回数量
            operation: 过滤操作类型
            since: 起始时间戳
            success_only: 只返回成功的
        """
        with self._lock:
            entries = list(self._buffer)

        if operation:
            entries = [e for e in entries if e.operation == operation]
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        if success_only:
            entries = [e for e in entries if e.success]

        return [
            {
                "timestamp": e.timestamp,
                "operation": e.operation,
                "details": e.details,
                "success": e.success,
                "error": e.error,
            }
            for e in entries[-limit:]
        ]

    def get_stats(self) -> Dict:
        """获取审计统计"""
        with self._lock:
            return {
                "total_entries": sum(self._counts.values()),
                "by_operation": dict(self._counts),
                "buffer_size": len(self._buffer),
                "buffer_capacity": self._buffer.maxlen,
            }
