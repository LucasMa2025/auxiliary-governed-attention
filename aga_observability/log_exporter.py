"""
aga_observability/log_exporter.py — 结构化日志导出

将 EventBus 事件转换为结构化日志输出。

支持格式:
  - JSON: 每行一个 JSON 对象（适合 ELK / Loki）
  - Text: 人类可读的文本格式

输出目标:
  - stderr: 标准错误输出（默认）
  - file: 文件输出（支持 rotation）
"""
import json
import time
import logging
import logging.handlers
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LogExporter:
    """
    结构化日志导出器

    使用方式:
        exporter = LogExporter(format="json", file="aga.log")
        exporter.subscribe(event_bus)
        # ... 推理 ...
        exporter.shutdown()
    """

    SUBSCRIBER_ID = "aga-observability-log"

    def __init__(
        self,
        format: str = "json",
        level: str = "INFO",
        file: Optional[str] = None,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
        logger_name: str = "aga.observability",
    ):
        """
        Args:
            format: 日志格式 ("json" / "text")
            level: 日志级别
            file: 日志文件路径（None=仅 stderr）
            max_bytes: 单文件最大大小
            backup_count: 保留的备份文件数
            logger_name: Logger 名称
        """
        self._format = format
        self._level = getattr(logging, level.upper(), logging.INFO)
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(self._level)
        self._logger.propagate = False  # 不传播到根 logger

        # 清除已有 handler（避免重复添加）
        self._logger.handlers.clear()

        # 创建 formatter
        if format == "json":
            formatter = _JsonFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(event_type)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )

        # stderr handler（始终添加）
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(self._level)
        self._logger.addHandler(stderr_handler)

        # file handler（可选）
        if file:
            file_handler = logging.handlers.RotatingFileHandler(
                file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self._level)
            self._logger.addHandler(file_handler)

        self._subscribed = False
        self._event_count = 0

    def subscribe(self, event_bus) -> None:
        """订阅 EventBus 所有事件"""
        event_bus.subscribe(
            "*",
            self._on_event,
            subscriber_id=self.SUBSCRIBER_ID,
        )
        self._subscribed = True
        logger.info(f"LogExporter 已订阅 EventBus (format={self._format})")

    def unsubscribe(self, event_bus) -> None:
        """取消订阅"""
        event_bus.unsubscribe(
            "*",
            subscriber_id=self.SUBSCRIBER_ID,
        )
        self._subscribed = False

    def _on_event(self, event) -> None:
        """处理所有事件"""
        try:
            self._event_count += 1

            # 根据事件类型选择日志级别
            level = self._get_event_level(event)

            # 构建日志记录
            extra = {
                "event_type": event.type,
                "event_source": event.source,
                "event_timestamp": event.timestamp,
                "event_data": event.data,
            }

            # 生成消息
            message = self._format_event(event)

            self._logger.log(level, message, extra=extra)

        except Exception as e:
            # 日志导出不应影响推理
            pass

    def _get_event_level(self, event) -> int:
        """根据事件类型和内容确定日志级别"""
        if event.type == "audit":
            success = event.data.get("success", True)
            return logging.WARNING if not success else logging.INFO
        elif event.type == "forward":
            return logging.DEBUG  # forward 事件量大，使用 DEBUG
        elif event.type == "retrieval":
            return logging.INFO
        return logging.INFO

    def _format_event(self, event) -> str:
        """格式化事件为消息字符串"""
        if event.type == "forward":
            d = event.data
            applied = "✓" if d.get("aga_applied") else "✗"
            return (
                f"layer={d.get('layer_idx', '?')} "
                f"applied={applied} "
                f"gate={d.get('gate_mean', 0):.3f} "
                f"entropy={d.get('entropy_mean', 0):.3f} "
                f"latency={d.get('latency_us', 0):.1f}μs"
            )
        elif event.type == "retrieval":
            d = event.data
            return (
                f"layer={d.get('layer_idx', '?')} "
                f"results={d.get('results_count', 0)} "
                f"injected={d.get('injected_count', 0)} "
                f"budget_left={d.get('budget_remaining', 0)}"
            )
        elif event.type == "audit":
            d = event.data
            op = d.get("operation", "?")
            success = "OK" if d.get("success", True) else "FAIL"
            details = d.get("details", {})
            return f"[{success}] {op}: {details}"
        else:
            return str(event.data)

    def get_stats(self) -> Dict[str, Any]:
        """获取导出器统计"""
        return {
            "subscribed": self._subscribed,
            "format": self._format,
            "event_count": self._event_count,
            "handlers": len(self._logger.handlers),
        }

    def shutdown(self) -> None:
        """关闭并刷新所有 handler"""
        for handler in self._logger.handlers:
            handler.flush()
            handler.close()
        self._logger.handlers.clear()
        logger.info("LogExporter 已关闭")


class _JsonFormatter(logging.Formatter):
    """JSON 格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加事件数据
        if hasattr(record, "event_type"):
            log_entry["event_type"] = record.event_type
        if hasattr(record, "event_source"):
            log_entry["event_source"] = record.event_source
        if hasattr(record, "event_timestamp"):
            log_entry["event_timestamp"] = record.event_timestamp
        if hasattr(record, "event_data"):
            log_entry["data"] = record.event_data

        return json.dumps(log_entry, ensure_ascii=False, default=str)
