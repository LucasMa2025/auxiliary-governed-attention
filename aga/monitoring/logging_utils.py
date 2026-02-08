"""
AGA 结构化日志模块

提供生产级结构化日志能力：
1. JSON 格式输出
2. 兼容 ELK Stack 和 Loki
3. 请求级别上下文
4. 自动 trace_id 注入

版本: v1.0
"""
import json
import logging
import sys
import threading
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass, field


# ==================== 配置 ====================

@dataclass
class LoggingConfig:
    """日志配置"""
    service_name: str = "aga"
    level: str = "INFO"
    format: str = "json"  # json, text
    output: str = "stdout"  # stdout, file, both
    log_file: Optional[str] = None
    include_extra: bool = True
    include_source: bool = True
    max_message_length: int = 10000
    
    # 第三方库日志级别
    third_party_level: str = "WARNING"


# ==================== 日志上下文 ====================

class LogContext:
    """
    线程安全的日志上下文管理器
    
    用于在日志中添加请求级别的上下文信息。
    
    Usage:
        ```python
        from aga.monitoring.logging_utils import LogContext
        
        # 设置上下文
        LogContext.set(
            trace_id="abc-123",
            namespace="production",
            user_id="user_001"
        )
        
        # 日志会自动包含这些上下文
        logger.info("Processing request")
        
        # 清除上下文
        LogContext.clear()
        ```
    """
    
    _local = threading.local()
    
    @classmethod
    def _get_context(cls) -> Dict[str, Any]:
        """获取当前线程的上下文"""
        if not hasattr(cls._local, "context"):
            cls._local.context = {}
        return cls._local.context
    
    @classmethod
    def set(cls, **kwargs):
        """设置上下文"""
        cls._get_context().update(kwargs)
    
    @classmethod
    def get(cls, key: str, default=None):
        """获取上下文值"""
        return cls._get_context().get(key, default)
    
    @classmethod
    def clear(cls):
        """清除上下文"""
        cls._local.context = {}
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """获取所有上下文"""
        return cls._get_context().copy()
    
    @classmethod
    def remove(cls, key: str):
        """移除指定上下文"""
        ctx = cls._get_context()
        if key in ctx:
            del ctx[key]


@contextmanager
def log_context(**kwargs):
    """
    日志上下文管理器
    
    Usage:
        ```python
        with log_context(trace_id="abc-123", namespace="production"):
            logger.info("Processing request")
            # 日志会包含 trace_id 和 namespace
        # 上下文自动清除
        ```
    """
    # 保存旧值
    old_values = {}
    for key in kwargs:
        old_values[key] = LogContext.get(key)
    
    # 设置新值
    LogContext.set(**kwargs)
    
    try:
        yield
    finally:
        # 恢复旧值
        for key, old_value in old_values.items():
            if old_value is None:
                LogContext.remove(key)
            else:
                LogContext.set(**{key: old_value})


def generate_trace_id() -> str:
    """生成 trace ID"""
    return str(uuid.uuid4())


# ==================== 结构化日志格式化器 ====================

class StructuredFormatter(logging.Formatter):
    """
    结构化 JSON 日志格式化器
    
    输出格式兼容 ELK Stack 和 Loki。
    """
    
    # 标准字段，不包含在 extra 中
    STANDARD_FIELDS = {
        "name", "msg", "args", "created", "filename",
        "funcName", "levelname", "levelno", "lineno",
        "module", "msecs", "pathname", "process",
        "processName", "relativeCreated", "stack_info",
        "exc_info", "exc_text", "thread", "threadName",
        "message", "taskName",
    }
    
    def __init__(
        self,
        service_name: str = "aga",
        include_extra: bool = True,
        include_source: bool = True,
        max_message_length: int = 10000,
    ):
        super().__init__()
        self.service_name = service_name
        self.include_extra = include_extra
        self.include_source = include_source
        self.max_message_length = max_message_length
    
    def format(self, record: logging.LogRecord) -> str:
        # 基础字段
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": self._truncate_message(record.getMessage()),
            "service": self.service_name,
        }
        
        # 添加 trace_id（如果存在）
        trace_id = LogContext.get("trace_id")
        if trace_id:
            log_data["trace_id"] = trace_id
        
        # 添加上下文
        context = LogContext.as_dict()
        if context:
            log_data["context"] = context
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        # 添加源位置
        if self.include_source:
            log_data["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        # 添加额外字段
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in self.STANDARD_FIELDS:
                    try:
                        # 确保可序列化
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False, default=str)
    
    def _truncate_message(self, message: str) -> str:
        """截断过长的消息"""
        if len(message) > self.max_message_length:
            return message[:self.max_message_length] + "...[truncated]"
        return message


class TextFormatter(logging.Formatter):
    """
    增强的文本日志格式化器
    
    包含上下文信息。
    """
    
    def __init__(
        self,
        service_name: str = "aga",
        include_context: bool = True,
    ):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.service_name = service_name
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        # 添加上下文到消息
        if self.include_context:
            context = LogContext.as_dict()
            if context:
                context_str = " | ".join(f"{k}={v}" for k, v in context.items())
                record.msg = f"[{context_str}] {record.msg}"
        
        return super().format(record)


# ==================== 日志配置函数 ====================

def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    配置 AGA 日志系统
    
    Args:
        config: 日志配置
    
    Usage:
        ```python
        from aga.monitoring.logging_utils import setup_logging, LoggingConfig
        
        # 使用默认配置
        setup_logging()
        
        # 使用自定义配置
        config = LoggingConfig(
            service_name="aga-portal",
            level="DEBUG",
            format="json",
            output="both",
            log_file="/var/log/aga/portal.log"
        )
        setup_logging(config)
        ```
    """
    config = config or LoggingConfig()
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 创建格式化器
    if config.format == "json":
        formatter = StructuredFormatter(
            service_name=config.service_name,
            include_extra=config.include_extra,
            include_source=config.include_source,
            max_message_length=config.max_message_length,
        )
    else:
        formatter = TextFormatter(
            service_name=config.service_name,
        )
    
    # stdout 处理器
    if config.output in ("stdout", "both"):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)
    
    # 文件处理器
    if config.output in ("file", "both") and config.log_file:
        import os
        os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
        file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库日志级别
    third_party_level = getattr(logging, config.third_party_level.upper())
    for lib in ["uvicorn", "fastapi", "httpx", "asyncio", "urllib3"]:
        logging.getLogger(lib).setLevel(third_party_level)
    
    logging.info(
        f"Logging configured: service={config.service_name}, "
        f"level={config.level}, format={config.format}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器
    
    Args:
        name: 日志器名称
    
    Returns:
        配置好的日志器
    """
    return logging.getLogger(name)


# ==================== 日志辅助函数 ====================

def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **extra,
):
    """
    带上下文的日志记录
    
    Args:
        logger: 日志器
        level: 日志级别
        message: 消息
        **extra: 额外字段
    """
    # 合并上下文
    context = LogContext.as_dict()
    context.update(extra)
    
    logger.log(level, message, extra={"context": context})


def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **extra,
):
    """
    记录 HTTP 请求日志
    """
    logger.info(
        f"{method} {path} {status_code} {duration_ms:.2f}ms",
        extra={
            "http": {
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
            **extra,
        }
    )


def log_operation(
    logger: logging.Logger,
    operation: str,
    success: bool,
    duration_ms: float,
    **extra,
):
    """
    记录操作日志
    """
    level = logging.INFO if success else logging.WARNING
    status = "success" if success else "failed"
    
    logger.log(
        level,
        f"Operation {operation} {status} in {duration_ms:.2f}ms",
        extra={
            "operation": {
                "name": operation,
                "success": success,
                "duration_ms": duration_ms,
            },
            **extra,
        }
    )


# ==================== 导出 ====================

__all__ = [
    # 配置
    "LoggingConfig",
    "setup_logging",
    "get_logger",
    # 上下文
    "LogContext",
    "log_context",
    "generate_trace_id",
    # 格式化器
    "StructuredFormatter",
    "TextFormatter",
    # 辅助函数
    "log_with_context",
    "log_request",
    "log_operation",
]
