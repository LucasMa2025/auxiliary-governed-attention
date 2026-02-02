"""
AGA 异常处理模块

提供结构化的异常类型和错误处理机制。

设计原则：
- 异常层次清晰，便于精确捕获
- 包含丰富的上下文信息
- 支持错误码和错误分类
- 便于日志记录和监控
"""
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import traceback
import json
import logging

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """错误分类"""
    CONFIGURATION = "configuration"      # 配置错误
    INJECTION = "injection"              # 知识注入错误
    COMPUTATION = "computation"          # 计算错误
    LIFECYCLE = "lifecycle"              # 生命周期错误
    PERSISTENCE = "persistence"          # 持久化错误
    CONCURRENCY = "concurrency"          # 并发错误
    RESOURCE = "resource"                # 资源错误
    VALIDATION = "validation"            # 验证错误
    GOVERNANCE = "governance"            # 治理错误
    UNKNOWN = "unknown"                  # 未知错误


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    DEBUG = "debug"           # 调试信息
    INFO = "info"             # 一般信息
    WARNING = "warning"       # 警告
    ERROR = "error"           # 错误
    CRITICAL = "critical"     # 严重错误


@dataclass
class ErrorContext:
    """错误上下文"""
    # 基本信息
    error_code: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    
    # 位置信息
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    
    # 业务上下文
    namespace: Optional[str] = None
    lu_id: Optional[str] = None
    slot_idx: Optional[int] = None
    request_id: Optional[str] = None
    
    # 时间和追踪
    timestamp: datetime = field(default_factory=datetime.now)
    trace_id: Optional[str] = None
    
    # 额外信息
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    # 恢复建议
    recovery_suggestion: Optional[str] = None
    retry_allowed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'error_code': self.error_code,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'namespace': self.namespace,
            'lu_id': self.lu_id,
            'slot_idx': self.slot_idx,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'trace_id': self.trace_id,
            'details': self.details,
            'stack_trace': self.stack_trace,
            'recovery_suggestion': self.recovery_suggestion,
            'retry_allowed': self.retry_allowed,
        }
    
    def to_json(self) -> str:
        """转换为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class AGAException(Exception):
    """
    AGA 基础异常类
    
    所有 AGA 相关异常的基类。
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "AGA_ERROR",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        **kwargs
    ):
        super().__init__(message)
        self.context = ErrorContext(
            error_code=error_code,
            category=category,
            severity=severity,
            message=message,
            stack_trace=traceback.format_exc(),
            **kwargs
        )
    
    def __str__(self) -> str:
        return f"[{self.context.error_code}] {self.context.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        return self.context.to_dict()
    
    def to_json(self) -> str:
        return self.context.to_json()


# ==================== 配置相关异常 ====================

class ConfigurationError(AGAException):
    """配置错误"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="AGA_CONFIG_ERROR",
            category=ErrorCategory.CONFIGURATION,
            details={'config_key': config_key, **kwargs.pop('details', {})},
            recovery_suggestion="检查配置参数是否正确",
            **kwargs
        )


class InvalidDimensionError(ConfigurationError):
    """维度不匹配错误"""
    
    def __init__(
        self,
        expected_dim: int,
        actual_dim: int,
        tensor_name: str = "tensor",
        **kwargs
    ):
        super().__init__(
            message=f"{tensor_name} 维度不匹配: 期望 {expected_dim}, 实际 {actual_dim}",
            error_code="AGA_DIM_MISMATCH",
            details={
                'expected_dim': expected_dim,
                'actual_dim': actual_dim,
                'tensor_name': tensor_name,
            },
            recovery_suggestion=f"确保 {tensor_name} 的维度为 {expected_dim}",
            **kwargs
        )


# ==================== 注入相关异常 ====================

class InjectionError(AGAException):
    """知识注入错误"""
    
    def __init__(self, message: str, lu_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="AGA_INJECTION_ERROR",
            category=ErrorCategory.INJECTION,
            lu_id=lu_id,
            **kwargs
        )


class SlotNotAvailableError(InjectionError):
    """无可用槽位错误"""
    
    def __init__(self, namespace: str, current_slots: int, max_slots: int, **kwargs):
        super().__init__(
            message=f"命名空间 '{namespace}' 无可用槽位: {current_slots}/{max_slots}",
            error_code="AGA_NO_SLOT",
            namespace=namespace,
            details={
                'current_slots': current_slots,
                'max_slots': max_slots,
            },
            recovery_suggestion="清理过期槽位或增加槽位容量",
            retry_allowed=False,
            **kwargs
        )


class DuplicateLUError(InjectionError):
    """重复 LU 错误"""
    
    def __init__(self, lu_id: str, existing_slot: int, **kwargs):
        super().__init__(
            message=f"LU '{lu_id}' 已存在于槽位 {existing_slot}",
            error_code="AGA_DUPLICATE_LU",
            lu_id=lu_id,
            slot_idx=existing_slot,
            recovery_suggestion="使用更新操作而非新建",
            retry_allowed=False,
            **kwargs
        )


class VectorValidationError(InjectionError):
    """向量验证错误"""
    
    def __init__(
        self,
        vector_type: str,
        issue: str,
        lu_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=f"{vector_type} 向量验证失败: {issue}",
            error_code="AGA_VECTOR_INVALID",
            lu_id=lu_id,
            details={'vector_type': vector_type, 'issue': issue},
            recovery_suggestion="检查向量是否包含 NaN/Inf 或范数异常",
            **kwargs
        )


class TrainingModeError(InjectionError):
    """训练模式错误"""
    
    def __init__(self, operation: str, **kwargs):
        super().__init__(
            message=f"不能在训练模式下执行 '{operation}' 操作",
            error_code="AGA_TRAINING_MODE",
            details={'operation': operation},
            recovery_suggestion="调用 model.eval() 切换到推理模式",
            retry_allowed=False,
            **kwargs
        )


# ==================== 生命周期相关异常 ====================

class LifecycleError(AGAException):
    """生命周期错误"""
    
    def __init__(self, message: str, lu_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="AGA_LIFECYCLE_ERROR",
            category=ErrorCategory.LIFECYCLE,
            lu_id=lu_id,
            **kwargs
        )


class InvalidStateTransitionError(LifecycleError):
    """无效状态转换错误"""
    
    def __init__(
        self,
        lu_id: str,
        current_state: str,
        target_state: str,
        **kwargs
    ):
        super().__init__(
            message=f"LU '{lu_id}' 无法从 '{current_state}' 转换到 '{target_state}'",
            error_code="AGA_INVALID_TRANSITION",
            lu_id=lu_id,
            details={
                'current_state': current_state,
                'target_state': target_state,
            },
            recovery_suggestion="检查状态转换规则",
            retry_allowed=False,
            **kwargs
        )


class QuarantineError(LifecycleError):
    """隔离错误"""
    
    def __init__(self, lu_id: str, reason: str, **kwargs):
        super().__init__(
            message=f"隔离 LU '{lu_id}' 失败: {reason}",
            error_code="AGA_QUARANTINE_FAILED",
            lu_id=lu_id,
            severity=ErrorSeverity.CRITICAL,
            details={'reason': reason},
            recovery_suggestion="检查槽位状态并重试",
            **kwargs
        )


# ==================== 持久化相关异常 ====================

class PersistenceError(AGAException):
    """持久化错误"""
    
    def __init__(self, message: str, operation: str = "unknown", **kwargs):
        super().__init__(
            message=message,
            error_code="AGA_PERSISTENCE_ERROR",
            category=ErrorCategory.PERSISTENCE,
            details={'operation': operation, **kwargs.pop('details', {})},
            **kwargs
        )


class DatabaseConnectionError(PersistenceError):
    """数据库连接错误"""
    
    def __init__(self, db_type: str, host: str, error: str, **kwargs):
        super().__init__(
            message=f"无法连接到 {db_type} ({host}): {error}",
            error_code="AGA_DB_CONNECTION",
            operation="connect",
            severity=ErrorSeverity.CRITICAL,
            details={
                'db_type': db_type,
                'host': host,
                'error': error,
            },
            recovery_suggestion="检查数据库服务状态和网络连接",
            **kwargs
        )


class DataCorruptionError(PersistenceError):
    """数据损坏错误"""
    
    def __init__(self, entity_type: str, entity_id: str, issue: str, **kwargs):
        super().__init__(
            message=f"{entity_type} '{entity_id}' 数据损坏: {issue}",
            error_code="AGA_DATA_CORRUPTION",
            operation="read",
            severity=ErrorSeverity.CRITICAL,
            details={
                'entity_type': entity_type,
                'entity_id': entity_id,
                'issue': issue,
            },
            recovery_suggestion="从备份恢复或重建数据",
            retry_allowed=False,
            **kwargs
        )


# ==================== 并发相关异常 ====================

class ConcurrencyError(AGAException):
    """并发错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="AGA_CONCURRENCY_ERROR",
            category=ErrorCategory.CONCURRENCY,
            **kwargs
        )


class LockTimeoutError(ConcurrencyError):
    """锁超时错误"""
    
    def __init__(
        self,
        resource: str,
        timeout_seconds: float,
        **kwargs
    ):
        super().__init__(
            message=f"获取资源 '{resource}' 的锁超时 ({timeout_seconds}s)",
            error_code="AGA_LOCK_TIMEOUT",
            details={
                'resource': resource,
                'timeout_seconds': timeout_seconds,
            },
            recovery_suggestion="稍后重试或检查是否存在死锁",
            **kwargs
        )


class VersionConflictError(ConcurrencyError):
    """版本冲突错误"""
    
    def __init__(
        self,
        entity_id: str,
        expected_version: int,
        actual_version: int,
        **kwargs
    ):
        super().__init__(
            message=f"实体 '{entity_id}' 版本冲突: 期望 {expected_version}, 实际 {actual_version}",
            error_code="AGA_VERSION_CONFLICT",
            details={
                'entity_id': entity_id,
                'expected_version': expected_version,
                'actual_version': actual_version,
            },
            recovery_suggestion="重新读取最新版本后重试",
            **kwargs
        )


# ==================== 资源相关异常 ====================

class ResourceError(AGAException):
    """资源错误"""
    
    def __init__(self, message: str, resource_type: str = "unknown", **kwargs):
        super().__init__(
            message=message,
            error_code="AGA_RESOURCE_ERROR",
            category=ErrorCategory.RESOURCE,
            details={'resource_type': resource_type, **kwargs.pop('details', {})},
            **kwargs
        )


class MemoryExhaustedError(ResourceError):
    """内存耗尽错误"""
    
    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(
            message=f"{device} 内存不足: 需要 {required_mb:.1f}MB, 可用 {available_mb:.1f}MB",
            error_code="AGA_OOM",
            resource_type="memory",
            severity=ErrorSeverity.CRITICAL,
            details={
                'required_mb': required_mb,
                'available_mb': available_mb,
                'device': device,
            },
            recovery_suggestion="减少批量大小或清理缓存",
            retry_allowed=False,
            **kwargs
        )


class QuotaExceededError(ResourceError):
    """配额超限错误"""
    
    def __init__(
        self,
        namespace: str,
        quota_type: str,
        current: float,
        limit: float,
        **kwargs
    ):
        super().__init__(
            message=f"命名空间 '{namespace}' 的 {quota_type} 配额超限: {current}/{limit}",
            error_code="AGA_QUOTA_EXCEEDED",
            resource_type="quota",
            namespace=namespace,
            details={
                'quota_type': quota_type,
                'current': current,
                'limit': limit,
            },
            recovery_suggestion="清理资源或申请更高配额",
            retry_allowed=False,
            **kwargs
        )


# ==================== 治理相关异常 ====================

class GovernanceError(AGAException):
    """治理错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="AGA_GOVERNANCE_ERROR",
            category=ErrorCategory.GOVERNANCE,
            **kwargs
        )


class ConflictDetectedError(GovernanceError):
    """知识冲突错误"""
    
    def __init__(
        self,
        new_lu_id: str,
        conflicting_lu_ids: List[str],
        conflict_type: str,
        **kwargs
    ):
        super().__init__(
            message=f"LU '{new_lu_id}' 与 {conflicting_lu_ids} 存在 {conflict_type} 冲突",
            error_code="AGA_KNOWLEDGE_CONFLICT",
            lu_id=new_lu_id,
            details={
                'conflicting_lu_ids': conflicting_lu_ids,
                'conflict_type': conflict_type,
            },
            recovery_suggestion="解决冲突后重试或使用强制覆盖",
            retry_allowed=False,
            **kwargs
        )


class ApprovalRequiredError(GovernanceError):
    """需要审批错误"""
    
    def __init__(
        self,
        lu_id: str,
        operation: str,
        reason: str,
        **kwargs
    ):
        super().__init__(
            message=f"LU '{lu_id}' 的 '{operation}' 操作需要人工审批: {reason}",
            error_code="AGA_APPROVAL_REQUIRED",
            lu_id=lu_id,
            severity=ErrorSeverity.WARNING,
            details={
                'operation': operation,
                'reason': reason,
            },
            recovery_suggestion="提交审批请求",
            retry_allowed=False,
            **kwargs
        )


# ==================== 结构化日志 ====================

class StructuredLogger:
    """
    结构化日志记录器
    
    提供统一的日志格式，便于日志聚合和分析。
    """
    
    def __init__(
        self,
        name: str,
        default_namespace: Optional[str] = None,
        default_trace_id: Optional[str] = None,
    ):
        self.logger = logging.getLogger(name)
        self.default_namespace = default_namespace
        self.default_trace_id = default_trace_id
    
    def _format_log(
        self,
        level: str,
        message: str,
        category: ErrorCategory,
        **kwargs
    ) -> Dict[str, Any]:
        """格式化日志"""
        return {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'category': category.value,
            'message': message,
            'namespace': kwargs.get('namespace', self.default_namespace),
            'trace_id': kwargs.get('trace_id', self.default_trace_id),
            'lu_id': kwargs.get('lu_id'),
            'slot_idx': kwargs.get('slot_idx'),
            'request_id': kwargs.get('request_id'),
            'details': kwargs.get('details', {}),
            'duration_ms': kwargs.get('duration_ms'),
        }
    
    def debug(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
        """调试日志"""
        log_data = self._format_log('DEBUG', message, category, **kwargs)
        self.logger.debug(json.dumps(log_data, ensure_ascii=False))
    
    def info(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
        """信息日志"""
        log_data = self._format_log('INFO', message, category, **kwargs)
        self.logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def warning(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
        """警告日志"""
        log_data = self._format_log('WARNING', message, category, **kwargs)
        self.logger.warning(json.dumps(log_data, ensure_ascii=False))
    
    def error(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
        """错误日志"""
        log_data = self._format_log('ERROR', message, category, **kwargs)
        if 'exception' in kwargs:
            log_data['stack_trace'] = traceback.format_exc()
        self.logger.error(json.dumps(log_data, ensure_ascii=False))
    
    def critical(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
        """严重错误日志"""
        log_data = self._format_log('CRITICAL', message, category, **kwargs)
        if 'exception' in kwargs:
            log_data['stack_trace'] = traceback.format_exc()
        self.logger.critical(json.dumps(log_data, ensure_ascii=False))
    
    def log_exception(self, exc: AGAException):
        """记录 AGA 异常"""
        log_data = exc.to_dict()
        log_data['timestamp'] = datetime.now().isoformat()
        
        level = exc.context.severity
        if level == ErrorSeverity.DEBUG:
            self.logger.debug(json.dumps(log_data, ensure_ascii=False))
        elif level == ErrorSeverity.INFO:
            self.logger.info(json.dumps(log_data, ensure_ascii=False))
        elif level == ErrorSeverity.WARNING:
            self.logger.warning(json.dumps(log_data, ensure_ascii=False))
        elif level == ErrorSeverity.ERROR:
            self.logger.error(json.dumps(log_data, ensure_ascii=False))
        elif level == ErrorSeverity.CRITICAL:
            self.logger.critical(json.dumps(log_data, ensure_ascii=False))
    
    def log_operation(
        self,
        operation: str,
        success: bool,
        duration_ms: float,
        **kwargs
    ):
        """记录操作日志"""
        log_data = self._format_log(
            'INFO' if success else 'ERROR',
            f"Operation '{operation}' {'succeeded' if success else 'failed'}",
            ErrorCategory.UNKNOWN,
            duration_ms=duration_ms,
            details={'operation': operation, 'success': success, **kwargs.get('details', {})},
            **kwargs
        )
        
        if success:
            self.logger.info(json.dumps(log_data, ensure_ascii=False))
        else:
            self.logger.error(json.dumps(log_data, ensure_ascii=False))


# ==================== 重试机制 ====================

class RetryPolicy:
    """重试策略"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        retryable_exceptions: Optional[List[type]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions or [
            AGAException,
            LockTimeoutError,
            DatabaseConnectionError,
        ]
    
    def get_delay(self, attempt: int) -> float:
        """计算延迟时间（指数退避）"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_retries:
            return False
        
        # 检查异常类型
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                # 检查 retry_allowed 标志
                if isinstance(exception, AGAException):
                    return exception.context.retry_allowed
                return True
        
        return False


def with_retry(policy: Optional[RetryPolicy] = None):
    """
    重试装饰器
    
    用法：
    ```python
    @with_retry(RetryPolicy(max_retries=3))
    def my_function():
        ...
    ```
    """
    import functools
    import time
    
    policy = policy or RetryPolicy()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(policy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not policy.should_retry(e, attempt):
                        raise
                    
                    delay = policy.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{policy.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    
    return decorator


# 便捷函数
def get_logger(name: str, **kwargs) -> StructuredLogger:
    """获取结构化日志记录器"""
    return StructuredLogger(name, **kwargs)

