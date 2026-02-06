"""
AGA 分布式追踪模块

提供 OpenTelemetry 集成，支持：
- 请求追踪
- 性能指标
- 错误追踪
- 上下文传播

版本: v1.0
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

# 可选依赖
try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    trace = None
    SpanKind = None
    Status = None
    StatusCode = None

try:
    from prometheus_client import Counter, Histogram, Gauge
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


@dataclass
class TracingConfig:
    """追踪配置"""
    enabled: bool = True
    service_name: str = "aga"
    
    # Span 配置
    record_gate_values: bool = True
    record_slot_info: bool = True
    record_latency: bool = True
    
    # 采样
    sample_rate: float = 1.0  # 1.0 = 100%
    
    # Prometheus 指标
    metrics_enabled: bool = True
    metrics_prefix: str = "aga"


class AGATracer:
    """
    AGA 追踪器
    
    提供分布式追踪和指标收集功能。
    
    使用示例：
        ```python
        tracer = AGATracer(TracingConfig(service_name="my-aga"))
        
        # 追踪 AGA 前向传播
        with tracer.trace_forward(namespace="default", batch_size=1) as span:
            result = aga_operator(hidden_states)
            span.set_attribute("aga.gate_mean", result.gate_mean)
        
        # 追踪知识注入
        with tracer.trace_injection(namespace="default", lu_id="rule_001"):
            aga_operator.inject_knowledge(...)
        ```
    """
    
    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        
        # 初始化 tracer
        if _HAS_OTEL and self.config.enabled:
            self._tracer = trace.get_tracer(self.config.service_name)
        else:
            self._tracer = None
        
        # 初始化 Prometheus 指标
        if _HAS_PROMETHEUS and self.config.metrics_enabled:
            self._init_metrics()
        else:
            self._metrics = None
    
    def _init_metrics(self):
        """初始化 Prometheus 指标"""
        prefix = self.config.metrics_prefix
        
        self._metrics = {
            # 计数器
            'forward_total': Counter(
                f'{prefix}_forward_total',
                'Total AGA forward calls',
                ['namespace', 'gate_result']
            ),
            'injection_total': Counter(
                f'{prefix}_injection_total',
                'Total knowledge injections',
                ['namespace', 'lifecycle_state']
            ),
            'error_total': Counter(
                f'{prefix}_error_total',
                'Total errors',
                ['namespace', 'error_type']
            ),
            
            # 直方图
            'forward_latency': Histogram(
                f'{prefix}_forward_latency_seconds',
                'AGA forward latency',
                ['namespace'],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
            ),
            'gate_value': Histogram(
                f'{prefix}_gate_value',
                'Gate value distribution',
                ['namespace'],
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            
            # 仪表
            'active_slots': Gauge(
                f'{prefix}_active_slots',
                'Number of active slots',
                ['namespace']
            ),
            'slot_hit_rate': Gauge(
                f'{prefix}_slot_hit_rate',
                'Slot hit rate',
                ['namespace']
            ),
        }
    
    @contextmanager
    def trace_forward(
        self,
        namespace: str = "default",
        batch_size: int = 1,
        seq_len: int = 0,
        **extra_attributes,
    ):
        """
        追踪 AGA 前向传播
        
        Args:
            namespace: 命名空间
            batch_size: 批次大小
            seq_len: 序列长度
            **extra_attributes: 额外属性
        
        Yields:
            Span 对象（如果启用追踪）
        """
        start_time = time.time()
        span = None
        
        try:
            if self._tracer:
                span = self._tracer.start_span(
                    "aga_forward",
                    kind=SpanKind.INTERNAL,
                )
                span.set_attribute("aga.namespace", namespace)
                span.set_attribute("aga.batch_size", batch_size)
                span.set_attribute("aga.seq_len", seq_len)
                
                for key, value in extra_attributes.items():
                    span.set_attribute(f"aga.{key}", value)
            
            yield SpanWrapper(span, self.config)
            
            # 记录成功
            if self._metrics:
                latency = time.time() - start_time
                self._metrics['forward_latency'].labels(namespace=namespace).observe(latency)
                self._metrics['forward_total'].labels(
                    namespace=namespace,
                    gate_result='success'
                ).inc()
            
            if span:
                span.set_status(Status(StatusCode.OK))
                
        except Exception as e:
            # 记录错误
            if self._metrics:
                self._metrics['forward_total'].labels(
                    namespace=namespace,
                    gate_result='error'
                ).inc()
                self._metrics['error_total'].labels(
                    namespace=namespace,
                    error_type=type(e).__name__
                ).inc()
            
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            
            raise
        finally:
            if span:
                span.end()
    
    @contextmanager
    def trace_injection(
        self,
        namespace: str = "default",
        lu_id: str = "",
        **extra_attributes,
    ):
        """
        追踪知识注入
        
        Args:
            namespace: 命名空间
            lu_id: 知识 ID
            **extra_attributes: 额外属性
        
        Yields:
            Span 对象
        """
        span = None
        
        try:
            if self._tracer:
                span = self._tracer.start_span(
                    "aga_injection",
                    kind=SpanKind.INTERNAL,
                )
                span.set_attribute("aga.namespace", namespace)
                span.set_attribute("aga.lu_id", lu_id)
                
                for key, value in extra_attributes.items():
                    span.set_attribute(f"aga.{key}", value)
            
            yield SpanWrapper(span, self.config)
            
            # 记录成功
            if self._metrics:
                self._metrics['injection_total'].labels(
                    namespace=namespace,
                    lifecycle_state='probationary'
                ).inc()
            
            if span:
                span.set_status(Status(StatusCode.OK))
                
        except Exception as e:
            if self._metrics:
                self._metrics['error_total'].labels(
                    namespace=namespace,
                    error_type=type(e).__name__
                ).inc()
            
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            
            raise
        finally:
            if span:
                span.end()
    
    @contextmanager
    def trace_gate(self, gate_name: str, namespace: str = "default"):
        """追踪门控检查"""
        span = None
        
        try:
            if self._tracer:
                span = self._tracer.start_span(
                    f"aga_gate_{gate_name}",
                    kind=SpanKind.INTERNAL,
                )
                span.set_attribute("aga.namespace", namespace)
                span.set_attribute("aga.gate_name", gate_name)
            
            yield SpanWrapper(span, self.config)
            
            if span:
                span.set_status(Status(StatusCode.OK))
                
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise
        finally:
            if span:
                span.end()
    
    def record_gate_value(self, namespace: str, gate_mean: float):
        """记录门控值"""
        if self._metrics:
            self._metrics['gate_value'].labels(namespace=namespace).observe(gate_mean)
    
    def record_active_slots(self, namespace: str, count: int):
        """记录活跃槽位数"""
        if self._metrics:
            self._metrics['active_slots'].labels(namespace=namespace).set(count)
    
    def record_hit_rate(self, namespace: str, rate: float):
        """记录命中率"""
        if self._metrics:
            self._metrics['slot_hit_rate'].labels(namespace=namespace).set(rate)


class SpanWrapper:
    """
    Span 包装器
    
    提供统一的接口，无论是否启用追踪。
    """
    
    def __init__(self, span, config: TracingConfig):
        self._span = span
        self._config = config
    
    def set_attribute(self, key: str, value: Any):
        """设置属性"""
        if self._span:
            self._span.set_attribute(key, value)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """添加事件"""
        if self._span:
            self._span.add_event(name, attributes or {})
    
    def record_gate_result(self, gate_name: str, result: str, value: float = 0.0):
        """记录门控结果"""
        if self._span and self._config.record_gate_values:
            self._span.set_attribute(f"aga.{gate_name}_result", result)
            self._span.set_attribute(f"aga.{gate_name}_value", value)
    
    def record_routing(self, routed_slots: int, total_slots: int):
        """记录路由结果"""
        if self._span and self._config.record_slot_info:
            self._span.set_attribute("aga.routed_slots", routed_slots)
            self._span.set_attribute("aga.total_slots", total_slots)


def traced(tracer: AGATracer, operation: str = "operation"):
    """
    追踪装饰器
    
    使用示例：
        ```python
        tracer = AGATracer()
        
        @traced(tracer, "my_operation")
        def my_function(x):
            return x * 2
        ```
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace_forward(namespace="default") as span:
                span.set_attribute("aga.operation", operation)
                return func(*args, **kwargs)
        return wrapper
    return decorator


class TracedAGAOperator:
    """
    带追踪的 AGA 算子包装器
    
    自动追踪所有 AGA 操作。
    """
    
    def __init__(
        self,
        aga_operator,
        tracer: Optional[AGATracer] = None,
        namespace: str = "default",
    ):
        self.aga = aga_operator
        self.tracer = tracer or AGATracer()
        self.namespace = namespace
    
    def __call__(self, hidden_states, **kwargs):
        """前向传播（带追踪）"""
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        with self.tracer.trace_forward(
            namespace=self.namespace,
            batch_size=batch_size,
            seq_len=seq_len,
        ) as span:
            # Gate-0
            with self.tracer.trace_gate("gate0", self.namespace):
                pass  # Gate-0 检查在 aga 内部
            
            result = self.aga(hidden_states, **kwargs)
            
            # 记录结果
            if hasattr(result, 'gate_mean'):
                span.set_attribute("aga.gate_mean", result.gate_mean)
                self.tracer.record_gate_value(self.namespace, result.gate_mean)
            
            if hasattr(result, 'routed_slots'):
                span.record_routing(
                    len(result.routed_slots) if result.routed_slots else 0,
                    getattr(self.aga, 'num_slots', 0)
                )
            
            return result
    
    def inject_knowledge(self, *args, **kwargs):
        """知识注入（带追踪）"""
        lu_id = args[0] if args else kwargs.get('lu_id', '')
        
        with self.tracer.trace_injection(
            namespace=self.namespace,
            lu_id=lu_id,
        ):
            return self.aga.inject_knowledge(*args, **kwargs)
