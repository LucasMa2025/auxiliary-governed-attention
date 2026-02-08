"""
AGA 统一指标注册中心

提供所有 AGA 组件的 Prometheus 指标注册和导出。

特性：
1. 单例模式，全局统一管理
2. 支持 Counter/Gauge/Histogram/Summary
3. 线程安全
4. 无 prometheus_client 时优雅降级
5. 支持动态标签

版本: v1.0
"""
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== 配置 ====================

@dataclass
class MetricsConfig:
    """指标配置"""
    enabled: bool = True
    prefix: str = "aga"
    include_process_metrics: bool = True
    include_platform_metrics: bool = True
    default_labels: Dict[str, str] = field(default_factory=dict)
    
    # 直方图桶配置
    latency_buckets: Tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0
    )
    size_buckets: Tuple[int, ...] = (
        10, 50, 100, 250, 500, 1000, 2500, 5000, 10000
    )
    
    # ANN 专用桶
    ann_latency_buckets: Tuple[float, ...] = (
        0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1
    )
    
    # 门控值桶
    gate_value_buckets: Tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    )


# ==================== 指标类型枚举 ====================

class MetricType(str, Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# ==================== 空操作指标（降级用）====================

class NoOpMetric:
    """空操作指标，用于 prometheus_client 不可用时"""
    
    def labels(self, **kwargs):
        return self
    
    def inc(self, amount: float = 1):
        pass
    
    def dec(self, amount: float = 1):
        pass
    
    def set(self, value: float):
        pass
    
    def observe(self, value: float):
        pass
    
    def time(self):
        """返回一个空的上下文管理器"""
        import contextlib
        return contextlib.nullcontext()


# ==================== 指标注册中心 ====================

class MetricsRegistry:
    """
    AGA 统一指标注册中心
    
    单例模式，提供所有 AGA 组件的指标注册和导出。
    
    使用示例：
        ```python
        from aga.monitoring.metrics import get_metrics_registry
        
        registry = get_metrics_registry()
        
        # 记录请求
        registry.get_metric("requests_total").labels(
            namespace="default",
            operation="forward",
            status="success"
        ).inc()
        
        # 记录延迟
        registry.get_metric("request_duration_seconds").labels(
            namespace="default",
            operation="forward"
        ).observe(0.05)
        
        # 导出指标
        text = registry.export_text()
        ```
    """
    
    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[MetricsConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        if self._initialized:
            return
        
        self.config = config or MetricsConfig()
        self._metrics: Dict[str, Any] = {}
        self._prometheus_available = False
        self._registry = None
        self._generate_latest = None
        self._content_type = "text/plain"
        
        self._init_prometheus()
        self._initialized = True
    
    def _init_prometheus(self):
        """初始化 Prometheus 客户端"""
        try:
            from prometheus_client import (
                Counter, Gauge, Histogram, Summary,
                CollectorRegistry, REGISTRY,
                generate_latest, CONTENT_TYPE_LATEST
            )
            self._prometheus_available = True
            self._registry = REGISTRY
            self._generate_latest = generate_latest
            self._content_type = CONTENT_TYPE_LATEST
            
            # 保存类引用
            self._Counter = Counter
            self._Gauge = Gauge
            self._Histogram = Histogram
            self._Summary = Summary
            
            # 注册所有 AGA 指标
            self._register_all_metrics()
            
            logger.info("Prometheus metrics initialized successfully")
            
        except ImportError:
            logger.warning(
                "prometheus_client not installed, metrics disabled. "
                "Install with: pip install prometheus_client"
            )
    
    def _register_all_metrics(self):
        """注册所有 AGA 指标"""
        prefix = self.config.prefix
        
        # ==================== 核心请求指标 ====================
        
        self._metrics["requests_total"] = self._Counter(
            f"{prefix}_requests_total",
            "Total number of AGA requests",
            ["namespace", "operation", "status"]
        )
        
        self._metrics["request_duration_seconds"] = self._Histogram(
            f"{prefix}_request_duration_seconds",
            "Request duration in seconds",
            ["namespace", "operation"],
            buckets=self.config.latency_buckets
        )
        
        self._metrics["errors_total"] = self._Counter(
            f"{prefix}_errors_total",
            "Total number of errors",
            ["namespace", "error_type", "component"]
        )
        
        # ==================== 门控指标 ====================
        
        self._metrics["gate_checks_total"] = self._Counter(
            f"{prefix}_gate_checks_total",
            "Total gate checks",
            ["namespace", "gate_type"]
        )
        
        self._metrics["gate_hits_total"] = self._Counter(
            f"{prefix}_gate_hits_total",
            "Total gate hits (AGA activated)",
            ["namespace", "gate_type"]
        )
        
        self._metrics["gate_bypasses_total"] = self._Counter(
            f"{prefix}_gate_bypasses_total",
            "Total gate bypasses",
            ["namespace", "reason"]
        )
        
        self._metrics["gate_early_exits_total"] = self._Counter(
            f"{prefix}_gate_early_exits_total",
            "Total early exits",
            ["namespace"]
        )
        
        self._metrics["gate_confidence"] = self._Histogram(
            f"{prefix}_gate_confidence",
            "Gate confidence distribution",
            ["namespace"],
            buckets=self.config.gate_value_buckets
        )
        
        self._metrics["entropy_value"] = self._Histogram(
            f"{prefix}_entropy_value",
            "Entropy value distribution",
            ["namespace"],
            buckets=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
        )
        
        # ==================== 槽位指标 ====================
        
        self._metrics["slots_total"] = self._Gauge(
            f"{prefix}_slots_total",
            "Total number of slots",
            ["namespace", "lifecycle_state"]
        )
        
        self._metrics["slots_active"] = self._Gauge(
            f"{prefix}_slots_active",
            "Number of active slots",
            ["namespace"]
        )
        
        self._metrics["slot_utilization"] = self._Gauge(
            f"{prefix}_slot_utilization",
            "Slot utilization ratio (0-1)",
            ["namespace"]
        )
        
        self._metrics["slot_hits_total"] = self._Counter(
            f"{prefix}_slot_hits_total",
            "Total slot hits",
            ["namespace"]
        )
        
        self._metrics["slots_by_trust_tier"] = self._Gauge(
            f"{prefix}_slots_by_trust_tier",
            "Slots by trust tier",
            ["namespace", "trust_tier"]
        )
        
        self._metrics["slot_hit_rate"] = self._Gauge(
            f"{prefix}_slot_hit_rate",
            "Slot hit rate",
            ["namespace"]
        )
        
        # ==================== ANN 索引指标 ====================
        
        self._metrics["ann_index_size"] = self._Gauge(
            f"{prefix}_ann_index_size",
            "ANN index size (number of vectors)",
            ["namespace", "backend"]
        )
        
        self._metrics["ann_search_total"] = self._Counter(
            f"{prefix}_ann_search_total",
            "Total ANN searches",
            ["namespace", "status"]
        )
        
        self._metrics["ann_search_duration_seconds"] = self._Histogram(
            f"{prefix}_ann_search_duration_seconds",
            "ANN search duration",
            ["namespace"],
            buckets=self.config.ann_latency_buckets
        )
        
        self._metrics["ann_candidates_returned"] = self._Histogram(
            f"{prefix}_ann_candidates_returned",
            "Number of candidates returned by ANN",
            ["namespace"],
            buckets=(10, 25, 50, 100, 150, 200, 300, 500)
        )
        
        self._metrics["ann_rebuild_total"] = self._Counter(
            f"{prefix}_ann_rebuild_total",
            "Total ANN index rebuilds",
            ["namespace", "trigger"]
        )
        
        self._metrics["ann_rebuild_duration_seconds"] = self._Histogram(
            f"{prefix}_ann_rebuild_duration_seconds",
            "ANN rebuild duration",
            ["namespace"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self._metrics["ann_deleted_ratio"] = self._Gauge(
            f"{prefix}_ann_deleted_ratio",
            "Ratio of deleted vectors in ANN index",
            ["namespace"]
        )
        
        self._metrics["ann_total_vectors"] = self._Gauge(
            f"{prefix}_ann_total_vectors",
            "Total vectors in ANN index",
            ["namespace"]
        )
        
        self._metrics["ann_active_vectors"] = self._Gauge(
            f"{prefix}_ann_active_vectors",
            "Active (non-deleted) vectors in ANN index",
            ["namespace"]
        )
        
        # ==================== 动态加载器指标 ====================
        
        self._metrics["loader_requests_total"] = self._Counter(
            f"{prefix}_loader_requests_total",
            "Total loader requests",
            ["namespace"]
        )
        
        self._metrics["loader_hot_hits_total"] = self._Counter(
            f"{prefix}_loader_hot_hits_total",
            "Hot tier hits",
            ["namespace"]
        )
        
        self._metrics["loader_warm_hits_total"] = self._Counter(
            f"{prefix}_loader_warm_hits_total",
            "Warm tier hits",
            ["namespace"]
        )
        
        self._metrics["loader_cold_loads_total"] = self._Counter(
            f"{prefix}_loader_cold_loads_total",
            "Cold tier loads",
            ["namespace"]
        )
        
        self._metrics["loader_failures_total"] = self._Counter(
            f"{prefix}_loader_failures_total",
            "Loader failures",
            ["namespace", "reason"]
        )
        
        self._metrics["loader_duration_seconds"] = self._Histogram(
            f"{prefix}_loader_duration_seconds",
            "Loader duration",
            ["namespace", "tier"],
            buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
        )
        
        self._metrics["warm_cache_size"] = self._Gauge(
            f"{prefix}_warm_cache_size",
            "Warm cache size",
            ["namespace"]
        )
        
        self._metrics["warm_cache_hit_rate"] = self._Gauge(
            f"{prefix}_warm_cache_hit_rate",
            "Warm cache hit rate",
            ["namespace"]
        )
        
        self._metrics["warm_cache_evictions_total"] = self._Counter(
            f"{prefix}_warm_cache_evictions_total",
            "Warm cache evictions",
            ["namespace"]
        )
        
        # ==================== 同步指标 ====================
        
        self._metrics["sync_lag_seconds"] = self._Gauge(
            f"{prefix}_sync_lag_seconds",
            "Sync lag in seconds",
            ["instance"]
        )
        
        self._metrics["sync_messages_total"] = self._Counter(
            f"{prefix}_sync_messages_total",
            "Total sync messages",
            ["type", "direction"]
        )
        
        self._metrics["sync_failures_total"] = self._Counter(
            f"{prefix}_sync_failures_total",
            "Total sync failures",
            ["type"]
        )
        
        self._metrics["sync_last_success_timestamp"] = self._Gauge(
            f"{prefix}_sync_last_success_timestamp",
            "Timestamp of last successful sync"
        )
        
        # ==================== 分布式指标 ====================
        
        self._metrics["partition_state"] = self._Gauge(
            f"{prefix}_partition_state",
            "Partition state (0=normal, 1=degraded, 2=partitioned)",
            ["instance"]
        )
        
        self._metrics["healthy_instances"] = self._Gauge(
            f"{prefix}_healthy_instances",
            "Number of healthy instances"
        )
        
        self._metrics["quorum_size"] = self._Gauge(
            f"{prefix}_quorum_size",
            "Required quorum size"
        )
        
        # ==================== 资源指标 ====================
        
        self._metrics["memory_usage_bytes"] = self._Gauge(
            f"{prefix}_memory_usage_bytes",
            "Memory usage in bytes",
            ["component"]
        )
        
        self._metrics["gpu_memory_usage_bytes"] = self._Gauge(
            f"{prefix}_gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            ["device"]
        )
        
        # ==================== 持久化指标 ====================
        
        self._metrics["persistence_operations_total"] = self._Counter(
            f"{prefix}_persistence_operations_total",
            "Total persistence operations",
            ["namespace", "operation", "status"]
        )
        
        self._metrics["persistence_operation_duration_seconds"] = self._Histogram(
            f"{prefix}_persistence_operation_duration_seconds",
            "Persistence operation duration",
            ["namespace", "operation"],
            buckets=self.config.latency_buckets
        )
        
        # ==================== 治理指标 ====================
        
        self._metrics["quarantine_total"] = self._Counter(
            f"{prefix}_quarantine_total",
            "Total quarantines",
            ["namespace", "reason"]
        )
        
        self._metrics["lifecycle_transitions_total"] = self._Counter(
            f"{prefix}_lifecycle_transitions_total",
            "Total lifecycle transitions",
            ["namespace", "from_state", "to_state"]
        )
        
        self._metrics["injection_total"] = self._Counter(
            f"{prefix}_injection_total",
            "Total knowledge injections",
            ["namespace", "lifecycle_state"]
        )
        
        # ==================== 缓存指标 ====================
        
        self._metrics["cache_hits_total"] = self._Counter(
            f"{prefix}_cache_hits_total",
            "Total cache hits",
            ["namespace", "cache_type"]
        )
        
        self._metrics["cache_misses_total"] = self._Counter(
            f"{prefix}_cache_misses_total",
            "Total cache misses",
            ["namespace", "cache_type"]
        )
        
        self._metrics["cache_size"] = self._Gauge(
            f"{prefix}_cache_size",
            "Cache size",
            ["namespace", "cache_type"]
        )
        
        logger.debug(f"Registered {len(self._metrics)} metrics")
    
    def get_metric(self, name: str) -> Any:
        """
        获取指标
        
        Args:
            name: 指标名称（不含前缀）
        
        Returns:
            Prometheus 指标对象，或 NoOpMetric（降级时）
        """
        if not self._prometheus_available:
            return NoOpMetric()
        
        metric = self._metrics.get(name)
        if metric is None:
            logger.warning(f"Metric '{name}' not found, returning NoOpMetric")
            return NoOpMetric()
        
        return metric
    
    def register_custom_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: List[str] = None,
        buckets: Tuple[float, ...] = None,
    ) -> Any:
        """
        注册自定义指标
        
        Args:
            name: 指标名称（不含前缀）
            metric_type: 指标类型
            description: 描述
            labels: 标签列表
            buckets: 直方图桶（仅 HISTOGRAM 类型）
        
        Returns:
            Prometheus 指标对象
        """
        if not self._prometheus_available:
            return NoOpMetric()
        
        if name in self._metrics:
            logger.warning(f"Metric '{name}' already exists, returning existing")
            return self._metrics[name]
        
        full_name = f"{self.config.prefix}_{name}"
        labels = labels or []
        
        if metric_type == MetricType.COUNTER:
            metric = self._Counter(full_name, description, labels)
        elif metric_type == MetricType.GAUGE:
            metric = self._Gauge(full_name, description, labels)
        elif metric_type == MetricType.HISTOGRAM:
            buckets = buckets or self.config.latency_buckets
            metric = self._Histogram(full_name, description, labels, buckets=buckets)
        elif metric_type == MetricType.SUMMARY:
            metric = self._Summary(full_name, description, labels)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        self._metrics[name] = metric
        logger.debug(f"Registered custom metric: {name}")
        return metric
    
    def export_text(self) -> str:
        """
        导出文本格式指标
        
        Returns:
            Prometheus 格式的指标文本
        """
        if not self._prometheus_available:
            return "# Prometheus metrics not available\n# Install: pip install prometheus_client\n"
        
        return self._generate_latest(self._registry).decode("utf-8")
    
    @property
    def content_type(self) -> str:
        """获取 Content-Type"""
        return self._content_type
    
    @property
    def enabled(self) -> bool:
        """是否启用"""
        return self._prometheus_available and self.config.enabled
    
    def get_all_metric_names(self) -> List[str]:
        """获取所有已注册的指标名称"""
        return list(self._metrics.keys())
    
    def reset(self):
        """
        重置注册中心（仅用于测试）
        
        警告：这会清除所有指标数据！
        """
        with self._lock:
            MetricsRegistry._instance = None
            self._initialized = False


# ==================== 全局访问函数 ====================

_registry: Optional[MetricsRegistry] = None
_registry_lock = threading.Lock()


def get_metrics_registry(config: Optional[MetricsConfig] = None) -> MetricsRegistry:
    """
    获取全局指标注册中心
    
    Args:
        config: 指标配置（仅首次调用时生效）
    
    Returns:
        MetricsRegistry 单例
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = MetricsRegistry(config)
    return _registry


def reset_metrics_registry():
    """
    重置全局指标注册中心（仅用于测试）
    """
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.reset()
        _registry = None


# ==================== 便捷函数 ====================

def inc_counter(name: str, labels: Dict[str, str] = None, amount: float = 1):
    """增加计数器"""
    metric = get_metrics_registry().get_metric(name)
    if labels:
        metric = metric.labels(**labels)
    metric.inc(amount)


def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """设置仪表值"""
    metric = get_metrics_registry().get_metric(name)
    if labels:
        metric = metric.labels(**labels)
    metric.set(value)


def observe_histogram(name: str, value: float, labels: Dict[str, str] = None):
    """记录直方图观测值"""
    metric = get_metrics_registry().get_metric(name)
    if labels:
        metric = metric.labels(**labels)
    metric.observe(value)


# ==================== 导出 ====================

__all__ = [
    "MetricsConfig",
    "MetricType",
    "MetricsRegistry",
    "NoOpMetric",
    "get_metrics_registry",
    "reset_metrics_registry",
    "inc_counter",
    "set_gauge",
    "observe_histogram",
]
