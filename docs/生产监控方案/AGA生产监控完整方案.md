# AGA 生产监控完整方案

## 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v1.1 |
| 日期 | 2026-02-09 |
| 状态 | **已实施** |
| 作者 | AGA Team |

---

## 一、现状分析

### 1.1 已实现的监控能力

| 模块 | 文件 | 能力 | 完成度 |
|------|------|------|--------|
| **告警规则** | `aga/monitoring/alerts.py` | Prometheus 告警规则定义、Grafana 仪表盘生成 | ✅ 100% |
| **简单 UI** | `aga/monitoring/simple_ui.py` | Flask 轻量级监控界面 | ✅ 100% |
| **持久化指标** | `aga/production/persistence.py` | `PrometheusMetrics` 类，槽位/操作/同步指标 | ✅ 80% |
| **分布式追踪** | `aga/api/tracing.py` | OpenTelemetry 集成、`AGATracer` 类 | ✅ 70% |
| **配置支持** | `aga/production/config.py` | `metrics_enabled`, `metrics_prefix` 配置 | ✅ 100% |

### 1.2 已定义的指标

#### 持久化层指标 (`aga/production/persistence.py`)

```python
# Gauge (仪表)
aga_slots_total{namespace, lifecycle_state}      # 槽位总数
aga_slots_active{namespace}                       # 活跃槽位数

# Counter (计数器)
aga_persistence_operations_total{namespace, operation, status}  # 持久化操作
aga_slot_hits_total{namespace}                    # 槽位命中
aga_sync_records_synced_total                     # 同步记录数
aga_sync_errors_total                             # 同步错误

# Histogram (直方图)
aga_persistence_operation_duration_seconds{namespace, operation}  # 操作延迟
```

#### 追踪层指标 (`aga/api/tracing.py`)

```python
# Counter
aga_forward_total{namespace, gate_result}         # 前向传播次数
aga_injection_total{namespace, lifecycle_state}   # 注入次数
aga_error_total{namespace, error_type}            # 错误次数

# Histogram
aga_forward_latency_seconds{namespace}            # 前向延迟
aga_gate_value{namespace}                         # 门控值分布

# Gauge
aga_active_slots{namespace}                       # 活跃槽位
aga_slot_hit_rate{namespace}                      # 命中率
```

### 1.3 已定义的告警规则

| 告警名称 | 严重级别 | 触发条件 |
|----------|----------|----------|
| `AGAServiceDown` | CRITICAL | 服务不可用 > 1m |
| `AGAHighErrorRate` | WARNING | 错误率 > 10% 持续 5m |
| `AGACriticalErrorRate` | CRITICAL | 错误率 > 50% 持续 2m |
| `AGAHighLatencyP95` | WARNING | P95 延迟 > 500ms 持续 5m |
| `AGAHighLatencyP99` | CRITICAL | P99 延迟 > 1s 持续 5m |
| `AGASlotUtilizationHigh` | WARNING | 槽位使用率 > 90% 持续 10m |
| `AGASlotUtilizationCritical` | CRITICAL | 槽位使用率 > 95% 持续 5m |
| `AGAGateBypassRateHigh` | WARNING | 旁路率 > 50% 持续 10m |
| `AGASyncLagHigh` | WARNING | 同步延迟 > 60s 持续 5m |
| `AGASyncFailed` | CRITICAL | 5m 内同步失败 > 5 次 |
| `AGAPartitionDetected` | CRITICAL | 检测到网络分区 |
| `AGAQuorumLost` | CRITICAL | 失去 Quorum |
| `AGAMemoryUsageHigh` | WARNING | 内存使用率 > 85% |
| `AGAQuarantineRateHigh` | WARNING | 1h 内隔离 > 10 个槽位 |

### 1.4 现有差距分析

| 能力 | 现状 | 差距 |
|------|------|------|
| **指标收集** | 分散在多个模块 | 缺少统一的指标注册中心 |
| **指标导出** | 需要手动集成 | 缺少独立的 `/metrics` HTTP 端点 |
| **ANN 索引监控** | 无 | 新增模块未集成监控 |
| **动态加载器监控** | 无 | 新增模块未集成监控 |
| **日志聚合** | 基础 logging | 缺少结构化日志和 ELK 集成 |
| **链路追踪** | 基础实现 | 缺少完整的 Span 传播 |
| **告警通知** | 本地 AlertManager | 缺少 Webhook/PagerDuty 集成 |
| **SLO/SLI** | 无 | 缺少服务级别目标定义 |

---

## 二、目标架构

### 2.1 监控架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGA 生产监控架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        可视化层 (Visualization)                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │  Grafana    │  │   Kibana    │  │   Jaeger    │  │ Simple UI  │  │    │
│  │  │  仪表盘     │  │   日志查询  │  │   链路追踪  │  │  轻量监控  │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↑                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        存储层 (Storage)                              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │ Prometheus  │  │Elasticsearch│  │   Jaeger    │  │ AlertMgr   │  │    │
│  │  │  时序数据   │  │   日志存储  │  │   追踪存储  │  │  告警管理  │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↑                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        采集层 (Collection)                           │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │  /metrics   │  │  Filebeat   │  │   OTEL      │  │  Health    │  │    │
│  │  │  HTTP 端点  │  │   日志采集  │  │   Collector │  │  Checks    │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↑                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        应用层 (Application)                          │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    AGA Metrics Registry                      │    │    │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │    │    │
│  │  │  │ Gate    │ │ SlotPool│ │ ANN     │ │ Loader  │ │ Sync   │ │    │    │
│  │  │  │ Metrics │ │ Metrics │ │ Metrics │ │ Metrics │ │ Metrics│ │    │    │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    AGA Core Components                       │    │    │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │    │    │
│  │  │  │ Operator│ │ Gate    │ │ SlotPool│ │ ANN     │ │ Loader │ │    │    │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **统一指标注册**：所有指标通过 `MetricsRegistry` 统一管理
2. **零侵入采集**：通过装饰器和上下文管理器自动采集
3. **Fail-Open**：监控失败不影响业务
4. **可观测性三支柱**：Metrics + Logs + Traces 完整覆盖
5. **SLO 驱动**：基于服务级别目标设计告警

---

## 三、详细设计

### 3.1 统一指标注册中心

```python
# aga/monitoring/metrics.py

from dataclasses import dataclass
from typing import Dict, Optional, List
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """指标配置"""
    enabled: bool = True
    prefix: str = "aga"
    include_process_metrics: bool = True
    include_platform_metrics: bool = True
    default_labels: Dict[str, str] = None
    
    # 直方图桶配置
    latency_buckets: tuple = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
    size_buckets: tuple = (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


class MetricsRegistry:
    """
    AGA 统一指标注册中心
    
    单例模式，提供所有 AGA 组件的指标注册和导出。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[MetricsConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        if self._initialized:
            return
        
        self.config = config or MetricsConfig()
        self._metrics: Dict[str, any] = {}
        self._prometheus_available = False
        
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
            
            # 注册所有 AGA 指标
            self._register_all_metrics()
            
            logger.info("Prometheus metrics initialized")
        except ImportError:
            logger.warning("prometheus_client not installed, metrics disabled")
    
    def _register_all_metrics(self):
        """注册所有 AGA 指标"""
        from prometheus_client import Counter, Gauge, Histogram
        
        prefix = self.config.prefix
        
        # ==================== 核心指标 ====================
        
        # 请求指标
        self._metrics["requests_total"] = Counter(
            f"{prefix}_requests_total",
            "Total number of AGA requests",
            ["namespace", "operation", "status"]
        )
        
        self._metrics["request_duration_seconds"] = Histogram(
            f"{prefix}_request_duration_seconds",
            "Request duration in seconds",
            ["namespace", "operation"],
            buckets=self.config.latency_buckets
        )
        
        # 错误指标
        self._metrics["errors_total"] = Counter(
            f"{prefix}_errors_total",
            "Total number of errors",
            ["namespace", "error_type", "component"]
        )
        
        # ==================== 门控指标 ====================
        
        self._metrics["gate_checks_total"] = Counter(
            f"{prefix}_gate_checks_total",
            "Total gate checks",
            ["namespace", "gate_type"]
        )
        
        self._metrics["gate_hits_total"] = Counter(
            f"{prefix}_gate_hits_total",
            "Total gate hits (AGA activated)",
            ["namespace", "gate_type"]
        )
        
        self._metrics["gate_bypasses_total"] = Counter(
            f"{prefix}_gate_bypasses_total",
            "Total gate bypasses",
            ["namespace", "reason"]
        )
        
        self._metrics["gate_early_exits_total"] = Counter(
            f"{prefix}_gate_early_exits_total",
            "Total early exits",
            ["namespace"]
        )
        
        self._metrics["gate_confidence"] = Histogram(
            f"{prefix}_gate_confidence",
            "Gate confidence distribution",
            ["namespace"],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self._metrics["entropy_value"] = Histogram(
            f"{prefix}_entropy_value",
            "Entropy value distribution",
            ["namespace"],
            buckets=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
        )
        
        # ==================== 槽位指标 ====================
        
        self._metrics["slots_total"] = Gauge(
            f"{prefix}_slots_total",
            "Total number of slots",
            ["namespace", "lifecycle_state"]
        )
        
        self._metrics["slots_active"] = Gauge(
            f"{prefix}_slots_active",
            "Number of active slots",
            ["namespace"]
        )
        
        self._metrics["slot_utilization"] = Gauge(
            f"{prefix}_slot_utilization",
            "Slot utilization ratio (0-1)",
            ["namespace"]
        )
        
        self._metrics["slot_hits_total"] = Counter(
            f"{prefix}_slot_hits_total",
            "Total slot hits",
            ["namespace", "lu_id"]
        )
        
        self._metrics["slots_by_trust_tier"] = Gauge(
            f"{prefix}_slots_by_trust_tier",
            "Slots by trust tier",
            ["namespace", "trust_tier"]
        )
        
        # ==================== ANN 索引指标 ====================
        
        self._metrics["ann_index_size"] = Gauge(
            f"{prefix}_ann_index_size",
            "ANN index size (number of vectors)",
            ["namespace", "backend"]
        )
        
        self._metrics["ann_search_total"] = Counter(
            f"{prefix}_ann_search_total",
            "Total ANN searches",
            ["namespace", "status"]
        )
        
        self._metrics["ann_search_duration_seconds"] = Histogram(
            f"{prefix}_ann_search_duration_seconds",
            "ANN search duration",
            ["namespace"],
            buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
        )
        
        self._metrics["ann_candidates_returned"] = Histogram(
            f"{prefix}_ann_candidates_returned",
            "Number of candidates returned by ANN",
            ["namespace"],
            buckets=(10, 25, 50, 100, 150, 200, 300, 500)
        )
        
        self._metrics["ann_rebuild_total"] = Counter(
            f"{prefix}_ann_rebuild_total",
            "Total ANN index rebuilds",
            ["namespace", "trigger"]
        )
        
        self._metrics["ann_deleted_ratio"] = Gauge(
            f"{prefix}_ann_deleted_ratio",
            "Ratio of deleted vectors in ANN index",
            ["namespace"]
        )
        
        # ==================== 动态加载器指标 ====================
        
        self._metrics["loader_requests_total"] = Counter(
            f"{prefix}_loader_requests_total",
            "Total loader requests",
            ["namespace"]
        )
        
        self._metrics["loader_hot_hits_total"] = Counter(
            f"{prefix}_loader_hot_hits_total",
            "Hot tier hits",
            ["namespace"]
        )
        
        self._metrics["loader_warm_hits_total"] = Counter(
            f"{prefix}_loader_warm_hits_total",
            "Warm tier hits",
            ["namespace"]
        )
        
        self._metrics["loader_cold_loads_total"] = Counter(
            f"{prefix}_loader_cold_loads_total",
            "Cold tier loads",
            ["namespace"]
        )
        
        self._metrics["loader_failures_total"] = Counter(
            f"{prefix}_loader_failures_total",
            "Loader failures",
            ["namespace", "reason"]
        )
        
        self._metrics["loader_duration_seconds"] = Histogram(
            f"{prefix}_loader_duration_seconds",
            "Loader duration",
            ["namespace", "tier"],
            buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
        )
        
        self._metrics["warm_cache_size"] = Gauge(
            f"{prefix}_warm_cache_size",
            "Warm cache size",
            ["namespace"]
        )
        
        self._metrics["warm_cache_hit_rate"] = Gauge(
            f"{prefix}_warm_cache_hit_rate",
            "Warm cache hit rate",
            ["namespace"]
        )
        
        # ==================== 同步指标 ====================
        
        self._metrics["sync_lag_seconds"] = Gauge(
            f"{prefix}_sync_lag_seconds",
            "Sync lag in seconds",
            ["instance"]
        )
        
        self._metrics["sync_messages_total"] = Counter(
            f"{prefix}_sync_messages_total",
            "Total sync messages",
            ["type", "direction"]
        )
        
        self._metrics["sync_failures_total"] = Counter(
            f"{prefix}_sync_failures_total",
            "Total sync failures",
            ["type"]
        )
        
        # ==================== 分布式指标 ====================
        
        self._metrics["partition_state"] = Gauge(
            f"{prefix}_partition_state",
            "Partition state (0=normal, 1=degraded, 2=partitioned)",
            ["instance"]
        )
        
        self._metrics["healthy_instances"] = Gauge(
            f"{prefix}_healthy_instances",
            "Number of healthy instances"
        )
        
        self._metrics["quorum_size"] = Gauge(
            f"{prefix}_quorum_size",
            "Required quorum size"
        )
        
        # ==================== 资源指标 ====================
        
        self._metrics["memory_usage_bytes"] = Gauge(
            f"{prefix}_memory_usage_bytes",
            "Memory usage in bytes",
            ["component"]
        )
        
        self._metrics["gpu_memory_usage_bytes"] = Gauge(
            f"{prefix}_gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            ["device"]
        )
        
        # ==================== 治理指标 ====================
        
        self._metrics["quarantine_total"] = Counter(
            f"{prefix}_quarantine_total",
            "Total quarantines",
            ["namespace", "reason"]
        )
        
        self._metrics["lifecycle_transitions_total"] = Counter(
            f"{prefix}_lifecycle_transitions_total",
            "Total lifecycle transitions",
            ["namespace", "from_state", "to_state"]
        )
    
    def get_metric(self, name: str):
        """获取指标"""
        return self._metrics.get(name)
    
    def export_text(self) -> str:
        """导出文本格式指标"""
        if not self._prometheus_available:
            return "# Prometheus metrics not available\n"
        return self._generate_latest(self._registry).decode("utf-8")
    
    @property
    def content_type(self) -> str:
        """获取 Content-Type"""
        return self._content_type if self._prometheus_available else "text/plain"
    
    @property
    def enabled(self) -> bool:
        """是否启用"""
        return self._prometheus_available and self.config.enabled


# 全局实例
_registry: Optional[MetricsRegistry] = None

def get_metrics_registry(config: Optional[MetricsConfig] = None) -> MetricsRegistry:
    """获取全局指标注册中心"""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry(config)
    return _registry
```

### 3.2 指标采集装饰器

```python
# aga/monitoring/decorators.py

import time
import functools
from typing import Optional, Callable
from contextlib import contextmanager

from .metrics import get_metrics_registry


def track_latency(
    operation: str,
    namespace_arg: str = "namespace",
    default_namespace: str = "default",
):
    """
    延迟追踪装饰器
    
    Usage:
        @track_latency("forward")
        def forward(self, hidden_states, namespace="default"):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            registry = get_metrics_registry()
            if not registry.enabled:
                return func(*args, **kwargs)
            
            # 获取 namespace
            namespace = kwargs.get(namespace_arg, default_namespace)
            
            start = time.perf_counter()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start
                registry.get_metric("request_duration_seconds").labels(
                    namespace=namespace,
                    operation=operation
                ).observe(duration)
                registry.get_metric("requests_total").labels(
                    namespace=namespace,
                    operation=operation,
                    status=status
                ).inc()
        
        return wrapper
    return decorator


def track_errors(
    component: str,
    namespace_arg: str = "namespace",
    default_namespace: str = "default",
):
    """
    错误追踪装饰器
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            registry = get_metrics_registry()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if registry.enabled:
                    namespace = kwargs.get(namespace_arg, default_namespace)
                    registry.get_metric("errors_total").labels(
                        namespace=namespace,
                        error_type=type(e).__name__,
                        component=component
                    ).inc()
                raise
        return wrapper
    return decorator


@contextmanager
def track_ann_search(namespace: str = "default"):
    """ANN 搜索追踪上下文"""
    registry = get_metrics_registry()
    if not registry.enabled:
        yield
        return
    
    start = time.perf_counter()
    status = "success"
    candidates = 0
    
    try:
        result = yield
        if hasattr(result, '__len__'):
            candidates = len(result)
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        registry.get_metric("ann_search_total").labels(
            namespace=namespace,
            status=status
        ).inc()
        registry.get_metric("ann_search_duration_seconds").labels(
            namespace=namespace
        ).observe(duration)
        if candidates > 0:
            registry.get_metric("ann_candidates_returned").labels(
                namespace=namespace
            ).observe(candidates)


@contextmanager
def track_loader(namespace: str = "default"):
    """动态加载器追踪上下文"""
    registry = get_metrics_registry()
    if not registry.enabled:
        yield {}
        return
    
    start = time.perf_counter()
    result = {}
    
    try:
        result = yield result
    finally:
        duration = time.perf_counter() - start
        registry.get_metric("loader_requests_total").labels(
            namespace=namespace
        ).inc()
        
        # 记录各层命中
        if result.get("hot_hits", 0) > 0:
            registry.get_metric("loader_hot_hits_total").labels(
                namespace=namespace
            ).inc(result["hot_hits"])
        
        if result.get("warm_hits", 0) > 0:
            registry.get_metric("loader_warm_hits_total").labels(
                namespace=namespace
            ).inc(result["warm_hits"])
        
        if result.get("cold_loads", 0) > 0:
            registry.get_metric("loader_cold_loads_total").labels(
                namespace=namespace
            ).inc(result["cold_loads"])
        
        registry.get_metric("loader_duration_seconds").labels(
            namespace=namespace,
            tier="total"
        ).observe(duration)
```

### 3.3 HTTP 指标端点

```python
# aga/monitoring/http.py

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Response
    from fastapi.responses import PlainTextResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .metrics import get_metrics_registry, MetricsConfig


def add_metrics_endpoint(
    app: "FastAPI",
    path: str = "/metrics",
    config: Optional[MetricsConfig] = None,
):
    """
    添加 Prometheus 指标端点到 FastAPI 应用
    
    Args:
        app: FastAPI 应用
        path: 端点路径
        config: 指标配置
    """
    if not HAS_FASTAPI:
        logger.warning("FastAPI not available, metrics endpoint not added")
        return
    
    registry = get_metrics_registry(config)
    
    @app.get(path, include_in_schema=False)
    async def metrics():
        """Prometheus 指标端点"""
        return Response(
            content=registry.export_text(),
            media_type=registry.content_type,
        )
    
    logger.info(f"Metrics endpoint added at {path}")


def create_metrics_app(
    config: Optional[MetricsConfig] = None,
    port: int = 9090,
) -> "FastAPI":
    """
    创建独立的指标服务应用
    
    用于需要独立指标端口的场景。
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")
    
    from fastapi import FastAPI
    
    app = FastAPI(
        title="AGA Metrics",
        docs_url=None,
        redoc_url=None,
    )
    
    add_metrics_endpoint(app, "/metrics", config)
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return app
```

### 3.4 结构化日志

```python
# aga/monitoring/logging.py

import json
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any


class StructuredFormatter(logging.Formatter):
    """
    结构化 JSON 日志格式化器
    
    输出格式兼容 ELK Stack 和 Loki。
    """
    
    def __init__(
        self,
        service_name: str = "aga",
        include_extra: bool = True,
    ):
        super().__init__()
        self.service_name = service_name
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }
        
        # 添加位置信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # 添加额外字段
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "filename",
                    "funcName", "levelname", "levelno", "lineno",
                    "module", "msecs", "pathname", "process",
                    "processName", "relativeCreated", "stack_info",
                    "exc_info", "exc_text", "thread", "threadName",
                    "message",
                ):
                    try:
                        json.dumps(value)  # 确保可序列化
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_structured_logging(
    service_name: str = "aga",
    level: str = "INFO",
    output: str = "stdout",
    log_file: Optional[str] = None,
):
    """
    配置结构化日志
    
    Args:
        service_name: 服务名称
        level: 日志级别
        output: 输出目标 (stdout, file, both)
        log_file: 日志文件路径
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    formatter = StructuredFormatter(service_name=service_name)
    
    # stdout 处理器
    if output in ("stdout", "both"):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)
    
    # 文件处理器
    if output in ("file", "both") and log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


class LogContext:
    """
    日志上下文管理器
    
    用于在日志中添加请求级别的上下文信息。
    """
    
    _context: Dict[str, Any] = {}
    
    @classmethod
    def set(cls, **kwargs):
        """设置上下文"""
        cls._context.update(kwargs)
    
    @classmethod
    def get(cls, key: str, default=None):
        """获取上下文"""
        return cls._context.get(key, default)
    
    @classmethod
    def clear(cls):
        """清除上下文"""
        cls._context.clear()
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """获取所有上下文"""
        return cls._context.copy()
```

### 3.5 SLO/SLI 定义

```python
# aga/monitoring/slo.py

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class SLOType(str, Enum):
    """SLO 类型"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class SLO:
    """服务级别目标"""
    name: str
    type: SLOType
    target: float
    window: str  # e.g., "30d", "7d"
    description: str
    
    # PromQL 表达式
    sli_query: str
    
    # 告警配置
    alert_burn_rate_1h: float = 14.4  # 1h 内消耗 2% 错误预算
    alert_burn_rate_6h: float = 6.0   # 6h 内消耗 5% 错误预算


# AGA 服务级别目标
AGA_SLOS = [
    SLO(
        name="aga_availability",
        type=SLOType.AVAILABILITY,
        target=0.999,  # 99.9%
        window="30d",
        description="AGA 服务可用性",
        sli_query='sum(rate(aga_requests_total{status="success"}[5m])) / sum(rate(aga_requests_total[5m]))',
    ),
    SLO(
        name="aga_latency_p99",
        type=SLOType.LATENCY,
        target=0.99,  # 99% 请求 < 100ms
        window="30d",
        description="AGA P99 延迟 < 100ms",
        sli_query='histogram_quantile(0.99, rate(aga_request_duration_seconds_bucket[5m])) < 0.1',
    ),
    SLO(
        name="aga_error_rate",
        type=SLOType.ERROR_RATE,
        target=0.001,  # < 0.1%
        window="30d",
        description="AGA 错误率 < 0.1%",
        sli_query='sum(rate(aga_errors_total[5m])) / sum(rate(aga_requests_total[5m]))',
    ),
    SLO(
        name="aga_ann_latency",
        type=SLOType.LATENCY,
        target=0.99,  # 99% ANN 搜索 < 10ms
        window="30d",
        description="ANN 搜索 P99 延迟 < 10ms",
        sli_query='histogram_quantile(0.99, rate(aga_ann_search_duration_seconds_bucket[5m])) < 0.01',
    ),
]


def generate_slo_recording_rules() -> Dict[str, Any]:
    """生成 SLO 记录规则"""
    rules = []
    
    for slo in AGA_SLOS:
        # SLI 记录规则
        rules.append({
            "record": f"sli:{slo.name}",
            "expr": slo.sli_query,
        })
        
        # 错误预算消耗率
        rules.append({
            "record": f"slo:{slo.name}:error_budget_remaining",
            "expr": f"1 - ((1 - sli:{slo.name}) / (1 - {slo.target}))",
        })
    
    return {
        "groups": [
            {
                "name": "aga_slo_rules",
                "interval": "30s",
                "rules": rules,
            }
        ]
    }


def generate_slo_alerts() -> List[Dict[str, Any]]:
    """生成 SLO 告警规则"""
    alerts = []
    
    for slo in AGA_SLOS:
        # 快速燃烧告警 (1h)
        alerts.append({
            "alert": f"{slo.name}_fast_burn",
            "expr": f"slo:{slo.name}:error_budget_remaining < {1 - slo.alert_burn_rate_1h * (1 - slo.target)}",
            "for": "2m",
            "labels": {
                "severity": "critical",
                "slo": slo.name,
            },
            "annotations": {
                "summary": f"SLO {slo.name} 快速燃烧",
                "description": f"错误预算在 1 小时内消耗过快，当前剩余: {{{{ $value | printf \"%.2f\" }}}}%",
            },
        })
        
        # 慢速燃烧告警 (6h)
        alerts.append({
            "alert": f"{slo.name}_slow_burn",
            "expr": f"slo:{slo.name}:error_budget_remaining < {1 - slo.alert_burn_rate_6h * (1 - slo.target)}",
            "for": "1h",
            "labels": {
                "severity": "warning",
                "slo": slo.name,
            },
            "annotations": {
                "summary": f"SLO {slo.name} 慢速燃烧",
                "description": f"错误预算在 6 小时内持续消耗，当前剩余: {{{{ $value | printf \"%.2f\" }}}}%",
            },
        })
    
    return alerts
```

---

## 四、新增告警规则

### 4.1 ANN 索引告警

```yaml
# ANN 索引告警规则
- alert: AGAANNSearchLatencyHigh
  expr: histogram_quantile(0.99, rate(aga_ann_search_duration_seconds_bucket[5m])) > 0.01
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "ANN 搜索延迟过高"
    description: "ANN 搜索 P99 延迟超过 10ms，当前值: {{ $value | printf \"%.3f\" }}s"

- alert: AGAANNSearchErrors
  expr: rate(aga_ann_search_total{status="error"}[5m]) > 0.01
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "ANN 搜索错误率过高"
    description: "ANN 搜索错误率超过 1%"

- alert: AGAANNIndexNeedsRebuild
  expr: aga_ann_deleted_ratio > 0.3
  for: 10m
  labels:
    severity: info
  annotations:
    summary: "ANN 索引需要重建"
    description: "ANN 索引删除比例超过 30%，建议重建"

- alert: AGAANNIndexSizeLow
  expr: aga_ann_index_size < 100
  for: 10m
  labels:
    severity: info
  annotations:
    summary: "ANN 索引规模过小"
    description: "ANN 索引向量数少于 100，可能未正确初始化"
```

### 4.2 动态加载器告警

```yaml
# 动态加载器告警规则
- alert: AGALoaderColdLoadRateHigh
  expr: |
    rate(aga_loader_cold_loads_total[5m]) / 
    rate(aga_loader_requests_total[5m]) > 0.3
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Cold 加载率过高"
    description: "超过 30% 的请求需要从 Cold 层加载，可能影响延迟"

- alert: AGALoaderFailureRateHigh
  expr: |
    rate(aga_loader_failures_total[5m]) / 
    rate(aga_loader_requests_total[5m]) > 0.05
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "加载器失败率过高"
    description: "动态加载器失败率超过 5%"

- alert: AGAWarmCacheHitRateLow
  expr: aga_warm_cache_hit_rate < 0.3
  for: 15m
  labels:
    severity: info
  annotations:
    summary: "Warm 缓存命中率过低"
    description: "Warm 缓存命中率低于 30%，可能需要调整缓存大小"
```

---

## 五、Grafana 仪表盘扩展

### 5.1 新增面板

#### ANN 索引面板

```json
{
  "title": "ANN 索引性能",
  "type": "timeseries",
  "targets": [
    {
      "expr": "histogram_quantile(0.50, rate(aga_ann_search_duration_seconds_bucket[5m]))",
      "legendFormat": "P50"
    },
    {
      "expr": "histogram_quantile(0.95, rate(aga_ann_search_duration_seconds_bucket[5m]))",
      "legendFormat": "P95"
    },
    {
      "expr": "histogram_quantile(0.99, rate(aga_ann_search_duration_seconds_bucket[5m]))",
      "legendFormat": "P99"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "s"
    }
  }
}
```

#### 动态加载器面板

```json
{
  "title": "加载器命中分布",
  "type": "piechart",
  "targets": [
    {
      "expr": "sum(rate(aga_loader_hot_hits_total[5m]))",
      "legendFormat": "Hot"
    },
    {
      "expr": "sum(rate(aga_loader_warm_hits_total[5m]))",
      "legendFormat": "Warm"
    },
    {
      "expr": "sum(rate(aga_loader_cold_loads_total[5m]))",
      "legendFormat": "Cold"
    }
  ]
}
```

---

## 六、开发计划

### 6.1 阶段划分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGA 监控开发计划                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: 基础设施 (2 周)                                                    │
│  ├── Week 1: 统一指标注册中心                                                │
│  │   ├── MetricsRegistry 实现                                               │
│  │   ├── 装饰器和上下文管理器                                                │
│  │   └── HTTP 端点集成                                                      │
│  └── Week 2: ANN/Loader 指标集成                                            │
│      ├── ann_index.py 指标埋点                                              │
│      ├── dynamic_loader.py 指标埋点                                         │
│      └── 单元测试                                                           │
│                                                                              │
│  Phase 2: 可视化 (1 周)                                                      │
│  ├── Grafana 仪表盘更新                                                      │
│  │   ├── ANN 索引面板                                                       │
│  │   ├── 动态加载器面板                                                      │
│  │   └── SLO 面板                                                           │
│  └── 告警规则更新                                                            │
│      ├── ANN 告警规则                                                        │
│      └── Loader 告警规则                                                     │
│                                                                              │
│  Phase 3: 日志与追踪 (1 周)                                                  │
│  ├── 结构化日志实现                                                          │
│  ├── OpenTelemetry 完整集成                                                  │
│  └── 日志上下文传播                                                          │
│                                                                              │
│  Phase 4: SLO/SLI (1 周)                                                     │
│  ├── SLO 定义和记录规则                                                      │
│  ├── 错误预算告警                                                            │
│  └── SLO 仪表盘                                                              │
│                                                                              │
│  Phase 5: 文档与测试 (1 周)                                                  │
│  ├── 运维手册                                                                │
│  ├── 告警响应指南                                                            │
│  └── 集成测试                                                                │
│                                                                              │
│  总计: 6 周                                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 详细任务清单

#### Phase 1: 基础设施 (Week 1-2)

| 任务 | 文件 | 优先级 | 预估工时 |
|------|------|--------|----------|
| 实现 MetricsRegistry | `aga/monitoring/metrics.py` | P0 | 4h |
| 实现装饰器 | `aga/monitoring/decorators.py` | P0 | 2h |
| 实现 HTTP 端点 | `aga/monitoring/http.py` | P0 | 2h |
| 集成到 FastAPI | `aga/api/app.py` | P0 | 1h |
| ANN 指标埋点 | `aga/retrieval/ann_index.py` | P0 | 3h |
| Loader 指标埋点 | `aga/retrieval/dynamic_loader.py` | P0 | 3h |
| Operator 指标集成 | `aga/production/operator.py` | P1 | 2h |
| 单元测试 | `tests/unit/monitoring/` | P1 | 4h |

#### Phase 2: 可视化 (Week 3)

| 任务 | 文件 | 优先级 | 预估工时 |
|------|------|--------|----------|
| 更新 Grafana 仪表盘 | `aga/monitoring/alerts.py` | P0 | 4h |
| ANN 告警规则 | `configs/prometheus/aga_alerts.yml` | P0 | 2h |
| Loader 告警规则 | `configs/prometheus/aga_alerts.yml` | P0 | 2h |
| 导出脚本 | `scripts/export_monitoring.py` | P1 | 2h |

#### Phase 3: 日志与追踪 (Week 4)

| 任务 | 文件 | 优先级 | 预估工时 |
|------|------|--------|----------|
| 结构化日志 | `aga/monitoring/logging.py` | P0 | 3h |
| 日志上下文 | `aga/monitoring/logging.py` | P1 | 2h |
| OTEL 完整集成 | `aga/api/tracing.py` | P1 | 4h |
| Span 传播 | `aga/production/operator.py` | P2 | 3h |

#### Phase 4: SLO/SLI (Week 5)

| 任务 | 文件 | 优先级 | 预估工时 |
|------|------|--------|----------|
| SLO 定义 | `aga/monitoring/slo.py` | P0 | 3h |
| 记录规则生成 | `aga/monitoring/slo.py` | P0 | 2h |
| 错误预算告警 | `aga/monitoring/slo.py` | P1 | 2h |
| SLO 仪表盘 | `aga/monitoring/alerts.py` | P1 | 3h |

#### Phase 5: 文档与测试 (Week 6)

| 任务 | 文件 | 优先级 | 预估工时 |
|------|------|--------|----------|
| 运维手册 | `docs/operations/monitoring.md` | P0 | 4h |
| 告警响应指南 | `docs/operations/alert_runbook.md` | P0 | 4h |
| 集成测试 | `tests/integration/test_monitoring.py` | P1 | 4h |
| 性能测试 | `tests/performance/test_metrics_overhead.py` | P2 | 2h |

### 6.3 里程碑

| 里程碑 | 日期 | 交付物 |
|--------|------|--------|
| M1: 指标基础 | Week 2 | MetricsRegistry, 装饰器, HTTP 端点 |
| M2: 可视化 | Week 3 | 更新的 Grafana 仪表盘, 新告警规则 |
| M3: 可观测性 | Week 4 | 结构化日志, OTEL 集成 |
| M4: SLO | Week 5 | SLO 定义, 错误预算告警 |
| M5: 文档 | Week 6 | 运维手册, 告警响应指南 |

---

## 七、部署配置

### 7.1 Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/rules/aga_alerts.yml
  - /etc/prometheus/rules/aga_slo.yml

scrape_configs:
  - job_name: 'aga'
    static_configs:
      - targets: ['aga-api:8081']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'aga-portal'
    static_configs:
      - targets: ['aga-portal:8081']
    metrics_path: /metrics

  - job_name: 'aga-runtime'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: aga-runtime
        action: keep
```

### 7.2 Alertmanager 配置

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://aga-alert-handler:8080/webhook'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<PAGERDUTY_KEY>'
        severity: critical

  - name: 'slack'
    slack_configs:
      - api_url: '<SLACK_WEBHOOK_URL>'
        channel: '#aga-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'
```

### 7.3 Grafana 数据源

```yaml
# grafana/provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: false

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: false
```

---

## 八、运维指南

### 8.1 常用查询

```promql
# 服务可用性
sum(rate(aga_requests_total{status="success"}[5m])) / sum(rate(aga_requests_total[5m]))

# P99 延迟
histogram_quantile(0.99, rate(aga_request_duration_seconds_bucket[5m]))

# 槽位使用率
sum(aga_slots_active) / sum(aga_slots_total)

# ANN 搜索延迟
histogram_quantile(0.95, rate(aga_ann_search_duration_seconds_bucket[5m]))

# 加载器命中率
sum(rate(aga_loader_hot_hits_total[5m])) / sum(rate(aga_loader_requests_total[5m]))

# 错误预算剩余
1 - ((1 - sli:aga_availability) / (1 - 0.999))
```

### 8.2 告警响应

| 告警 | 响应步骤 |
|------|----------|
| `AGAServiceDown` | 1. 检查服务日志 2. 检查资源使用 3. 重启服务 |
| `AGAHighLatencyP99` | 1. 检查 ANN 索引状态 2. 检查 Cold 加载率 3. 扩容 |
| `AGASlotUtilizationCritical` | 1. 清理过期槽位 2. 扩展槽位容量 3. 检查泄漏 |
| `AGAANNSearchLatencyHigh` | 1. 检查索引大小 2. 触发重建 3. 调整 nprobe |
| `AGALoaderColdLoadRateHigh` | 1. 增加 Warm 缓存 2. 预热热点知识 3. 检查访问模式 |

---

## 九、总结

本方案提供了 AGA 生产监控的完整设计，包括：

1. **统一指标注册中心**：解决指标分散问题
2. **新增 ANN/Loader 监控**：覆盖新增模块
3. **SLO/SLI 体系**：建立服务级别目标
4. **结构化日志**：支持 ELK 集成
5. **完整告警规则**：覆盖所有关键场景
6. **6 周开发计划**：分阶段实施

预期收益：
- 问题发现时间从小时级降低到分钟级
- 故障定位效率提升 80%
- SLO 达成率可量化追踪
