"""
AGA 指标采集装饰器和上下文管理器

提供零侵入的指标采集能力：
1. 函数装饰器：自动记录延迟和错误
2. 上下文管理器：灵活的指标采集
3. 支持同步和异步函数

版本: v1.0
"""
import time
import functools
import asyncio
import logging
from typing import Optional, Callable, Dict, Any, TypeVar, Union
from contextlib import contextmanager, asynccontextmanager

from .metrics import get_metrics_registry

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ==================== 延迟追踪装饰器 ====================

def track_latency(
    operation: str,
    namespace_arg: str = "namespace",
    default_namespace: str = "default",
    record_errors: bool = True,
):
    """
    延迟追踪装饰器
    
    自动记录函数执行延迟和请求计数。
    
    Args:
        operation: 操作名称
        namespace_arg: 命名空间参数名
        default_namespace: 默认命名空间
        record_errors: 是否记录错误
    
    Usage:
        ```python
        @track_latency("forward")
        def forward(self, hidden_states, namespace="default"):
            ...
        
        @track_latency("inject", namespace_arg="ns")
        async def inject_knowledge(self, data, ns="default"):
            ...
        ```
    """
    def decorator(func: F) -> F:
        # 检查是否是异步函数
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                registry = get_metrics_registry()
                if not registry.enabled:
                    return await func(*args, **kwargs)
                
                # 获取 namespace
                namespace = _extract_namespace(args, kwargs, namespace_arg, default_namespace, func)
                
                start = time.perf_counter()
                status = "success"
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    if record_errors:
                        registry.get_metric("errors_total").labels(
                            namespace=namespace,
                            error_type=type(e).__name__,
                            component=operation
                        ).inc()
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
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                registry = get_metrics_registry()
                if not registry.enabled:
                    return func(*args, **kwargs)
                
                # 获取 namespace
                namespace = _extract_namespace(args, kwargs, namespace_arg, default_namespace, func)
                
                start = time.perf_counter()
                status = "success"
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    if record_errors:
                        registry.get_metric("errors_total").labels(
                            namespace=namespace,
                            error_type=type(e).__name__,
                            component=operation
                        ).inc()
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
            
            return sync_wrapper
    
    return decorator


def _extract_namespace(
    args: tuple,
    kwargs: dict,
    namespace_arg: str,
    default_namespace: str,
    func: Callable,
) -> str:
    """从函数参数中提取 namespace"""
    # 先从 kwargs 中查找
    if namespace_arg in kwargs:
        return kwargs[namespace_arg]
    
    # 尝试从位置参数中查找
    try:
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if namespace_arg in params:
            idx = params.index(namespace_arg)
            # 考虑 self 参数
            if idx < len(args):
                return args[idx]
    except Exception:
        pass
    
    return default_namespace


# ==================== 错误追踪装饰器 ====================

def track_errors(
    component: str,
    namespace_arg: str = "namespace",
    default_namespace: str = "default",
    reraise: bool = True,
):
    """
    错误追踪装饰器
    
    记录函数执行中的错误。
    
    Args:
        component: 组件名称
        namespace_arg: 命名空间参数名
        default_namespace: 默认命名空间
        reraise: 是否重新抛出异常
    
    Usage:
        ```python
        @track_errors("persistence")
        def save_slot(self, slot, namespace="default"):
            ...
        ```
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                registry = get_metrics_registry()
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if registry.enabled:
                        namespace = _extract_namespace(
                            args, kwargs, namespace_arg, default_namespace, func
                        )
                        registry.get_metric("errors_total").labels(
                            namespace=namespace,
                            error_type=type(e).__name__,
                            component=component
                        ).inc()
                    if reraise:
                        raise
                    logger.exception(f"Error in {component}: {e}")
                    return None
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                registry = get_metrics_registry()
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if registry.enabled:
                        namespace = _extract_namespace(
                            args, kwargs, namespace_arg, default_namespace, func
                        )
                        registry.get_metric("errors_total").labels(
                            namespace=namespace,
                            error_type=type(e).__name__,
                            component=component
                        ).inc()
                    if reraise:
                        raise
                    logger.exception(f"Error in {component}: {e}")
                    return None
            return sync_wrapper
    return decorator


# ==================== 上下文管理器 ====================

@contextmanager
def track_operation(
    operation: str,
    namespace: str = "default",
    extra_labels: Dict[str, str] = None,
):
    """
    通用操作追踪上下文管理器
    
    Args:
        operation: 操作名称
        namespace: 命名空间
        extra_labels: 额外标签
    
    Usage:
        ```python
        with track_operation("batch_save", namespace="production"):
            save_batch(slots)
        ```
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        yield {}
        return
    
    start = time.perf_counter()
    status = "success"
    context = {"start_time": start}
    
    try:
        yield context
    except Exception as e:
        status = "error"
        context["error"] = str(e)
        context["error_type"] = type(e).__name__
        registry.get_metric("errors_total").labels(
            namespace=namespace,
            error_type=type(e).__name__,
            component=operation
        ).inc()
        raise
    finally:
        duration = time.perf_counter() - start
        context["duration"] = duration
        context["status"] = status
        
        registry.get_metric("request_duration_seconds").labels(
            namespace=namespace,
            operation=operation
        ).observe(duration)
        registry.get_metric("requests_total").labels(
            namespace=namespace,
            operation=operation,
            status=status
        ).inc()


@contextmanager
def track_ann_search(namespace: str = "default", backend: str = "faiss"):
    """
    ANN 搜索追踪上下文管理器
    
    Args:
        namespace: 命名空间
        backend: ANN 后端类型
    
    Usage:
        ```python
        with track_ann_search(namespace="default") as ctx:
            results = index.search(query, k=100)
            ctx["candidates"] = len(results)
        ```
    
    Yields:
        上下文字典，可设置 candidates 等信息
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        yield {}
        return
    
    start = time.perf_counter()
    status = "success"
    context = {"candidates": 0}
    
    try:
        yield context
    except Exception as e:
        status = "error"
        registry.get_metric("errors_total").labels(
            namespace=namespace,
            error_type=type(e).__name__,
            component="ann_search"
        ).inc()
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
        
        if context.get("candidates", 0) > 0:
            registry.get_metric("ann_candidates_returned").labels(
                namespace=namespace
            ).observe(context["candidates"])


@contextmanager
def track_loader(namespace: str = "default"):
    """
    动态加载器追踪上下文管理器
    
    Args:
        namespace: 命名空间
    
    Usage:
        ```python
        with track_loader(namespace="default") as ctx:
            result = loader.load_candidates(lu_ids)
            ctx["hot_hits"] = result.hot_hits
            ctx["warm_hits"] = result.warm_hits
            ctx["cold_loads"] = result.cold_loads
        ```
    
    Yields:
        上下文字典，可设置 hot_hits, warm_hits, cold_loads 等信息
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        yield {}
        return
    
    start = time.perf_counter()
    context = {
        "hot_hits": 0,
        "warm_hits": 0,
        "cold_loads": 0,
        "failures": 0,
    }
    
    try:
        yield context
    except Exception as e:
        context["failures"] += 1
        registry.get_metric("loader_failures_total").labels(
            namespace=namespace,
            reason=type(e).__name__
        ).inc()
        raise
    finally:
        duration = time.perf_counter() - start
        
        registry.get_metric("loader_requests_total").labels(
            namespace=namespace
        ).inc()
        
        # 记录各层命中
        if context.get("hot_hits", 0) > 0:
            registry.get_metric("loader_hot_hits_total").labels(
                namespace=namespace
            ).inc(context["hot_hits"])
        
        if context.get("warm_hits", 0) > 0:
            registry.get_metric("loader_warm_hits_total").labels(
                namespace=namespace
            ).inc(context["warm_hits"])
        
        if context.get("cold_loads", 0) > 0:
            registry.get_metric("loader_cold_loads_total").labels(
                namespace=namespace
            ).inc(context["cold_loads"])
        
        registry.get_metric("loader_duration_seconds").labels(
            namespace=namespace,
            tier="total"
        ).observe(duration)


@contextmanager
def track_gate(namespace: str = "default", gate_type: str = "gate1"):
    """
    门控追踪上下文管理器
    
    Args:
        namespace: 命名空间
        gate_type: 门控类型 (gate0, gate1, gate2)
    
    Usage:
        ```python
        with track_gate(namespace="default", gate_type="gate1") as ctx:
            result = gate.check(hidden_states)
            ctx["hit"] = result.should_activate
            ctx["confidence"] = result.confidence
            ctx["bypass"] = result.bypass
            ctx["bypass_reason"] = result.bypass_reason
        ```
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        yield {}
        return
    
    context = {
        "hit": False,
        "confidence": 0.0,
        "bypass": False,
        "bypass_reason": None,
        "early_exit": False,
    }
    
    try:
        yield context
    finally:
        registry.get_metric("gate_checks_total").labels(
            namespace=namespace,
            gate_type=gate_type
        ).inc()
        
        if context.get("hit"):
            registry.get_metric("gate_hits_total").labels(
                namespace=namespace,
                gate_type=gate_type
            ).inc()
        
        if context.get("bypass"):
            reason = context.get("bypass_reason", "unknown")
            registry.get_metric("gate_bypasses_total").labels(
                namespace=namespace,
                reason=reason
            ).inc()
        
        if context.get("early_exit"):
            registry.get_metric("gate_early_exits_total").labels(
                namespace=namespace
            ).inc()
        
        if "confidence" in context and context["confidence"] is not None:
            registry.get_metric("gate_confidence").labels(
                namespace=namespace
            ).observe(context["confidence"])


@contextmanager
def track_persistence(
    operation: str,
    namespace: str = "default",
):
    """
    持久化操作追踪上下文管理器
    
    Args:
        operation: 操作类型 (save, load, delete, etc.)
        namespace: 命名空间
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        yield {}
        return
    
    start = time.perf_counter()
    status = "success"
    context = {}
    
    try:
        yield context
    except Exception as e:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        
        registry.get_metric("persistence_operations_total").labels(
            namespace=namespace,
            operation=operation,
            status=status
        ).inc()
        
        registry.get_metric("persistence_operation_duration_seconds").labels(
            namespace=namespace,
            operation=operation
        ).observe(duration)


# ==================== 异步上下文管理器 ====================

@asynccontextmanager
async def async_track_operation(
    operation: str,
    namespace: str = "default",
):
    """
    异步操作追踪上下文管理器
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        yield {}
        return
    
    start = time.perf_counter()
    status = "success"
    context = {}
    
    try:
        yield context
    except Exception as e:
        status = "error"
        registry.get_metric("errors_total").labels(
            namespace=namespace,
            error_type=type(e).__name__,
            component=operation
        ).inc()
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


# ==================== 辅助函数 ====================

def record_slot_metrics(
    namespace: str,
    total_slots: int,
    active_slots: int,
    slots_by_state: Dict[str, int] = None,
    slots_by_tier: Dict[str, int] = None,
):
    """
    记录槽位指标
    
    Args:
        namespace: 命名空间
        total_slots: 总槽位数
        active_slots: 活跃槽位数
        slots_by_state: 按状态分布
        slots_by_tier: 按信任层级分布
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        return
    
    registry.get_metric("slots_active").labels(
        namespace=namespace
    ).set(active_slots)
    
    if total_slots > 0:
        registry.get_metric("slot_utilization").labels(
            namespace=namespace
        ).set(active_slots / total_slots)
    
    if slots_by_state:
        for state, count in slots_by_state.items():
            registry.get_metric("slots_total").labels(
                namespace=namespace,
                lifecycle_state=state
            ).set(count)
    
    if slots_by_tier:
        for tier, count in slots_by_tier.items():
            registry.get_metric("slots_by_trust_tier").labels(
                namespace=namespace,
                trust_tier=tier
            ).set(count)


def record_ann_index_metrics(
    namespace: str,
    backend: str,
    total_vectors: int,
    active_vectors: int,
    deleted_ratio: float,
):
    """
    记录 ANN 索引指标
    
    Args:
        namespace: 命名空间
        backend: 后端类型
        total_vectors: 总向量数
        active_vectors: 活跃向量数
        deleted_ratio: 删除比例
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        return
    
    registry.get_metric("ann_index_size").labels(
        namespace=namespace,
        backend=backend
    ).set(total_vectors)
    
    registry.get_metric("ann_total_vectors").labels(
        namespace=namespace
    ).set(total_vectors)
    
    registry.get_metric("ann_active_vectors").labels(
        namespace=namespace
    ).set(active_vectors)
    
    registry.get_metric("ann_deleted_ratio").labels(
        namespace=namespace
    ).set(deleted_ratio)


def record_cache_metrics(
    namespace: str,
    cache_type: str,
    size: int,
    hit_rate: float = None,
):
    """
    记录缓存指标
    
    Args:
        namespace: 命名空间
        cache_type: 缓存类型
        size: 缓存大小
        hit_rate: 命中率
    """
    registry = get_metrics_registry()
    if not registry.enabled:
        return
    
    registry.get_metric("cache_size").labels(
        namespace=namespace,
        cache_type=cache_type
    ).set(size)
    
    if cache_type == "warm":
        registry.get_metric("warm_cache_size").labels(
            namespace=namespace
        ).set(size)
        
        if hit_rate is not None:
            registry.get_metric("warm_cache_hit_rate").labels(
                namespace=namespace
            ).set(hit_rate)


# ==================== 导出 ====================

__all__ = [
    # 装饰器
    "track_latency",
    "track_errors",
    # 上下文管理器
    "track_operation",
    "track_ann_search",
    "track_loader",
    "track_gate",
    "track_persistence",
    "async_track_operation",
    # 辅助函数
    "record_slot_metrics",
    "record_ann_index_metrics",
    "record_cache_metrics",
]
