"""
aga_observability/prometheus_exporter.py — Prometheus 指标导出

订阅 aga-core 的 EventBus 事件，转换为 Prometheus 指标。

指标清单:
  Counters:
    - aga_forward_total{layer, applied}     — forward 调用总次数
    - aga_retrieval_total{layer}            — 召回器调用总次数
    - aga_retrieval_injected_total{layer}   — 召回注入知识总数
    - aga_audit_operations_total{operation, success} — 审计操作总数

  Histograms:
    - aga_forward_latency_us{layer}         — forward 延迟分布 (μs)
    - aga_gate_value{layer}                 — 门控值分布
    - aga_entropy_value{layer}              — 熵值分布
    - aga_retrieval_score                   — 召回分数分布

  Gauges:
    - aga_knowledge_count                   — 当前知识数量
    - aga_knowledge_utilization             — KVStore 利用率
    - aga_knowledge_pinned_count            — 锁定知识数量
    - aga_activation_rate                   — 激活率（滑动窗口）
    - aga_slot_change_rate                  — Slot 变化率
"""
import time
import logging
import threading
from typing import Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

# 尝试导入 prometheus_client（可选依赖）
_PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        start_http_server,
        CollectorRegistry,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    pass


class PrometheusExporter:
    """
    Prometheus 指标导出器

    订阅 EventBus 的 forward / retrieval / audit 事件，
    转换为标准 Prometheus 指标。

    使用方式:
        exporter = PrometheusExporter(config)
        exporter.subscribe(event_bus)
        exporter.start_server()  # 启动 HTTP 端点
        # ... 推理 ...
        exporter.shutdown()
    """

    SUBSCRIBER_ID = "aga-observability-prometheus"

    def __init__(
        self,
        prefix: str = "aga",
        port: int = 9090,
        labels: Optional[Dict[str, str]] = None,
        registry: Optional[Any] = None,
    ):
        if not _PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client 未安装。请运行: pip install prometheus-client"
            )

        self._prefix = prefix
        self._port = port
        self._global_labels = labels or {}
        self._registry = registry or REGISTRY
        self._server_started = False
        self._subscribed = False

        # 滑动窗口（用于计算实时激活率）
        self._window_size = 1000
        self._recent_applied = deque(maxlen=self._window_size)

        # 创建指标
        self._create_metrics()

        logger.info(
            f"PrometheusExporter 初始化: prefix={prefix}, port={port}"
        )

    def _create_metrics(self):
        """创建所有 Prometheus 指标"""
        p = self._prefix
        reg = self._registry

        # === Counters ===
        self.forward_total = Counter(
            f"{p}_forward_total",
            "AGA forward 调用总次数",
            ["layer", "applied"],
            registry=reg,
        )

        self.retrieval_total = Counter(
            f"{p}_retrieval_total",
            "召回器调用总次数",
            ["layer"],
            registry=reg,
        )

        self.retrieval_injected_total = Counter(
            f"{p}_retrieval_injected_total",
            "召回注入知识总数",
            ["layer"],
            registry=reg,
        )

        self.audit_operations_total = Counter(
            f"{p}_audit_operations_total",
            "审计操作总数",
            ["operation", "success"],
            registry=reg,
        )

        # === Histograms ===
        self.forward_latency = Histogram(
            f"{p}_forward_latency_us",
            "AGA forward 延迟 (μs)",
            ["layer"],
            buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=reg,
        )

        self.gate_value = Histogram(
            f"{p}_gate_value",
            "门控值分布",
            ["layer"],
            buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=reg,
        )

        self.entropy_value = Histogram(
            f"{p}_entropy_value",
            "熵值分布",
            ["layer"],
            buckets=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0],
            registry=reg,
        )

        self.retrieval_score = Histogram(
            f"{p}_retrieval_score",
            "召回分数分布",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=reg,
        )

        # === Gauges ===
        self.knowledge_count = Gauge(
            f"{p}_knowledge_count",
            "当前知识数量",
            registry=reg,
        )

        self.knowledge_utilization = Gauge(
            f"{p}_knowledge_utilization",
            "KVStore 利用率",
            registry=reg,
        )

        self.knowledge_pinned = Gauge(
            f"{p}_knowledge_pinned_count",
            "锁定知识数量",
            registry=reg,
        )

        self.activation_rate = Gauge(
            f"{p}_activation_rate",
            "激活率（滑动窗口）",
            registry=reg,
        )

        self.slot_change_rate = Gauge(
            f"{p}_slot_change_rate",
            "Slot 变化率",
            registry=reg,
        )

        # === Info ===
        self.info = Info(
            f"{p}_build",
            "AGA 构建信息",
            registry=reg,
        )

    def subscribe(self, event_bus) -> None:
        """
        订阅 EventBus 事件

        Args:
            event_bus: aga-core 的 EventBus 实例
        """
        event_bus.subscribe(
            "forward",
            self._on_forward,
            subscriber_id=self.SUBSCRIBER_ID,
        )
        event_bus.subscribe(
            "retrieval",
            self._on_retrieval,
            subscriber_id=self.SUBSCRIBER_ID,
        )
        event_bus.subscribe(
            "audit",
            self._on_audit,
            subscriber_id=self.SUBSCRIBER_ID,
        )
        self._subscribed = True
        logger.info("PrometheusExporter 已订阅 EventBus")

    def unsubscribe(self, event_bus) -> None:
        """取消订阅"""
        event_bus.unsubscribe(
            "forward",
            subscriber_id=self.SUBSCRIBER_ID,
        )
        event_bus.unsubscribe(
            "retrieval",
            subscriber_id=self.SUBSCRIBER_ID,
        )
        event_bus.unsubscribe(
            "audit",
            subscriber_id=self.SUBSCRIBER_ID,
        )
        self._subscribed = False

    def _on_forward(self, event) -> None:
        """处理 forward 事件"""
        try:
            data = event.data
            layer = str(data.get("layer_idx", 0))
            applied = data.get("aga_applied", False)
            gate_mean = data.get("gate_mean", 0.0)
            entropy_mean = data.get("entropy_mean", 0.0)
            latency_us = data.get("latency_us", 0.0)

            # Counter
            self.forward_total.labels(
                layer=layer,
                applied=str(applied).lower(),
            ).inc()

            # Histograms
            if latency_us > 0:
                self.forward_latency.labels(layer=layer).observe(latency_us)
            self.gate_value.labels(layer=layer).observe(gate_mean)
            if entropy_mean > 0:
                self.entropy_value.labels(layer=layer).observe(entropy_mean)

            # 滑动窗口激活率
            self._recent_applied.append(1 if applied else 0)
            if len(self._recent_applied) > 0:
                rate = sum(self._recent_applied) / len(self._recent_applied)
                self.activation_rate.set(rate)

        except Exception as e:
            logger.debug(f"PrometheusExporter forward 事件处理异常: {e}")

    def _on_retrieval(self, event) -> None:
        """处理 retrieval 事件"""
        try:
            data = event.data
            layer = str(data.get("layer_idx", 0))
            injected = data.get("injected_count", 0)
            scores = data.get("scores", [])

            # Counter
            self.retrieval_total.labels(layer=layer).inc()
            self.retrieval_injected_total.labels(layer=layer).inc(injected)

            # Histogram
            for score in scores:
                self.retrieval_score.observe(score)

        except Exception as e:
            logger.debug(f"PrometheusExporter retrieval 事件处理异常: {e}")

    def _on_audit(self, event) -> None:
        """处理 audit 事件"""
        try:
            data = event.data
            operation = data.get("operation", "unknown")
            success = data.get("success", True)

            self.audit_operations_total.labels(
                operation=operation,
                success=str(success).lower(),
            ).inc()

        except Exception as e:
            logger.debug(f"PrometheusExporter audit 事件处理异常: {e}")

    def update_gauges(self, plugin) -> None:
        """
        从 AGAPlugin 更新 Gauge 指标

        建议在定时任务中调用（如每 5 秒一次），
        而非在热路径上调用。

        Args:
            plugin: AGAPlugin 实例
        """
        try:
            stats = plugin.get_store_stats()
            self.knowledge_count.set(stats.get("count", 0))
            self.knowledge_utilization.set(stats.get("utilization", 0.0))
            self.knowledge_pinned.set(stats.get("pinned_count", 0))

            diag = plugin.get_diagnostics()
            slot_gov = diag.get("slot_governance", {})
            self.slot_change_rate.set(slot_gov.get("slot_change_rate", 0.0))

        except Exception as e:
            logger.debug(f"PrometheusExporter gauge 更新异常: {e}")

    def set_build_info(self, info: Dict[str, str]) -> None:
        """设置构建信息"""
        self.info.info(info)

    def start_server(self) -> None:
        """启动 Prometheus HTTP 端点"""
        if self._server_started:
            logger.warning("Prometheus HTTP 端点已启动")
            return

        try:
            start_http_server(self._port, registry=self._registry)
            self._server_started = True
            logger.info(f"Prometheus HTTP 端点已启动: http://0.0.0.0:{self._port}/metrics")
        except Exception as e:
            logger.error(f"Prometheus HTTP 端点启动失败: {e}")

    def shutdown(self) -> None:
        """关闭"""
        self._server_started = False
        logger.info("PrometheusExporter 已关闭")

    def get_stats(self) -> Dict[str, Any]:
        """获取导出器统计"""
        return {
            "server_started": self._server_started,
            "subscribed": self._subscribed,
            "port": self._port,
            "prefix": self._prefix,
            "window_size": len(self._recent_applied),
        }
