"""
aga/instrumentation/forward_metrics.py — Forward 指标收集器

在每次 forward 调用时收集关键指标，存储在内存中。
不依赖任何外部库，通过 EventBus 发射事件供外部消费。
"""
import threading
from typing import Dict, Any

from .event_bus import EventBus


class ForwardMetrics:
    """
    Forward 指标收集器

    收集的指标:
      - forward_total: 总调用次数
      - forward_applied: AGA 实际注入次数
      - forward_bypassed: 旁路次数（early exit）
      - gate_mean_avg: 平均门控值
      - entropy_mean_avg: 平均熵值
      - activation_rate: 激活率 (applied / total)
      - latency_p50/p95/p99: 延迟百分位

    所有指标始终可用（内存计算），不依赖 Prometheus。
    """

    def __init__(self, event_bus: EventBus):
        self._bus = event_bus
        self._lock = threading.Lock()

        # 计数器
        self._total = 0
        self._applied = 0
        self._bypassed = 0

        # 累积值（用于计算平均）
        self._gate_sum = 0.0
        self._entropy_sum = 0.0

        # 延迟追踪（滑动窗口）
        self._latencies: list = []
        self._max_latencies = 1000

        # 按层统计
        self._layer_stats: Dict[int, Dict] = {}

    def record(
        self,
        aga_applied: bool,
        gate_mean: float,
        entropy_mean: float = 0.0,
        layer_idx: int = 0,
        latency_us: float = 0.0,
    ):
        """
        记录一次 forward（热路径，必须极快）
        """
        with self._lock:
            self._total += 1
            if aga_applied:
                self._applied += 1
            else:
                self._bypassed += 1

            self._gate_sum += gate_mean
            self._entropy_sum += entropy_mean

            if latency_us > 0:
                self._latencies.append(latency_us)
                if len(self._latencies) > self._max_latencies:
                    self._latencies = self._latencies[-self._max_latencies:]

            # 按层统计
            if layer_idx not in self._layer_stats:
                self._layer_stats[layer_idx] = {"total": 0, "applied": 0}
            self._layer_stats[layer_idx]["total"] += 1
            if aga_applied:
                self._layer_stats[layer_idx]["applied"] += 1

        # 发射事件（供 aga-observability 消费）
        self._bus.emit("forward", {
            "aga_applied": aga_applied,
            "gate_mean": gate_mean,
            "entropy_mean": entropy_mean,
            "layer_idx": layer_idx,
            "latency_us": latency_us,
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要（始终可用）

        Returns:
            {
                "forward_total": 15000,
                "forward_applied": 6000,
                "forward_bypassed": 9000,
                "activation_rate": 0.4,
                "gate_mean_avg": 0.35,
                "entropy_mean_avg": 1.2,
                "latency_p50_us": 45.0,
                "latency_p95_us": 120.0,
                "latency_p99_us": 250.0,
                "layer_stats": {0: {"total": 5000, "applied": 2000, "rate": 0.4}, ...}
            }
        """
        with self._lock:
            result: Dict[str, Any] = {
                "forward_total": self._total,
                "forward_applied": self._applied,
                "forward_bypassed": self._bypassed,
                "activation_rate": self._applied / max(self._total, 1),
                "gate_mean_avg": self._gate_sum / max(self._total, 1),
                "entropy_mean_avg": self._entropy_sum / max(self._total, 1),
            }

            # 延迟百分位
            if self._latencies:
                sorted_lat = sorted(self._latencies)
                n = len(sorted_lat)
                result["latency_p50_us"] = sorted_lat[int(n * 0.5)]
                result["latency_p95_us"] = sorted_lat[int(n * 0.95)]
                result["latency_p99_us"] = sorted_lat[min(int(n * 0.99), n - 1)]

            # 按层统计
            result["layer_stats"] = {}
            for layer_idx, stats in self._layer_stats.items():
                result["layer_stats"][layer_idx] = {
                    **stats,
                    "rate": stats["applied"] / max(stats["total"], 1),
                }

            return result

    def reset(self):
        """重置指标（用于测试或周期性重置）"""
        with self._lock:
            self._total = 0
            self._applied = 0
            self._bypassed = 0
            self._gate_sum = 0.0
            self._entropy_sum = 0.0
            self._latencies.clear()
            self._layer_stats.clear()
