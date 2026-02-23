"""
aga_observability/alert_manager.py — SLO/SLI 告警管理

基于 EventBus 事件流的实时告警系统。

支持的告警指标:
  - activation_rate: 激活率（滑动窗口）
  - latency_p99: P99 延迟
  - gate_mean: 平均门控值
  - entropy_mean: 平均熵值
  - knowledge_utilization: KVStore 利用率
  - retrieval_failure_rate: 召回失败率
  - slot_change_rate: Slot 变化率

告警级别:
  - info: 信息通知
  - warning: 警告（需关注）
  - critical: 严重（需立即处理）

告警通道:
  - logging: Python 日志（默认）
  - webhook: HTTP POST 通知
  - callback: 自定义回调函数
"""
import time
import logging
import threading
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """
    告警规则

    Attributes:
        name: 规则名称
        metric: 监控指标名
        operator: 比较运算符 (> / < / >= / <= / ==)
        threshold: 阈值
        window_seconds: 评估窗口（秒）
        severity: 告警级别
        message: 告警消息模板（支持 {value}, {threshold}, {metric}）
        cooldown_seconds: 冷却期（秒），同一规则在冷却期内不重复告警
    """
    name: str
    metric: str
    operator: str = ">"
    threshold: float = 0.0
    window_seconds: int = 60
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    cooldown_seconds: int = 300

    def evaluate(self, value: float) -> bool:
        """评估规则是否触发"""
        ops = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: abs(v - t) < 1e-9,
        }
        op_fn = ops.get(self.operator)
        if op_fn is None:
            logger.warning(f"未知运算符: {self.operator}")
            return False
        return op_fn(value, self.threshold)

    def format_message(self, value: float) -> str:
        """格式化告警消息"""
        if self.message:
            return self.message.format(
                value=value,
                threshold=self.threshold,
                metric=self.metric,
                name=self.name,
            )
        return (
            f"[{self.severity.value.upper()}] {self.name}: "
            f"{self.metric} = {value:.4f} {self.operator} {self.threshold}"
        )


@dataclass
class AlertEvent:
    """告警事件"""
    rule_name: str
    metric: str
    value: float
    threshold: float
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


class AlertManager:
    """
    SLO/SLI 告警管理器

    使用方式:
        manager = AlertManager()

        # 添加规则
        manager.add_rule(AlertRule(
            name="high_latency",
            metric="latency_p99",
            operator=">",
            threshold=500.0,
            severity=AlertSeverity.WARNING,
            message="P99 延迟过高: {value:.1f}μs > {threshold:.1f}μs",
        ))

        manager.add_rule(AlertRule(
            name="low_activation",
            metric="activation_rate",
            operator="<",
            threshold=0.01,
            severity=AlertSeverity.CRITICAL,
            message="激活率过低: {value:.2%}，AGA 可能未正常工作",
        ))

        # 订阅事件
        manager.subscribe(event_bus)

        # 添加通知回调
        manager.add_callback(my_alert_handler)

        # 查询告警历史
        alerts = manager.get_alerts(severity=AlertSeverity.CRITICAL)
    """

    SUBSCRIBER_ID = "aga-observability-alert"

    # 默认 SLO 规则
    DEFAULT_RULES = [
        AlertRule(
            name="slo_latency_p99",
            metric="latency_p99",
            operator=">",
            threshold=1000.0,
            window_seconds=60,
            severity=AlertSeverity.WARNING,
            message="SLO 违规: P99 延迟 {value:.1f}μs 超过 {threshold:.0f}μs",
            cooldown_seconds=300,
        ),
        AlertRule(
            name="slo_latency_critical",
            metric="latency_p99",
            operator=">",
            threshold=5000.0,
            window_seconds=30,
            severity=AlertSeverity.CRITICAL,
            message="严重: P99 延迟 {value:.1f}μs 远超阈值 {threshold:.0f}μs",
            cooldown_seconds=60,
        ),
        AlertRule(
            name="high_utilization",
            metric="knowledge_utilization",
            operator=">",
            threshold=0.95,
            window_seconds=60,
            severity=AlertSeverity.WARNING,
            message="KVStore 利用率过高: {value:.1%}，可能导致频繁淘汰",
            cooldown_seconds=600,
        ),
        AlertRule(
            name="slot_thrashing",
            metric="slot_change_rate",
            operator=">",
            threshold=0.5,
            window_seconds=30,
            severity=AlertSeverity.WARNING,
            message="Slot Thrashing 风险: 变化率 {value:.2f} 超过阈值 {threshold:.2f}",
            cooldown_seconds=120,
        ),
    ]

    def __init__(
        self,
        rules: Optional[List[AlertRule]] = None,
        use_defaults: bool = True,
        max_history: int = 1000,
        webhook_url: Optional[str] = None,
    ):
        self._rules: Dict[str, AlertRule] = {}
        self._history: deque = deque(maxlen=max_history)
        self._last_fired: Dict[str, float] = {}  # rule_name → last fire timestamp
        self._callbacks: List[Callable[[AlertEvent], None]] = []
        self._webhook_url = webhook_url
        self._lock = threading.Lock()

        # 实时指标缓存（滑动窗口）
        self._metrics_window: Dict[str, deque] = {}
        self._window_size = 1000

        # 加载默认规则
        if use_defaults:
            for rule in self.DEFAULT_RULES:
                self.add_rule(rule)

        # 加载自定义规则
        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self._rules[rule.name] = rule
        logger.debug(f"告警规则已添加: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        """移除告警规则"""
        return self._rules.pop(name, None) is not None

    def add_callback(self, callback: Callable[[AlertEvent], None]) -> None:
        """添加告警回调"""
        self._callbacks.append(callback)

    def subscribe(self, event_bus) -> None:
        """订阅 EventBus 事件"""
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
        logger.info(f"AlertManager 已订阅 EventBus（{len(self._rules)} 条规则）")

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

    def _on_forward(self, event) -> None:
        """处理 forward 事件，更新指标并评估规则"""
        try:
            data = event.data
            self._update_metric("latency_p99", data.get("latency_us", 0.0))
            self._update_metric("gate_mean", data.get("gate_mean", 0.0))
            self._update_metric("entropy_mean", data.get("entropy_mean", 0.0))

            applied = 1.0 if data.get("aga_applied", False) else 0.0
            self._update_metric("activation_rate", applied)

            # 评估所有规则
            self._evaluate_rules()

        except Exception as e:
            logger.debug(f"AlertManager forward 事件处理异常: {e}")

    def _on_retrieval(self, event) -> None:
        """处理 retrieval 事件"""
        try:
            data = event.data
            budget = data.get("budget_remaining", 0)
            injected = data.get("injected_count", 0)
            results = data.get("results_count", 0)

            if results > 0:
                failure_rate = 1.0 - (injected / results)
                self._update_metric("retrieval_failure_rate", failure_rate)

        except Exception as e:
            logger.debug(f"AlertManager retrieval 事件处理异常: {e}")

    def update_plugin_metrics(self, plugin) -> None:
        """
        从 AGAPlugin 更新额外指标

        建议在定时任务中调用。

        Args:
            plugin: AGAPlugin 实例
        """
        try:
            stats = plugin.get_store_stats()
            self._update_metric(
                "knowledge_utilization",
                stats.get("utilization", 0.0),
            )

            diag = plugin.get_diagnostics()
            slot_gov = diag.get("slot_governance", {})
            self._update_metric(
                "slot_change_rate",
                slot_gov.get("slot_change_rate", 0.0),
            )

            self._evaluate_rules()

        except Exception as e:
            logger.debug(f"AlertManager plugin 指标更新异常: {e}")

    def _update_metric(self, name: str, value: float) -> None:
        """更新指标滑动窗口"""
        if name not in self._metrics_window:
            self._metrics_window[name] = deque(maxlen=self._window_size)
        self._metrics_window[name].append((time.time(), value))

    def _get_metric_value(self, name: str, window_seconds: int = 60) -> Optional[float]:
        """获取指标在窗口内的聚合值"""
        window = self._metrics_window.get(name)
        if not window:
            return None

        cutoff = time.time() - window_seconds
        values = [v for t, v in window if t >= cutoff]
        if not values:
            return None

        # 根据指标类型选择聚合方式
        if name in ("latency_p99",):
            # 延迟取 P99
            sorted_vals = sorted(values)
            idx = min(int(len(sorted_vals) * 0.99), len(sorted_vals) - 1)
            return sorted_vals[idx]
        elif name in ("activation_rate",):
            # 激活率取平均
            return sum(values) / len(values)
        else:
            # 其他取最新值
            return values[-1]

    def _evaluate_rules(self) -> None:
        """评估所有告警规则"""
        now = time.time()

        for name, rule in self._rules.items():
            value = self._get_metric_value(rule.metric, rule.window_seconds)
            if value is None:
                continue

            if rule.evaluate(value):
                # 检查冷却期
                last = self._last_fired.get(name, 0)
                if now - last < rule.cooldown_seconds:
                    continue

                # 触发告警
                alert = AlertEvent(
                    rule_name=name,
                    metric=rule.metric,
                    value=value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=rule.format_message(value),
                    timestamp=now,
                )

                self._fire_alert(alert)
                self._last_fired[name] = now

    def _fire_alert(self, alert: AlertEvent) -> None:
        """触发告警"""
        with self._lock:
            self._history.append(alert)

        # 1. 日志输出
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)

        logger.log(log_level, f"[AGA ALERT] {alert.message}")

        # 2. 回调通知
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.debug(f"告警回调异常: {e}")

        # 3. Webhook 通知
        if self._webhook_url:
            self._send_webhook(alert)

    def _send_webhook(self, alert: AlertEvent) -> None:
        """发送 Webhook 通知"""
        try:
            import urllib.request
            import json

            payload = json.dumps({
                "rule": alert.rule_name,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
            }).encode("utf-8")

            req = urllib.request.Request(
                self._webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # 非阻塞发送（不等待响应）
            threading.Thread(
                target=lambda: urllib.request.urlopen(req, timeout=5),
                daemon=True,
            ).start()

        except Exception as e:
            logger.debug(f"Webhook 发送失败: {e}")

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询告警历史

        Args:
            severity: 过滤告警级别
            limit: 返回数量
            since: 起始时间戳

        Returns:
            告警事件列表
        """
        with self._lock:
            alerts = list(self._history)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        return [
            {
                "rule_name": a.rule_name,
                "metric": a.metric,
                "value": a.value,
                "threshold": a.threshold,
                "severity": a.severity.value,
                "message": a.message,
                "timestamp": a.timestamp,
            }
            for a in alerts[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        with self._lock:
            severity_counts = {}
            for alert in self._history:
                sev = alert.severity.value
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "rules_count": len(self._rules),
            "total_alerts": len(self._history),
            "by_severity": severity_counts,
            "active_metrics": list(self._metrics_window.keys()),
            "callbacks_count": len(self._callbacks),
        }

    def shutdown(self) -> None:
        """关闭"""
        self._callbacks.clear()
        logger.info("AlertManager 已关闭")
