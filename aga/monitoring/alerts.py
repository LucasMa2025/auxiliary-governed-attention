"""
AGA 告警规则配置与仪表盘定义

提供生产环境所需的监控告警和可视化配置：
1. Prometheus 告警规则
2. Grafana 仪表盘配置
3. 告警通知渠道
4. 自定义指标

版本: v3.4.1
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
import time

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PAGE = "page"  # 需要立即处理


class AlertState(str, Enum):
    """告警状态"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    expr: str  # PromQL 表达式
    duration: str  # 持续时间 (e.g., "5m")
    severity: AlertSeverity
    summary: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def to_prometheus_rule(self) -> Dict[str, Any]:
        """转换为 Prometheus 规则格式"""
        return {
            "alert": self.name,
            "expr": self.expr,
            "for": self.duration,
            "labels": {
                "severity": self.severity.value,
                **self.labels,
            },
            "annotations": {
                "summary": self.summary,
                "description": self.description,
                **self.annotations,
            },
        }


# ==================== 预定义告警规则 ====================

AGA_ALERT_RULES = [
    # 可用性告警
    AlertRule(
        name="AGAServiceDown",
        expr='up{job="aga"} == 0',
        duration="1m",
        severity=AlertSeverity.CRITICAL,
        summary="AGA 服务不可用",
        description="AGA 服务 {{ $labels.instance }} 已停止响应超过 1 分钟",
    ),
    
    AlertRule(
        name="AGAHighErrorRate",
        expr='rate(aga_errors_total[5m]) > 0.1',
        duration="5m",
        severity=AlertSeverity.WARNING,
        summary="AGA 错误率过高",
        description="AGA 服务错误率超过 10%，当前值: {{ $value | printf \"%.2f\" }}",
    ),
    
    AlertRule(
        name="AGACriticalErrorRate",
        expr='rate(aga_errors_total[5m]) > 0.5',
        duration="2m",
        severity=AlertSeverity.CRITICAL,
        summary="AGA 错误率严重过高",
        description="AGA 服务错误率超过 50%，需要立即处理",
    ),
    
    # 延迟告警
    AlertRule(
        name="AGAHighLatencyP95",
        expr='histogram_quantile(0.95, rate(aga_request_duration_seconds_bucket[5m])) > 0.5',
        duration="5m",
        severity=AlertSeverity.WARNING,
        summary="AGA P95 延迟过高",
        description="AGA 请求 P95 延迟超过 500ms，当前值: {{ $value | printf \"%.3f\" }}s",
    ),
    
    AlertRule(
        name="AGAHighLatencyP99",
        expr='histogram_quantile(0.99, rate(aga_request_duration_seconds_bucket[5m])) > 1.0',
        duration="5m",
        severity=AlertSeverity.CRITICAL,
        summary="AGA P99 延迟严重过高",
        description="AGA 请求 P99 延迟超过 1s，当前值: {{ $value | printf \"%.3f\" }}s",
    ),
    
    # 槽位告警
    AlertRule(
        name="AGASlotUtilizationHigh",
        expr='aga_slot_utilization > 0.9',
        duration="10m",
        severity=AlertSeverity.WARNING,
        summary="AGA 槽位使用率过高",
        description="命名空间 {{ $labels.namespace }} 槽位使用率超过 90%",
    ),
    
    AlertRule(
        name="AGASlotUtilizationCritical",
        expr='aga_slot_utilization > 0.95',
        duration="5m",
        severity=AlertSeverity.CRITICAL,
        summary="AGA 槽位即将耗尽",
        description="命名空间 {{ $labels.namespace }} 槽位使用率超过 95%，需要扩容",
    ),
    
    AlertRule(
        name="AGANoActiveSlots",
        expr='aga_active_slots == 0',
        duration="5m",
        severity=AlertSeverity.WARNING,
        summary="AGA 无活跃槽位",
        description="命名空间 {{ $labels.namespace }} 没有活跃的知识槽位",
    ),
    
    # 门控告警
    AlertRule(
        name="AGAGateHitRateLow",
        expr='rate(aga_gate_hits_total[5m]) / rate(aga_gate_checks_total[5m]) < 0.1',
        duration="15m",
        severity=AlertSeverity.INFO,
        summary="AGA 门控命中率过低",
        description="AGA 门控命中率低于 10%，可能需要调整阈值或知识",
    ),
    
    AlertRule(
        name="AGAGateBypassRateHigh",
        expr='rate(aga_gate_bypasses_total[5m]) / rate(aga_gate_checks_total[5m]) > 0.5',
        duration="10m",
        severity=AlertSeverity.WARNING,
        summary="AGA 门控旁路率过高",
        description="超过 50% 的请求绕过了 AGA 门控",
    ),
    
    # 同步告警
    AlertRule(
        name="AGASyncLagHigh",
        expr='aga_sync_lag_seconds > 60',
        duration="5m",
        severity=AlertSeverity.WARNING,
        summary="AGA 同步延迟过高",
        description="实例 {{ $labels.instance }} 同步延迟超过 60 秒",
    ),
    
    AlertRule(
        name="AGASyncFailed",
        expr='increase(aga_sync_failures_total[5m]) > 5',
        duration="5m",
        severity=AlertSeverity.CRITICAL,
        summary="AGA 同步失败",
        description="5 分钟内同步失败超过 5 次",
    ),
    
    # 分布式告警
    AlertRule(
        name="AGAPartitionDetected",
        expr='aga_partition_state == 2',  # PARTITIONED
        duration="1m",
        severity=AlertSeverity.CRITICAL,
        summary="AGA 检测到网络分区",
        description="检测到网络分区，部分实例不可达",
    ),
    
    AlertRule(
        name="AGAQuorumLost",
        expr='aga_healthy_instances < aga_quorum_size',
        duration="2m",
        severity=AlertSeverity.CRITICAL,
        summary="AGA 失去 Quorum",
        description="健康实例数低于 quorum 要求，无法进行治理决策",
    ),
    
    # 资源告警
    AlertRule(
        name="AGAMemoryUsageHigh",
        expr='aga_memory_usage_bytes / aga_memory_limit_bytes > 0.85',
        duration="10m",
        severity=AlertSeverity.WARNING,
        summary="AGA 内存使用率过高",
        description="AGA 内存使用率超过 85%",
    ),
    
    AlertRule(
        name="AGACacheHitRateLow",
        expr='rate(aga_cache_hits_total[5m]) / rate(aga_cache_requests_total[5m]) < 0.5',
        duration="15m",
        severity=AlertSeverity.INFO,
        summary="AGA 缓存命中率过低",
        description="缓存命中率低于 50%，可能影响性能",
    ),
    
    # 隔离告警
    AlertRule(
        name="AGAQuarantineRateHigh",
        expr='increase(aga_quarantine_total[1h]) > 10',
        duration="5m",
        severity=AlertSeverity.WARNING,
        summary="AGA 隔离率过高",
        description="1 小时内隔离了超过 10 个知识槽位",
    ),
]


def generate_prometheus_rules(
    rules: List[AlertRule] = None,
    group_name: str = "aga_alerts",
) -> Dict[str, Any]:
    """
    生成 Prometheus 告警规则文件
    
    Args:
        rules: 告警规则列表
        group_name: 规则组名称
    
    Returns:
        Prometheus 规则配置
    """
    if rules is None:
        rules = AGA_ALERT_RULES
    
    return {
        "groups": [
            {
                "name": group_name,
                "interval": "30s",
                "rules": [rule.to_prometheus_rule() for rule in rules],
            }
        ]
    }


def generate_prometheus_rules_yaml(
    rules: List[AlertRule] = None,
    group_name: str = "aga_alerts",
) -> str:
    """生成 Prometheus 告警规则 YAML"""
    import yaml
    config = generate_prometheus_rules(rules, group_name)
    # sort_keys=False 避免每次生成的 YAML 字段顺序不同导致 git diff 噪音
    return yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ==================== Grafana 仪表盘 ====================

@dataclass
class GrafanaPanel:
    """Grafana 面板"""
    title: str
    panel_type: str  # graph, stat, gauge, table, heatmap
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)


def create_aga_dashboard() -> Dict[str, Any]:
    """
    创建 AGA Grafana 仪表盘
    
    Returns:
        Grafana 仪表盘 JSON
    """
    panels = []
    panel_id = 1
    
    # ==================== 概览行 ====================
    
    # 服务状态
    panels.append({
        "id": panel_id,
        "title": "服务状态",
        "type": "stat",
        "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
        "targets": [
            {
                "expr": 'up{job="aga"}',
                "legendFormat": "{{ instance }}",
            }
        ],
        "options": {
            "colorMode": "background",
            "graphMode": "none",
            "justifyMode": "center",
            "textMode": "value_and_name",
        },
        "fieldConfig": {
            "defaults": {
                "mappings": [
                    {"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}}},
                    {"type": "value", "options": {"1": {"text": "UP", "color": "green"}}},
                ],
            }
        },
    })
    panel_id += 1
    
    # 请求速率
    panels.append({
        "id": panel_id,
        "title": "请求速率",
        "type": "stat",
        "gridPos": {"h": 4, "w": 4, "x": 4, "y": 0},
        "targets": [
            {
                "expr": 'sum(rate(aga_requests_total[5m]))',
                "legendFormat": "req/s",
            }
        ],
        "options": {
            "colorMode": "value",
            "graphMode": "area",
        },
        "fieldConfig": {
            "defaults": {
                "unit": "reqps",
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 100},
                        {"color": "red", "value": 500},
                    ],
                },
            }
        },
    })
    panel_id += 1
    
    # 错误率
    panels.append({
        "id": panel_id,
        "title": "错误率",
        "type": "gauge",
        "gridPos": {"h": 4, "w": 4, "x": 8, "y": 0},
        "targets": [
            {
                "expr": 'sum(rate(aga_errors_total[5m])) / sum(rate(aga_requests_total[5m])) * 100',
                "legendFormat": "Error %",
            }
        ],
        "options": {
            "showThresholdLabels": False,
            "showThresholdMarkers": True,
        },
        "fieldConfig": {
            "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 100,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 1},
                        {"color": "orange", "value": 5},
                        {"color": "red", "value": 10},
                    ],
                },
            }
        },
    })
    panel_id += 1
    
    # 槽位使用率
    panels.append({
        "id": panel_id,
        "title": "槽位使用率",
        "type": "gauge",
        "gridPos": {"h": 4, "w": 4, "x": 12, "y": 0},
        "targets": [
            {
                "expr": 'avg(aga_slot_utilization) * 100',
                "legendFormat": "Utilization",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 100,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 70},
                        {"color": "orange", "value": 85},
                        {"color": "red", "value": 95},
                    ],
                },
            }
        },
    })
    panel_id += 1
    
    # 门控命中率
    panels.append({
        "id": panel_id,
        "title": "门控命中率",
        "type": "stat",
        "gridPos": {"h": 4, "w": 4, "x": 16, "y": 0},
        "targets": [
            {
                "expr": 'sum(rate(aga_gate_hits_total[5m])) / sum(rate(aga_gate_checks_total[5m])) * 100',
                "legendFormat": "Hit Rate",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "percent",
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 20},
                        {"color": "green", "value": 50},
                    ],
                },
            }
        },
    })
    panel_id += 1
    
    # 活跃实例数
    panels.append({
        "id": panel_id,
        "title": "活跃实例",
        "type": "stat",
        "gridPos": {"h": 4, "w": 4, "x": 20, "y": 0},
        "targets": [
            {
                "expr": 'count(up{job="aga"} == 1)',
                "legendFormat": "Instances",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 1},
                        {"color": "green", "value": 2},
                    ],
                },
            }
        },
    })
    panel_id += 1
    
    # ==================== 延迟行 ====================
    
    # 延迟分布
    panels.append({
        "id": panel_id,
        "title": "请求延迟分布",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
        "targets": [
            {
                "expr": 'histogram_quantile(0.50, rate(aga_request_duration_seconds_bucket[5m]))',
                "legendFormat": "P50",
            },
            {
                "expr": 'histogram_quantile(0.95, rate(aga_request_duration_seconds_bucket[5m]))',
                "legendFormat": "P95",
            },
            {
                "expr": 'histogram_quantile(0.99, rate(aga_request_duration_seconds_bucket[5m]))',
                "legendFormat": "P99",
            },
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "s",
            }
        },
    })
    panel_id += 1
    
    # 延迟热力图
    panels.append({
        "id": panel_id,
        "title": "延迟热力图",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
        "targets": [
            {
                "expr": 'sum(increase(aga_request_duration_seconds_bucket[1m])) by (le)',
                "legendFormat": "{{ le }}",
                "format": "heatmap",
            }
        ],
        "options": {
            "calculate": False,
            "color": {
                "scheme": "Spectral",
            },
        },
    })
    panel_id += 1
    
    # ==================== 槽位行 ====================
    
    # 槽位状态分布
    panels.append({
        "id": panel_id,
        "title": "槽位状态分布",
        "type": "piechart",
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 12},
        "targets": [
            {
                "expr": 'sum(aga_slots_by_state) by (state)',
                "legendFormat": "{{ state }}",
            }
        ],
        "options": {
            "legend": {
                "displayMode": "table",
                "placement": "right",
            },
        },
    })
    panel_id += 1
    
    # 槽位命中趋势
    panels.append({
        "id": panel_id,
        "title": "槽位命中趋势",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 12},
        "targets": [
            {
                "expr": 'sum(rate(aga_slot_hits_total[5m])) by (namespace)',
                "legendFormat": "{{ namespace }}",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "ops",
            }
        },
    })
    panel_id += 1
    
    # 信任层级分布
    panels.append({
        "id": panel_id,
        "title": "信任层级分布",
        "type": "barchart",
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 12},
        "targets": [
            {
                "expr": 'sum(aga_slots_by_trust_tier) by (trust_tier)',
                "legendFormat": "{{ trust_tier }}",
            }
        ],
        "options": {
            "orientation": "horizontal",
        },
    })
    panel_id += 1
    
    # ==================== 门控行 ====================
    
    # 门控决策
    panels.append({
        "id": panel_id,
        "title": "门控决策",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 20},
        "targets": [
            {
                "expr": 'sum(rate(aga_gate_hits_total[5m]))',
                "legendFormat": "Hits",
            },
            {
                "expr": 'sum(rate(aga_gate_bypasses_total[5m]))',
                "legendFormat": "Bypasses",
            },
            {
                "expr": 'sum(rate(aga_gate_early_exits_total[5m]))',
                "legendFormat": "Early Exits",
            },
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "ops",
            }
        },
    })
    panel_id += 1
    
    # 熵分布
    panels.append({
        "id": panel_id,
        "title": "熵值分布",
        "type": "histogram",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 20},
        "targets": [
            {
                "expr": 'aga_entropy_histogram_bucket',
                "legendFormat": "{{ le }}",
            }
        ],
    })
    panel_id += 1
    
    # ==================== 同步行 ====================
    
    # 同步状态
    panels.append({
        "id": panel_id,
        "title": "同步延迟",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 28},
        "targets": [
            {
                "expr": 'aga_sync_lag_seconds',
                "legendFormat": "{{ instance }}",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "s",
            }
        },
    })
    panel_id += 1
    
    # 同步消息
    panels.append({
        "id": panel_id,
        "title": "同步消息",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 28},
        "targets": [
            {
                "expr": 'sum(rate(aga_sync_messages_total[5m])) by (type)',
                "legendFormat": "{{ type }}",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "ops",
            }
        },
    })
    panel_id += 1
    
    # 构建仪表盘
    dashboard = {
        "title": "AGA 监控仪表盘",
        "uid": "aga-dashboard",
        "version": 1,
        "schemaVersion": 38,
        "tags": ["aga", "knowledge", "llm"],
        "timezone": "browser",
        "refresh": "30s",
        "time": {
            "from": "now-1h",
            "to": "now",
        },
        "panels": panels,
        "templating": {
            "list": [
                {
                    "name": "namespace",
                    "type": "query",
                    "datasource": "Prometheus",
                    "query": 'label_values(aga_slots_total, namespace)',
                    "refresh": 2,
                    "multi": True,
                    "includeAll": True,
                },
                {
                    "name": "instance",
                    "type": "query",
                    "datasource": "Prometheus",
                    "query": 'label_values(up{job="aga"}, instance)',
                    "refresh": 2,
                    "multi": True,
                    "includeAll": True,
                },
            ]
        },
        "annotations": {
            "list": [
                {
                    "name": "Alerts",
                    "datasource": "Prometheus",
                    "enable": True,
                    "expr": 'ALERTS{alertstate="firing", job="aga"}',
                    "titleFormat": "{{ alertname }}",
                    "textFormat": "{{ description }}",
                }
            ]
        },
    }
    
    return dashboard


def export_dashboard_json(output_path: str = None) -> str:
    """
    导出 Grafana 仪表盘 JSON
    
    Args:
        output_path: 输出文件路径（可选）
    
    Returns:
        JSON 字符串
    """
    dashboard = create_aga_dashboard()
    json_str = json.dumps(dashboard, indent=2, ensure_ascii=False)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        logger.info(f"Dashboard exported to {output_path}")
    
    return json_str


# ==================== 告警管理器 ====================

class AlertManager:
    """
    告警管理器
    
    本地告警检测和通知。
    """
    
    def __init__(
        self,
        instance_id: str,
        check_interval: float = 30.0,
    ):
        """
        初始化告警管理器
        
        Args:
            instance_id: 实例 ID
            check_interval: 检查间隔（秒）
        """
        self.instance_id = instance_id
        self.check_interval = check_interval
        
        # 告警规则
        self._rules: List[AlertRule] = []
        
        # 当前告警
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        
        # 告警历史
        self._alert_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        
        # 通知回调
        self._notification_callbacks: List[Callable] = []
        
        # 指标获取回调
        self._metrics_callback: Optional[Callable] = None
        
        # 运行状态
        self._running = False
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self._rules.append(rule)
    
    def add_rules(self, rules: List[AlertRule]):
        """批量添加告警规则"""
        self._rules.extend(rules)
    
    def set_metrics_callback(self, callback: Callable):
        """设置指标获取回调"""
        self._metrics_callback = callback
    
    def add_notification_callback(self, callback: Callable):
        """添加通知回调"""
        self._notification_callbacks.append(callback)
    
    async def start(self):
        """启动告警检查"""
        if self._running:
            return
        
        self._running = True
        asyncio.create_task(self._check_loop())
        logger.info("Alert manager started")
    
    async def stop(self):
        """停止告警检查"""
        self._running = False
        logger.info("Alert manager stopped")
    
    async def _check_loop(self):
        """告警检查循环"""
        while self._running:
            try:
                await self._check_alerts()
            except Exception as e:
                logger.error(f"Alert check error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_alerts(self):
        """检查所有告警规则"""
        if not self._metrics_callback:
            return
        
        # 获取当前指标
        metrics = await self._metrics_callback()
        
        for rule in self._rules:
            try:
                # 评估规则（简化版，实际应使用 PromQL 解析器）
                is_firing = self._evaluate_rule(rule, metrics)
                
                alert_key = rule.name
                
                if is_firing:
                    if alert_key not in self._active_alerts:
                        # 新告警
                        alert = {
                            "name": rule.name,
                            "severity": rule.severity.value,
                            "summary": rule.summary,
                            "description": rule.description,
                            "state": AlertState.FIRING.value,
                            "started_at": time.time(),
                            "instance": self.instance_id,
                        }
                        self._active_alerts[alert_key] = alert
                        
                        # 发送通知
                        await self._send_notification(alert)
                        
                        # 记录历史
                        self._record_history(alert)
                else:
                    if alert_key in self._active_alerts:
                        # 告警恢复
                        alert = self._active_alerts.pop(alert_key)
                        alert["state"] = AlertState.RESOLVED.value
                        alert["resolved_at"] = time.time()
                        
                        # 发送恢复通知
                        await self._send_notification(alert)
                        
                        # 记录历史
                        self._record_history(alert)
            
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """
        评估告警规则（简化版）
        
        注意：这是一个简化实现，只支持基础比较。
        生产环境应使用 Prometheus Alertmanager 进行完整的 PromQL 评估。
        """
        expr = rule.expr
        
        try:
            # 提取指标名（去除 labels 和函数调用）
            metric_name = expr.split("{")[0].split("(")[-1].strip()
            
            # 简单的指标比较
            if ">=" in expr:
                parts = expr.split(">=")
                threshold = float(parts[1].strip())
                value = metrics.get(metric_name, 0)
                return value >= threshold
            
            elif "<=" in expr:
                parts = expr.split("<=")
                threshold = float(parts[1].strip())
                value = metrics.get(metric_name, 0)
                return value <= threshold
            
            elif ">" in expr and "=" not in expr.split(">")[1][:1]:
                parts = expr.split(">")
                threshold = float(parts[1].strip())
                value = metrics.get(metric_name, 0)
                return value > threshold
            
            elif "<" in expr and "=" not in expr.split("<")[1][:1]:
                parts = expr.split("<")
                threshold = float(parts[1].strip())
                value = metrics.get(metric_name, 0)
                return value < threshold
            
            elif "==" in expr:
                parts = expr.split("==")
                expected = float(parts[1].strip())
                value = metrics.get(metric_name, 0)
                return value == expected
            
        except (ValueError, IndexError) as e:
            # 复杂表达式无法本地评估，返回 False（安全默认）
            logger.debug(f"Cannot evaluate complex expr '{expr}': {e}")
            return False
        
        return False
    
    async def _send_notification(self, alert: Dict[str, Any]):
        """发送告警通知"""
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")
    
    def _record_history(self, alert: Dict[str, Any]):
        """记录告警历史"""
        self._alert_history.append({
            **alert,
            "recorded_at": time.time(),
        })
        
        # 限制历史大小
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取当前活跃告警"""
        return list(self._active_alerts.values())
    
    def get_alert_history(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Dict[str, Any]]:
        """获取告警历史"""
        history = self._alert_history
        
        if severity:
            history = [a for a in history if a.get("severity") == severity.value]
        
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "active_alerts": len(self._active_alerts),
            "total_rules": len(self._rules),
            "history_size": len(self._alert_history),
        }


# 导出
__all__ = [
    "AlertSeverity",
    "AlertState",
    "AlertRule",
    "AGA_ALERT_RULES",
    "generate_prometheus_rules",
    "generate_prometheus_rules_yaml",
    "GrafanaPanel",
    "create_aga_dashboard",
    "export_dashboard_json",
    "AlertManager",
]
