"""
AGA SLO/SLI 定义模块

提供服务级别目标（SLO）和服务级别指标（SLI）的定义和管理：
1. SLO 数据结构
2. 预定义 AGA SLOs
3. Prometheus 记录规则生成
4. 错误预算告警生成

版本: v1.0
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


# ==================== 枚举定义 ====================

class SLOType(str, Enum):
    """SLO 类型"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class SLOWindow(str, Enum):
    """SLO 时间窗口"""
    ROLLING_7D = "7d"
    ROLLING_28D = "28d"
    ROLLING_30D = "30d"
    CALENDAR_MONTH = "calendar_month"


# ==================== SLO 数据结构 ====================

@dataclass
class SLO:
    """
    服务级别目标
    
    定义一个 SLO，包括目标值、时间窗口和告警配置。
    """
    name: str
    type: SLOType
    target: float  # 目标值 (0-1)
    window: str  # 时间窗口
    description: str
    
    # PromQL 表达式
    sli_query: str  # SLI 查询（返回 0-1 的值）
    
    # 告警配置
    alert_burn_rate_1h: float = 14.4  # 1h 内消耗 2% 错误预算
    alert_burn_rate_6h: float = 6.0   # 6h 内消耗 5% 错误预算
    
    # 元数据
    labels: Dict[str, str] = field(default_factory=dict)
    
    @property
    def error_budget(self) -> float:
        """错误预算 (1 - target)"""
        return 1 - self.target
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.type.value,
            "target": self.target,
            "window": self.window,
            "description": self.description,
            "sli_query": self.sli_query,
            "error_budget": self.error_budget,
        }


# ==================== 预定义 AGA SLOs ====================

AGA_SLOS = [
    # 可用性 SLO
    SLO(
        name="aga_availability",
        type=SLOType.AVAILABILITY,
        target=0.999,  # 99.9%
        window="30d",
        description="AGA 服务可用性 - 99.9% 的请求成功",
        sli_query=(
            'sum(rate(aga_requests_total{status="success"}[5m])) / '
            'sum(rate(aga_requests_total[5m]))'
        ),
        labels={"tier": "critical"},
    ),
    
    # P99 延迟 SLO
    SLO(
        name="aga_latency_p99",
        type=SLOType.LATENCY,
        target=0.99,  # 99% 请求 < 100ms
        window="30d",
        description="AGA P99 延迟 - 99% 的请求延迟 < 100ms",
        sli_query=(
            'sum(rate(aga_request_duration_seconds_bucket{le="0.1"}[5m])) / '
            'sum(rate(aga_request_duration_seconds_count[5m]))'
        ),
        labels={"tier": "critical"},
    ),
    
    # P95 延迟 SLO
    SLO(
        name="aga_latency_p95",
        type=SLOType.LATENCY,
        target=0.95,  # 95% 请求 < 50ms
        window="30d",
        description="AGA P95 延迟 - 95% 的请求延迟 < 50ms",
        sli_query=(
            'sum(rate(aga_request_duration_seconds_bucket{le="0.05"}[5m])) / '
            'sum(rate(aga_request_duration_seconds_count[5m]))'
        ),
        labels={"tier": "high"},
    ),
    
    # 错误率 SLO
    SLO(
        name="aga_error_rate",
        type=SLOType.ERROR_RATE,
        target=0.999,  # 错误率 < 0.1%
        window="30d",
        description="AGA 错误率 - 错误率 < 0.1%",
        sli_query=(
            '1 - (sum(rate(aga_errors_total[5m])) / '
            'sum(rate(aga_requests_total[5m])))'
        ),
        labels={"tier": "critical"},
    ),
    
    # ANN 搜索延迟 SLO
    SLO(
        name="aga_ann_latency_p99",
        type=SLOType.LATENCY,
        target=0.99,  # 99% ANN 搜索 < 10ms
        window="30d",
        description="ANN 搜索 P99 延迟 - 99% 的搜索延迟 < 10ms",
        sli_query=(
            'sum(rate(aga_ann_search_duration_seconds_bucket{le="0.01"}[5m])) / '
            'sum(rate(aga_ann_search_duration_seconds_count[5m]))'
        ),
        labels={"tier": "high"},
    ),
    
    # 动态加载器 SLO
    SLO(
        name="aga_loader_success_rate",
        type=SLOType.AVAILABILITY,
        target=0.999,  # 99.9% 加载成功
        window="30d",
        description="动态加载器成功率 - 99.9% 的加载请求成功",
        sli_query=(
            '1 - (sum(rate(aga_loader_failures_total[5m])) / '
            'sum(rate(aga_loader_requests_total[5m])))'
        ),
        labels={"tier": "high"},
    ),
    
    # 槽位命中率 SLO
    SLO(
        name="aga_slot_hit_rate",
        type=SLOType.THROUGHPUT,
        target=0.5,  # 50% 命中率
        window="30d",
        description="槽位命中率 - 至少 50% 的请求命中知识槽位",
        sli_query=(
            'sum(rate(aga_gate_hits_total[5m])) / '
            'sum(rate(aga_gate_checks_total[5m]))'
        ),
        alert_burn_rate_1h=5.0,  # 更宽松的告警
        alert_burn_rate_6h=2.0,
        labels={"tier": "medium"},
    ),
]


# ==================== 规则生成函数 ====================

def generate_slo_recording_rules(
    slos: List[SLO] = None,
    group_name: str = "aga_slo_rules",
) -> Dict[str, Any]:
    """
    生成 SLO 记录规则
    
    Args:
        slos: SLO 列表
        group_name: 规则组名称
    
    Returns:
        Prometheus 记录规则配置
    """
    if slos is None:
        slos = AGA_SLOS
    
    rules = []
    
    for slo in slos:
        # SLI 记录规则
        rules.append({
            "record": f"sli:{slo.name}",
            "expr": slo.sli_query,
            "labels": {
                "slo": slo.name,
                "type": slo.type.value,
                **slo.labels,
            },
        })
        
        # 错误预算剩余（基于短期窗口估算）
        rules.append({
            "record": f"slo:{slo.name}:error_budget_remaining",
            "expr": f"clamp_min(1 - ((1 - sli:{slo.name}) / {slo.error_budget}), 0)",
            "labels": {
                "slo": slo.name,
                **slo.labels,
            },
        })
        
        # 燃烧率（1h 窗口）
        rules.append({
            "record": f"slo:{slo.name}:burn_rate_1h",
            "expr": f"(1 - sli:{slo.name}) / {slo.error_budget}",
            "labels": {
                "slo": slo.name,
                "window": "1h",
                **slo.labels,
            },
        })
        
        # 燃烧率（6h 窗口）
        rules.append({
            "record": f"slo:{slo.name}:burn_rate_6h",
            "expr": f"avg_over_time(slo:{slo.name}:burn_rate_1h[6h])",
            "labels": {
                "slo": slo.name,
                "window": "6h",
                **slo.labels,
            },
        })
    
    return {
        "groups": [
            {
                "name": group_name,
                "interval": "30s",
                "rules": rules,
            }
        ]
    }


def generate_slo_alerts(
    slos: List[SLO] = None,
    group_name: str = "aga_slo_alerts",
) -> Dict[str, Any]:
    """
    生成 SLO 告警规则
    
    基于多窗口多燃烧率告警策略。
    
    Args:
        slos: SLO 列表
        group_name: 规则组名称
    
    Returns:
        Prometheus 告警规则配置
    """
    if slos is None:
        slos = AGA_SLOS
    
    alerts = []
    
    for slo in slos:
        # 快速燃烧告警 (1h 窗口，高燃烧率)
        # 如果 1h 内燃烧率超过阈值，说明错误预算消耗过快
        alerts.append({
            "alert": f"{slo.name}_fast_burn",
            "expr": f"slo:{slo.name}:burn_rate_1h > {slo.alert_burn_rate_1h}",
            "for": "2m",
            "labels": {
                "severity": "critical",
                "slo": slo.name,
                "alert_type": "fast_burn",
                **slo.labels,
            },
            "annotations": {
                "summary": f"SLO {slo.name} 快速燃烧",
                "description": (
                    f"SLO {slo.name} 错误预算在 1 小时内消耗过快。"
                    f"当前燃烧率: {{{{ $value | printf \"%.2f\" }}}}x，"
                    f"阈值: {slo.alert_burn_rate_1h}x。"
                    f"目标: {slo.target * 100:.1f}%"
                ),
                "runbook_url": f"https://docs.aga.io/runbooks/{slo.name}",
            },
        })
        
        # 慢速燃烧告警 (6h 窗口，中等燃烧率)
        alerts.append({
            "alert": f"{slo.name}_slow_burn",
            "expr": f"slo:{slo.name}:burn_rate_6h > {slo.alert_burn_rate_6h}",
            "for": "1h",
            "labels": {
                "severity": "warning",
                "slo": slo.name,
                "alert_type": "slow_burn",
                **slo.labels,
            },
            "annotations": {
                "summary": f"SLO {slo.name} 慢速燃烧",
                "description": (
                    f"SLO {slo.name} 错误预算在 6 小时内持续消耗。"
                    f"当前燃烧率: {{{{ $value | printf \"%.2f\" }}}}x，"
                    f"阈值: {slo.alert_burn_rate_6h}x。"
                    f"目标: {slo.target * 100:.1f}%"
                ),
                "runbook_url": f"https://docs.aga.io/runbooks/{slo.name}",
            },
        })
        
        # 错误预算耗尽告警
        alerts.append({
            "alert": f"{slo.name}_budget_exhausted",
            "expr": f"slo:{slo.name}:error_budget_remaining < 0.1",
            "for": "5m",
            "labels": {
                "severity": "critical",
                "slo": slo.name,
                "alert_type": "budget_exhausted",
                **slo.labels,
            },
            "annotations": {
                "summary": f"SLO {slo.name} 错误预算即将耗尽",
                "description": (
                    f"SLO {slo.name} 错误预算剩余不足 10%。"
                    f"当前剩余: {{{{ $value | printf \"%.1f\" }}}}%。"
                    f"需要立即采取措施。"
                ),
                "runbook_url": f"https://docs.aga.io/runbooks/{slo.name}",
            },
        })
    
    return {
        "groups": [
            {
                "name": group_name,
                "rules": alerts,
            }
        ]
    }


def generate_slo_rules_yaml(
    slos: List[SLO] = None,
    include_alerts: bool = True,
) -> str:
    """
    生成完整的 SLO 规则 YAML
    
    Args:
        slos: SLO 列表
        include_alerts: 是否包含告警规则
    
    Returns:
        YAML 格式的规则配置
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")
    
    if slos is None:
        slos = AGA_SLOS
    
    # 生成记录规则
    recording_rules = generate_slo_recording_rules(slos)
    
    # 合并告警规则
    if include_alerts:
        alert_rules = generate_slo_alerts(slos)
        recording_rules["groups"].extend(alert_rules["groups"])
    
    return yaml.dump(
        recording_rules,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


# ==================== SLO 仪表盘生成 ====================

def create_slo_dashboard_panels(slos: List[SLO] = None) -> List[Dict[str, Any]]:
    """
    创建 SLO Grafana 仪表盘面板
    
    Args:
        slos: SLO 列表
    
    Returns:
        Grafana 面板配置列表
    """
    if slos is None:
        slos = AGA_SLOS
    
    panels = []
    panel_id = 1
    y_pos = 0
    
    # 概览行
    panels.append({
        "id": panel_id,
        "title": "SLO 概览",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "collapsed": False,
    })
    panel_id += 1
    y_pos += 1
    
    # 每个 SLO 的状态卡片
    x_pos = 0
    for slo in slos:
        panels.append({
            "id": panel_id,
            "title": slo.name,
            "type": "stat",
            "gridPos": {"h": 4, "w": 4, "x": x_pos, "y": y_pos},
            "targets": [
                {
                    "expr": f"sli:{slo.name}",
                    "legendFormat": "SLI",
                }
            ],
            "options": {
                "colorMode": "background",
                "graphMode": "none",
            },
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": slo.target - 0.01},
                            {"color": "green", "value": slo.target},
                        ],
                    },
                }
            },
        })
        panel_id += 1
        x_pos += 4
        if x_pos >= 24:
            x_pos = 0
            y_pos += 4
    
    y_pos += 4
    
    # 错误预算行
    panels.append({
        "id": panel_id,
        "title": "错误预算",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "collapsed": False,
    })
    panel_id += 1
    y_pos += 1
    
    # 错误预算剩余图表
    panels.append({
        "id": panel_id,
        "title": "错误预算剩余",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_pos},
        "targets": [
            {
                "expr": f"slo:{slo.name}:error_budget_remaining",
                "legendFormat": slo.name,
            }
            for slo in slos
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "percentunit",
                "min": 0,
                "max": 1,
            }
        },
    })
    panel_id += 1
    
    # 燃烧率图表
    panels.append({
        "id": panel_id,
        "title": "燃烧率 (1h)",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_pos},
        "targets": [
            {
                "expr": f"slo:{slo.name}:burn_rate_1h",
                "legendFormat": slo.name,
            }
            for slo in slos
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "short",
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 1},
                        {"color": "red", "value": 5},
                    ],
                },
            }
        },
    })
    
    return panels


def export_slo_dashboard(
    slos: List[SLO] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    导出 SLO Grafana 仪表盘
    
    Args:
        slos: SLO 列表
        output_path: 输出文件路径
    
    Returns:
        JSON 格式的仪表盘配置
    """
    panels = create_slo_dashboard_panels(slos)
    
    dashboard = {
        "title": "AGA SLO 仪表盘",
        "uid": "aga-slo-dashboard",
        "version": 1,
        "schemaVersion": 38,
        "tags": ["aga", "slo", "reliability"],
        "timezone": "browser",
        "refresh": "1m",
        "time": {
            "from": "now-24h",
            "to": "now",
        },
        "panels": panels,
        "templating": {
            "list": [
                {
                    "name": "slo",
                    "type": "custom",
                    "options": [
                        {"text": slo.name, "value": slo.name}
                        for slo in (slos or AGA_SLOS)
                    ],
                    "multi": True,
                    "includeAll": True,
                }
            ]
        },
    }
    
    json_str = json.dumps(dashboard, indent=2, ensure_ascii=False)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
    
    return json_str


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "SLOType",
    "SLOWindow",
    # 数据结构
    "SLO",
    # 预定义 SLOs
    "AGA_SLOS",
    # 规则生成
    "generate_slo_recording_rules",
    "generate_slo_alerts",
    "generate_slo_rules_yaml",
    # 仪表盘
    "create_slo_dashboard_panels",
    "export_slo_dashboard",
]
