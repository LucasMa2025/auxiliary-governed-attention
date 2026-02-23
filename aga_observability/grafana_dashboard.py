"""
aga_observability/grafana_dashboard.py — Grafana Dashboard 自动生成

生成标准的 Grafana Dashboard JSON，可直接导入 Grafana。

Dashboard 包含:
  - 概览面板: 激活率、知识数量、利用率
  - Forward 面板: 延迟分布、门控值、熵值
  - 召回器面板: 召回次数、注入数量、分数分布
  - 审计面板: 操作统计
  - 告警面板: SLO 状态
"""
import json
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class GrafanaDashboardGenerator:
    """
    Grafana Dashboard JSON 生成器

    使用方式:
        gen = GrafanaDashboardGenerator(prefix="aga")
        dashboard_json = gen.generate()

        # 保存到文件
        gen.save("aga_dashboard.json")

        # 或获取 Python dict
        dashboard = gen.to_dict()
    """

    def __init__(
        self,
        prefix: str = "aga",
        datasource: str = "Prometheus",
        title: str = "AGA Observability Dashboard",
        refresh: str = "5s",
    ):
        self._prefix = prefix
        self._datasource = datasource
        self._title = title
        self._refresh = refresh

    def generate(self) -> str:
        """生成 Dashboard JSON 字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        """生成 Dashboard 字典"""
        p = self._prefix
        ds = self._datasource

        panels = []
        y_pos = 0

        # === Row 1: 概览 ===
        panels.append(self._row("📊 概览", y_pos))
        y_pos += 1

        panels.append(self._stat_panel(
            title="激活率",
            expr=f"{p}_activation_rate",
            unit="percentunit",
            x=0, y=y_pos, w=6, h=4,
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 0.3},
                {"color": "red", "value": 0.7},
            ],
        ))

        panels.append(self._stat_panel(
            title="知识数量",
            expr=f"{p}_knowledge_count",
            unit="short",
            x=6, y=y_pos, w=6, h=4,
        ))

        panels.append(self._stat_panel(
            title="KVStore 利用率",
            expr=f"{p}_knowledge_utilization",
            unit="percentunit",
            x=12, y=y_pos, w=6, h=4,
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 0.7},
                {"color": "red", "value": 0.9},
            ],
        ))

        panels.append(self._stat_panel(
            title="锁定知识",
            expr=f"{p}_knowledge_pinned_count",
            unit="short",
            x=18, y=y_pos, w=6, h=4,
        ))
        y_pos += 4

        # === Row 2: Forward 性能 ===
        panels.append(self._row("⚡ Forward 性能", y_pos))
        y_pos += 1

        panels.append(self._graph_panel(
            title="Forward 延迟 (P50/P95/P99)",
            exprs=[
                (f"histogram_quantile(0.50, rate({p}_forward_latency_us_bucket[5m]))", "P50"),
                (f"histogram_quantile(0.95, rate({p}_forward_latency_us_bucket[5m]))", "P95"),
                (f"histogram_quantile(0.99, rate({p}_forward_latency_us_bucket[5m]))", "P99"),
            ],
            unit="µs",
            x=0, y=y_pos, w=12, h=8,
        ))

        panels.append(self._graph_panel(
            title="Forward QPS (applied vs bypassed)",
            exprs=[
                (f'rate({p}_forward_total{{applied="true"}}[1m])', "Applied"),
                (f'rate({p}_forward_total{{applied="false"}}[1m])', "Bypassed"),
            ],
            unit="ops",
            x=12, y=y_pos, w=12, h=8,
        ))
        y_pos += 8

        # === Row 3: 门控与熵 ===
        panels.append(self._row("🔑 门控与熵", y_pos))
        y_pos += 1

        panels.append(self._heatmap_panel(
            title="门控值分布",
            expr=f"rate({p}_gate_value_bucket[5m])",
            x=0, y=y_pos, w=12, h=8,
        ))

        panels.append(self._heatmap_panel(
            title="熵值分布",
            expr=f"rate({p}_entropy_value_bucket[5m])",
            x=12, y=y_pos, w=12, h=8,
        ))
        y_pos += 8

        # === Row 4: 召回器 ===
        panels.append(self._row("🔍 召回器", y_pos))
        y_pos += 1

        panels.append(self._graph_panel(
            title="召回器调用频率",
            exprs=[
                (f"rate({p}_retrieval_total[5m])", "Retrieval QPS"),
            ],
            unit="ops",
            x=0, y=y_pos, w=8, h=8,
        ))

        panels.append(self._graph_panel(
            title="召回注入数量",
            exprs=[
                (f"rate({p}_retrieval_injected_total[5m])", "Injected/s"),
            ],
            unit="ops",
            x=8, y=y_pos, w=8, h=8,
        ))

        panels.append(self._graph_panel(
            title="Slot 变化率",
            exprs=[
                (f"{p}_slot_change_rate", "Change Rate"),
            ],
            unit="short",
            x=16, y=y_pos, w=8, h=8,
        ))
        y_pos += 8

        # === Row 5: 审计 ===
        panels.append(self._row("📋 审计", y_pos))
        y_pos += 1

        panels.append(self._graph_panel(
            title="审计操作统计",
            exprs=[
                (f'rate({p}_audit_operations_total{{operation="register"}}[5m])', "Register"),
                (f'rate({p}_audit_operations_total{{operation="unregister"}}[5m])', "Unregister"),
                (f'rate({p}_audit_operations_total{{operation="load_from"}}[5m])', "Load"),
                (f'rate({p}_audit_operations_total{{operation="clear"}}[5m])', "Clear"),
            ],
            unit="ops",
            x=0, y=y_pos, w=24, h=8,
        ))
        y_pos += 8

        # 分配 panel ID
        for i, panel in enumerate(panels):
            panel["id"] = i + 1

        return {
            "dashboard": {
                "id": None,
                "uid": f"aga-observability-{int(time.time())}",
                "title": self._title,
                "tags": ["aga", "observability", "llm"],
                "timezone": "browser",
                "refresh": self._refresh,
                "schemaVersion": 39,
                "version": 1,
                "panels": panels,
                "time": {"from": "now-1h", "to": "now"},
                "templating": {
                    "list": [
                        {
                            "name": "datasource",
                            "type": "datasource",
                            "query": "prometheus",
                            "current": {"text": self._datasource, "value": self._datasource},
                        }
                    ]
                },
            },
            "overwrite": True,
        }

    def save(self, path: str) -> None:
        """保存 Dashboard JSON 到文件"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate())
        logger.info(f"Grafana Dashboard 已保存: {path}")

    # ========== 面板构建器 ==========

    def _row(self, title: str, y: int) -> Dict:
        return {
            "type": "row",
            "title": title,
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
            "panels": [],
        }

    def _stat_panel(
        self,
        title: str,
        expr: str,
        unit: str,
        x: int,
        y: int,
        w: int,
        h: int,
        thresholds: Optional[List[Dict]] = None,
    ) -> Dict:
        if thresholds is None:
            thresholds = [{"color": "green", "value": None}]

        return {
            "type": "stat",
            "title": title,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": [
                {
                    "expr": expr,
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds,
                    },
                },
            },
        }

    def _graph_panel(
        self,
        title: str,
        exprs: List[tuple],
        unit: str,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Dict:
        targets = []
        for expr, legend in exprs:
            targets.append({
                "expr": expr,
                "legendFormat": legend,
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
            })

        return {
            "type": "timeseries",
            "title": title,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": targets,
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "fillOpacity": 10,
                    },
                },
            },
        }

    def _heatmap_panel(
        self,
        title: str,
        expr: str,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Dict:
        return {
            "type": "heatmap",
            "title": title,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": [
                {
                    "expr": expr,
                    "format": "heatmap",
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                }
            ],
            "options": {
                "calculate": False,
                "color": {"scheme": "Oranges"},
            },
        }
