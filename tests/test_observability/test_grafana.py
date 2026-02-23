"""
tests/test_observability/test_grafana.py — GrafanaDashboardGenerator 测试
"""
import json
import pytest
from aga_observability.grafana_dashboard import GrafanaDashboardGenerator


class TestGrafanaDashboardGenerator:
    """GrafanaDashboardGenerator 测试"""

    def test_generate_json(self):
        """生成 JSON 字符串"""
        gen = GrafanaDashboardGenerator()
        result = gen.generate()
        assert isinstance(result, str)

        # 验证是有效 JSON
        data = json.loads(result)
        assert "dashboard" in data
        assert "panels" in data["dashboard"]

    def test_to_dict(self):
        """生成字典"""
        gen = GrafanaDashboardGenerator()
        data = gen.to_dict()
        assert isinstance(data, dict)
        assert "dashboard" in data

    def test_panels_exist(self):
        """面板存在"""
        gen = GrafanaDashboardGenerator()
        data = gen.to_dict()
        panels = data["dashboard"]["panels"]
        assert len(panels) > 0

        # 检查面板类型
        panel_types = {p["type"] for p in panels}
        assert "row" in panel_types
        assert "stat" in panel_types
        assert "timeseries" in panel_types

    def test_custom_prefix(self):
        """自定义前缀"""
        gen = GrafanaDashboardGenerator(prefix="myaga")
        result = gen.generate()
        assert "myaga" in result

    def test_custom_title(self):
        """自定义标题"""
        gen = GrafanaDashboardGenerator(title="My AGA Dashboard")
        data = gen.to_dict()
        assert data["dashboard"]["title"] == "My AGA Dashboard"

    def test_panel_ids_unique(self):
        """面板 ID 唯一"""
        gen = GrafanaDashboardGenerator()
        data = gen.to_dict()
        ids = [p["id"] for p in data["dashboard"]["panels"]]
        assert len(ids) == len(set(ids))

    def test_save_to_file(self, tmp_path):
        """保存到文件"""
        gen = GrafanaDashboardGenerator()
        path = str(tmp_path / "dashboard.json")
        gen.save(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "dashboard" in data

    def test_datasource_template(self):
        """数据源模板变量"""
        gen = GrafanaDashboardGenerator(datasource="MyPrometheus")
        data = gen.to_dict()
        templating = data["dashboard"]["templating"]["list"]
        assert len(templating) > 0
        assert templating[0]["name"] == "datasource"

    def test_refresh_interval(self):
        """刷新间隔"""
        gen = GrafanaDashboardGenerator(refresh="10s")
        data = gen.to_dict()
        assert data["dashboard"]["refresh"] == "10s"

    def test_all_metric_prefixes(self):
        """所有指标使用正确前缀"""
        gen = GrafanaDashboardGenerator(prefix="test_aga")
        result = gen.generate()
        # 确保所有 Prometheus 查询使用正确前缀
        assert "test_aga_forward" in result
        assert "test_aga_retrieval" in result
        assert "test_aga_knowledge" in result
        assert "test_aga_activation" in result
