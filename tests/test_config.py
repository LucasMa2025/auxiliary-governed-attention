"""
AGAConfig 单元测试
"""
import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aga.config import AGAConfig
from aga.exceptions import ConfigError


class TestAGAConfig:
    """AGAConfig 测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = AGAConfig()
        assert config.hidden_dim == 4096
        assert config.bottleneck_dim == 64
        assert config.max_slots == 256
        assert config.tau_low == 0.5
        assert config.tau_high == 2.0
        assert config.max_gate == 0.8
        assert config.fail_open is True
        assert config.decay_enabled is True
        assert config.instrumentation_enabled is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = AGAConfig(
            hidden_dim=2048,
            bottleneck_dim=32,
            max_slots=512,
            tau_low=0.3,
            tau_high=1.5,
        )
        assert config.hidden_dim == 2048
        assert config.bottleneck_dim == 32
        assert config.max_slots == 512
        assert config.tau_low == 0.3
        assert config.tau_high == 1.5

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "hidden_dim": 2048,
            "bottleneck_dim": 32,
            "max_slots": 128,
        }
        config = AGAConfig.from_dict(data)
        assert config.hidden_dim == 2048
        assert config.bottleneck_dim == 32
        assert config.max_slots == 128

    def test_from_dict_nested_gate(self):
        """测试嵌套 gate 配置展平"""
        data = {
            "hidden_dim": 4096,
            "gate": {
                "tau_low": 0.3,
                "tau_high": 1.5,
                "max_gate": 0.7,
            }
        }
        config = AGAConfig.from_dict(data)
        assert config.tau_low == 0.3
        assert config.tau_high == 1.5
        assert config.max_gate == 0.7

    def test_from_dict_nested_decay(self):
        """测试嵌套 decay 配置展平"""
        data = {
            "hidden_dim": 4096,
            "decay": {
                "enabled": True,
                "strategy": "linear",
                "gamma": 0.85,
            }
        }
        config = AGAConfig.from_dict(data)
        assert config.decay_enabled is True
        assert config.decay_strategy == "linear"
        assert config.decay_gamma == 0.85

    def test_from_dict_ignores_unknown_fields(self):
        """测试忽略未知字段"""
        data = {
            "hidden_dim": 4096,
            "unknown_field": "should_be_ignored",
            "another_unknown": 42,
        }
        config = AGAConfig.from_dict(data)
        assert config.hidden_dim == 4096
        assert not hasattr(config, "unknown_field")

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = AGAConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_config(self):
        """测试验证无效配置"""
        config = AGAConfig(
            hidden_dim=-1,
            bottleneck_dim=0,
            max_slots=0,
            tau_low=3.0,
            tau_high=1.0,
        )
        errors = config.validate()
        assert len(errors) > 0

    def test_to_dict(self):
        """测试导出为字典"""
        config = AGAConfig(hidden_dim=2048)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["hidden_dim"] == 2048
        assert "tau_low" in d
        assert "max_slots" in d

    def test_from_yaml_nonexistent(self):
        """测试加载不存在的 YAML 文件"""
        with pytest.raises(ConfigError):
            AGAConfig.from_yaml("nonexistent.yaml")
