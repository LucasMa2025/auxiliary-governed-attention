"""
aga-knowledge 配置测试
"""

import pytest
import os
import tempfile

from aga_knowledge.config import (
    PortalConfig,
    ServerConfig,
    PersistenceDBConfig,
    MessagingConfig,
    RegistryConfig,
    GovernanceConfig,
    load_config,
)


class TestPortalConfig:
    """测试 Portal 配置"""

    def test_default_config(self):
        config = PortalConfig()
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8081
        assert config.persistence.type == "sqlite"
        assert config.messaging.backend == "redis"
        assert config.environment == "development"

    def test_for_development(self):
        config = PortalConfig.for_development()
        assert config.server.host == "127.0.0.1"
        assert config.server.reload is True
        assert config.persistence.type == "sqlite"
        assert config.messaging.backend == "memory"
        assert config.environment == "development"

    def test_for_production(self):
        config = PortalConfig.for_production(
            postgres_url="postgresql://localhost/aga",
            redis_host="redis-server",
        )
        assert config.persistence.type == "postgres"
        assert config.persistence.postgres_url == "postgresql://localhost/aga"
        assert config.messaging.backend == "redis"
        assert config.messaging.redis_host == "redis-server"
        assert config.environment == "production"

    def test_to_dict(self):
        config = PortalConfig.for_development()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "server" in d
        assert "persistence" in d
        assert "messaging" in d

    def test_from_dict(self):
        data = {
            "server": {"host": "10.0.0.1", "port": 9090},
            "persistence": {"type": "memory"},
            "messaging": {"backend": "memory"},
            "environment": "staging",
        }
        config = PortalConfig.from_dict(data)
        assert config.server.host == "10.0.0.1"
        assert config.server.port == 9090
        assert config.persistence.type == "memory"
        assert config.environment == "staging"

    def test_roundtrip(self):
        original = PortalConfig.for_production(
            postgres_url="postgresql://localhost/aga",
        )
        d = original.to_dict()
        restored = PortalConfig.from_dict(d)
        assert restored.persistence.postgres_url == original.persistence.postgres_url
        assert restored.environment == original.environment


class TestLoadConfig:
    """测试从 YAML 加载配置"""

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
server:
  host: "10.0.0.1"
  port: 9090
  workers: 2

persistence:
  type: "memory"

messaging:
  backend: "memory"

environment: "test"
"""
        config_path = str(tmp_path / "test_config.yaml")
        with open(config_path, "w") as f:
            f.write(yaml_content)

        try:
            config = load_config(config_path)
            assert config.server.host == "10.0.0.1"
            assert config.server.port == 9090
            assert config.persistence.type == "memory"
            assert config.environment == "test"
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_env_override(self, tmp_path, monkeypatch):
        yaml_content = """
server:
  host: "127.0.0.1"
  port: 8081

environment: "development"
"""
        config_path = str(tmp_path / "test_config.yaml")
        with open(config_path, "w") as f:
            f.write(yaml_content)

        monkeypatch.setenv("AGA_PORTAL_HOST", "0.0.0.0")
        monkeypatch.setenv("AGA_PORTAL_PORT", "9999")
        monkeypatch.setenv("AGA_ENVIRONMENT", "production")

        try:
            config = load_config(config_path)
            assert config.server.host == "0.0.0.0"
            assert config.server.port == 9999
            assert config.environment == "production"
        except ImportError:
            pytest.skip("PyYAML not installed")


class TestConfigAdapter:
    """测试配置适配器"""

    def test_env_adapter(self, monkeypatch):
        from aga_knowledge.config_adapter import EnvConfigAdapter

        monkeypatch.setenv("AGA_PORTAL_HOST", "10.0.0.1")
        monkeypatch.setenv("AGA_PORTAL_PORT", "9090")
        monkeypatch.setenv("AGA_PERSISTENCE_TYPE", "memory")

        adapter = EnvConfigAdapter()
        config = adapter.load()
        assert config.server.host == "10.0.0.1"
        assert config.server.port == 9090
        assert config.persistence.type == "memory"

    def test_env_summary(self, monkeypatch):
        from aga_knowledge.config_adapter import EnvConfigAdapter

        monkeypatch.setenv("AGA_PORTAL_HOST", "10.0.0.1")
        monkeypatch.setenv("AGA_MESSAGING_REDIS_PASSWORD", "secret123")

        adapter = EnvConfigAdapter()
        summary = adapter.get_env_summary()
        assert summary["AGA_PORTAL_HOST"] == "10.0.0.1"
        assert summary["AGA_MESSAGING_REDIS_PASSWORD"] == "***"

    def test_yaml_adapter(self, tmp_path):
        from aga_knowledge.config_adapter import YAMLConfigAdapter

        yaml_content = """
server:
  host: "10.0.0.1"
persistence:
  type: "memory"
"""
        config_path = str(tmp_path / "test.yaml")
        with open(config_path, "w") as f:
            f.write(yaml_content)

        try:
            adapter = YAMLConfigAdapter(config_path)
            config = adapter.load()
            assert config.server.host == "10.0.0.1"
            assert config.persistence.type == "memory"
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_chain_adapters(self, tmp_path, monkeypatch):
        from aga_knowledge.config_adapter import (
            ConfigAdapter, YAMLConfigAdapter, EnvConfigAdapter
        )

        yaml_content = """
server:
  host: "127.0.0.1"
  port: 8081
persistence:
  type: "sqlite"
"""
        config_path = str(tmp_path / "test.yaml")
        with open(config_path, "w") as f:
            f.write(yaml_content)

        monkeypatch.setenv("AGA_PORTAL_PORT", "9999")

        try:
            config = ConfigAdapter.chain([
                YAMLConfigAdapter(config_path),
                EnvConfigAdapter(),
            ])
            assert config.server.host == "127.0.0.1"  # from YAML
            assert config.server.port == 9999           # from env override
        except ImportError:
            pytest.skip("PyYAML not installed")
