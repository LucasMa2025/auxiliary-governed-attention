"""
AGA 烟雾测试 - 公共导出验证

确保包入口可用，关键符号可导入。
此测试应在 CI 中优先执行。

运行方式：
    pytest tests/test_smoke.py -v
"""
import pytest


class TestPackageImport:
    """测试包导入"""
    
    def test_import_aga(self):
        """测试 import aga 成功"""
        import aga
        assert aga is not None
    
    def test_version_exists(self):
        """测试版本号存在"""
        import aga
        assert hasattr(aga, '__version__')
        assert isinstance(aga.__version__, str)
        assert len(aga.__version__) > 0
        # 版本格式检查 (x.y.z)
        parts = aga.__version__.split('.')
        assert len(parts) >= 2, f"版本格式应为 x.y 或 x.y.z，实际: {aga.__version__}"


class TestCoreExports:
    """测试核心导出符号"""
    
    def test_types_export(self):
        """测试类型导出"""
        from aga import (
            LifecycleState,
            Slot,
            GateContext,
            KnowledgeSlotInfo,
            AGADiagnostics,
        )
        assert LifecycleState is not None
        assert Slot is not None
    
    def test_config_export(self):
        """测试配置导出"""
        from aga import (
            AGAConfig,
            GateConfig,
            SlotPoolConfig,
            PersistenceConfig,
        )
        assert AGAConfig is not None
    
    def test_operator_export(self):
        """测试算子导出"""
        from aga import (
            AGAOperator,
            AGAManager,
        )
        assert AGAOperator is not None
        assert AGAManager is not None
    
    def test_persistence_export(self):
        """测试持久化导出"""
        from aga import (
            PersistenceAdapter,
            KnowledgeRecord,
            SQLiteAdapter,
            MemoryAdapter,
            PersistenceManager,
            create_adapter,
        )
        assert PersistenceAdapter is not None
        assert create_adapter is not None
    
    def test_exception_export(self):
        """测试异常导出"""
        from aga import (
            AGAException,
            AGAConfigError,
            AGAInjectionError,
            AGAPersistenceError,
        )
        assert AGAException is not None
        # 验证别名正确
        from aga.exceptions import ConfigurationError
        assert AGAConfigError is ConfigurationError


class TestOptionalExports:
    """测试可选模块导出（可能因依赖缺失而为 None）"""
    
    def test_api_export(self):
        """测试 API 导出（需要 fastapi）"""
        from aga import create_api_app, AGAAPIService
        # 可能为 None（如果没有安装 fastapi）
        # 但导入本身不应失败
    
    def test_portal_export(self):
        """测试 Portal 导出（需要 fastapi）"""
        from aga import create_portal_app, PortalService
    
    def test_runtime_export(self):
        """测试 Runtime 导出"""
        from aga import RuntimeAgent, AGARuntime, LocalCache
    
    def test_encoder_export(self):
        """测试编码器导出"""
        from aga import (
            BaseEncoder,
            EncoderType,
            EncoderFactory,
        )


class TestCreateAdapter:
    """测试适配器工厂函数"""
    
    def test_create_memory_adapter(self):
        """测试创建内存适配器"""
        from aga import create_adapter
        adapter = create_adapter("memory")
        assert adapter is not None
    
    def test_create_sqlite_adapter(self, tmp_path):
        """测试创建 SQLite 适配器"""
        from aga import create_adapter
        db_path = tmp_path / "test.db"
        adapter = create_adapter("sqlite", db_path=str(db_path))
        assert adapter is not None
    
    def test_create_unknown_adapter_raises(self):
        """测试创建未知适配器抛出异常"""
        from aga import create_adapter
        with pytest.raises(ValueError, match="Unknown adapter type"):
            create_adapter("unknown_type")


class TestBasicFunctionality:
    """测试基本功能（不需要 GPU）"""
    
    def test_create_config(self):
        """测试创建配置"""
        from aga import AGAConfig
        config = AGAConfig()
        # AGAConfig 使用嵌套配置结构
        assert config.slot_pool is not None
        assert config.gate is not None
        assert config.num_heads > 0
    
    def test_lifecycle_state_enum(self):
        """测试生命周期状态枚举"""
        from aga import LifecycleState
        assert LifecycleState.PROBATIONARY.value == "probationary"
        assert LifecycleState.CONFIRMED.value == "confirmed"
        assert LifecycleState.DEPRECATED.value == "deprecated"
        assert LifecycleState.QUARANTINED.value == "quarantined"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
