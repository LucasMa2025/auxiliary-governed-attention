"""
Portal 模块组件集成测试

测试 Portal 服务的端到端功能。
"""
import pytest
import asyncio
import tempfile
import os

from aga.config.portal import PortalConfig, PersistenceDBConfig, MessagingConfig
from aga.portal.service import PortalService
from aga.portal.registry import RuntimeRegistry


@pytest.fixture
def temp_db():
    """创建临时数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def portal_config(temp_db):
    """创建测试配置"""
    return PortalConfig(
        persistence=PersistenceDBConfig(
            type="sqlite",
            sqlite_path=temp_db,
        ),
        messaging=MessagingConfig(
            backend="memory",
        ),
    )


@pytest.fixture
async def portal_service(portal_config):
    """创建并初始化 Portal 服务"""
    service = PortalService(portal_config)
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.mark.component
class TestPortalServiceIntegration:
    """Portal 服务集成测试"""
    
    @pytest.mark.asyncio
    async def test_inject_and_get(self, portal_service):
        """测试注入和获取"""
        # 1. 注入知识
        inject_result = await portal_service.inject_knowledge_text(
            lu_id="LU_PORTAL_LC",
            condition="用户询问天气",
            decision="提供天气信息",
        )
        assert inject_result["success"] is True
        
        # 2. 获取知识
        knowledge = await portal_service.get_knowledge("LU_PORTAL_LC", "default")
        assert knowledge is not None
    
    @pytest.mark.asyncio
    async def test_batch_inject_text(self, portal_service):
        """测试批量文本注入"""
        items = [
            {
                "lu_id": f"LU_BATCH_{i}",
                "condition": f"条件{i}",
                "decision": f"决策{i}",
            }
            for i in range(5)
        ]
        
        result = await portal_service.batch_inject_text(items)
        
        assert result["success_count"] >= 5
    
    @pytest.mark.asyncio
    async def test_health_check(self, portal_service):
        """测试健康检查"""
        health = await portal_service.health_check()
        
        assert health["status"] == "healthy"
        assert "uptime_seconds" in health
    
    @pytest.mark.asyncio
    async def test_encoder_signature(self, portal_service):
        """测试编码器签名"""
        signature = portal_service.get_encoder_signature()
        
        assert "encoder_type" in signature
        assert "native_dim" in signature


@pytest.mark.component
class TestRuntimeRegistryIntegration:
    """RuntimeRegistry 集成测试"""
    
    @pytest.mark.asyncio
    async def test_registry_lifecycle(self):
        """测试注册表生命周期"""
        registry = RuntimeRegistry(
            heartbeat_timeout=5,
            cleanup_interval=1,
        )
        
        await registry.start()
        
        # 注册 Runtime
        registry.register(
            instance_id="runtime-001",
            namespaces=["default", "test"],
            host="localhost",
            port=8080,
        )
        
        # 验证注册
        info = registry.get_runtime("runtime-001")
        assert info is not None
        assert info.namespaces == ["default", "test"]
        
        # 心跳
        registry.heartbeat("runtime-001")
        
        # 列出
        all_runtimes = registry.get_all_runtimes()
        assert len(all_runtimes) == 1
        
        # 注销
        registry.deregister("runtime-001")
        assert registry.get_runtime("runtime-001") is None
        
        await registry.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_runtimes(self):
        """测试多个 Runtime"""
        registry = RuntimeRegistry()
        
        await registry.start()
        
        # 注册多个 Runtime
        for i in range(3):
            registry.register(
                instance_id=f"runtime-{i:03d}",
                namespaces=["default"],
            )
        
        assert len(registry.get_all_runtimes()) == 3
        
        # 按命名空间查询
        runtimes = registry.get_runtimes_for_namespace("default")
        assert len(runtimes) == 3
        
        await registry.stop()
    
    @pytest.mark.asyncio
    async def test_heartbeat_timeout(self):
        """测试心跳超时"""
        registry = RuntimeRegistry(
            heartbeat_timeout=0.1,
            cleanup_interval=0.05,
        )
        
        await registry.start()
        
        registry.register(
            instance_id="runtime-timeout",
            namespaces=["default"],
        )
        
        # 等待超时
        await asyncio.sleep(0.2)
        
        # 应该被清理
        assert registry.get_runtime("runtime-timeout") is None
        
        await registry.stop()


@pytest.mark.component
class TestPortalPersistence:
    """Portal 持久化测试"""
    
    @pytest.mark.asyncio
    async def test_persistence_roundtrip(self, portal_config):
        """测试持久化往返"""
        # 第一个服务 - 保存
        service1 = PortalService(portal_config)
        await service1.initialize()
        
        await service1.inject_knowledge_text(
            lu_id="LU_PERSIST",
            condition="持久化测试",
            decision="测试决策",
        )
        
        await service1.shutdown()
        
        # 第二个服务 - 加载
        service2 = PortalService(portal_config)
        await service2.initialize()
        
        knowledge = await service2.get_knowledge("LU_PERSIST", "default")
        
        assert knowledge is not None
        
        await service2.shutdown()
