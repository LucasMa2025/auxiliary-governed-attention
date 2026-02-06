"""
API 服务层单元测试

测试 AGAService 的核心业务逻辑。
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

from aga.api.service import AGAService, ServiceConfig
from aga.types import LifecycleState


@pytest.fixture
def service_config():
    """创建测试配置"""
    return ServiceConfig(
        hidden_dim=256,
        bottleneck_dim=32,
        num_slots=10,
        num_heads=4,
        persistence_enabled=False,
        writer_enabled=False,
        encoder_type="hash",
    )


@pytest.fixture
def service(service_config):
    """创建服务实例"""
    # 重置单例
    AGAService.reset_instance()
    svc = AGAService(service_config)
    yield svc
    # 清理
    AGAService.reset_instance()


@pytest.mark.unit
class TestServiceConfig:
    """ServiceConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = ServiceConfig()
        
        assert config.hidden_dim == 4096
        assert config.bottleneck_dim == 64
        assert config.num_slots == 100
        assert config.persistence_enabled is True
        assert config.writer_enabled is True
        assert config.encoder_type == "hash"
    
    def test_custom_values(self):
        """测试自定义值"""
        config = ServiceConfig(
            hidden_dim=2048,
            bottleneck_dim=128,
            num_slots=50,
            persistence_type="postgres",
        )
        
        assert config.hidden_dim == 2048
        assert config.bottleneck_dim == 128
        assert config.num_slots == 50
        assert config.persistence_type == "postgres"


@pytest.mark.unit
class TestAGAServiceSingleton:
    """AGAService 单例测试"""
    
    def test_get_instance(self, service_config):
        """测试获取单例"""
        AGAService.reset_instance()
        
        instance1 = AGAService.get_instance(service_config)
        instance2 = AGAService.get_instance()
        
        assert instance1 is instance2
        
        AGAService.reset_instance()
    
    def test_reset_instance(self, service_config):
        """测试重置单例"""
        AGAService.reset_instance()
        
        instance1 = AGAService.get_instance(service_config)
        AGAService.reset_instance()
        instance2 = AGAService.get_instance(service_config)
        
        assert instance1 is not instance2
        
        AGAService.reset_instance()


@pytest.mark.unit
class TestAGAServiceInitialization:
    """AGAService 初始化测试"""
    
    def test_initial_state(self, service):
        """测试初始状态"""
        assert service._initialized is False
        assert service._aga_instances == {}
        assert service._persistence_adapter is None
    
    @pytest.mark.asyncio
    async def test_initialize_without_persistence(self, service):
        """测试不带持久化的初始化"""
        result = await service.initialize()
        
        assert result is True
        assert service._initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, service):
        """测试初始化幂等性"""
        await service.initialize()
        result = await service.initialize()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shutdown(self, service):
        """测试关闭"""
        await service.initialize()
        await service.shutdown()
        
        # 应该能正常关闭


@pytest.mark.unit
class TestAGAServiceEncoder:
    """AGAService 编码器测试"""
    
    @pytest.mark.asyncio
    async def test_encoder_initialization(self, service):
        """测试编码器初始化"""
        await service.initialize()
        
        assert service._encoder is not None
    
    @pytest.mark.asyncio
    async def test_encoder_property(self, service):
        """测试编码器属性"""
        await service.initialize()
        
        encoder = service.encoder
        assert encoder is not None
    
    @pytest.mark.asyncio
    async def test_get_encoder_signature(self, service):
        """测试获取编码器签名"""
        await service.initialize()
        
        signature = service.get_encoder_signature()
        assert "encoder_type" in signature


@pytest.mark.unit
class TestAGAServiceAGAManagement:
    """AGAService AGA 实例管理测试"""
    
    @pytest.mark.asyncio
    async def test_get_or_create_aga(self, service):
        """测试获取或创建 AGA 实例"""
        await service.initialize()
        
        aga1 = service._get_or_create_aga("default")
        aga2 = service._get_or_create_aga("default")
        
        assert aga1 is aga2
    
    @pytest.mark.asyncio
    async def test_get_or_create_aga_different_namespaces(self, service):
        """测试不同命名空间的 AGA 实例"""
        await service.initialize()
        
        aga1 = service._get_or_create_aga("ns1")
        aga2 = service._get_or_create_aga("ns2")
        
        assert aga1 is not aga2
    
    @pytest.mark.asyncio
    async def test_get_namespaces(self, service):
        """测试列出命名空间"""
        await service.initialize()
        
        service._get_or_create_aga("ns1")
        service._get_or_create_aga("ns2")
        
        namespaces = service.get_namespaces()
        
        assert "ns1" in namespaces
        assert "ns2" in namespaces


@pytest.mark.unit
class TestAGAServiceKnowledgeOperations:
    """AGAService 知识操作测试"""
    
    @pytest.mark.asyncio
    async def test_inject_knowledge(self, service):
        """测试注入知识"""
        await service.initialize()
        
        result = await service.inject_knowledge(
            lu_id="LU_001",
            namespace="default",
            condition="用户询问天气",
            decision="提供天气信息",
            key_vector=[0.1] * service.config.bottleneck_dim,
            value_vector=[0.2] * service.config.hidden_dim,
        )
        
        assert result["success"] is True
        assert result["lu_id"] == "LU_001"
        assert "slot_idx" in result
    
    @pytest.mark.asyncio
    async def test_inject_knowledge_text(self, service):
        """测试文本知识注入"""
        await service.initialize()
        
        result = await service.inject_knowledge_text(
            lu_id="LU_002",
            namespace="default",
            condition="用户询问时间",
            decision="提供当前时间",
        )
        
        assert result["success"] is True
        assert result["lu_id"] == "LU_002"
    
    @pytest.mark.asyncio
    async def test_inject_duplicate_lu_id(self, service):
        """测试重复 LU ID 注入"""
        await service.initialize()
        
        # 第一次注入
        await service.inject_knowledge_text(
            lu_id="LU_DUP",
            namespace="default",
            condition="条件1",
            decision="决策1",
        )
        
        # 第二次注入相同 LU ID - 会创建新槽位
        result = await service.inject_knowledge_text(
            lu_id="LU_DUP",
            namespace="default",
            condition="条件2",
            decision="决策2",
        )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_get_knowledge(self, service):
        """测试获取知识"""
        await service.initialize()
        
        # 先注入
        await service.inject_knowledge_text(
            lu_id="LU_GET",
            namespace="default",
            condition="条件",
            decision="决策",
        )
        
        # 获取
        result = await service.get_knowledge("default", "LU_GET")
        
        assert result is not None
        assert result["lu_id"] == "LU_GET"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_knowledge(self, service):
        """测试获取不存在的知识"""
        await service.initialize()
        
        result = await service.get_knowledge("default", "NONEXISTENT")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_query_knowledge(self, service):
        """测试查询知识列表"""
        await service.initialize()
        
        # 先注入
        await service.inject_knowledge_text(
            lu_id="LU_QUERY",
            namespace="default",
            condition="条件",
            decision="决策",
        )
        
        # 查询
        results = await service.query_knowledge("default")
        
        assert len(results) >= 1


@pytest.mark.unit
class TestAGAServiceLifecycleOperations:
    """AGAService 生命周期操作测试"""
    
    @pytest.mark.asyncio
    async def test_update_lifecycle(self, service):
        """测试更新生命周期"""
        await service.initialize()
        
        # 先注入
        await service.inject_knowledge_text(
            lu_id="LU_LC",
            namespace="default",
            condition="条件",
            decision="决策",
        )
        
        # 更新生命周期
        result = await service.update_lifecycle(
            lu_id="LU_LC",
            namespace="default",
            new_state="confirmed",
            reason="测试确认",
        )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_quarantine_knowledge(self, service):
        """测试隔离知识"""
        await service.initialize()
        
        # 先注入
        await service.inject_knowledge_text(
            lu_id="LU_QT",
            namespace="default",
            condition="条件",
            decision="决策",
        )
        
        # 隔离
        result = await service.quarantine_knowledge(
            lu_id="LU_QT",
            namespace="default",
            reason="检测到异常",
        )
        
        assert result["success"] is True


@pytest.mark.unit
class TestAGAServiceStatistics:
    """AGAService 统计测试"""
    
    @pytest.mark.asyncio
    async def test_get_namespace_statistics(self, service):
        """测试获取命名空间统计"""
        await service.initialize()
        
        # 注入一些知识
        await service.inject_knowledge_text(
            lu_id="LU_STAT1",
            namespace="default",
            condition="条件1",
            decision="决策1",
        )
        
        stats = await service.get_namespace_statistics("default")
        
        assert stats is not None
        assert "total_slots" in stats
        assert "active_slots" in stats
    
    @pytest.mark.asyncio
    async def test_get_all_statistics(self, service):
        """测试获取所有统计"""
        await service.initialize()
        
        # 创建多个命名空间
        await service.inject_knowledge_text(
            lu_id="LU_S1",
            namespace="ns1",
            condition="条件",
            decision="决策",
        )
        await service.inject_knowledge_text(
            lu_id="LU_S2",
            namespace="ns2",
            condition="条件",
            decision="决策",
        )
        
        all_stats = await service.get_all_statistics()
        
        assert "namespaces" in all_stats
        assert "ns1" in all_stats["namespaces"]
        assert "ns2" in all_stats["namespaces"]


@pytest.mark.unit
class TestAGAServiceHealth:
    """AGAService 健康检查测试"""
    
    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """测试健康检查"""
        await service.initialize()
        
        health = await service.health_check()
        
        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert "uptime_seconds" in health
    
    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, service):
        """测试未初始化时的健康检查"""
        health = await service.health_check()
        
        assert health["initialized"] is False
