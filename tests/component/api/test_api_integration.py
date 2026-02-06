"""
API 模块组件集成测试

测试 API 服务的端到端功能。
"""
import pytest
import asyncio
import tempfile
import os

from aga.api.service import AGAService, ServiceConfig


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
def service_config(temp_db):
    """创建测试配置"""
    return ServiceConfig(
        hidden_dim=256,
        bottleneck_dim=32,
        num_slots=20,
        num_heads=4,
        persistence_enabled=True,
        persistence_type="sqlite",
        persistence_path=temp_db,
        writer_enabled=False,
        encoder_type="hash",
    )


@pytest.fixture
async def service(service_config):
    """创建并初始化服务"""
    AGAService.reset_instance()
    svc = AGAService(service_config)
    await svc.initialize()
    yield svc
    await svc.shutdown()
    AGAService.reset_instance()


@pytest.mark.component
class TestAPIServiceIntegration:
    """API 服务集成测试"""
    
    @pytest.mark.asyncio
    async def test_inject_and_query(self, service):
        """测试注入和查询"""
        # 1. 注入知识
        inject_result = await service.inject_knowledge_text(
            lu_id="LU_LIFECYCLE",
            namespace="default",
            condition="用户询问天气",
            decision="提供天气信息",
        )
        assert inject_result["success"] is True
        
        # 2. 查询知识
        knowledge = await service.query_knowledge("default")
        assert len(knowledge) >= 1
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, service):
        """测试批量操作"""
        # 批量注入
        items = [
            {
                "lu_id": f"LU_BATCH_{i}",
                "condition": f"条件{i}",
                "decision": f"决策{i}",
            }
            for i in range(5)
        ]
        
        for item in items:
            result = await service.inject_knowledge_text(
                lu_id=item["lu_id"],
                namespace="default",
                condition=item["condition"],
                decision=item["decision"],
            )
            assert result["success"] is True
        
        # 获取统计
        stats = await service.get_namespace_statistics("default")
        assert stats["active_slots"] >= 5
    
    @pytest.mark.asyncio
    async def test_multi_namespace(self, service):
        """测试多命名空间"""
        # 在不同命名空间注入
        await service.inject_knowledge_text(
            lu_id="LU_NS1",
            namespace="ns1",
            condition="条件1",
            decision="决策1",
        )
        
        await service.inject_knowledge_text(
            lu_id="LU_NS2",
            namespace="ns2",
            condition="条件2",
            decision="决策2",
        )
        
        # 验证命名空间隔离
        namespaces = service.get_namespaces()
        assert "ns1" in namespaces
        assert "ns2" in namespaces
    
    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """测试健康检查"""
        health = await service.health_check()
        
        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert health["uptime_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_statistics(self, service):
        """测试统计功能"""
        # 注入一些知识
        for i in range(3):
            await service.inject_knowledge_text(
                lu_id=f"LU_STAT_{i}",
                namespace="default",
                condition=f"条件{i}",
                decision=f"决策{i}",
            )
        
        stats = await service.get_namespace_statistics("default")
        
        assert stats["total_slots"] > 0
        assert stats["active_slots"] >= 3
        assert "state_distribution" in stats


@pytest.mark.component
class TestAPIServicePersistence:
    """API 服务持久化测试"""
    
    @pytest.mark.asyncio
    async def test_persistence_save_load(self, service_config):
        """测试持久化保存和加载"""
        # 第一个服务实例 - 保存数据
        AGAService.reset_instance()
        service1 = AGAService(service_config)
        await service1.initialize()
        
        await service1.inject_knowledge_text(
            lu_id="LU_PERSIST",
            namespace="default",
            condition="持久化测试条件",
            decision="持久化测试决策",
        )
        
        await service1.shutdown()
        AGAService.reset_instance()


@pytest.mark.component
class TestAPIServiceEncoding:
    """API 服务编码测试"""
    
    @pytest.mark.asyncio
    async def test_text_encoding(self, service):
        """测试文本编码"""
        # 使用文本 API 注入
        result = await service.inject_knowledge_text(
            lu_id="LU_ENCODE",
            namespace="default",
            condition="这是一个测试条件",
            decision="这是一个测试决策",
        )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_vector_injection(self, service):
        """测试向量注入"""
        # 直接使用向量 API 注入
        result = await service.inject_knowledge(
            lu_id="LU_VECTOR",
            namespace="default",
            condition="条件",
            decision="决策",
            key_vector=[0.1] * service.config.bottleneck_dim,
            value_vector=[0.2] * service.config.hidden_dim,
        )
        
        assert result["success"] is True
