"""
Portal 服务层单元测试

测试 PortalService 的核心业务逻辑。
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

from aga.config.portal import PortalConfig, PersistenceDBConfig, MessagingConfig


@pytest.fixture
def portal_config():
    """创建测试配置"""
    return PortalConfig(
        persistence=PersistenceDBConfig(
            type="memory",
        ),
        messaging=MessagingConfig(
            backend="memory",
        ),
    )


@pytest.fixture
def mock_persistence():
    """创建模拟持久化适配器"""
    adapter = AsyncMock()
    adapter.connect = AsyncMock()
    adapter.disconnect = AsyncMock()
    adapter.save_knowledge = AsyncMock(return_value=True)
    adapter.load_knowledge = AsyncMock(return_value=None)
    adapter.delete_knowledge = AsyncMock(return_value=True)
    adapter.update_knowledge = AsyncMock(return_value=True)
    adapter.query_knowledge = AsyncMock(return_value=[])
    adapter.get_namespaces = AsyncMock(return_value=["default"])
    return adapter


@pytest.fixture
def mock_publisher():
    """创建模拟消息发布器"""
    publisher = AsyncMock()
    publisher.connect = AsyncMock()
    publisher.disconnect = AsyncMock()
    publisher.publish = AsyncMock(return_value=True)
    publisher.publish_inject = AsyncMock(return_value={"success": True})
    publisher.publish_update = AsyncMock(return_value={"success": True})
    publisher.publish_delete = AsyncMock(return_value={"success": True})
    publisher.publish_quarantine = AsyncMock(return_value={"success": True})
    return publisher


@pytest.fixture
def mock_encoder():
    """创建模拟编码器"""
    encoder = Mock()
    encoder.encoder_type = Mock()
    encoder.encoder_type.value = "hash"
    encoder.native_dim = 768
    encoder.key_dim = 64
    encoder.value_dim = 4096
    encoder.model_name = "hash-encoder"
    encoder.is_available = True
    encoder.initialize = Mock()
    encoder.encode = Mock(return_value=[0.1] * 768)
    encoder.encode_key = Mock(return_value=[0.1] * 64)
    encoder.encode_value = Mock(return_value=[0.2] * 4096)
    encoder.encode_constraint = Mock(return_value=([0.1] * 64, [0.2] * 4096))
    encoder.get_signature = Mock(return_value=Mock(to_dict=Mock(return_value={
        "type": "hash",
        "native_dim": 768,
        "key_dim": 64,
        "value_dim": 4096,
    })))
    return encoder


@pytest.mark.unit
class TestPortalConfig:
    """PortalConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PortalConfig()
        
        assert config.persistence is not None
        assert config.messaging is not None
    
    def test_custom_persistence(self):
        """测试自定义持久化配置"""
        config = PortalConfig(
            persistence=PersistenceDBConfig(
                type="sqlite",
                sqlite_path="/tmp/test.db",
            ),
        )
        
        assert config.persistence.type == "sqlite"
        assert config.persistence.sqlite_path == "/tmp/test.db"
    
    def test_custom_messaging(self):
        """测试自定义消息配置"""
        config = PortalConfig(
            messaging=MessagingConfig(
                backend="redis",
                redis_host="localhost",
                redis_port=6379,
            ),
        )
        
        assert config.messaging.backend == "redis"
        assert config.messaging.redis_host == "localhost"


@pytest.mark.unit
class TestPortalServiceInitialization:
    """PortalService 初始化测试"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, portal_config, mock_persistence, mock_publisher, mock_encoder):
        """测试初始化"""
        from aga.portal.service import PortalService
        
        with patch.object(PortalService, '_init_persistence', new_callable=AsyncMock) as mock_init_p, \
             patch.object(PortalService, '_init_publisher', new_callable=AsyncMock) as mock_init_pub, \
             patch.object(PortalService, '_init_encoder', new_callable=AsyncMock) as mock_init_enc:
            
            service = PortalService(portal_config)
            await service.initialize()
            
            assert service._initialized is True
            mock_init_p.assert_called_once()
            mock_init_pub.assert_called_once()
            mock_init_enc.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, portal_config):
        """测试初始化幂等性"""
        from aga.portal.service import PortalService
        
        with patch.object(PortalService, '_init_persistence', new_callable=AsyncMock), \
             patch.object(PortalService, '_init_publisher', new_callable=AsyncMock), \
             patch.object(PortalService, '_init_encoder', new_callable=AsyncMock):
            
            service = PortalService(portal_config)
            await service.initialize()
            await service.initialize()  # 第二次调用
            
            # 应该只初始化一次
            assert service._initialized is True
    
    @pytest.mark.asyncio
    async def test_shutdown(self, portal_config, mock_persistence, mock_publisher):
        """测试关闭"""
        from aga.portal.service import PortalService
        
        service = PortalService(portal_config)
        service._persistence = mock_persistence
        service._publisher = mock_publisher
        service._initialized = True
        
        await service.shutdown()
        
        mock_persistence.disconnect.assert_called_once()
        mock_publisher.disconnect.assert_called_once()
        assert service._initialized is False


@pytest.mark.unit
class TestPortalServiceEncoder:
    """PortalService 编码器测试"""
    
    def test_get_encoder_signature(self, portal_config, mock_encoder):
        """测试获取编码器签名"""
        from aga.portal.service import PortalService
        
        service = PortalService(portal_config)
        service._encoder = mock_encoder
        service._initialized = True
        
        signature = service.get_encoder_signature()
        
        assert "type" in signature
    
    def test_encoder_property_not_initialized(self, portal_config):
        """测试未初始化时访问编码器"""
        from aga.portal.service import PortalService
        
        service = PortalService(portal_config)
        
        with pytest.raises(RuntimeError):
            _ = service.encoder


@pytest.mark.unit
class TestPortalServiceKnowledgeOperations:
    """PortalService 知识操作测试"""
    
    @pytest.mark.asyncio
    async def test_inject_knowledge(self, portal_config, mock_persistence, mock_publisher):
        """测试注入知识"""
        from aga.portal.service import PortalService
        
        service = PortalService(portal_config)
        service._persistence = mock_persistence
        service._publisher = mock_publisher
        service._initialized = True
        
        result = await service.inject_knowledge(
            lu_id="LU_001",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 4096,
            condition="条件",
            decision="决策",
        )
        
        assert result["success"] is True
        assert result["lu_id"] == "LU_001"
        mock_persistence.save_knowledge.assert_called_once()
        mock_publisher.publish_inject.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_inject_knowledge_text(self, portal_config, mock_persistence, mock_publisher, mock_encoder):
        """测试文本知识注入"""
        from aga.portal.service import PortalService
        
        service = PortalService(portal_config)
        service._persistence = mock_persistence
        service._publisher = mock_publisher
        service._encoder = mock_encoder
        service._initialized = True
        
        result = await service.inject_knowledge_text(
            lu_id="LU_001",
            condition="用户询问天气",
            decision="提供天气信息",
        )
        
        assert result["success"] is True
        mock_encoder.encode_constraint.assert_called_once_with("用户询问天气", "提供天气信息")
    
    @pytest.mark.asyncio
    async def test_get_knowledge(self, portal_config, mock_persistence):
        """测试获取知识"""
        from aga.portal.service import PortalService
        
        # 模拟返回数据 - 返回字典而不是 KnowledgeRecord
        mock_persistence.load_knowledge = AsyncMock(return_value={
            "lu_id": "LU_001",
            "slot_idx": 0,
            "namespace": "default",
            "key_vector": [0.1] * 64,
            "value_vector": [0.2] * 4096,
            "condition": "条件",
            "decision": "决策",
            "lifecycle_state": "probationary",
        })
        
        service = PortalService(portal_config)
        service._persistence = mock_persistence
        service._initialized = True
        
        result = await service.get_knowledge("LU_001", "default")
        
        assert result is not None
        # 向量应该被移除（include_vectors=False）
        mock_persistence.load_knowledge.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_knowledge(self, portal_config, mock_persistence, mock_publisher):
        """测试删除知识"""
        from aga.portal.service import PortalService
        
        # 模拟存在的记录
        mock_persistence.load_knowledge = AsyncMock(return_value={
            "lu_id": "LU_001",
            "namespace": "default",
            "lifecycle_state": "probationary",
        })
        
        service = PortalService(portal_config)
        service._persistence = mock_persistence
        service._publisher = mock_publisher
        service._initialized = True
        
        result = await service.delete_knowledge("LU_001", "default")
        
        assert result["success"] is True
        mock_persistence.delete_knowledge.assert_called_once()


@pytest.mark.unit
class TestPortalServiceLifecycle:
    """PortalService 生命周期操作测试"""
    
    @pytest.mark.asyncio
    async def test_update_lifecycle(self, portal_config, mock_persistence, mock_publisher):
        """测试更新生命周期"""
        from aga.portal.service import PortalService
        
        # 模拟加载现有知识
        mock_persistence.load_knowledge = AsyncMock(return_value={
            "lu_id": "LU_001",
            "slot_idx": 0,
            "namespace": "default",
            "key_vector": [0.1] * 64,
            "value_vector": [0.2] * 4096,
            "condition": "条件",
            "decision": "决策",
            "lifecycle_state": "probationary",
        })
        
        service = PortalService(portal_config)
        service._persistence = mock_persistence
        service._publisher = mock_publisher
        service._initialized = True
        
        result = await service.update_lifecycle(
            lu_id="LU_001",
            namespace="default",
            new_state="confirmed",
        )
        
        assert result["success"] is True


@pytest.mark.unit
class TestPortalServiceHealth:
    """PortalService 健康检查测试"""
    
    @pytest.mark.asyncio
    async def test_health_check(self, portal_config):
        """测试健康检查"""
        from aga.portal.service import PortalService
        
        service = PortalService(portal_config)
        service._initialized = True
        
        health = await service.health_check()
        
        assert health["status"] == "healthy"
        assert "uptime_seconds" in health
    
    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, portal_config):
        """测试未初始化时的健康检查"""
        from aga.portal.service import PortalService
        
        service = PortalService(portal_config)
        
        health = await service.health_check()
        
        # 未初始化时状态为 not_initialized
        assert health["status"] == "not_initialized"


@pytest.mark.unit
class TestPortalServiceBatchOperations:
    """PortalService 批量操作测试"""
    
    @pytest.mark.asyncio
    async def test_batch_inject_text(self, portal_config, mock_persistence, mock_publisher, mock_encoder):
        """测试批量文本注入"""
        from aga.portal.service import PortalService
        
        service = PortalService(portal_config)
        service._persistence = mock_persistence
        service._publisher = mock_publisher
        service._encoder = mock_encoder
        service._initialized = True
        
        items = [
            {"lu_id": "LU_001", "condition": "条件1", "decision": "决策1"},
            {"lu_id": "LU_002", "condition": "条件2", "decision": "决策2"},
        ]
        
        result = await service.batch_inject_text(items)
        
        assert result["total"] == 2
        assert result["success_count"] == 2
        assert result["failed_count"] == 0
