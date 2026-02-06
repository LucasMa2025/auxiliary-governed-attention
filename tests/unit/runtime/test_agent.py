"""
Runtime Agent 单元测试

测试 RuntimeAgent 的核心功能。
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from aga.config.runtime import RuntimeConfig, AGAModuleConfig, SyncClientConfig
from aga.sync.protocol import SyncMessage, MessageType


@pytest.fixture
def runtime_config():
    """创建测试配置"""
    return RuntimeConfig(
        instance_id="runtime-001",
        namespace="default",
        namespaces=["default"],
        aga=AGAModuleConfig(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=10,
        ),
        sync=SyncClientConfig(
            backend="memory",
            sync_on_start=False,
        ),
        device="cpu",
        dtype="float32",
    )


@pytest.mark.unit
class TestRuntimeAgentBasic:
    """RuntimeAgent 基本功能测试"""
    
    def test_create_agent(self, runtime_config):
        """测试创建代理"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        
        assert agent.instance_id == "runtime-001"
        assert agent.namespaces == ["default"]
        assert agent._initialized is False
        assert agent._running is False
    
    def test_initial_stats(self, runtime_config):
        """测试初始统计"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        
        assert agent._stats["messages_received"] == 0
        assert agent._stats["inject_count"] == 0
        assert agent._stats["errors"] == 0


@pytest.mark.unit
class TestRuntimeAgentInitialization:
    """RuntimeAgent 初始化测试"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, runtime_config):
        """测试初始化"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        
        await agent.initialize()
        
        assert agent._initialized is True
        assert agent._cache is not None
        assert agent._runtime is not None
        assert agent._subscriber is not None
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, runtime_config):
        """测试初始化幂等性"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        
        await agent.initialize()
        await agent.initialize()  # 第二次调用
        
        assert agent._initialized is True


@pytest.mark.unit
class TestRuntimeAgentSync:
    """RuntimeAgent 同步测试"""
    
    @pytest.fixture
    def mock_subscriber(self):
        """创建 Mock 订阅器"""
        subscriber = AsyncMock()
        subscriber.connect = AsyncMock()
        subscriber.disconnect = AsyncMock()
        subscriber.start = AsyncMock()
        subscriber.stop = AsyncMock()
        subscriber.on_inject = Mock()
        subscriber.on_update = Mock()
        subscriber.on_quarantine = Mock()
        subscriber.on_delete = Mock()
        subscriber.on_batch_inject = Mock()
        subscriber.on_full_sync = Mock()
        return subscriber
    
    @pytest.mark.asyncio
    async def test_start(self, runtime_config, mock_subscriber):
        """测试启动"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        await agent.initialize()
        agent._subscriber = mock_subscriber
        
        # Mock Portal 注册
        with patch.object(agent, '_register_to_portal', new_callable=AsyncMock):
            await agent.start()
        
        assert agent._running is True
        mock_subscriber.connect.assert_called_once()
        mock_subscriber.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop(self, runtime_config, mock_subscriber):
        """测试停止"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        await agent.initialize()
        agent._subscriber = mock_subscriber
        agent._running = True
        
        # Mock Portal 注销
        with patch.object(agent, '_deregister_from_portal', new_callable=AsyncMock):
            await agent.stop()
        
        assert agent._running is False
        mock_subscriber.stop.assert_called_once()
        mock_subscriber.disconnect.assert_called_once()


@pytest.mark.unit
class TestRuntimeAgentMessageHandling:
    """RuntimeAgent 消息处理测试"""
    
    @pytest.fixture
    async def initialized_agent(self, runtime_config):
        """创建已初始化的代理"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_handle_inject(self, initialized_agent):
        """测试处理注入消息"""
        message = SyncMessage(
            message_type=MessageType.INJECT,
            lu_id="LU_001",
            namespace="default",
            key_vector=[0.1] * 32,
            value_vector=[0.2] * 256,
            condition="条件",
            decision="决策",
        )
        
        await initialized_agent._handle_inject(message)
        
        assert initialized_agent._stats["inject_count"] == 1
        assert initialized_agent._cache.contains("LU_001")
    
    @pytest.mark.asyncio
    async def test_handle_update(self, initialized_agent):
        """测试处理更新消息"""
        # 先注入
        inject_msg = SyncMessage(
            message_type=MessageType.INJECT,
            lu_id="LU_001",
            namespace="default",
            key_vector=[0.1] * 32,
            value_vector=[0.2] * 256,
        )
        await initialized_agent._handle_inject(inject_msg)
        
        # 再更新
        update_msg = SyncMessage(
            message_type=MessageType.UPDATE,
            lu_id="LU_001",
            namespace="default",
            lifecycle_state="confirmed",
        )
        await initialized_agent._handle_update(update_msg)
        
        assert initialized_agent._stats["update_count"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_delete(self, initialized_agent):
        """测试处理删除消息"""
        # 先注入
        inject_msg = SyncMessage(
            message_type=MessageType.INJECT,
            lu_id="LU_001",
            namespace="default",
            key_vector=[0.1] * 32,
            value_vector=[0.2] * 256,
        )
        await initialized_agent._handle_inject(inject_msg)
        
        # 再删除
        delete_msg = SyncMessage(
            message_type=MessageType.DELETE,
            lu_id="LU_001",
            namespace="default",
        )
        await initialized_agent._handle_delete(delete_msg)
        
        assert not initialized_agent._cache.contains("LU_001")


@pytest.mark.unit
class TestRuntimeAgentCallbacks:
    """RuntimeAgent 回调测试"""
    
    @pytest.mark.asyncio
    async def test_on_inject_callback(self, runtime_config):
        """测试注入回调"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        await agent.initialize()
        
        callback = Mock()
        agent.on_inject(callback)
        
        message = SyncMessage(
            message_type=MessageType.INJECT,
            lu_id="LU_001",
            namespace="default",
            key_vector=[0.1] * 32,
            value_vector=[0.2] * 256,
        )
        await agent._handle_inject(message)
        
        callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_on_update_callback(self, runtime_config):
        """测试更新回调"""
        from aga.runtime.agent import RuntimeAgent
        
        agent = RuntimeAgent(runtime_config)
        await agent.initialize()
        
        callback = Mock()
        agent.on_update(callback)
        
        # 先注入
        inject_msg = SyncMessage(
            message_type=MessageType.INJECT,
            lu_id="LU_001",
            namespace="default",
            key_vector=[0.1] * 32,
            value_vector=[0.2] * 256,
        )
        await agent._handle_inject(inject_msg)
        
        # 再更新
        update_msg = SyncMessage(
            message_type=MessageType.UPDATE,
            lu_id="LU_001",
            namespace="default",
            lifecycle_state="confirmed",
        )
        await agent._handle_update(update_msg)
        
        callback.assert_called_once()
