"""
分布式协调器单元测试

测试 InstanceCoordinator 的核心功能。
使用 MockRedis 模拟 Redis 后端。
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from aga.distributed.coordinator import InstanceCoordinator, InstanceInfo


@pytest.mark.unit
class TestInstanceInfo:
    """InstanceInfo 测试"""
    
    def test_create_instance_info(self):
        """测试创建实例信息"""
        info = InstanceInfo(
            instance_id="instance-001",
            namespace="default",
            host="localhost",
            port=8000,
        )
        
        assert info.instance_id == "instance-001"
        assert info.namespace == "default"
        assert info.host == "localhost"
        assert info.port == 8000
        assert info.status == "unknown"
    
    def test_instance_info_healthy(self):
        """测试健康状态"""
        info = InstanceInfo(
            instance_id="instance-001",
            namespace="default",
            host="localhost",
            port=8000,
            status="healthy",
        )
        
        assert info.is_healthy is True
        
        info.status = "unhealthy"
        assert info.is_healthy is False
    
    def test_instance_info_heartbeat(self):
        """测试心跳时间"""
        info = InstanceInfo(
            instance_id="instance-001",
            namespace="default",
            host="localhost",
            port=8000,
        )
        
        time.sleep(0.01)
        assert info.seconds_since_heartbeat > 0
    
    def test_instance_info_to_dict(self):
        """测试转换为字典"""
        info = InstanceInfo(
            instance_id="instance-001",
            namespace="default",
            host="localhost",
            port=8000,
            capabilities=["inference", "sync"],
            metadata={"version": "1.0"},
        )
        
        d = info.to_dict()
        
        assert d["instance_id"] == "instance-001"
        assert d["host"] == "localhost"
        assert d["capabilities"] == ["inference", "sync"]
        assert d["metadata"]["version"] == "1.0"


@pytest.mark.unit
class TestInstanceCoordinatorBasic:
    """InstanceCoordinator 基本功能测试"""
    
    def test_create_coordinator(self):
        """测试创建协调器"""
        coordinator = InstanceCoordinator(
            instance_id="instance-001",
            namespace="default",
            backend="redis",
            config={"heartbeat_interval": 5},
        )
        
        assert coordinator.instance_id == "instance-001"
        assert coordinator.namespace == "default"
        assert coordinator.backend == "redis"
        assert coordinator.heartbeat_interval == 5
    
    def test_create_coordinator_default_config(self):
        """测试默认配置"""
        coordinator = InstanceCoordinator(
            instance_id="instance-001",
            namespace="default",
            config={},  # 空配置
        )
        
        assert coordinator.heartbeat_interval == 10
        assert coordinator.instance_timeout == 30


@pytest.mark.unit
class TestInstanceCoordinatorWithMock:
    """使用 Mock 的协调器测试"""
    
    @pytest.fixture
    def mock_redis(self):
        """创建 Mock Redis"""
        from tests.mocks import MockRedis
        return MockRedis()
    
    @pytest.fixture
    def coordinator(self, mock_redis):
        """创建协调器"""
        coord = InstanceCoordinator(
            instance_id="instance-001",
            namespace="default",
            backend="redis",
            config={
                "heartbeat_interval": 1,
                "instance_timeout": 3,
            },
        )
        coord._client = mock_redis
        return coord
    
    def test_get_instances_empty(self, coordinator):
        """测试获取空实例列表"""
        instances = coordinator.get_instances()
        assert instances == []
    
    def test_get_healthy_instances_empty(self, coordinator):
        """测试获取空健康实例列表"""
        instances = coordinator.get_healthy_instances()
        assert instances == []
    
    def test_get_self_info_before_start(self, coordinator):
        """测试启动前获取本实例信息"""
        info = coordinator.get_self_info()
        assert info is None
    
    def test_update_self_stats(self, coordinator):
        """测试更新本实例统计"""
        # 先设置 self_info
        coordinator._self_info = InstanceInfo(
            instance_id="instance-001",
            namespace="default",
            host="localhost",
            port=8000,
        )
        
        coordinator.update_self_stats(
            active_slots=10,
            total_requests=100,
            avg_latency_ms=5.5,
        )
        
        assert coordinator._self_info.active_slots == 10
        assert coordinator._self_info.total_requests == 100
        assert coordinator._self_info.avg_latency_ms == 5.5
    
    def test_callbacks_registration(self, coordinator):
        """测试回调注册"""
        join_callback = Mock()
        leave_callback = Mock()
        leader_callback = Mock()
        
        coordinator.on_instance_join(join_callback)
        coordinator.on_instance_leave(leave_callback)
        coordinator.on_leader_change(leader_callback)
        
        assert coordinator._on_instance_join is join_callback
        assert coordinator._on_instance_leave is leave_callback
        assert coordinator._on_leader_change is leader_callback


@pytest.mark.unit
class TestInstanceCoordinatorAsync:
    """协调器异步功能测试"""
    
    @pytest.fixture
    def mock_redis(self):
        """创建 Mock Redis"""
        from tests.mocks import MockRedis
        redis = MockRedis()
        # 添加异步方法
        redis.set = AsyncMock(return_value=True)
        redis.get = AsyncMock(return_value=None)
        redis.delete = AsyncMock(return_value=True)
        redis.keys = AsyncMock(return_value=[])
        redis.close = AsyncMock()
        return redis
    
    @pytest.mark.asyncio
    async def test_register_self(self, mock_redis):
        """测试注册本实例"""
        coordinator = InstanceCoordinator(
            instance_id="instance-001",
            namespace="default",
            backend="redis",
            config={},
        )
        coordinator._client = mock_redis
        coordinator._self_info = InstanceInfo(
            instance_id="instance-001",
            namespace="default",
            host="localhost",
            port=8000,
            status="healthy",
        )
        
        await coordinator._register_self()
        
        mock_redis.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deregister_self(self, mock_redis):
        """测试注销本实例"""
        coordinator = InstanceCoordinator(
            instance_id="instance-001",
            namespace="default",
            backend="redis",
            config={},
        )
        coordinator._client = mock_redis
        
        await coordinator._deregister_self()
        
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_instance_health(self, mock_redis):
        """测试检查实例健康"""
        coordinator = InstanceCoordinator(
            instance_id="instance-001",
            namespace="default",
            backend="redis",
            config={"instance_timeout": 0.01},
        )
        coordinator._client = mock_redis
        
        # 添加一个过期的实例
        old_info = InstanceInfo(
            instance_id="instance-002",
            namespace="default",
            host="localhost",
            port=8001,
            status="healthy",
        )
        old_info.last_heartbeat = time.time() - 1  # 1秒前
        coordinator._instances["instance-002"] = old_info
        
        # 注册离开回调
        leave_callback = AsyncMock()
        coordinator._on_instance_leave = leave_callback
        
        await coordinator._check_instance_health()
        
        # 实例应该被移除
        assert "instance-002" not in coordinator._instances
        leave_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop(self, mock_redis):
        """测试停止协调器"""
        coordinator = InstanceCoordinator(
            instance_id="instance-001",
            namespace="default",
            backend="redis",
            config={},
        )
        coordinator._client = mock_redis
        coordinator._running = True
        
        await coordinator.stop()
        
        assert coordinator._running is False
        mock_redis.delete.assert_called_once()
        mock_redis.close.assert_called_once()
