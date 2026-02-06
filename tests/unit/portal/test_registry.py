"""
Portal RuntimeRegistry 单元测试

测试 Runtime 注册表的核心功能。
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from aga.portal.registry import RuntimeRegistry, RuntimeInfo


@pytest.mark.unit
class TestRuntimeInfo:
    """RuntimeInfo 测试"""
    
    def test_create_runtime_info(self):
        """测试创建 RuntimeInfo"""
        info = RuntimeInfo(
            instance_id="runtime-001",
            namespaces=["default", "test"],
        )
        
        assert info.instance_id == "runtime-001"
        assert info.namespaces == ["default", "test"]
        assert info.status == "active"
    
    def test_runtime_info_with_host(self):
        """测试带主机信息的 RuntimeInfo"""
        info = RuntimeInfo(
            instance_id="runtime-001",
            namespaces=["default"],
            host="192.168.1.100",
            port=8080,
        )
        
        assert info.host == "192.168.1.100"
        assert info.port == 8080
    
    def test_runtime_info_to_dict(self):
        """测试转换为字典"""
        info = RuntimeInfo(
            instance_id="runtime-001",
            namespaces=["default"],
            metadata={"version": "1.0"},
        )
        
        d = info.to_dict()
        
        assert d["instance_id"] == "runtime-001"
        assert d["namespaces"] == ["default"]
        assert d["metadata"]["version"] == "1.0"


@pytest.mark.unit
class TestRuntimeRegistryBasic:
    """RuntimeRegistry 基本功能测试"""
    
    def test_create_registry(self):
        """测试创建注册表"""
        registry = RuntimeRegistry(
            heartbeat_timeout=60,
            cleanup_interval=30,
        )
        
        assert registry.heartbeat_timeout == 60
        assert registry.cleanup_interval == 30
    
    def test_register_runtime(self):
        """测试注册 Runtime"""
        registry = RuntimeRegistry()
        
        registry.register(
            instance_id="runtime-001",
            namespaces=["default"],
        )
        
        assert "runtime-001" in registry._runtimes
    
    def test_register_runtime_with_metadata(self):
        """测试带元数据的注册"""
        registry = RuntimeRegistry()
        
        registry.register(
            instance_id="runtime-001",
            namespaces=["default"],
            host="localhost",
            port=8080,
            metadata={"version": "1.0"},
        )
        
        info = registry._runtimes["runtime-001"]
        assert info.host == "localhost"
        assert info.port == 8080
        assert info.metadata["version"] == "1.0"
    
    def test_deregister_runtime(self):
        """测试注销 Runtime"""
        registry = RuntimeRegistry()
        
        registry.register(
            instance_id="runtime-001",
            namespaces=["default"],
        )
        
        registry.deregister("runtime-001")
        
        assert "runtime-001" not in registry._runtimes
    
    def test_deregister_nonexistent(self):
        """测试注销不存在的 Runtime"""
        registry = RuntimeRegistry()
        
        # 不应该抛出异常
        registry.deregister("nonexistent")
    
    def test_get_runtime(self):
        """测试获取 Runtime"""
        registry = RuntimeRegistry()
        
        registry.register(
            instance_id="runtime-001",
            namespaces=["default"],
        )
        
        info = registry.get_runtime("runtime-001")
        
        assert info is not None
        assert info.instance_id == "runtime-001"
    
    def test_get_nonexistent_runtime(self):
        """测试获取不存在的 Runtime"""
        registry = RuntimeRegistry()
        
        info = registry.get_runtime("nonexistent")
        
        assert info is None


@pytest.mark.unit
class TestRuntimeRegistryHeartbeat:
    """RuntimeRegistry 心跳测试"""
    
    def test_heartbeat(self):
        """测试心跳"""
        registry = RuntimeRegistry()
        
        registry.register(
            instance_id="runtime-001",
            namespaces=["default"],
        )
        
        old_heartbeat = registry._runtimes["runtime-001"].last_heartbeat
        time.sleep(0.01)
        
        registry.heartbeat("runtime-001")
        
        new_heartbeat = registry._runtimes["runtime-001"].last_heartbeat
        assert new_heartbeat > old_heartbeat
    
    def test_heartbeat_nonexistent(self):
        """测试不存在的 Runtime 心跳"""
        registry = RuntimeRegistry()
        
        # 不应该抛出异常
        registry.heartbeat("nonexistent")


@pytest.mark.unit
class TestRuntimeRegistryQuery:
    """RuntimeRegistry 查询测试"""
    
    def test_get_all_runtimes(self):
        """测试列出所有 Runtime"""
        registry = RuntimeRegistry()
        
        registry.register("runtime-001", ["default"])
        registry.register("runtime-002", ["test"])
        
        all_runtimes = registry.get_all_runtimes()
        
        assert len(all_runtimes) == 2
    
    def test_get_runtimes_for_namespace(self):
        """测试按命名空间列出"""
        registry = RuntimeRegistry()
        
        registry.register("runtime-001", ["default", "ns1"])
        registry.register("runtime-002", ["default", "ns2"])
        registry.register("runtime-003", ["ns3"])
        
        runtimes = registry.get_runtimes_for_namespace("default")
        
        assert len(runtimes) == 2
        assert "runtime-003" not in [r.instance_id for r in runtimes]
    
    def test_get_active_runtimes(self):
        """测试列出活跃 Runtime"""
        registry = RuntimeRegistry()
        
        registry.register("runtime-001", ["default"])
        registry.register("runtime-002", ["default"])
        
        # 设置一个为不活跃
        registry._runtimes["runtime-002"].status = "inactive"
        
        active = registry.get_active_runtimes()
        
        assert len(active) == 1
        assert active[0].instance_id == "runtime-001"
    
    def test_get_stats(self):
        """测试获取统计"""
        registry = RuntimeRegistry()
        
        registry.register("runtime-001", ["default"])
        registry.register("runtime-002", ["default"])
        registry._runtimes["runtime-002"].status = "inactive"
        
        stats = registry.get_stats()
        
        assert stats["total_runtimes"] == 2
        assert stats["active_runtimes"] == 1
        assert stats["inactive_runtimes"] == 1


@pytest.mark.unit
class TestRuntimeRegistryAsync:
    """RuntimeRegistry 异步功能测试"""
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动和停止"""
        registry = RuntimeRegistry(cleanup_interval=1)
        
        await registry.start()
        assert registry._running is True
        
        await registry.stop()
        assert registry._running is False
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive(self):
        """测试清理不活跃 Runtime"""
        registry = RuntimeRegistry(heartbeat_timeout=0.01)
        
        registry.register("runtime-001", ["default"])
        
        # 等待超时
        await asyncio.sleep(0.02)
        
        await registry._cleanup_inactive()
        
        assert "runtime-001" not in registry._runtimes


@pytest.mark.unit
class TestRuntimeRegistryMessageTracking:
    """RuntimeRegistry 消息跟踪测试"""
    
    def test_record_message(self):
        """测试记录消息"""
        registry = RuntimeRegistry()
        
        registry.register("runtime-001", ["default"])
        
        registry.record_message("runtime-001")
        
        info = registry._runtimes["runtime-001"]
        assert info.messages_received == 1
        assert info.last_message_at is not None
    
    def test_record_multiple_messages(self):
        """测试记录多条消息"""
        registry = RuntimeRegistry()
        
        registry.register("runtime-001", ["default"])
        
        for _ in range(5):
            registry.record_message("runtime-001")
        
        info = registry._runtimes["runtime-001"]
        assert info.messages_received == 5
