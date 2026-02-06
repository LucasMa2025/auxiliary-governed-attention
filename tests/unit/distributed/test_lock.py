"""
分布式锁单元测试

测试 DistributedLock 和 LockManager 的核心功能。
使用 MockRedis 模拟 Redis 后端。
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from aga.distributed.lock import DistributedLock, LockManager


@pytest.mark.unit
class TestDistributedLockBasic:
    """DistributedLock 基本功能测试"""
    
    def test_create_lock(self):
        """测试创建锁"""
        mock_client = Mock()
        
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
            timeout=30,
            retry_interval=0.1,
        )
        
        assert lock.name == "test-lock"
        assert lock.timeout == 30
        assert lock.retry_interval == 0.1
        assert lock._acquired is False
    
    def test_lock_key(self):
        """测试锁的 Redis 键"""
        mock_client = Mock()
        
        lock = DistributedLock(
            name="my-resource",
            client=mock_client,
        )
        
        assert lock.key == "aga:lock:my-resource"
    
    def test_lock_token_unique(self):
        """测试锁令牌唯一性"""
        mock_client = Mock()
        
        lock1 = DistributedLock(name="lock1", client=mock_client)
        lock2 = DistributedLock(name="lock2", client=mock_client)
        
        assert lock1._token != lock2._token


@pytest.mark.unit
class TestDistributedLockAsync:
    """DistributedLock 异步功能测试"""
    
    @pytest.fixture
    def mock_client(self):
        """创建 Mock Redis 客户端"""
        client = AsyncMock()
        client.set = AsyncMock(return_value=True)
        client.eval = AsyncMock(return_value=1)
        return client
    
    @pytest.mark.asyncio
    async def test_acquire_success(self, mock_client):
        """测试成功获取锁"""
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
            auto_extend=False,
        )
        
        result = await lock.acquire(blocking=False)
        
        assert result is True
        assert lock._acquired is True
        mock_client.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_acquire_fail_non_blocking(self, mock_client):
        """测试非阻塞获取失败"""
        mock_client.set.return_value = False
        
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
            auto_extend=False,
        )
        
        result = await lock.acquire(blocking=False)
        
        assert result is False
        assert lock._acquired is False
    
    @pytest.mark.asyncio
    async def test_acquire_with_timeout(self, mock_client):
        """测试带超时的获取"""
        mock_client.set.return_value = False
        
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
            retry_interval=0.01,
            auto_extend=False,
        )
        
        result = await lock.acquire(blocking=True, timeout=0.05)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_release_success(self, mock_client):
        """测试成功释放锁"""
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
            auto_extend=False,
        )
        lock._acquired = True
        
        result = await lock.release()
        
        assert result is True
        assert lock._acquired is False
        mock_client.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_release_not_acquired(self, mock_client):
        """测试释放未获取的锁"""
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
        )
        
        result = await lock.release()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_extend_success(self, mock_client):
        """测试成功延长锁"""
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
        )
        lock._acquired = True
        
        result = await lock.extend(additional_time=60)
        
        assert result is True
        mock_client.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extend_not_acquired(self, mock_client):
        """测试延长未获取的锁"""
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
        )
        
        result = await lock.extend()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_client):
        """测试上下文管理器"""
        lock = DistributedLock(
            name="test-lock",
            client=mock_client,
            auto_extend=False,
        )
        
        async with lock:
            assert lock._acquired is True
        
        assert lock._acquired is False


@pytest.mark.unit
class TestLockManager:
    """LockManager 测试"""
    
    @pytest.fixture
    def mock_client(self):
        """创建 Mock Redis 客户端"""
        client = AsyncMock()
        client.set = AsyncMock(return_value=True)
        client.eval = AsyncMock(return_value=1)
        return client
    
    def test_create_manager(self, mock_client):
        """测试创建管理器"""
        manager = LockManager(
            client=mock_client,
            default_timeout=60,
            default_retry_interval=0.2,
        )
        
        assert manager.default_timeout == 60
        assert manager.default_retry_interval == 0.2
    
    def test_get_lock(self, mock_client):
        """测试获取锁"""
        manager = LockManager(client=mock_client)
        
        lock = manager.get_lock("resource-1")
        
        assert lock.name == "resource-1"
        assert lock in manager._locks.values()
    
    def test_get_lock_cached(self, mock_client):
        """测试获取缓存的锁"""
        manager = LockManager(client=mock_client)
        
        lock1 = manager.get_lock("resource-1")
        lock2 = manager.get_lock("resource-1")
        
        assert lock1 is lock2
    
    def test_get_lock_custom_config(self, mock_client):
        """测试自定义配置的锁"""
        manager = LockManager(client=mock_client)
        
        lock = manager.get_lock(
            "resource-1",
            timeout=120,
            retry_interval=0.5,
            auto_extend=False,
        )
        
        assert lock.timeout == 120
        assert lock.retry_interval == 0.5
        assert lock.auto_extend is False
    
    @pytest.mark.asyncio
    async def test_acquire_lock(self, mock_client):
        """测试通过管理器获取锁"""
        manager = LockManager(client=mock_client)
        
        result = await manager.acquire_lock("resource-1", blocking=False)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_release_lock(self, mock_client):
        """测试通过管理器释放锁"""
        manager = LockManager(client=mock_client)
        
        # 先获取
        await manager.acquire_lock("resource-1", blocking=False)
        
        # 再释放
        result = await manager.release_lock("resource-1")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_release_nonexistent_lock(self, mock_client):
        """测试释放不存在的锁"""
        manager = LockManager(client=mock_client)
        
        result = await manager.release_lock("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_release_all(self, mock_client):
        """测试释放所有锁"""
        manager = LockManager(client=mock_client)
        
        # 获取多个锁
        await manager.acquire_lock("resource-1", blocking=False)
        await manager.acquire_lock("resource-2", blocking=False)
        
        # 释放所有
        await manager.release_all()
        
        # 验证都被释放
        for lock in manager._locks.values():
            assert lock._acquired is False
