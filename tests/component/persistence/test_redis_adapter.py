"""
Redis 适配器组件测试（使用 Mock）

测试 Redis 持久化适配器的功能。
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from aga import LifecycleState
from aga.persistence.base import KnowledgeRecord
from tests.mocks.redis_mock import MockRedis

# 尝试导入 Redis 适配器
try:
    from aga.persistence.redis_adapter import RedisAdapter
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    RedisAdapter = None


@pytest.mark.component
@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_REDIS, reason="Redis adapter not available")
class TestRedisAdapterWithMock:
    """Redis 适配器测试（使用 Mock）"""
    
    @pytest.fixture
    def mock_redis(self):
        """创建 Mock Redis"""
        return MockRedis()
    
    @pytest.fixture
    async def adapter(self, mock_redis):
        """创建适配器"""
        adapter = RedisAdapter(host="localhost", port=6379)
        # 注入 Mock
        adapter._client = mock_redis
        adapter._connected = True
        yield adapter
    
    async def test_save_and_load_slot(self, adapter, sample_knowledge_record):
        """测试保存和加载槽位"""
        namespace = sample_knowledge_record.namespace
        await adapter.save_slot(namespace, sample_knowledge_record)
        
        loaded = await adapter.load_slot(
            namespace,
            sample_knowledge_record.lu_id,
        )
        
        assert loaded is not None
        assert loaded.lu_id == sample_knowledge_record.lu_id
    
    async def test_delete_slot(self, adapter, sample_knowledge_record):
        """测试删除槽位"""
        namespace = sample_knowledge_record.namespace
        await adapter.save_slot(namespace, sample_knowledge_record)
        
        await adapter.delete_slot(
            namespace,
            sample_knowledge_record.lu_id,
        )
        
        loaded = await adapter.load_slot(
            namespace,
            sample_knowledge_record.lu_id,
        )
        assert loaded is None
    
    async def test_save_batch(self, adapter, sample_knowledge_records):
        """测试批量保存"""
        namespace = sample_knowledge_records[0].namespace
        await adapter.save_batch(namespace, sample_knowledge_records)
        
        for record in sample_knowledge_records:
            loaded = await adapter.load_slot(record.namespace, record.lu_id)
            assert loaded is not None


@pytest.mark.component
@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_REDIS, reason="Redis adapter not available")
class TestRedisAdapterFailure:
    """Redis 适配器故障测试"""
    
    @pytest.fixture
    def failing_redis(self):
        """创建会失败的 Mock Redis"""
        return MockRedis(fail_after=2)
    
    @pytest.fixture
    async def adapter(self, failing_redis):
        """创建适配器"""
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = failing_redis
        adapter._connected = True
        yield adapter
    
    async def test_connection_failure(self, adapter, sample_knowledge_record):
        """测试连接失败"""
        namespace = sample_knowledge_record.namespace
        
        # 前两次操作成功
        await adapter.save_slot(namespace, sample_knowledge_record)
        await adapter.load_slot(namespace, sample_knowledge_record.lu_id)
        
        # 第三次操作应该失败
        with pytest.raises(ConnectionError):
            await adapter.save_slot(namespace, sample_knowledge_record)
    
    async def test_reconnect_after_failure(self, adapter, sample_knowledge_record, failing_redis):
        """测试失败后重连"""
        namespace = sample_knowledge_record.namespace
        
        # 触发失败
        try:
            for _ in range(5):
                await adapter.save_slot(namespace, sample_knowledge_record)
        except ConnectionError:
            pass
        
        # 重连
        failing_redis.reconnect()
        failing_redis._operation_count = 0
        failing_redis._fail_after = 0  # 不再失败
        
        # 应该可以正常操作
        await adapter.save_slot(namespace, sample_knowledge_record)
        loaded = await adapter.load_slot(
            namespace,
            sample_knowledge_record.lu_id,
        )
        assert loaded is not None
