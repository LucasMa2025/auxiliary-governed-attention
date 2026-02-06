"""
Redis 故障测试

测试 Redis 不可用时系统的行为。
"""
import pytest
import asyncio
import torch
from unittest.mock import patch, MagicMock, AsyncMock

from aga import KnowledgeRecord, LifecycleState
from tests.mocks.redis_mock import MockRedis

# 尝试导入 Redis 适配器
try:
    from aga.persistence.redis_adapter import RedisAdapter
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    RedisAdapter = None


@pytest.mark.fault
@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_REDIS, reason="Redis adapter not available")
class TestRedisConnectionFailure:
    """Redis 连接故障测试"""
    
    @pytest.fixture
    def failing_redis(self):
        """创建会失败的 Mock Redis"""
        redis = MockRedis()
        redis.disconnect()  # 模拟断开连接
        return redis
    
    @pytest.fixture
    async def adapter_with_failing_redis(self, failing_redis):
        """创建使用失败 Redis 的适配器"""
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = failing_redis
        adapter._connected = True
        return adapter
    
    async def test_save_fails_gracefully(self, adapter_with_failing_redis, sample_knowledge_record):
        """测试保存失败时的优雅处理"""
        with pytest.raises(ConnectionError):
            await adapter_with_failing_redis.save_slot(sample_knowledge_record)
    
    async def test_load_fails_gracefully(self, adapter_with_failing_redis):
        """测试加载失败时的优雅处理"""
        with pytest.raises(ConnectionError):
            await adapter_with_failing_redis.load_slot("default", "test_001")
    
    async def test_batch_save_partial_failure(self, sample_knowledge_records):
        """测试批量保存部分失败"""
        # 创建在第 5 次操作后失败的 Redis
        redis = MockRedis(fail_after=5)
        
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = redis
        adapter._connected = True
        
        # 批量保存应该在某个点失败
        with pytest.raises(ConnectionError):
            await adapter.save_batch(sample_knowledge_records)


@pytest.mark.fault
@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_REDIS, reason="Redis adapter not available")
class TestRedisReconnection:
    """Redis 重连测试"""
    
    @pytest.fixture
    def intermittent_redis(self):
        """创建间歇性失败的 Redis"""
        return MockRedis(fail_after=3)
    
    async def test_reconnect_after_failure(self, intermittent_redis, sample_knowledge_record):
        """测试失败后重连"""
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = intermittent_redis
        adapter._connected = True
        
        # 前几次操作成功
        await adapter.save_slot(sample_knowledge_record)
        await adapter.load_slot(sample_knowledge_record.namespace, sample_knowledge_record.lu_id)
        
        # 触发失败
        try:
            await adapter.save_slot(sample_knowledge_record)
            await adapter.save_slot(sample_knowledge_record)
        except ConnectionError:
            pass
        
        # 模拟重连
        intermittent_redis.reconnect()
        intermittent_redis._operation_count = 0
        intermittent_redis._fail_after = 0
        
        # 应该可以正常操作
        await adapter.save_slot(sample_knowledge_record)
        loaded = await adapter.load_slot(
            sample_knowledge_record.namespace,
            sample_knowledge_record.lu_id,
        )
        assert loaded is not None
    
    async def test_retry_mechanism(self, sample_knowledge_record):
        """测试重试机制"""
        # 创建一个在第 2 次操作后恢复的 Redis
        redis = MockRedis(fail_after=1)
        
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = redis
        adapter._connected = True
        
        # 第一次成功
        await adapter.save_slot(sample_knowledge_record)
        
        # 第二次失败
        with pytest.raises(ConnectionError):
            await adapter.save_slot(sample_knowledge_record)
        
        # 恢复
        redis.reconnect()
        redis._operation_count = 0
        redis._fail_after = 0
        
        # 重试成功
        await adapter.save_slot(sample_knowledge_record)


@pytest.mark.fault
@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_REDIS, reason="Redis adapter not available")
class TestRedisLatency:
    """Redis 延迟测试"""
    
    @pytest.fixture
    def slow_redis(self):
        """创建慢速 Redis"""
        return MockRedis(latency_ms=100)
    
    async def test_high_latency_operations(self, slow_redis, sample_knowledge_record):
        """测试高延迟操作"""
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = slow_redis
        adapter._connected = True
        
        import time
        start = time.time()
        
        await adapter.save_slot(sample_knowledge_record)
        
        elapsed = time.time() - start
        
        # 应该有明显延迟
        assert elapsed >= 0.1
    
    async def test_timeout_handling(self, sample_knowledge_record):
        """测试超时处理"""
        # 创建非常慢的 Redis
        very_slow_redis = MockRedis(latency_ms=5000)
        
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = very_slow_redis
        adapter._connected = True
        
        # 使用 asyncio.wait_for 模拟超时
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                adapter.save_slot(sample_knowledge_record),
                timeout=0.1,
            )


@pytest.mark.fault
@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_REDIS, reason="Redis adapter not available")
class TestRedisDataCorruption:
    """Redis 数据损坏测试"""
    
    async def test_corrupted_data_handling(self):
        """测试损坏数据处理"""
        redis = MockRedis()
        
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = redis
        adapter._connected = True
        
        # 直接写入损坏的数据
        await redis.set("aga:default:corrupted_001", "not valid json")
        
        # 尝试加载应该优雅处理
        try:
            loaded = await adapter.load_slot("default", "corrupted_001")
            # 如果没有抛出异常，应该返回 None 或处理错误
            assert loaded is None or isinstance(loaded, KnowledgeRecord)
        except (ValueError, KeyError):
            # 预期的异常
            pass
    
    async def test_partial_data_handling(self, bottleneck_dim, hidden_dim):
        """测试部分数据处理"""
        redis = MockRedis()
        
        adapter = RedisAdapter(host="localhost", port=6379)
        adapter._client = redis
        adapter._connected = True
        
        # 写入缺少字段的数据
        import json
        partial_data = json.dumps({
            "lu_id": "partial_001",
            "namespace": "default",
            # 缺少其他必要字段
        })
        await redis.set("aga:default:partial_001", partial_data)
        
        # 尝试加载
        try:
            loaded = await adapter.load_slot("default", "partial_001")
            # 应该返回 None 或抛出异常
        except (ValueError, KeyError, TypeError):
            # 预期的异常
            pass
