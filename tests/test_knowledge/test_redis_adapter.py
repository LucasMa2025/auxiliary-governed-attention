"""
aga-knowledge Redis 持久化适配器测试

使用 mock 测试，不需要真实的 Redis 实例。
所有 redis.asyncio 调用均被 mock。
"""

import json
import sys
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


# ==================== 辅助：异步上下文管理器 mock ====================

class AsyncContextManagerMock:
    """可用于 async with 的 mock"""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        return False


def make_pipe_mock():
    """创建 pipeline mock"""
    pipe = MagicMock()
    pipe.set = MagicMock()
    pipe.delete = MagicMock()
    pipe.sadd = MagicMock()
    pipe.srem = MagicMock()
    pipe.lpush = MagicMock()
    pipe.ltrim = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    return pipe


def make_client_mock(pipe=None):
    """创建 Redis 客户端 mock"""
    if pipe is None:
        pipe = make_pipe_mock()

    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=0)
    client.sadd = AsyncMock(return_value=1)
    client.srem = AsyncMock(return_value=1)
    client.smembers = AsyncMock(return_value=set())
    client.scard = AsyncMock(return_value=0)
    client.mget = AsyncMock(return_value=[])
    client.info = AsyncMock(return_value={"redis_version": "7.0.0"})
    client.eval = AsyncMock(return_value=0)
    client.publish = AsyncMock(return_value=1)
    client.lrange = AsyncMock(return_value=[])
    client.close = AsyncMock()

    # pipeline() 是同步调用，返回 async context manager
    # 必须用 MagicMock 覆盖，否则 AsyncMock 会让 pipeline() 返回 coroutine
    client.pipeline = MagicMock(return_value=AsyncContextManagerMock(pipe))

    # pubsub() 也是同步调用，返回一个对象
    mock_pubsub = MagicMock()
    mock_pubsub.subscribe = AsyncMock()
    client.pubsub = MagicMock(return_value=mock_pubsub)

    return client, pipe


# ==================== Fixtures ====================

@pytest.fixture
def mock_redis_module():
    """Patch redis 模块"""
    mock_redis = MagicMock()
    mock_aioredis = MagicMock()
    mock_aioredis.Redis = MagicMock
    mock_aioredis.from_url = MagicMock()
    mock_redis.asyncio = mock_aioredis
    with patch.dict("sys.modules", {
        "redis": mock_redis,
        "redis.asyncio": mock_aioredis,
    }):
        if "aga_knowledge.persistence.redis_adapter" in sys.modules:
            del sys.modules["aga_knowledge.persistence.redis_adapter"]
        yield mock_redis, mock_aioredis


@pytest.fixture
def make_adapter(mock_redis_module):
    """创建 RedisAdapter 实例的工厂"""
    from aga_knowledge.persistence.redis_adapter import RedisAdapter
    return RedisAdapter


@pytest.fixture
def adapter(make_adapter):
    """创建默认配置的 RedisAdapter"""
    return make_adapter()


@pytest.fixture
def pipe_mock():
    return make_pipe_mock()


@pytest.fixture
def client_mock(pipe_mock):
    client, pipe = make_client_mock(pipe_mock)
    return client, pipe


@pytest.fixture
def connected_adapter(adapter, client_mock):
    """创建已连接的 RedisAdapter"""
    client, pipe = client_mock
    adapter._client = client
    adapter._connected = True
    return adapter, client, pipe


# ==================== 配置测试 ====================

class TestRedisAdapterConfig:
    """测试 Redis 适配器配置"""

    def test_default_config(self, make_adapter):
        a = make_adapter()
        assert a.host == "localhost"
        assert a.port == 6379
        assert a.db == 0
        assert a.password is None
        assert a.key_prefix == "aga_knowledge"
        assert a.ttl_seconds == 30 * 86400
        assert a.pool_size == 10
        assert a.enable_audit is True
        assert a.audit_max_entries == 1000
        assert a.url is None

    def test_custom_config(self, make_adapter):
        a = make_adapter(
            host="redis.example.com",
            port=6380,
            db=2,
            password="secret",
            key_prefix="my_app",
            ttl_days=7,
            pool_size=20,
            enable_audit=False,
            audit_max_entries=500,
        )
        assert a.host == "redis.example.com"
        assert a.port == 6380
        assert a.db == 2
        assert a.password == "secret"
        assert a.key_prefix == "my_app"
        assert a.ttl_seconds == 7 * 86400
        assert a.pool_size == 20
        assert a.enable_audit is False
        assert a.audit_max_entries == 500

    def test_no_ttl(self, make_adapter):
        a = make_adapter(ttl_days=0)
        assert a.ttl_seconds == 0

    def test_url_config(self, make_adapter):
        a = make_adapter(url="redis://user:pass@host:6379/0")
        assert a.url == "redis://user:pass@host:6379/0"

    def test_not_connected_initially(self, adapter):
        assert adapter._connected is False
        assert adapter._client is None


# ==================== 键名生成测试 ====================

class TestRedisKeyGeneration:
    """测试键名生成"""

    def test_make_key(self, adapter):
        key = adapter._make_key("default", "lu_001")
        assert key == "aga_knowledge:default:knowledge:lu_001"

    def test_make_index_key(self, adapter):
        key = adapter._make_index_key("default")
        assert key == "aga_knowledge:default:index"

    def test_make_audit_key(self, adapter):
        key = adapter._make_audit_key("default")
        assert key == "aga_knowledge:default:audit"

    def test_custom_prefix(self, make_adapter):
        a = make_adapter(key_prefix="my_app")
        assert a._make_key("ns", "id") == "my_app:ns:knowledge:id"
        assert a._make_index_key("ns") == "my_app:ns:index"
        assert a._make_audit_key("ns") == "my_app:ns:audit"


# ==================== 连接管理测试 ====================

class TestRedisConnection:
    """测试连接管理"""

    @pytest.mark.asyncio
    async def test_connect_with_params(self, adapter, mock_redis_module):
        mock_redis, mock_aioredis = mock_redis_module
        mock_client_instance = AsyncMock()
        mock_client_instance.ping = AsyncMock(return_value=True)
        mock_aioredis.Redis = MagicMock(return_value=mock_client_instance)

        result = await adapter.connect()

        assert result is True
        assert adapter._connected is True

    @pytest.mark.asyncio
    async def test_connect_with_url(self, make_adapter, mock_redis_module):
        mock_redis, mock_aioredis = mock_redis_module
        mock_client_instance = AsyncMock()
        mock_client_instance.ping = AsyncMock(return_value=True)
        mock_aioredis.from_url = MagicMock(return_value=mock_client_instance)

        a = make_adapter(url="redis://host:6379/0")
        result = await a.connect()

        assert result is True
        mock_aioredis.from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, adapter, mock_redis_module):
        mock_redis, mock_aioredis = mock_redis_module
        mock_client_instance = AsyncMock()
        mock_client_instance.ping = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mock_aioredis.Redis = MagicMock(return_value=mock_client_instance)

        result = await adapter.connect()

        assert result is False
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        await adapter.disconnect()

        assert adapter._connected is False
        assert adapter._client is None
        client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_without_client(self, adapter):
        await adapter.disconnect()
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_is_connected_true(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        result = await adapter.is_connected()
        assert result is True
        client.ping.assert_called()

    @pytest.mark.asyncio
    async def test_is_connected_no_client(self, adapter):
        result = await adapter.is_connected()
        assert result is False

    @pytest.mark.asyncio
    async def test_is_connected_ping_fails(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.ping = AsyncMock(side_effect=Exception("Timeout"))

        result = await adapter.is_connected()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        result = await adapter.health_check()

        assert result["status"] == "healthy"
        assert result["adapter"] == "redis"
        assert result["redis_version"] == "7.0.0"

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, adapter):
        result = await adapter.health_check()
        assert result["status"] == "disconnected"


# ==================== CRUD 测试 ====================

class TestRedisCRUD:
    """测试知识 CRUD 操作"""

    @pytest.mark.asyncio
    async def test_save_knowledge_new(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.get = AsyncMock(return_value=None)

        result = await adapter.save_knowledge(
            "default", "lu_001",
            {
                "condition": "when X happens",
                "decision": "do Y",
                "lifecycle_state": "active",
            },
        )

        assert result is True
        pipe.set.assert_called()
        pipe.sadd.assert_called()

    @pytest.mark.asyncio
    async def test_save_knowledge_update_existing(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        existing = json.dumps({
            "lu_id": "lu_001",
            "version": 2,
            "hit_count": 5,
            "created_at": "2025-01-01T00:00:00",
        })
        client.get = AsyncMock(return_value=existing)

        result = await adapter.save_knowledge(
            "default", "lu_001",
            {"condition": "updated", "decision": "updated"},
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_save_knowledge_no_client(self, adapter):
        result = await adapter.save_knowledge(
            "default", "lu_001", {"condition": "c", "decision": "d"}
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_load_knowledge(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        record = {
            "lu_id": "lu_001",
            "namespace": "default",
            "condition": "when X",
            "decision": "do Y",
            "lifecycle_state": "active",
        }
        client.get = AsyncMock(return_value=json.dumps(record))

        result = await adapter.load_knowledge("default", "lu_001")

        assert result is not None
        assert result["lu_id"] == "lu_001"
        assert result["condition"] == "when X"

    @pytest.mark.asyncio
    async def test_load_knowledge_not_found(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.get = AsyncMock(return_value=None)

        result = await adapter.load_knowledge("default", "lu_999")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_knowledge_no_client(self, adapter):
        result = await adapter.load_knowledge("default", "lu_001")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_knowledge(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        result = await adapter.delete_knowledge("default", "lu_001")

        assert result is True
        pipe.delete.assert_called()
        pipe.srem.assert_called()

    @pytest.mark.asyncio
    async def test_delete_knowledge_no_client(self, adapter):
        result = await adapter.delete_knowledge("default", "lu_001")
        assert result is False

    @pytest.mark.asyncio
    async def test_knowledge_exists_true(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.exists = AsyncMock(return_value=1)

        result = await adapter.knowledge_exists("default", "lu_001")
        assert result is True

    @pytest.mark.asyncio
    async def test_knowledge_exists_false(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.exists = AsyncMock(return_value=0)

        result = await adapter.knowledge_exists("default", "lu_999")
        assert result is False

    @pytest.mark.asyncio
    async def test_knowledge_exists_no_client(self, adapter):
        result = await adapter.knowledge_exists("default", "lu_001")
        assert result is False


# ==================== 批量操作测试 ====================

class TestRedisBatch:
    """测试批量操作"""

    @pytest.mark.asyncio
    async def test_save_batch(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        records = [
            {"lu_id": f"lu_{i}", "condition": f"c{i}", "decision": f"d{i}"}
            for i in range(5)
        ]

        result = await adapter.save_batch("default", records)
        assert result == 5

    @pytest.mark.asyncio
    async def test_save_batch_empty(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        result = await adapter.save_batch("default", [])
        assert result == 0

    @pytest.mark.asyncio
    async def test_save_batch_no_client(self, adapter):
        result = await adapter.save_batch("default", [{"lu_id": "lu_1"}])
        assert result == 0

    @pytest.mark.asyncio
    async def test_save_batch_skips_empty_lu_id(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        records = [
            {"lu_id": "lu_1", "condition": "c1", "decision": "d1"},
            {"lu_id": "", "condition": "c2", "decision": "d2"},
            {"condition": "c3", "decision": "d3"},
        ]

        result = await adapter.save_batch("default", records)
        assert result == 1

    @pytest.mark.asyncio
    async def test_load_active_knowledge(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value={"lu_001", "lu_002"})
        records = [
            json.dumps({
                "lu_id": "lu_001",
                "lifecycle_state": "active",
            }),
            json.dumps({
                "lu_id": "lu_002",
                "lifecycle_state": "quarantined",
            }),
        ]
        client.mget = AsyncMock(return_value=records)

        result = await adapter.load_active_knowledge("default")

        assert len(result) == 1
        assert result[0]["lu_id"] == "lu_001"

    @pytest.mark.asyncio
    async def test_load_active_knowledge_empty(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value=set())

        result = await adapter.load_active_knowledge("default")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_active_knowledge_no_client(self, adapter):
        result = await adapter.load_active_knowledge("default")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_all_knowledge(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value={"lu_001", "lu_002"})
        records = [
            json.dumps({"lu_id": "lu_001", "lifecycle_state": "active"}),
            json.dumps({"lu_id": "lu_002", "lifecycle_state": "quarantined"}),
        ]
        client.mget = AsyncMock(return_value=records)

        result = await adapter.load_all_knowledge("default")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_load_all_knowledge_with_none_values(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value={"lu_001", "lu_002"})
        records = [
            json.dumps({"lu_id": "lu_001"}),
            None,
        ]
        client.mget = AsyncMock(return_value=records)

        result = await adapter.load_all_knowledge("default")
        assert len(result) == 1


# ==================== 查询测试 ====================

class TestRedisQuery:
    """测试查询功能"""

    @pytest.mark.asyncio
    async def test_query_knowledge_basic(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value=set())

        result = await adapter.query_knowledge("default")
        assert result == []

    @pytest.mark.asyncio
    async def test_query_knowledge_with_state_filter(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value={"lu_001", "lu_002", "lu_003"})
        records = [
            json.dumps({
                "lu_id": "lu_001",
                "lifecycle_state": "active",
                "trust_tier": "standard",
            }),
            json.dumps({
                "lu_id": "lu_002",
                "lifecycle_state": "probationary",
                "trust_tier": "standard",
            }),
            json.dumps({
                "lu_id": "lu_003",
                "lifecycle_state": "active",
                "trust_tier": "critical",
            }),
        ]
        client.mget = AsyncMock(return_value=records)

        result = await adapter.query_knowledge(
            "default",
            lifecycle_states=["active"],
        )
        assert len(result) == 2
        assert all(r["lifecycle_state"] == "active" for r in result)

    @pytest.mark.asyncio
    async def test_query_knowledge_with_tier_filter(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value={"lu_001", "lu_002"})
        records = [
            json.dumps({
                "lu_id": "lu_001",
                "lifecycle_state": "active",
                "trust_tier": "standard",
            }),
            json.dumps({
                "lu_id": "lu_002",
                "lifecycle_state": "active",
                "trust_tier": "critical",
            }),
        ]
        client.mget = AsyncMock(return_value=records)

        result = await adapter.query_knowledge(
            "default",
            trust_tiers=["critical"],
        )
        assert len(result) == 1
        assert result[0]["trust_tier"] == "critical"

    @pytest.mark.asyncio
    async def test_query_knowledge_with_pagination(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(
            return_value={f"lu_{i:03d}" for i in range(10)}
        )
        records = [
            json.dumps({
                "lu_id": f"lu_{i:03d}",
                "lifecycle_state": "active",
                "trust_tier": "standard",
            })
            for i in range(10)
        ]
        client.mget = AsyncMock(return_value=records)

        result = await adapter.query_knowledge(
            "default", limit=3, offset=2
        )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_query_knowledge_no_client(self, adapter):
        result = await adapter.query_knowledge("default")
        assert result == []


# ==================== 生命周期管理测试 ====================

class TestRedisLifecycle:
    """测试生命周期管理"""

    @pytest.mark.asyncio
    async def test_update_lifecycle(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        record = json.dumps({
            "lu_id": "lu_001",
            "lifecycle_state": "probationary",
            "updated_at": "2025-01-01T00:00:00",
        })
        client.get = AsyncMock(return_value=record)

        result = await adapter.update_lifecycle(
            "default", "lu_001", "active"
        )
        assert result is True
        client.set.assert_called()

    @pytest.mark.asyncio
    async def test_update_lifecycle_not_found(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.get = AsyncMock(return_value=None)

        result = await adapter.update_lifecycle(
            "default", "lu_999", "active"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_lifecycle_no_client(self, adapter):
        result = await adapter.update_lifecycle(
            "default", "lu_001", "active"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_trust_tier(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        record = json.dumps({
            "lu_id": "lu_001",
            "trust_tier": "standard",
            "updated_at": "2025-01-01T00:00:00",
        })
        client.get = AsyncMock(return_value=record)

        result = await adapter.update_trust_tier(
            "default", "lu_001", "critical"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_trust_tier_not_found(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.get = AsyncMock(return_value=None)

        result = await adapter.update_trust_tier(
            "default", "lu_999", "critical"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_trust_tier_no_client(self, adapter):
        result = await adapter.update_trust_tier(
            "default", "lu_001", "critical"
        )
        assert result is False


# ==================== 统计测试 ====================

class TestRedisStatistics:
    """测试统计功能"""

    @pytest.mark.asyncio
    async def test_get_knowledge_count_total(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.scard = AsyncMock(return_value=42)

        result = await adapter.get_knowledge_count("default")
        assert result == 42

    @pytest.mark.asyncio
    async def test_get_knowledge_count_with_state(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value={"lu_001", "lu_002", "lu_003"})
        records = [
            json.dumps({"lu_id": "lu_001", "lifecycle_state": "active"}),
            json.dumps({"lu_id": "lu_002", "lifecycle_state": "active"}),
            json.dumps({"lu_id": "lu_003", "lifecycle_state": "probationary"}),
        ]
        client.mget = AsyncMock(return_value=records)

        result = await adapter.get_knowledge_count("default", state="active")
        assert result == 2

    @pytest.mark.asyncio
    async def test_get_knowledge_count_no_client(self, adapter):
        result = await adapter.get_knowledge_count("default")
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_statistics(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.smembers = AsyncMock(return_value={"lu_001", "lu_002"})
        records = [
            json.dumps({
                "lu_id": "lu_001",
                "lifecycle_state": "active",
                "hit_count": 10,
            }),
            json.dumps({
                "lu_id": "lu_002",
                "lifecycle_state": "probationary",
                "hit_count": 5,
            }),
        ]
        client.mget = AsyncMock(return_value=records)

        result = await adapter.get_statistics("default")

        assert result["total_knowledge"] == 2
        assert result["total_hits"] == 15
        assert result["adapter"] == "redis"
        assert "state_distribution" in result
        assert result["state_distribution"]["active"] == 1
        assert result["state_distribution"]["probationary"] == 1

    @pytest.mark.asyncio
    async def test_get_statistics_no_client(self, adapter):
        result = await adapter.get_statistics("default")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_increment_hit_count(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        client.eval = AsyncMock(return_value=2)

        result = await adapter.increment_hit_count(
            "default", ["lu_001", "lu_002"]
        )
        assert result is True
        client.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_increment_hit_count_empty(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        result = await adapter.increment_hit_count("default", [])
        assert result is True

    @pytest.mark.asyncio
    async def test_increment_hit_count_no_client(self, adapter):
        result = await adapter.increment_hit_count("default", ["lu_001"])
        assert result is True


# ==================== 命名空间测试 ====================

class TestRedisNamespaces:
    """测试命名空间管理"""

    @pytest.mark.asyncio
    async def test_get_namespaces(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        async def mock_scan_iter(match=None):
            keys = [
                "aga_knowledge:default:index",
                "aga_knowledge:medical:index",
                "aga_knowledge:default:index",
            ]
            for k in keys:
                yield k

        client.scan_iter = mock_scan_iter

        result = await adapter.get_namespaces()
        assert len(result) == 2
        assert "default" in result
        assert "medical" in result

    @pytest.mark.asyncio
    async def test_get_namespaces_no_client(self, adapter):
        result = await adapter.get_namespaces()
        assert result == []


# ==================== 审计日志测试 ====================

class TestRedisAudit:
    """测试审计日志"""

    @pytest.mark.asyncio
    async def test_log_audit(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        await adapter._log_audit(
            "default", "lu_001", "SAVE",
            new_state="active",
        )

        pipe.lpush.assert_called()
        pipe.ltrim.assert_called()

    @pytest.mark.asyncio
    async def test_log_audit_disabled(self, make_adapter):
        a = make_adapter(enable_audit=False)
        client, pipe = make_client_mock()
        a._client = client
        a._connected = True

        await a._log_audit("default", "lu_001", "SAVE")

        pipe.lpush.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_audit_log(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        result = await adapter.save_audit_log({
            "namespace": "default",
            "lu_id": "lu_001",
            "action": "SAVE",
            "new_state": "active",
        })
        assert result is True

    @pytest.mark.asyncio
    async def test_save_audit_log_disabled(self, make_adapter):
        a = make_adapter(enable_audit=False)
        client, pipe = make_client_mock()
        a._client = client
        a._connected = True

        result = await a.save_audit_log({
            "namespace": "default",
            "action": "SAVE",
        })
        assert result is True

    @pytest.mark.asyncio
    async def test_query_audit_log_with_namespace(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        entries = [
            json.dumps({
                "namespace": "default",
                "lu_id": "lu_001",
                "action": "SAVE",
                "timestamp": "2025-01-01T00:00:00",
            }),
            json.dumps({
                "namespace": "default",
                "lu_id": "lu_002",
                "action": "DELETE",
                "timestamp": "2025-01-02T00:00:00",
            }),
        ]
        client.lrange = AsyncMock(return_value=entries)

        result = await adapter.query_audit_log(namespace="default")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_query_audit_log_with_lu_id_filter(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        entries = [
            json.dumps({
                "namespace": "default",
                "lu_id": "lu_001",
                "action": "SAVE",
                "timestamp": "2025-01-01T00:00:00",
            }),
            json.dumps({
                "namespace": "default",
                "lu_id": "lu_002",
                "action": "DELETE",
                "timestamp": "2025-01-02T00:00:00",
            }),
        ]
        client.lrange = AsyncMock(return_value=entries)

        result = await adapter.query_audit_log(
            namespace="default", lu_id="lu_001"
        )
        assert len(result) == 1
        assert result[0]["lu_id"] == "lu_001"

    @pytest.mark.asyncio
    async def test_query_audit_log_disabled(self, make_adapter):
        a = make_adapter(enable_audit=False)
        client, pipe = make_client_mock()
        a._client = client
        a._connected = True

        result = await a.query_audit_log()
        assert result == []


# ==================== Pub/Sub 测试 ====================

class TestRedisPubSub:
    """测试 Pub/Sub 功能"""

    @pytest.mark.asyncio
    async def test_publish_event(self, connected_adapter):
        adapter, client, pipe = connected_adapter

        result = await adapter.publish_event(
            "aga_knowledge:events",
            {"type": "knowledge_updated", "lu_id": "lu_001"},
        )
        assert result is True
        client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_event_no_client(self, adapter):
        result = await adapter.publish_event("channel", {"data": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_subscribe(self, connected_adapter):
        adapter, client, pipe = connected_adapter
        # pubsub mock 已在 make_client_mock 中设置
        result = await adapter.subscribe("aga_knowledge:events")
        assert result is not None
        client.pubsub.assert_called_once()
        client.pubsub.return_value.subscribe.assert_called_once_with(
            "aga_knowledge:events"
        )

    @pytest.mark.asyncio
    async def test_subscribe_no_client(self, adapter):
        result = await adapter.subscribe("channel")
        assert result is None


# ==================== 工厂函数测试 ====================

class TestRedisFactory:
    """测试通过工厂函数创建 Redis 适配器"""

    def test_create_redis_adapter(self, mock_redis_module):
        from aga_knowledge.persistence import create_adapter

        a = create_adapter({
            "type": "redis",
            "redis_host": "redis.example.com",
            "redis_port": 6380,
            "redis_db": 2,
            "redis_password": "secret",
            "redis_key_prefix": "my_app",
            "redis_ttl_days": 7,
            "redis_pool_size": 20,
        })

        from aga_knowledge.persistence.redis_adapter import RedisAdapter
        assert isinstance(a, RedisAdapter)
        assert a.host == "redis.example.com"
        assert a.port == 6380
        assert a.db == 2
        assert a.password == "secret"
        assert a.key_prefix == "my_app"
        assert a.ttl_seconds == 7 * 86400
        assert a.pool_size == 20

    def test_create_redis_adapter_with_url(self, mock_redis_module):
        from aga_knowledge.persistence import create_adapter

        a = create_adapter({
            "type": "redis",
            "redis_url": "redis://user:pass@host:6379/0",
        })

        assert a.url == "redis://user:pass@host:6379/0"

    def test_create_redis_adapter_defaults(self, mock_redis_module):
        from aga_knowledge.persistence import create_adapter

        a = create_adapter({"type": "redis"})
        assert a.host == "localhost"
        assert a.port == 6379
        assert a.db == 0
