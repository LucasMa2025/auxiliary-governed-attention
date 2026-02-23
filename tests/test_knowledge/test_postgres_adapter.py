"""
aga-knowledge PostgreSQL 持久化适配器测试

使用 mock 测试，不需要真实的 PostgreSQL 实例。
所有 asyncpg 调用均被 mock。
"""

import json
import sys
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from contextlib import asynccontextmanager


# ==================== 辅助：异步上下文管理器 mock ====================

class AsyncContextManagerMock:
    """可用于 async with 的 mock"""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        return False


def make_pool_mock(conn_mock):
    """创建一个正确支持 async with pool.acquire() 的 pool mock"""
    pool = MagicMock()
    pool.acquire.return_value = AsyncContextManagerMock(conn_mock)
    pool.close = AsyncMock()
    pool.get_size.return_value = 5
    return pool


def make_conn_mock():
    """创建 connection mock"""
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    # transaction() 是同步调用，返回 async context manager
    # 必须用 MagicMock 覆盖，否则 AsyncMock 会让 transaction() 返回 coroutine
    conn.transaction = MagicMock(return_value=AsyncContextManagerMock(None))
    return conn


# ==================== Fixtures ====================

@pytest.fixture
def mock_asyncpg():
    """Patch asyncpg 模块"""
    mock_module = MagicMock()
    mock_module.Pool = MagicMock
    mock_module.create_pool = AsyncMock()
    with patch.dict("sys.modules", {"asyncpg": mock_module}):
        if "aga_knowledge.persistence.postgres_adapter" in sys.modules:
            del sys.modules["aga_knowledge.persistence.postgres_adapter"]
        yield mock_module


@pytest.fixture
def make_adapter(mock_asyncpg):
    """创建 PostgresAdapter 实例的工厂"""
    from aga_knowledge.persistence.postgres_adapter import PostgresAdapter
    return PostgresAdapter


@pytest.fixture
def adapter(make_adapter):
    """创建默认配置的 PostgresAdapter"""
    return make_adapter()


@pytest.fixture
def conn_mock():
    return make_conn_mock()


@pytest.fixture
def pool_mock(conn_mock):
    return make_pool_mock(conn_mock)


@pytest.fixture
def connected_adapter(adapter, pool_mock, conn_mock):
    """创建已连接的 PostgresAdapter"""
    adapter._pool = pool_mock
    adapter._connected = True
    return adapter, pool_mock, conn_mock


# ==================== 配置测试 ====================

class TestPostgresAdapterConfig:
    """测试 PostgreSQL 适配器配置"""

    def test_default_config(self, make_adapter):
        a = make_adapter()
        assert a.host == "localhost"
        assert a.port == 5432
        assert a.database == "aga_knowledge"
        assert a.user == "aga"
        assert a.password is None
        assert a.pool_size == 5
        assert a.max_overflow == 10
        assert a.enable_audit is True
        assert a.dsn is None

    def test_custom_config(self, make_adapter):
        a = make_adapter(
            host="db.example.com",
            port=5433,
            database="my_knowledge",
            user="admin",
            password="secret",
            pool_size=20,
            max_overflow=30,
            enable_audit=False,
        )
        assert a.host == "db.example.com"
        assert a.port == 5433
        assert a.database == "my_knowledge"
        assert a.user == "admin"
        assert a.password == "secret"
        assert a.pool_size == 20
        assert a.max_overflow == 30
        assert a.enable_audit is False

    def test_dsn_config(self, make_adapter):
        dsn = "postgresql://user:pass@host:5432/db"
        a = make_adapter(dsn=dsn)
        assert a.dsn == dsn

    def test_not_connected_initially(self, adapter):
        assert adapter._connected is False
        assert adapter._pool is None


# ==================== 连接管理测试 ====================

class TestPostgresConnection:
    """测试连接管理"""

    @pytest.mark.asyncio
    async def test_connect_with_params(self, adapter, mock_asyncpg):
        conn = make_conn_mock()
        pool = make_pool_mock(conn)
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)

        result = await adapter.connect()

        assert result is True
        assert adapter._connected is True

    @pytest.mark.asyncio
    async def test_connect_with_dsn(self, make_adapter, mock_asyncpg):
        conn = make_conn_mock()
        pool = make_pool_mock(conn)
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)

        a = make_adapter(dsn="postgresql://user:pass@host:5432/db")
        result = await a.connect()

        assert result is True
        mock_asyncpg.create_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, adapter, mock_asyncpg):
        mock_asyncpg.create_pool = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        result = await adapter.connect()

        assert result is False
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        await adapter.disconnect()

        assert adapter._connected is False
        assert adapter._pool is None
        pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_without_pool(self, adapter):
        await adapter.disconnect()
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_is_connected_true(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        result = await adapter.is_connected()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_connected_no_pool(self, adapter):
        result = await adapter.is_connected()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetchrow = AsyncMock(return_value=["PostgreSQL 15.0"])

        result = await adapter.health_check()

        assert result["status"] == "healthy"
        assert result["adapter"] == "postgres"
        assert result["host"] == "localhost"

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, adapter):
        result = await adapter.health_check()
        assert result["status"] == "disconnected"
        assert result["adapter"] == "postgres"


# ==================== CRUD 测试 ====================

class TestPostgresCRUD:
    """测试知识 CRUD 操作"""

    @pytest.mark.asyncio
    async def test_save_knowledge(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        result = await adapter.save_knowledge(
            "default", "lu_001",
            {
                "condition": "when X happens",
                "decision": "do Y",
                "lifecycle_state": "active",
                "trust_tier": "standard",
            },
        )

        assert result is True
        assert conn.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_save_knowledge_no_pool(self, adapter):
        result = await adapter.save_knowledge(
            "default", "lu_001", {"condition": "c", "decision": "d"}
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_save_knowledge_with_metadata(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        result = await adapter.save_knowledge(
            "default", "lu_001",
            {
                "condition": "when X",
                "decision": "do Y",
                "metadata": {"source": "test", "priority": 1},
            },
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_save_knowledge_error(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.execute = AsyncMock(side_effect=Exception("DB error"))

        result = await adapter.save_knowledge(
            "default", "lu_001", {"condition": "c", "decision": "d"}
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_load_knowledge(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        mock_row = {
            "lu_id": "lu_001",
            "namespace": "default",
            "condition_text": "when X",
            "decision_text": "do Y",
            "lifecycle_state": "active",
            "trust_tier": "standard",
            "hit_count": 5,
            "consecutive_misses": 0,
            "version": 1,
            "created_at": datetime(2025, 1, 1),
            "updated_at": datetime(2025, 1, 2),
            "metadata": None,
        }
        conn.fetchrow = AsyncMock(return_value=mock_row)

        result = await adapter.load_knowledge("default", "lu_001")

        assert result is not None
        assert result["lu_id"] == "lu_001"
        assert result["condition"] == "when X"
        assert result["decision"] == "do Y"
        assert result["hit_count"] == 5

    @pytest.mark.asyncio
    async def test_load_knowledge_not_found(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetchrow = AsyncMock(return_value=None)

        result = await adapter.load_knowledge("default", "lu_999")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_knowledge_no_pool(self, adapter):
        result = await adapter.load_knowledge("default", "lu_001")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_knowledge(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        result = await adapter.delete_knowledge("default", "lu_001")
        assert result is True
        assert conn.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_knowledge_no_pool(self, adapter):
        result = await adapter.delete_knowledge("default", "lu_001")
        assert result is False

    @pytest.mark.asyncio
    async def test_knowledge_exists_true(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetchrow = AsyncMock(return_value=[1])

        result = await adapter.knowledge_exists("default", "lu_001")
        assert result is True

    @pytest.mark.asyncio
    async def test_knowledge_exists_false(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetchrow = AsyncMock(return_value=None)

        result = await adapter.knowledge_exists("default", "lu_999")
        assert result is False

    @pytest.mark.asyncio
    async def test_knowledge_exists_no_pool(self, adapter):
        result = await adapter.knowledge_exists("default", "lu_001")
        assert result is False


# ==================== 批量操作测试 ====================

class TestPostgresBatch:
    """测试批量操作"""

    @pytest.mark.asyncio
    async def test_save_batch(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        records = [
            {"lu_id": f"lu_{i}", "condition": f"c{i}", "decision": f"d{i}"}
            for i in range(5)
        ]

        result = await adapter.save_batch("default", records)
        assert result == 5

    @pytest.mark.asyncio
    async def test_save_batch_empty(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        result = await adapter.save_batch("default", [])
        assert result == 0

    @pytest.mark.asyncio
    async def test_save_batch_no_pool(self, adapter):
        result = await adapter.save_batch("default", [{"lu_id": "lu_1"}])
        assert result == 0

    @pytest.mark.asyncio
    async def test_save_batch_skips_empty_lu_id(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        records = [
            {"lu_id": "lu_1", "condition": "c1", "decision": "d1"},
            {"lu_id": "", "condition": "c2", "decision": "d2"},
            {"condition": "c3", "decision": "d3"},  # 无 lu_id
        ]

        result = await adapter.save_batch("default", records)
        assert result == 1

    @pytest.mark.asyncio
    async def test_load_active_knowledge(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        mock_rows = [
            {
                "lu_id": "lu_001",
                "namespace": "default",
                "condition_text": "c1",
                "decision_text": "d1",
                "lifecycle_state": "active",
                "trust_tier": "standard",
                "hit_count": 0,
                "consecutive_misses": 0,
                "version": 1,
                "created_at": datetime(2025, 1, 1),
                "updated_at": datetime(2025, 1, 1),
                "metadata": None,
            },
        ]
        conn.fetch = AsyncMock(return_value=mock_rows)

        result = await adapter.load_active_knowledge("default")
        assert len(result) == 1
        assert result[0]["lu_id"] == "lu_001"

    @pytest.mark.asyncio
    async def test_load_active_knowledge_no_pool(self, adapter):
        result = await adapter.load_active_knowledge("default")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_all_knowledge(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        mock_rows = [
            {
                "lu_id": f"lu_{i}",
                "namespace": "default",
                "condition_text": f"c{i}",
                "decision_text": f"d{i}",
                "lifecycle_state": "active",
                "trust_tier": "standard",
                "hit_count": i,
                "consecutive_misses": 0,
                "version": 1,
                "created_at": datetime(2025, 1, 1),
                "updated_at": datetime(2025, 1, 1),
                "metadata": None,
            }
            for i in range(3)
        ]
        conn.fetch = AsyncMock(return_value=mock_rows)

        result = await adapter.load_all_knowledge("default")
        assert len(result) == 3


# ==================== 查询测试 ====================

class TestPostgresQuery:
    """测试查询功能"""

    @pytest.mark.asyncio
    async def test_query_knowledge_basic(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetch = AsyncMock(return_value=[])

        result = await adapter.query_knowledge("default")
        assert result == []

    @pytest.mark.asyncio
    async def test_query_knowledge_with_filters(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetch = AsyncMock(return_value=[])

        result = await adapter.query_knowledge(
            "default",
            lifecycle_states=["active", "probationary"],
            trust_tiers=["standard"],
            limit=50,
            offset=10,
        )
        assert result == []
        conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_knowledge_no_pool(self, adapter):
        result = await adapter.query_knowledge("default")
        assert result == []


# ==================== 生命周期管理测试 ====================

class TestPostgresLifecycle:
    """测试生命周期管理"""

    @pytest.mark.asyncio
    async def test_update_lifecycle(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetchrow = AsyncMock(
            return_value={"lifecycle_state": "probationary"}
        )

        result = await adapter.update_lifecycle(
            "default", "lu_001", "active"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_lifecycle_no_pool(self, adapter):
        result = await adapter.update_lifecycle(
            "default", "lu_001", "active"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_trust_tier(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        result = await adapter.update_trust_tier(
            "default", "lu_001", "critical"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_trust_tier_no_pool(self, adapter):
        result = await adapter.update_trust_tier(
            "default", "lu_001", "critical"
        )
        assert result is False


# ==================== 统计测试 ====================

class TestPostgresStatistics:
    """测试统计功能"""

    @pytest.mark.asyncio
    async def test_get_knowledge_count(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetchrow = AsyncMock(return_value={"count": 42})

        result = await adapter.get_knowledge_count("default")
        assert result == 42

    @pytest.mark.asyncio
    async def test_get_knowledge_count_with_state(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetchrow = AsyncMock(return_value={"count": 10})

        result = await adapter.get_knowledge_count("default", state="active")
        assert result == 10

    @pytest.mark.asyncio
    async def test_get_knowledge_count_no_pool(self, adapter):
        result = await adapter.get_knowledge_count("default")
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_statistics(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetchrow = AsyncMock(
            side_effect=[
                {"count": 100},
                {"total_hits": 500, "avg_hits": 5.0, "max_hits": 50},
            ]
        )
        conn.fetch = AsyncMock(
            return_value=[
                {"lifecycle_state": "active", "count": 80},
                {"lifecycle_state": "probationary", "count": 20},
            ]
        )

        result = await adapter.get_statistics("default")
        assert result["total_knowledge"] == 100
        assert result["adapter"] == "postgres"
        assert "state_distribution" in result

    @pytest.mark.asyncio
    async def test_get_statistics_no_pool(self, adapter):
        result = await adapter.get_statistics("default")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_increment_hit_count(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        result = await adapter.increment_hit_count(
            "default", ["lu_001", "lu_002"]
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_increment_hit_count_empty(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        result = await adapter.increment_hit_count("default", [])
        assert result is True

    @pytest.mark.asyncio
    async def test_increment_hit_count_no_pool(self, adapter):
        result = await adapter.increment_hit_count("default", ["lu_001"])
        assert result is True


# ==================== 命名空间测试 ====================

class TestPostgresNamespaces:
    """测试命名空间管理"""

    @pytest.mark.asyncio
    async def test_get_namespaces(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetch = AsyncMock(
            return_value=[
                {"namespace": "default"},
                {"namespace": "medical"},
            ]
        )

        result = await adapter.get_namespaces()
        assert len(result) == 2
        assert "default" in result
        assert "medical" in result

    @pytest.mark.asyncio
    async def test_get_namespaces_no_pool(self, adapter):
        result = await adapter.get_namespaces()
        assert result == []


# ==================== 审计日志测试 ====================

class TestPostgresAudit:
    """测试审计日志"""

    @pytest.mark.asyncio
    async def test_save_audit_log(self, connected_adapter):
        adapter, pool, conn = connected_adapter

        result = await adapter.save_audit_log({
            "namespace": "default",
            "lu_id": "lu_001",
            "action": "SAVE",
            "new_state": "active",
        })
        assert result is True

    @pytest.mark.asyncio
    async def test_save_audit_log_disabled(self, make_adapter, mock_asyncpg):
        a = make_adapter(enable_audit=False)
        conn = make_conn_mock()
        a._pool = make_pool_mock(conn)
        a._connected = True

        result = await a.save_audit_log({
            "namespace": "default",
            "action": "SAVE",
        })
        assert result is True

    @pytest.mark.asyncio
    async def test_query_audit_log(self, connected_adapter):
        adapter, pool, conn = connected_adapter
        conn.fetch = AsyncMock(return_value=[])

        result = await adapter.query_audit_log(namespace="default")
        assert result == []

    @pytest.mark.asyncio
    async def test_query_audit_log_disabled(self, make_adapter, mock_asyncpg):
        a = make_adapter(enable_audit=False)
        conn = make_conn_mock()
        a._pool = make_pool_mock(conn)
        a._connected = True

        result = await a.query_audit_log()
        assert result == []


# ==================== 工具方法测试 ====================

class TestPostgresUtils:
    """测试工具方法"""

    def test_row_to_dict(self, adapter):
        row = {
            "lu_id": "lu_001",
            "namespace": "default",
            "condition_text": "when X",
            "decision_text": "do Y",
            "lifecycle_state": "active",
            "trust_tier": "standard",
            "hit_count": 5,
            "consecutive_misses": 1,
            "version": 3,
            "created_at": datetime(2025, 1, 1, 12, 0, 0),
            "updated_at": datetime(2025, 6, 15, 8, 30, 0),
            "metadata": None,
        }

        result = adapter._row_to_dict(row)

        assert result["lu_id"] == "lu_001"
        assert result["condition"] == "when X"
        assert result["decision"] == "do Y"
        assert result["lifecycle_state"] == "active"
        assert result["hit_count"] == 5
        assert result["version"] == 3
        assert result["metadata"] is None

    def test_row_to_dict_with_json_metadata(self, adapter):
        row = {
            "lu_id": "lu_001",
            "namespace": "default",
            "condition_text": "c",
            "decision_text": "d",
            "lifecycle_state": "active",
            "trust_tier": "standard",
            "hit_count": 0,
            "consecutive_misses": 0,
            "version": 1,
            "created_at": datetime(2025, 1, 1),
            "updated_at": datetime(2025, 1, 1),
            "metadata": '{"source": "test", "priority": 1}',
        }

        result = adapter._row_to_dict(row)
        assert result["metadata"] == {"source": "test", "priority": 1}

    def test_row_to_dict_with_invalid_json_metadata(self, adapter):
        row = {
            "lu_id": "lu_001",
            "namespace": "default",
            "condition_text": "c",
            "decision_text": "d",
            "lifecycle_state": "active",
            "trust_tier": "standard",
            "hit_count": 0,
            "consecutive_misses": 0,
            "version": 1,
            "created_at": None,
            "updated_at": None,
            "metadata": "not-valid-json",
        }

        result = adapter._row_to_dict(row)
        assert result["metadata"] is None
        assert result["created_at"] is None

    def test_row_to_dict_empty_texts(self, adapter):
        row = {
            "lu_id": "lu_001",
            "namespace": "default",
            "condition_text": None,
            "decision_text": None,
            "lifecycle_state": "probationary",
            "trust_tier": "standard",
            "hit_count": 0,
            "consecutive_misses": 0,
            "version": 1,
            "created_at": datetime(2025, 1, 1),
            "updated_at": datetime(2025, 1, 1),
            "metadata": None,
        }

        result = adapter._row_to_dict(row)
        assert result["condition"] == ""
        assert result["decision"] == ""


# ==================== 工厂函数测试 ====================

class TestPostgresFactory:
    """测试通过工厂函数创建 PostgreSQL 适配器"""

    def test_create_postgres_adapter(self, mock_asyncpg):
        from aga_knowledge.persistence import create_adapter

        a = create_adapter({
            "type": "postgres",
            "postgres_host": "db.example.com",
            "postgres_port": 5433,
            "postgres_database": "test_db",
            "postgres_user": "test_user",
            "postgres_password": "secret",
            "postgres_pool_size": 20,
        })

        from aga_knowledge.persistence.postgres_adapter import PostgresAdapter
        assert isinstance(a, PostgresAdapter)
        assert a.host == "db.example.com"
        assert a.port == 5433
        assert a.database == "test_db"
        assert a.user == "test_user"
        assert a.password == "secret"
        assert a.pool_size == 20

    def test_create_postgres_adapter_with_url(self, mock_asyncpg):
        from aga_knowledge.persistence import create_adapter

        a = create_adapter({
            "type": "postgres",
            "postgres_url": "postgresql://user:pass@host:5432/db",
        })

        assert a.dsn == "postgresql://user:pass@host:5432/db"

    def test_create_postgres_adapter_defaults(self, mock_asyncpg):
        from aga_knowledge.persistence import create_adapter

        a = create_adapter({"type": "postgres"})
        assert a.host == "localhost"
        assert a.port == 5432
        assert a.database == "aga_knowledge"


# ==================== 无 asyncpg 时的行为 ====================

class TestPostgresWithoutAsyncpg:
    """测试在没有 asyncpg 时的行为"""

    def test_import_error_without_asyncpg(self):
        original_modules = {}
        for key in list(sys.modules.keys()):
            if "asyncpg" in key or "postgres_adapter" in key:
                original_modules[key] = sys.modules.pop(key)

        try:
            with patch.dict("sys.modules", {"asyncpg": None}):
                if "aga_knowledge.persistence.postgres_adapter" in sys.modules:
                    del sys.modules["aga_knowledge.persistence.postgres_adapter"]

                try:
                    from aga_knowledge.persistence.postgres_adapter import PostgresAdapter
                    PostgresAdapter()
                except ImportError:
                    pass  # 预期行为
        finally:
            sys.modules.update(original_modules)
