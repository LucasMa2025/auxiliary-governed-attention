"""
API 路由层单元测试

测试 FastAPI 路由的 HTTP 协议转换。
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from aga.api.routes import create_routers, HAS_FASTAPI as ROUTES_HAS_FASTAPI
    from aga.api.service import AGAService, ServiceConfig
except ImportError:
    ROUTES_HAS_FASTAPI = False


@pytest.fixture
def mock_service():
    """创建模拟服务"""
    service = Mock(spec=AGAService)
    
    # 配置异步方法
    service.health_check = AsyncMock(return_value={
        "status": "healthy",
        "version": "3.1",
        "aga_initialized": True,
        "namespaces": ["default"],
        "total_knowledge": 10,
        "uptime_seconds": 100.0,
        "timestamp": "2024-01-01T00:00:00",
    })
    
    service.get_config = Mock(return_value={
        "hidden_dim": 4096,
        "bottleneck_dim": 64,
    })
    
    service.get_namespaces = Mock(return_value=["default", "test"])
    
    service.inject_knowledge = AsyncMock(return_value={
        "success": True,
        "lu_id": "LU_001",
        "slot_idx": 0,
    })
    
    service.inject_knowledge_text = AsyncMock(return_value={
        "success": True,
        "lu_id": "LU_001",
        "slot_idx": 0,
    })
    
    service.get_knowledge = AsyncMock(return_value={
        "lu_id": "LU_001",
        "namespace": "default",
        "condition": "条件",
        "decision": "决策",
        "lifecycle_state": "probationary",
    })
    
    service.delete_knowledge = AsyncMock(return_value={
        "success": True,
    })
    
    service.update_lifecycle = AsyncMock(return_value={
        "success": True,
    })
    
    service.quarantine_knowledge = AsyncMock(return_value={
        "success": True,
    })
    
    service.get_statistics = AsyncMock(return_value={
        "namespace": "default",
        "total_slots": 100,
        "active_slots": 10,
        "free_slots": 90,
        "state_distribution": {"probationary": 5, "confirmed": 5},
        "total_hits": 100,
        "avg_reliability": 0.9,
        "avg_key_norm": 1.0,
        "avg_value_norm": 1.0,
    })
    
    service.get_namespace_statistics = AsyncMock(return_value={
        "namespace": "default",
        "total_slots": 100,
        "active_slots": 10,
        "free_slots": 90,
        "state_distribution": {"probationary": 5, "confirmed": 5},
        "trust_tier_distribution": {"s0_acceleration": 3, "s1_experience": 7},
        "total_hits": 100,
        "avg_reliability": 0.9,
        "avg_key_norm": 1.0,
        "avg_value_norm": 1.0,
    })
    
    return service


@pytest.fixture
def app(mock_service):
    """创建测试应用"""
    if not HAS_FASTAPI or not ROUTES_HAS_FASTAPI:
        pytest.skip("FastAPI not available")
    
    app = FastAPI()
    
    # 覆盖依赖注入
    from aga.api import routes
    original_get_service = routes.get_service
    routes.get_service = lambda: mock_service
    
    # 创建路由
    routers = create_routers(mock_service)
    
    # 注册路由
    app.include_router(routers["health"])
    app.include_router(routers["namespace"])
    app.include_router(routers["knowledge"])
    app.include_router(routers["lifecycle"])
    app.include_router(routers["statistics"])
    
    yield app
    
    # 恢复
    routes.get_service = original_get_service


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestHealthRoutes:
    """健康检查路由测试"""
    
    def test_health_check(self, client, mock_service):
        """测试健康检查"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        mock_service.health_check.assert_called_once()
    
    def test_get_config(self, client, mock_service):
        """测试获取配置"""
        response = client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "hidden_dim" in data


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestNamespaceRoutes:
    """命名空间路由测试"""
    
    def test_list_namespaces(self, client, mock_service):
        """测试列出命名空间"""
        response = client.get("/namespaces")
        
        assert response.status_code == 200
        data = response.json()
        assert "namespaces" in data
        assert "default" in data["namespaces"]


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestKnowledgeRoutes:
    """知识管理路由测试"""
    
    def test_inject_knowledge(self, client, mock_service):
        """测试注入知识"""
        response = client.post("/knowledge/inject", json={
            "lu_id": "LU_001",
            "condition": "条件",
            "decision": "决策",
            "key_vector": [0.1] * 64,
            "value_vector": [0.2] * 4096,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_inject_knowledge_text(self, client, mock_service):
        """测试文本知识注入"""
        response = client.post("/knowledge/inject-text", json={
            "lu_id": "LU_001",
            "condition": "用户询问天气",
            "decision": "提供天气信息",
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_inject_knowledge_missing_field(self, client, mock_service):
        """测试缺少必填字段"""
        response = client.post("/knowledge/inject-text", json={
            "lu_id": "LU_001",
            # 缺少 condition 和 decision
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_get_knowledge(self, client, mock_service):
        """测试获取知识"""
        response = client.get("/knowledge/default/LU_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["lu_id"] == "LU_001"
    
    def test_delete_knowledge(self, client, mock_service):
        """测试删除知识"""
        response = client.delete("/knowledge/default/LU_001")
        
        assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestLifecycleRoutes:
    """生命周期路由测试"""
    
    def test_update_lifecycle(self, client, mock_service):
        """测试更新生命周期"""
        response = client.post("/lifecycle/update", json={
            "lu_id": "LU_001",
            "new_state": "confirmed",
        })
        
        assert response.status_code == 200
    
    def test_quarantine(self, client, mock_service):
        """测试隔离"""
        response = client.post("/lifecycle/quarantine", json={
            "lu_id": "LU_001",
            "reason": "检测到异常",
        })
        
        assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestStatisticsRoutes:
    """统计路由测试"""
    
    def test_get_statistics(self, client, mock_service):
        """测试获取统计"""
        response = client.get("/statistics/default")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_slots" in data


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestErrorHandling:
    """错误处理测试"""
    
    def test_inject_value_error(self, client, mock_service):
        """测试值错误处理"""
        mock_service.inject_knowledge.side_effect = ValueError("Invalid vector dimension")
        
        response = client.post("/knowledge/inject", json={
            "lu_id": "LU_001",
            "condition": "条件",
            "decision": "决策",
            "key_vector": [0.1] * 64,
            "value_vector": [0.2] * 4096,
        })
        
        assert response.status_code == 400
    
    def test_inject_runtime_error(self, client, mock_service):
        """测试运行时错误处理"""
        mock_service.inject_knowledge.side_effect = RuntimeError("Service unavailable")
        
        response = client.post("/knowledge/inject", json={
            "lu_id": "LU_001",
            "condition": "条件",
            "decision": "决策",
            "key_vector": [0.1] * 64,
            "value_vector": [0.2] * 4096,
        })
        
        assert response.status_code == 503
