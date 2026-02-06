"""
Portal 路由层单元测试

测试 FastAPI 路由的 HTTP 协议转换。
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from aga.portal.routes import create_portal_routers, HAS_FASTAPI as ROUTES_HAS_FASTAPI
    from aga.portal.service import PortalService
except ImportError:
    ROUTES_HAS_FASTAPI = False


@pytest.fixture
def mock_portal_service():
    """创建模拟 Portal 服务"""
    service = Mock(spec=PortalService)
    
    # 配置异步方法 - 返回字典而不是 AsyncMock
    service.health_check = AsyncMock(return_value={
        "status": "healthy",
        "uptime_seconds": 100.0,
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    service.inject_knowledge = AsyncMock(return_value={
        "success": True,
        "lu_id": "LU_001",
        "message": "Knowledge injected",
    })
    
    service.inject_knowledge_text = AsyncMock(return_value={
        "success": True,
        "lu_id": "LU_001",
        "message": "Knowledge injected",
    })
    
    service.batch_inject_text = AsyncMock(return_value={
        "success": True,
        "total": 2,
        "success_count": 2,
        "failed_count": 0,
    })
    
    service.get_knowledge = AsyncMock(return_value={
        "lu_id": "LU_001",
        "namespace": "default",
        "condition": "条件",
        "decision": "决策",
        "lifecycle_state": "probationary",
    })
    
    service.query_knowledge = AsyncMock(return_value={
        "items": [],
        "count": 0,
        "limit": 100,
        "offset": 0,
    })
    
    service.delete_knowledge = AsyncMock(return_value={
        "success": True,
    })
    
    service.update_lifecycle = AsyncMock(return_value={
        "success": True,
        "lu_id": "LU_001",
        "old_state": "probationary",
        "new_state": "confirmed",
    })
    
    service.quarantine = AsyncMock(return_value={
        "success": True,
        "lu_id": "LU_001",
        "new_state": "quarantined",
    })
    
    service.get_encoder_signature = Mock(return_value={
        "type": "hash",
        "native_dim": 768,
    })
    
    return service


@pytest.fixture
def app(mock_portal_service):
    """创建测试应用"""
    if not HAS_FASTAPI or not ROUTES_HAS_FASTAPI:
        pytest.skip("FastAPI not available")
    
    app = FastAPI()
    
    # 创建路由
    routers = create_portal_routers(mock_portal_service)
    
    # 注册路由
    app.include_router(routers["health"])
    app.include_router(routers["knowledge"])
    app.include_router(routers["lifecycle"])
    
    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestPortalHealthRoutes:
    """Portal 健康检查路由测试"""
    
    def test_health_check(self, client, mock_portal_service):
        """测试健康检查"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_readiness_check(self, client, mock_portal_service):
        """测试就绪检查"""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
    
    def test_readiness_check_unhealthy(self, client, mock_portal_service):
        """测试不健康时的就绪检查"""
        mock_portal_service.health_check.return_value = {
            "status": "unhealthy",
        }
        
        response = client.get("/health/ready")
        
        assert response.status_code == 503


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestPortalKnowledgeRoutes:
    """Portal 知识管理路由测试"""
    
    def test_inject_knowledge(self, client, mock_portal_service):
        """测试注入知识"""
        response = client.post("/knowledge/inject", json={
            "lu_id": "LU_001",
            "key_vector": [0.1] * 64,
            "value_vector": [0.2] * 4096,
            "condition": "条件",
            "decision": "决策",
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_inject_knowledge_text(self, client, mock_portal_service):
        """测试文本知识注入"""
        response = client.post("/knowledge/inject-text", json={
            "lu_id": "LU_001",
            "condition": "用户询问天气",
            "decision": "提供天气信息",
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_batch_inject_text(self, client, mock_portal_service):
        """测试批量文本注入"""
        # 注意：路由是 /batch-text 而不是 /batch-inject-text
        response = client.post("/knowledge/batch-text", json={
            "items": [
                {
                    "lu_id": "LU_001",
                    "condition": "条件1",
                    "decision": "决策1",
                },
                {
                    "lu_id": "LU_002",
                    "condition": "条件2",
                    "decision": "决策2",
                },
            ],
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_get_knowledge(self, client, mock_portal_service):
        """测试获取知识"""
        response = client.get("/knowledge/default/LU_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["lu_id"] == "LU_001"
    
    def test_list_knowledge(self, client, mock_portal_service):
        """测试列出知识"""
        response = client.get("/knowledge/default")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "items" in data["data"]
    
    def test_delete_knowledge(self, client, mock_portal_service):
        """测试删除知识"""
        response = client.delete("/knowledge/default/LU_001")
        
        assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestPortalLifecycleRoutes:
    """Portal 生命周期路由测试"""
    
    def test_update_lifecycle(self, client, mock_portal_service):
        """测试更新生命周期"""
        # 注意：使用 PUT 方法而不是 POST
        response = client.put("/lifecycle/update", json={
            "lu_id": "LU_001",
            "new_state": "confirmed",
        })
        
        assert response.status_code == 200
    
    def test_quarantine(self, client, mock_portal_service):
        """测试隔离"""
        response = client.post("/lifecycle/quarantine", json={
            "lu_id": "LU_001",
            "reason": "检测到异常",
        })
        
        assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestPortalValidation:
    """Portal 请求验证测试"""
    
    def test_inject_missing_lu_id(self, client, mock_portal_service):
        """测试缺少 lu_id"""
        response = client.post("/knowledge/inject-text", json={
            "condition": "条件",
            "decision": "决策",
        })
        
        assert response.status_code == 422
    
    def test_inject_missing_condition(self, client, mock_portal_service):
        """测试缺少 condition"""
        response = client.post("/knowledge/inject-text", json={
            "lu_id": "LU_001",
            "decision": "决策",
        })
        
        assert response.status_code == 422
    
    def test_quarantine_missing_reason(self, client, mock_portal_service):
        """测试隔离缺少原因"""
        response = client.post("/lifecycle/quarantine", json={
            "lu_id": "LU_001",
        })
        
        assert response.status_code == 422
