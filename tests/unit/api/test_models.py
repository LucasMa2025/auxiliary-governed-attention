"""
API 数据模型单元测试

测试 Pydantic 模型的验证和序列化。
"""
import pytest
from datetime import datetime

# 尝试导入 Pydantic 模型
try:
    from aga.api.models import (
        InjectKnowledgeRequest,
        InjectKnowledgeTextRequest,
        BatchInjectRequest,
        UpdateLifecycleRequest,
        QuarantineRequest,
        QueryKnowledgeRequest,
        APIResponse,
        KnowledgeResponse,
        StatisticsResponse,
        LifecycleStateEnum,
        TrustTierEnum,
        HAS_PYDANTIC,
    )
except ImportError:
    HAS_PYDANTIC = False


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestInjectKnowledgeRequest:
    """InjectKnowledgeRequest 测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = InjectKnowledgeRequest(
            lu_id="LU_001",
            condition="用户询问天气",
            decision="提供天气信息",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 4096,
        )
        
        assert request.lu_id == "LU_001"
        assert request.namespace == "default"
        assert request.lifecycle_state == LifecycleStateEnum.PROBATIONARY
    
    def test_custom_namespace(self):
        """测试自定义命名空间"""
        request = InjectKnowledgeRequest(
            lu_id="LU_001",
            namespace="custom_ns",
            condition="条件",
            decision="决策",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 4096,
        )
        
        assert request.namespace == "custom_ns"
    
    def test_lifecycle_state(self):
        """测试生命周期状态"""
        request = InjectKnowledgeRequest(
            lu_id="LU_001",
            condition="条件",
            decision="决策",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 4096,
            lifecycle_state=LifecycleStateEnum.CONFIRMED,
        )
        
        assert request.lifecycle_state == LifecycleStateEnum.CONFIRMED
    
    def test_trust_tier(self):
        """测试信任层级"""
        request = InjectKnowledgeRequest(
            lu_id="LU_001",
            condition="条件",
            decision="决策",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 4096,
            trust_tier=TrustTierEnum.S2_POLICY,
        )
        
        assert request.trust_tier == TrustTierEnum.S2_POLICY
    
    def test_metadata(self):
        """测试元数据"""
        request = InjectKnowledgeRequest(
            lu_id="LU_001",
            condition="条件",
            decision="决策",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 4096,
            metadata={"source": "test", "version": 1},
        )
        
        assert request.metadata["source"] == "test"
        assert request.metadata["version"] == 1
    
    def test_missing_required_field(self):
        """测试缺少必填字段"""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            InjectKnowledgeRequest(
                lu_id="LU_001",
                # 缺少 condition, decision, key_vector, value_vector
            )


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestInjectKnowledgeTextRequest:
    """InjectKnowledgeTextRequest 测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = InjectKnowledgeTextRequest(
            lu_id="LU_001",
            condition="用户询问天气",
            decision="提供天气信息",
        )
        
        assert request.lu_id == "LU_001"
        assert request.condition == "用户询问天气"
        assert request.decision == "提供天气信息"
    
    def test_no_vectors_required(self):
        """测试不需要向量"""
        # TextRequest 不需要 key_vector 和 value_vector
        request = InjectKnowledgeTextRequest(
            lu_id="LU_001",
            condition="条件",
            decision="决策",
        )
        
        assert not hasattr(request, 'key_vector')
        assert not hasattr(request, 'value_vector')


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestUpdateLifecycleRequest:
    """UpdateLifecycleRequest 测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = UpdateLifecycleRequest(
            lu_id="LU_001",
            new_state=LifecycleStateEnum.CONFIRMED,
        )
        
        assert request.lu_id == "LU_001"
        assert request.new_state == LifecycleStateEnum.CONFIRMED
    
    def test_with_reason(self):
        """测试带原因"""
        request = UpdateLifecycleRequest(
            lu_id="LU_001",
            new_state=LifecycleStateEnum.DEPRECATED,
            reason="规则已过时",
        )
        
        assert request.reason == "规则已过时"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestQuarantineRequest:
    """QuarantineRequest 测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = QuarantineRequest(
            lu_id="LU_001",
            reason="检测到异常行为",
        )
        
        assert request.lu_id == "LU_001"
        assert request.reason == "检测到异常行为"
    
    def test_with_source_instance(self):
        """测试带来源实例"""
        request = QuarantineRequest(
            lu_id="LU_001",
            reason="检测到异常行为",
            source_instance="runtime-001",
        )
        
        assert request.source_instance == "runtime-001"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestQueryKnowledgeRequest:
    """QueryKnowledgeRequest 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        request = QueryKnowledgeRequest()
        
        assert request.namespace == "default"
        assert request.limit == 100
        assert request.offset == 0
        assert request.include_vectors is False
    
    def test_filter_by_lifecycle(self):
        """测试按生命周期筛选"""
        request = QueryKnowledgeRequest(
            lifecycle_states=[LifecycleStateEnum.CONFIRMED, LifecycleStateEnum.PROBATIONARY],
        )
        
        assert len(request.lifecycle_states) == 2
    
    def test_pagination(self):
        """测试分页"""
        request = QueryKnowledgeRequest(
            limit=50,
            offset=100,
        )
        
        assert request.limit == 50
        assert request.offset == 100


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestAPIResponse:
    """APIResponse 测试"""
    
    def test_success_response(self):
        """测试成功响应"""
        response = APIResponse(
            success=True,
            message="操作成功",
            data={"lu_id": "LU_001"},
        )
        
        assert response.success is True
        assert response.message == "操作成功"
        assert response.data["lu_id"] == "LU_001"
    
    def test_error_response(self):
        """测试错误响应"""
        response = APIResponse(
            success=False,
            message="操作失败",
        )
        
        assert response.success is False
        assert response.data is None
    
    def test_timestamp(self):
        """测试时间戳"""
        response = APIResponse(success=True)
        
        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestLifecycleStateEnum:
    """LifecycleStateEnum 测试"""
    
    def test_all_states(self):
        """测试所有状态"""
        assert LifecycleStateEnum.PROBATIONARY.value == "probationary"
        assert LifecycleStateEnum.CONFIRMED.value == "confirmed"
        assert LifecycleStateEnum.DEPRECATED.value == "deprecated"
        assert LifecycleStateEnum.QUARANTINED.value == "quarantined"
    
    def test_from_string(self):
        """测试从字符串创建"""
        state = LifecycleStateEnum("confirmed")
        assert state == LifecycleStateEnum.CONFIRMED


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestTrustTierEnum:
    """TrustTierEnum 测试"""
    
    def test_all_tiers(self):
        """测试所有层级"""
        assert TrustTierEnum.S0_ACCELERATION.value == "s0_acceleration"
        assert TrustTierEnum.S1_EXPERIENCE.value == "s1_experience"
        assert TrustTierEnum.S2_POLICY.value == "s2_policy"
        assert TrustTierEnum.S3_IMMUTABLE.value == "s3_immutable"
