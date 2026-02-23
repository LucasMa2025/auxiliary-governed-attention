"""
aga-knowledge 类型测试
"""

import json
import pytest

from aga_knowledge.types import (
    KnowledgeRecord,
    LifecycleState,
    TrustTier,
    LIFECYCLE_RELIABILITY,
    TRUST_TIER_PRIORITY,
)


class TestLifecycleState:
    """测试生命周期状态"""

    def test_all_states(self):
        assert LifecycleState.PROBATIONARY.value == "probationary"
        assert LifecycleState.CONFIRMED.value == "confirmed"
        assert LifecycleState.DEPRECATED.value == "deprecated"
        assert LifecycleState.QUARANTINED.value == "quarantined"

    def test_reliability_mapping(self):
        assert LIFECYCLE_RELIABILITY[LifecycleState.PROBATIONARY] == 0.3
        assert LIFECYCLE_RELIABILITY[LifecycleState.CONFIRMED] == 1.0
        assert LIFECYCLE_RELIABILITY[LifecycleState.DEPRECATED] == 0.1
        assert LIFECYCLE_RELIABILITY[LifecycleState.QUARANTINED] == 0.0


class TestTrustTier:
    """测试信任层级"""

    def test_all_tiers(self):
        assert TrustTier.SYSTEM.value == "system"
        assert TrustTier.VERIFIED.value == "verified"
        assert TrustTier.STANDARD.value == "standard"
        assert TrustTier.EXPERIMENTAL.value == "experimental"
        assert TrustTier.UNTRUSTED.value == "untrusted"

    def test_priority_ordering(self):
        assert TRUST_TIER_PRIORITY[TrustTier.SYSTEM] > TRUST_TIER_PRIORITY[TrustTier.VERIFIED]
        assert TRUST_TIER_PRIORITY[TrustTier.VERIFIED] > TRUST_TIER_PRIORITY[TrustTier.STANDARD]
        assert TRUST_TIER_PRIORITY[TrustTier.STANDARD] > TRUST_TIER_PRIORITY[TrustTier.EXPERIMENTAL]
        assert TRUST_TIER_PRIORITY[TrustTier.EXPERIMENTAL] > TRUST_TIER_PRIORITY[TrustTier.UNTRUSTED]


class TestKnowledgeRecord:
    """测试知识记录"""

    def test_create_default(self):
        record = KnowledgeRecord(
            lu_id="test_001",
            condition="when user asks about X",
            decision="respond with Y",
        )
        assert record.lu_id == "test_001"
        assert record.condition == "when user asks about X"
        assert record.decision == "respond with Y"
        assert record.namespace == "default"
        assert record.lifecycle_state == "probationary"
        assert record.trust_tier == "standard"
        assert record.hit_count == 0

    def test_to_dict(self):
        record = KnowledgeRecord(
            lu_id="test_001",
            condition="cond",
            decision="dec",
            namespace="ns1",
        )
        d = record.to_dict()
        assert d["lu_id"] == "test_001"
        assert d["condition"] == "cond"
        assert d["decision"] == "dec"
        assert d["namespace"] == "ns1"

    def test_from_dict(self):
        data = {
            "lu_id": "test_002",
            "condition": "cond2",
            "decision": "dec2",
            "namespace": "ns2",
            "lifecycle_state": "confirmed",
            "trust_tier": "verified",
        }
        record = KnowledgeRecord.from_dict(data)
        assert record.lu_id == "test_002"
        assert record.lifecycle_state == "confirmed"
        assert record.trust_tier == "verified"

    def test_json_roundtrip(self):
        record = KnowledgeRecord(
            lu_id="test_003",
            condition="cond3",
            decision="dec3",
            metadata={"key": "value"},
        )
        json_str = record.to_json()
        restored = KnowledgeRecord.from_json(json_str)
        assert restored.lu_id == record.lu_id
        assert restored.condition == record.condition
        assert restored.metadata == {"key": "value"}

    def test_reliability_property(self):
        record = KnowledgeRecord(lu_id="r1", condition="c", decision="d")
        record.lifecycle_state = "probationary"
        assert record.reliability == 0.3

        record.lifecycle_state = "confirmed"
        assert record.reliability == 1.0

        record.lifecycle_state = "quarantined"
        assert record.reliability == 0.0

        record.lifecycle_state = "unknown_state"
        assert record.reliability == 0.5
