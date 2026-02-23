"""
aga-knowledge 数据类型定义

明文 KV 版本：知识以 condition/decision 文本对形式存储，
不涉及向量化编码，由 aga-core 在推理时按需处理。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


class LifecycleState(str, Enum):
    """知识生命周期状态"""
    PROBATIONARY = "probationary"  # 试用期
    CONFIRMED = "confirmed"        # 已确认
    DEPRECATED = "deprecated"      # 已弃用
    QUARANTINED = "quarantined"    # 已隔离


class TrustTier(str, Enum):
    """
    信任层级

    层级说明：
    - SYSTEM: 系统级，最高信任，核心规则
    - VERIFIED: 已验证，经过人工审核确认
    - STANDARD: 标准级，默认层级
    - EXPERIMENTAL: 实验性，尚在测试
    - UNTRUSTED: 不可信，需谨慎使用
    """
    SYSTEM = "system"
    VERIFIED = "verified"
    STANDARD = "standard"
    EXPERIMENTAL = "experimental"
    UNTRUSTED = "untrusted"


# 信任层级优先级（数值越大，信任度越高）
TRUST_TIER_PRIORITY = {
    TrustTier.SYSTEM: 100,
    TrustTier.VERIFIED: 80,
    TrustTier.STANDARD: 50,
    TrustTier.EXPERIMENTAL: 30,
    TrustTier.UNTRUSTED: 10,
}

# 生命周期状态对应的默认可靠性
LIFECYCLE_RELIABILITY = {
    LifecycleState.PROBATIONARY: 0.3,
    LifecycleState.CONFIRMED: 1.0,
    LifecycleState.DEPRECATED: 0.1,
    LifecycleState.QUARANTINED: 0.0,
}


@dataclass
class KnowledgeRecord:
    """
    知识记录（明文 KV 版本）

    以 condition/decision 文本对形式存储知识。
    不包含向量化数据，向量化由 aga-core 在推理时按需处理。
    """
    lu_id: str                          # Learning Unit ID
    condition: str                      # 触发条件描述（明文）
    decision: str                       # 决策描述（明文）
    namespace: str = "default"          # 命名空间
    lifecycle_state: str = LifecycleState.PROBATIONARY.value
    trust_tier: str = TrustTier.STANDARD.value
    hit_count: int = 0                  # 命中计数
    consecutive_misses: int = 0         # 连续未命中计数
    version: int = 1                    # 版本号
    created_at: Optional[str] = None    # 创建时间
    updated_at: Optional[str] = None    # 更新时间
    metadata: Optional[Dict[str, Any]] = None  # 扩展元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'lu_id': self.lu_id,
            'condition': self.condition,
            'decision': self.decision,
            'namespace': self.namespace,
            'lifecycle_state': self.lifecycle_state,
            'trust_tier': self.trust_tier,
            'hit_count': self.hit_count,
            'consecutive_misses': self.consecutive_misses,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeRecord":
        """从字典创建"""
        return cls(
            lu_id=data['lu_id'],
            condition=data.get('condition', ''),
            decision=data.get('decision', ''),
            namespace=data.get('namespace', 'default'),
            lifecycle_state=data.get('lifecycle_state', LifecycleState.PROBATIONARY.value),
            trust_tier=data.get('trust_tier', TrustTier.STANDARD.value),
            hit_count=data.get('hit_count', 0),
            consecutive_misses=data.get('consecutive_misses', 0),
            version=data.get('version', 1),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            metadata=data.get('metadata'),
        )

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeRecord":
        """从 JSON 反序列化"""
        return cls.from_dict(json.loads(json_str))

    @property
    def reliability(self) -> float:
        """根据生命周期状态获取可靠性"""
        try:
            state = LifecycleState(self.lifecycle_state)
            return LIFECYCLE_RELIABILITY.get(state, 0.5)
        except ValueError:
            return 0.5
