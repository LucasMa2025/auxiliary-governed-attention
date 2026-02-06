"""
测试数据 Fixtures

提供测试用的示例数据。
"""
import torch
import json
from pathlib import Path
from typing import List, Dict

from aga import KnowledgeRecord, LifecycleState


def create_sample_records(
    count: int,
    namespace: str = "default",
    bottleneck_dim: int = 64,
    hidden_dim: int = 768,
) -> List[KnowledgeRecord]:
    """
    创建示例知识记录
    
    Args:
        count: 记录数量
        namespace: 命名空间
        bottleneck_dim: key 向量维度
        hidden_dim: value 向量维度
        
    Returns:
        知识记录列表
    """
    records = []
    for i in range(count):
        records.append(KnowledgeRecord(
            lu_id=f"sample_{i:04d}",
            namespace=namespace,
            condition=f"示例条件 {i}",
            decision=f"示例决策 {i}",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY if i % 3 == 0 else LifecycleState.CONFIRMED,
            reliability=0.3 if i % 3 == 0 else 1.0,
        ))
    return records


def create_diverse_records(
    namespaces: List[str],
    records_per_namespace: int,
    bottleneck_dim: int = 64,
    hidden_dim: int = 768,
) -> Dict[str, List[KnowledgeRecord]]:
    """
    创建多命名空间的多样化记录
    
    Args:
        namespaces: 命名空间列表
        records_per_namespace: 每个命名空间的记录数
        bottleneck_dim: key 向量维度
        hidden_dim: value 向量维度
        
    Returns:
        按命名空间分组的记录字典
    """
    result = {}
    for ns in namespaces:
        result[ns] = create_sample_records(
            count=records_per_namespace,
            namespace=ns,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
        )
    return result


# 预定义的测试场景
SCENARIOS = {
    "small": {
        "num_slots": 16,
        "hidden_dim": 256,
        "bottleneck_dim": 32,
        "batch_size": 1,
        "seq_len": 16,
    },
    "medium": {
        "num_slots": 64,
        "hidden_dim": 768,
        "bottleneck_dim": 64,
        "batch_size": 4,
        "seq_len": 32,
    },
    "large": {
        "num_slots": 256,
        "hidden_dim": 1024,
        "bottleneck_dim": 128,
        "batch_size": 8,
        "seq_len": 64,
    },
}


def get_scenario(name: str) -> Dict:
    """获取预定义场景配置"""
    return SCENARIOS.get(name, SCENARIOS["medium"])
