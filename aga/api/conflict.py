"""
AGA 知识冲突检测模块

提供知识注入时的冲突检测和处理：
- 语义冲突检测（条件相似但决策不同）
- 重复检测（条件和决策都相似）
- 冲突解决策略

版本: v1.0
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import logging

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    """冲突类型"""
    NONE = "none"                    # 无冲突
    DUPLICATE = "duplicate"          # 重复（条件和决策都相似）
    SEMANTIC = "semantic"            # 语义冲突（条件相似但决策不同）
    PARTIAL = "partial"              # 部分冲突（条件部分重叠）


class ConflictResolution(str, Enum):
    """冲突解决策略"""
    REJECT = "reject"                # 拒绝新知识
    REPLACE = "replace"              # 替换旧知识
    MERGE = "merge"                  # 合并（平均向量）
    KEEP_BOTH = "keep_both"          # 保留两者
    MANUAL = "manual"                # 需要人工处理


@dataclass
class ConflictInfo:
    """冲突信息"""
    conflict_type: ConflictType
    conflicting_lu_id: str
    key_similarity: float
    value_similarity: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'conflict_type': self.conflict_type.value,
            'conflicting_lu_id': self.conflicting_lu_id,
            'key_similarity': self.key_similarity,
            'value_similarity': self.value_similarity,
            'details': self.details,
        }


@dataclass
class ConflictDetectorConfig:
    """冲突检测配置"""
    # 相似度阈值
    key_similarity_threshold: float = 0.9      # Key 相似度阈值
    value_similarity_threshold: float = 0.9    # Value 相似度阈值
    
    # 冲突判定
    duplicate_threshold: float = 0.95          # 重复判定阈值
    semantic_conflict_threshold: float = 0.85  # 语义冲突判定阈值
    
    # 默认解决策略
    default_resolution: ConflictResolution = ConflictResolution.REJECT
    
    # 自动解决
    auto_resolve_duplicates: bool = True       # 自动处理重复
    auto_resolve_strategy: ConflictResolution = ConflictResolution.REPLACE


class ConflictDetector:
    """
    知识冲突检测器
    
    检测新知识与现有知识的冲突，支持多种冲突类型和解决策略。
    
    使用示例：
        ```python
        detector = ConflictDetector(ConflictDetectorConfig())
        
        # 检测冲突
        conflicts = detector.detect_conflicts(
            new_key=new_key_vector,
            new_value=new_value_vector,
            existing_keys=existing_keys,
            existing_values=existing_values,
            existing_lu_ids=existing_lu_ids,
        )
        
        if conflicts:
            for conflict in conflicts:
                print(f"Conflict with {conflict.conflicting_lu_id}: {conflict.conflict_type}")
        ```
    """
    
    def __init__(self, config: Optional[ConflictDetectorConfig] = None):
        self.config = config or ConflictDetectorConfig()
    
    def detect_conflicts(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        existing_keys: torch.Tensor,
        existing_values: torch.Tensor,
        existing_lu_ids: List[str],
    ) -> List[ConflictInfo]:
        """
        检测新知识与现有知识的冲突
        
        Args:
            new_key: 新知识的 Key 向量 [key_dim]
            new_value: 新知识的 Value 向量 [value_dim]
            existing_keys: 现有知识的 Key 向量 [num_slots, key_dim]
            existing_values: 现有知识的 Value 向量 [num_slots, value_dim]
            existing_lu_ids: 现有知识的 ID 列表
        
        Returns:
            冲突信息列表
        """
        if existing_keys.shape[0] == 0:
            return []
        
        conflicts = []
        
        # 计算 Key 相似度
        key_similarities = F.cosine_similarity(
            new_key.unsqueeze(0),
            existing_keys,
            dim=-1
        )
        
        # 计算 Value 相似度
        value_similarities = F.cosine_similarity(
            new_value.unsqueeze(0),
            existing_values,
            dim=-1
        )
        
        for i, (key_sim, value_sim) in enumerate(zip(key_similarities, value_similarities)):
            key_sim_val = key_sim.item()
            value_sim_val = value_sim.item()
            
            # 检查是否超过阈值
            if key_sim_val < self.config.key_similarity_threshold:
                continue
            
            # 判定冲突类型
            if key_sim_val >= self.config.duplicate_threshold and value_sim_val >= self.config.duplicate_threshold:
                # 重复
                conflict_type = ConflictType.DUPLICATE
            elif key_sim_val >= self.config.semantic_conflict_threshold and value_sim_val < self.config.value_similarity_threshold:
                # 语义冲突
                conflict_type = ConflictType.SEMANTIC
            elif key_sim_val >= self.config.key_similarity_threshold:
                # 部分冲突
                conflict_type = ConflictType.PARTIAL
            else:
                continue
            
            conflicts.append(ConflictInfo(
                conflict_type=conflict_type,
                conflicting_lu_id=existing_lu_ids[i],
                key_similarity=key_sim_val,
                value_similarity=value_sim_val,
                details={
                    'slot_idx': i,
                },
            ))
        
        return conflicts
    
    def detect_conflicts_batch(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_lu_ids: List[str],
        existing_keys: torch.Tensor,
        existing_values: torch.Tensor,
        existing_lu_ids: List[str],
    ) -> Dict[str, List[ConflictInfo]]:
        """
        批量检测冲突
        
        Args:
            new_keys: 新知识的 Key 向量 [batch, key_dim]
            new_values: 新知识的 Value 向量 [batch, value_dim]
            new_lu_ids: 新知识的 ID 列表
            existing_keys: 现有知识的 Key 向量 [num_slots, key_dim]
            existing_values: 现有知识的 Value 向量 [num_slots, value_dim]
            existing_lu_ids: 现有知识的 ID 列表
        
        Returns:
            {lu_id: [ConflictInfo]} 字典
        """
        results = {}
        
        for i, (key, value, lu_id) in enumerate(zip(new_keys, new_values, new_lu_ids)):
            conflicts = self.detect_conflicts(
                key, value,
                existing_keys, existing_values, existing_lu_ids
            )
            if conflicts:
                results[lu_id] = conflicts
        
        return results
    
    def resolve_conflict(
        self,
        conflict: ConflictInfo,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        existing_key: torch.Tensor,
        existing_value: torch.Tensor,
        resolution: Optional[ConflictResolution] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """
        解决冲突
        
        Args:
            conflict: 冲突信息
            new_key: 新知识的 Key 向量
            new_value: 新知识的 Value 向量
            existing_key: 现有知识的 Key 向量
            existing_value: 现有知识的 Value 向量
            resolution: 解决策略（None 使用默认）
        
        Returns:
            (resolved_key, resolved_value, action)
            - resolved_key: 解决后的 Key 向量（None 表示不更新）
            - resolved_value: 解决后的 Value 向量（None 表示不更新）
            - action: 执行的操作描述
        """
        if resolution is None:
            # 自动选择策略
            if conflict.conflict_type == ConflictType.DUPLICATE:
                if self.config.auto_resolve_duplicates:
                    resolution = self.config.auto_resolve_strategy
                else:
                    resolution = ConflictResolution.MANUAL
            else:
                resolution = self.config.default_resolution
        
        if resolution == ConflictResolution.REJECT:
            return None, None, "rejected"
        
        elif resolution == ConflictResolution.REPLACE:
            return new_key, new_value, "replaced"
        
        elif resolution == ConflictResolution.MERGE:
            # 平均合并
            merged_key = (new_key + existing_key) / 2
            merged_value = (new_value + existing_value) / 2
            # 归一化
            merged_key = F.normalize(merged_key, dim=-1)
            merged_value = F.normalize(merged_value, dim=-1)
            return merged_key, merged_value, "merged"
        
        elif resolution == ConflictResolution.KEEP_BOTH:
            return new_key, new_value, "kept_both"
        
        elif resolution == ConflictResolution.MANUAL:
            return None, None, "manual_review_required"
        
        return None, None, "unknown"
    
    def get_conflict_summary(self, conflicts: List[ConflictInfo]) -> Dict[str, Any]:
        """获取冲突摘要"""
        if not conflicts:
            return {'total': 0, 'by_type': {}}
        
        by_type = {}
        for conflict in conflicts:
            t = conflict.conflict_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            'total': len(conflicts),
            'by_type': by_type,
            'most_similar': max(conflicts, key=lambda c: c.key_similarity).to_dict(),
        }


class ConflictAwareInjector:
    """
    冲突感知的知识注入器
    
    在注入前自动检测和处理冲突。
    """
    
    def __init__(
        self,
        detector: Optional[ConflictDetector] = None,
        auto_resolve: bool = True,
    ):
        self.detector = detector or ConflictDetector()
        self.auto_resolve = auto_resolve
        
        # 统计
        self._injection_count = 0
        self._conflict_count = 0
        self._resolved_count = 0
    
    def inject_with_conflict_check(
        self,
        aga_operator,
        lu_id: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        condition: str = "",
        decision: str = "",
        resolution: Optional[ConflictResolution] = None,
    ) -> Dict[str, Any]:
        """
        带冲突检测的知识注入
        
        Args:
            aga_operator: AGA 算子
            lu_id: 知识 ID
            key_vector: Key 向量
            value_vector: Value 向量
            condition: 条件文本
            decision: 决策文本
            resolution: 冲突解决策略
        
        Returns:
            注入结果
        """
        self._injection_count += 1
        
        # 获取现有知识
        existing_keys = aga_operator.aux_keys
        existing_values = aga_operator.aux_values
        existing_lu_ids = list(aga_operator.slot_lu_ids.values())
        
        # 检测冲突
        conflicts = self.detector.detect_conflicts(
            key_vector, value_vector,
            existing_keys, existing_values, existing_lu_ids
        )
        
        result = {
            'lu_id': lu_id,
            'conflicts': [c.to_dict() for c in conflicts],
            'injected': False,
            'action': None,
        }
        
        if conflicts:
            self._conflict_count += 1
            
            if not self.auto_resolve and resolution is None:
                result['action'] = 'conflict_detected_manual_review'
                return result
            
            # 处理第一个冲突（最严重的）
            conflict = max(conflicts, key=lambda c: c.key_similarity)
            
            # 获取冲突槽位的向量
            slot_idx = conflict.details.get('slot_idx', 0)
            existing_key = existing_keys[slot_idx]
            existing_value = existing_values[slot_idx]
            
            # 解决冲突
            resolved_key, resolved_value, action = self.detector.resolve_conflict(
                conflict, key_vector, value_vector,
                existing_key, existing_value, resolution
            )
            
            result['action'] = action
            
            if resolved_key is not None:
                # 更新或注入
                if action == "replaced":
                    aga_operator.aux_keys.data[slot_idx] = resolved_key
                    aga_operator.aux_values.data[slot_idx] = resolved_value
                    result['injected'] = True
                    result['slot_idx'] = slot_idx
                elif action == "merged":
                    aga_operator.aux_keys.data[slot_idx] = resolved_key
                    aga_operator.aux_values.data[slot_idx] = resolved_value
                    result['injected'] = True
                    result['slot_idx'] = slot_idx
                elif action == "kept_both":
                    # 注入到新槽位
                    slot_idx = aga_operator.find_free_slot()
                    if slot_idx >= 0:
                        aga_operator.inject_knowledge(
                            lu_id, key_vector.tolist(), value_vector.tolist(),
                            condition, decision
                        )
                        result['injected'] = True
                        result['slot_idx'] = slot_idx
                
                self._resolved_count += 1
        else:
            # 无冲突，直接注入
            slot_idx = aga_operator.inject_knowledge(
                lu_id, key_vector.tolist(), value_vector.tolist(),
                condition, decision
            )
            result['injected'] = slot_idx >= 0
            result['slot_idx'] = slot_idx
            result['action'] = 'injected'
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            'injection_count': self._injection_count,
            'conflict_count': self._conflict_count,
            'resolved_count': self._resolved_count,
            'conflict_rate': self._conflict_count / max(1, self._injection_count),
        }
