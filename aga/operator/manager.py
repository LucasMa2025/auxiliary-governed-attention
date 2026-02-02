"""
AGA 多实例管理器

管理多个 AGA 算子实例，支持 Transformer 模型集成。

版本: v3.0
"""
from typing import Optional, List, Dict, Any
import logging

import torch
import torch.nn as nn

from .aga_operator import AGAOperator
from .transformer import AGAAugmentedTransformerLayer
from ..types import LifecycleState, DecayContext
from ..unified_config import AGAConfig

logger = logging.getLogger(__name__)


class AGAManager:
    """
    AGA 多实例管理器
    
    功能：
    - 将 AGA 挂载到 Transformer 模型的指定层
    - 管理多个 AGA 实例
    - 提供统一的知识注入接口
    - 支持持久化衰减的跨层协调
    """
    
    def __init__(self, config: Optional[AGAConfig] = None):
        """
        初始化管理器
        
        Args:
            config: AGA 配置
        """
        self.config = config or AGAConfig()
        self.aga_modules: Dict[int, AGAOperator] = {}
        self.original_layers: Dict[int, nn.Module] = {}
        self.model = None
        self._decay_context: Optional[DecayContext] = None
    
    def attach_to_model(
        self,
        model: nn.Module,
        layer_indices: List[int],
        hidden_dim: Optional[int] = None,
        bottleneck_dim: Optional[int] = None,
        num_slots: Optional[int] = None,
        num_heads: Optional[int] = None,
        config: Optional[AGAConfig] = None,
    ) -> Dict[int, AGAOperator]:
        """
        将 AGA 挂载到模型
        
        Args:
            model: Transformer 模型
            layer_indices: 要挂载的层索引（支持负索引）
            hidden_dim: 隐藏维度（自动检测）
            bottleneck_dim: 瓶颈维度
            num_slots: 槽位数量
            num_heads: 注意力头数（自动检测）
            config: AGA 配置
        
        Returns:
            层索引到 AGA 模块的映射
        """
        self.model = model
        
        # 使用配置或参数
        if config is not None:
            self.config = config
        
        # 自动检测模型参数
        if hidden_dim is None:
            if hasattr(model.config, 'hidden_size'):
                hidden_dim = model.config.hidden_size
            elif hasattr(model.config, 'n_embd'):
                hidden_dim = model.config.n_embd
            else:
                raise ValueError("Cannot detect hidden_dim, please specify")
        
        if num_heads is None:
            if hasattr(model.config, 'num_attention_heads'):
                num_heads = model.config.num_attention_heads
            elif hasattr(model.config, 'n_head'):
                num_heads = model.config.n_head
            else:
                num_heads = 32
        
        # 更新配置
        self.config.slot_pool.hidden_dim = hidden_dim
        if bottleneck_dim:
            self.config.slot_pool.bottleneck_dim = bottleneck_dim
        if num_slots:
            self.config.slot_pool.max_slots = num_slots
        self.config.num_heads = num_heads
        
        # 获取层列表
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")
        
        num_layers = len(layers)
        
        # 转换负索引
        resolved_indices = []
        for idx in layer_indices:
            if idx < 0:
                resolved_indices.append(num_layers + idx)
            else:
                resolved_indices.append(idx)
        
        # 验证配置
        errors = self.config.validate()
        if errors:
            logger.warning(f"Config validation warnings: {errors}")
        
        # 挂载 AGA
        for idx in resolved_indices:
            if idx >= num_layers:
                raise ValueError(f"Layer index {idx} out of range")
            
            aga = AGAOperator(config=self.config)
            aga.eval()
            aga.to(next(model.parameters()).device)
            
            require_attn = self.config.gate.gate1_uncertainty_source.value == "attention_entropy"
            
            self.original_layers[idx] = layers[idx]
            layers[idx] = AGAAugmentedTransformerLayer(
                layers[idx], aga, require_attention_weights=require_attn
            )
            self.aga_modules[idx] = aga
        
        logger.info(f"Attached AGA to layers: {list(self.aga_modules.keys())}")
        return self.aga_modules
    
    def detach_from_model(self, layer_indices: Optional[List[int]] = None):
        """
        从模型卸载 AGA
        
        Args:
            layer_indices: 要卸载的层索引，None 表示全部
        """
        if self.model is None:
            return
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            return
        
        indices = layer_indices or list(self.original_layers.keys())
        
        for idx in indices:
            if idx in self.original_layers:
                layers[idx] = self.original_layers[idx]
                del self.original_layers[idx]
                del self.aga_modules[idx]
        
        logger.info(f"Detached AGA from layers: {indices}")
    
    # ==================== 知识注入 ====================
    
    def inject_knowledge_to_all(
        self,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lu_id: str,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Dict[int, int]:
        """
        向所有 AGA 模块注入知识
        
        Returns:
            层索引到槽位索引的映射
        """
        result = {}
        for layer_idx, aga in self.aga_modules.items():
            slot_idx = aga.find_free_slot()
            if slot_idx is not None:
                aga.inject_knowledge(
                    slot_idx, key_vector, value_vector, lu_id,
                    lifecycle_state, condition, decision
                )
                result[layer_idx] = slot_idx
        return result
    
    def inject_knowledge_to_layer(
        self,
        layer_idx: int,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lu_id: str,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Optional[int]:
        """
        向指定层的 AGA 模块注入知识
        
        Returns:
            槽位索引或 None
        """
        if layer_idx not in self.aga_modules:
            return None
        
        aga = self.aga_modules[layer_idx]
        slot_idx = aga.find_free_slot()
        if slot_idx is not None:
            aga.inject_knowledge(
                slot_idx, key_vector, value_vector, lu_id,
                lifecycle_state, condition, decision
            )
        return slot_idx
    
    # ==================== 生命周期管理 ====================
    
    def update_lifecycle_all(
        self,
        lu_id: str,
        new_state: LifecycleState,
    ) -> Dict[int, List[int]]:
        """
        更新所有层中指定 LU 的生命周期
        
        Returns:
            层索引到更新的槽位索引列表的映射
        """
        result = {}
        for layer_idx, aga in self.aga_modules.items():
            slots = aga.get_slot_by_lu_id(lu_id)
            for slot_idx in slots:
                aga.update_lifecycle(slot_idx, new_state)
            if slots:
                result[layer_idx] = slots
        return result
    
    def quarantine_by_lu_id(self, lu_id: str) -> Dict[int, List[int]]:
        """
        按 LU ID 隔离所有层中的知识
        
        Returns:
            层索引到隔离的槽位索引列表的映射
        """
        result = {}
        for layer_idx, aga in self.aga_modules.items():
            quarantined = aga.quarantine_by_lu_id(lu_id)
            if quarantined:
                result[layer_idx] = quarantined
        return result
    
    # ==================== 衰减管理 ====================
    
    def create_decay_context(self) -> DecayContext:
        """创建新的衰减上下文"""
        self._decay_context = DecayContext()
        return self._decay_context
    
    def get_decay_context(self) -> Optional[DecayContext]:
        """获取当前衰减上下文"""
        return self._decay_context
    
    def reset_decay_context(self):
        """重置衰减上下文"""
        if self._decay_context:
            self._decay_context.reset()
    
    def advance_decay_layer(self):
        """推进衰减层索引"""
        if self._decay_context:
            self._decay_context.layer_idx += 1
    
    # ==================== 统计和查询 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取所有 AGA 模块的统计信息"""
        return {
            'attached_layers': list(self.aga_modules.keys()),
            'total_modules': len(self.aga_modules),
            'per_layer_stats': {
                idx: aga.get_statistics()
                for idx, aga in self.aga_modules.items()
            },
        }
    
    def get_total_active_slots(self) -> int:
        """获取所有层的活跃槽位总数"""
        return sum(aga.get_active_slots() for aga in self.aga_modules.values())
    
    def get_total_hits(self) -> int:
        """获取所有层的总命中数"""
        return sum(
            sum(aga.slot_hit_counts)
            for aga in self.aga_modules.values()
        )
    
    # ==================== 状态导出/导入 ====================
    
    def export_all_states(self) -> Dict[int, Dict[str, Any]]:
        """导出所有 AGA 状态"""
        states = {}
        for idx, aga in self.aga_modules.items():
            states[idx] = {
                'config': self.config.to_dict(),
                'aux_keys': aga.aux_keys.cpu().tolist(),
                'aux_values': aga.aux_values.cpu().tolist(),
                'slot_lifecycle': [s.value for s in aga.slot_lifecycle],
                'slot_lu_ids': aga.slot_lu_ids,
                'slot_conditions': aga.slot_conditions,
                'slot_decisions': aga.slot_decisions,
                'slot_hit_counts': aga.slot_hit_counts,
                'slot_consecutive_misses': aga.slot_consecutive_misses,
            }
        return states
    
    def import_all_states(self, states: Dict[int, Dict[str, Any]]):
        """导入所有 AGA 状态"""
        for idx, state in states.items():
            if idx not in self.aga_modules:
                continue
            
            aga = self.aga_modules[idx]
            device = aga.aux_keys.device
            dtype = aga.aux_keys.dtype
            
            aga.aux_keys.data = torch.tensor(state['aux_keys'], device=device, dtype=dtype)
            aga.aux_values.data = torch.tensor(state['aux_values'], device=device, dtype=dtype)
            aga.slot_lifecycle = [LifecycleState(s) for s in state['slot_lifecycle']]
            aga.slot_lu_ids = state['slot_lu_ids']
            aga.slot_conditions = state.get('slot_conditions', [None] * aga.num_slots)
            aga.slot_decisions = state.get('slot_decisions', [None] * aga.num_slots)
            aga.slot_hit_counts = state.get('slot_hit_counts', [0] * aga.num_slots)
            aga.slot_consecutive_misses = state.get('slot_consecutive_misses', [0] * aga.num_slots)
            aga._invalidate_cache()
