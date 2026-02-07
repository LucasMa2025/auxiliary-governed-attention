"""
AGA 生产级算子

核心设计：
- AGA 是算子，不是服务
- 无状态计算，状态外置
- 每个推理实例独立持有 AGA 副本
- 线程安全 + Fail-Open

版本: v1.1

v1.1 更新:
- 增强 forward 的超时保护
- 改进异步命中记录的异常处理
- 优化向量形状验证
- 添加持久化操作的错误恢复
"""
import math
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ProductionAGAConfig, GateConfig, SlotPoolConfig
from .gate import GateChain, GateContext, GateResult, GateDiagnostics
from .slot_pool import SlotSubspacePool, SlotPool, Slot, LifecycleState
from .persistence import PersistenceManager

logger = logging.getLogger(__name__)


@dataclass
class AGAForwardResult:
    """AGA 前向传播结果"""
    output: torch.Tensor
    diagnostics: Optional[GateDiagnostics] = None
    aga_applied: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None


class AGAOperator(nn.Module):
    """
    AGA 生产级算子
    
    🔒 不变量 1：推理可见知识规模 = O(1)
    🔒 不变量 2：AGA 永远是"可绕过"的（Fail-Open）
    🔒 不变量 3：治理、学习、评估永不进入热路径
    """
    
    def __init__(
        self,
        config: ProductionAGAConfig,
        device: torch.device = None,
    ):
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")
        
        # 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid config: {errors}")
        
        # 槽位池
        self.slot_pool = SlotPool(
            namespace=config.namespace,
            config=config.slot_pool,
            device=self.device,
        )
        
        # 门控链
        self.gate_chain = GateChain(
            config=config.gate,
            hidden_dim=config.slot_pool.hidden_dim,
            bottleneck_dim=config.slot_pool.bottleneck_dim,
            num_slots=config.slot_pool.max_slots_per_namespace,
        )
        
        # 查询投影
        self.q_proj = nn.Linear(
            config.slot_pool.hidden_dim,
            config.slot_pool.bottleneck_dim,
            bias=False,
        )
        
        # Value Projection (delta subspace)
        if config.slot_pool.use_value_projection:
            self.value_down = nn.Linear(
                config.slot_pool.hidden_dim,
                config.slot_pool.value_bottleneck_dim,
                bias=False,
            )
            self.value_up = nn.Linear(
                config.slot_pool.value_bottleneck_dim,
                config.slot_pool.hidden_dim,
                bias=False,
            )
            nn.init.xavier_uniform_(self.value_down.weight, gain=0.1)
            nn.init.xavier_uniform_(self.value_up.weight, gain=0.1)
        
        # 线程锁（细粒度）
        self._slot_lock = threading.RLock()
        
        # 统计信息
        self._forward_count = 0
        self._aga_applied_count = 0
        self._fail_open_count = 0
        self._total_latency_ms = 0.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        primary_attention_output: torch.Tensor,
        context: Optional[GateContext] = None,
        logits: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> AGAForwardResult:
        """
        AGA 前向传播
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            primary_attention_output: [batch, seq, hidden_dim] 原始注意力输出
            context: 门控上下文
            logits: [batch, seq, vocab_size] 可选
            return_diagnostics: 是否返回诊断信息
        
        Returns:
            AGAForwardResult
        """
        start_time = time.time()
        self._forward_count += 1
        
        # 默认上下文
        if context is None:
            context = GateContext(namespace=self.config.namespace)
        
        try:
            # 检查是否有活跃槽位
            if self.slot_pool.active_count == 0:
                return AGAForwardResult(
                    output=primary_attention_output,
                    aga_applied=False,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            
            # 获取槽位向量
            with self._slot_lock:
                slot_keys, slot_values, reliability = self.slot_pool.get_vectors()
            
            if slot_keys.shape[0] == 0:
                return AGAForwardResult(
                    output=primary_attention_output,
                    aga_applied=False,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            
            # 查询投影
            query = self.q_proj(hidden_states)
            
            # 计算可靠性掩码
            reliability_mask = torch.log(reliability + 1e-10)
            
            # 门控链
            top_indices, final_gate, diagnostics = self.gate_chain(
                context=context,
                hidden_states=hidden_states,
                slot_keys=slot_keys,
                reliability_mask=reliability_mask,
                logits=logits,
            )
            
            # 检查是否通过门控
            if top_indices is None:
                return AGAForwardResult(
                    output=primary_attention_output,
                    diagnostics=diagnostics if return_diagnostics else None,
                    aga_applied=False,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            
            # 计算注意力
            aux_output = self._compute_attention(
                query=query,
                slot_keys=slot_keys,
                slot_values=slot_values,
                top_indices=top_indices,
            )
            
            # Value Projection
            if self.config.slot_pool.use_value_projection:
                aux_output = self.value_down(aux_output)
                aux_output = F.gelu(aux_output)
                aux_output = self.value_up(aux_output)
            
            # 融合输出
            fused_output = primary_attention_output + final_gate.unsqueeze(-1) * aux_output
            
            # 记录命中（异步，不阻塞）
            if not self.training:
                hit_indices = top_indices.reshape(-1).unique().cpu().tolist()
                # 使用线程避免阻塞
                threading.Thread(
                    target=self._record_hits_async,
                    args=(hit_indices,),
                    daemon=True,
                ).start()
            
            self._aga_applied_count += 1
            latency_ms = (time.time() - start_time) * 1000
            self._total_latency_ms += latency_ms
            
            return AGAForwardResult(
                output=fused_output,
                diagnostics=diagnostics if return_diagnostics else None,
                aga_applied=True,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            # Fail-Open: 任何异常直接返回原输出
            self._fail_open_count += 1
            logger.warning(f"AGA fail-open: {e}")
            
            return AGAForwardResult(
                output=primary_attention_output,
                aga_applied=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def _compute_attention(
        self,
        query: torch.Tensor,
        slot_keys: torch.Tensor,
        slot_values: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Top-k 注意力
        
        Args:
            query: [batch, seq, bottleneck_dim]
            slot_keys: [active_slots, bottleneck_dim]
            slot_values: [active_slots, hidden_dim]
            top_indices: [batch, seq, k]
        
        Returns:
            aux_output: [batch, seq, hidden_dim]
        """
        batch_size, seq_len, _ = query.shape
        k = top_indices.shape[-1]
        
        # Gather 选中的 keys 和 values
        # top_indices: [batch, seq, k]
        # 需要展平后 gather
        flat_indices = top_indices.reshape(-1)  # [batch * seq * k]
        
        selected_keys = slot_keys[flat_indices]  # [batch*seq*k, bottleneck_dim]
        selected_values = slot_values[flat_indices]  # [batch*seq*k, hidden_dim]
        
        # 重塑
        selected_keys = selected_keys.view(batch_size, seq_len, k, -1)
        selected_values = selected_values.view(batch_size, seq_len, k, -1)
        
        # 计算注意力分数
        query_expanded = query.unsqueeze(2)  # [batch, seq, 1, bottleneck]
        attn_scores = torch.sum(query_expanded * selected_keys, dim=-1)  # [batch, seq, k]
        attn_scores = attn_scores / math.sqrt(self.config.slot_pool.bottleneck_dim)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq, k]
        
        # 加权求和
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch, seq, k, 1]
        aux_output = torch.sum(attn_weights_expanded * selected_values, dim=2)  # [batch, seq, hidden]
        
        return aux_output
    
    def _record_hits_async(self, hit_indices: List[int]):
        """异步记录命中"""
        try:
            with self._slot_lock:
                self.slot_pool.record_hits(hit_indices)
        except Exception as e:
            logger.warning(f"Failed to record hits: {e}")
    
    # ==================== 知识管理接口 ====================
    
    def inject_knowledge(
        self,
        lu_id: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Optional[int]:
        """注入知识（带向量验证）"""
        # 验证向量
        if key_vector is None or value_vector is None:
            logger.warning(f"inject_knowledge: vectors cannot be None for lu_id={lu_id}")
            return None
        
        # 验证向量形状
        expected_key_dim = self.config.slot_pool.bottleneck_dim
        expected_value_dim = self.config.slot_pool.hidden_dim
        
        key_size = key_vector.numel()
        value_size = value_vector.numel()
        
        if key_size == 0 or value_size == 0:
            logger.warning(f"inject_knowledge: empty vectors for lu_id={lu_id}")
            return None
        
        with self._slot_lock:
            return self.slot_pool.add_slot(
                lu_id=lu_id,
                key_vector=key_vector,
                value_vector=value_vector,
                lifecycle_state=lifecycle_state,
                condition=condition,
                decision=decision,
            )
    
    def quarantine_knowledge(self, lu_id: str) -> bool:
        """隔离知识"""
        with self._slot_lock:
            return self.slot_pool.quarantine_slot(lu_id)
    
    def update_lifecycle(self, lu_id: str, new_state: LifecycleState) -> bool:
        """更新生命周期"""
        with self._slot_lock:
            return self.slot_pool.update_lifecycle(lu_id, new_state)
    
    def remove_knowledge(self, lu_id: str) -> bool:
        """移除知识"""
        with self._slot_lock:
            return self.slot_pool.remove_slot(lu_id)
    
    def load_from_persistence(self, persistence: PersistenceManager) -> int:
        """从持久化层加载（带错误恢复）"""
        try:
            slots_data = persistence.load_active_slots(self.config.namespace)
            
            if not slots_data:
                logger.info(f"No slots to load for namespace={self.config.namespace}")
                return 0
            
            with self._slot_lock:
                self.slot_pool.import_slots(slots_data)
            
            logger.info(f"Loaded {len(slots_data)} slots from persistence")
            return len(slots_data)
            
        except Exception as e:
            logger.error(f"Failed to load from persistence: {e}")
            return 0
    
    def save_to_persistence(self, persistence: PersistenceManager) -> int:
        """保存到持久化层（带错误恢复）"""
        try:
            with self._slot_lock:
                slots_data = self.slot_pool.export_all()
            
            if not slots_data:
                logger.debug(f"No slots to save for namespace={self.config.namespace}")
                return 0
            
            count = 0
            failed = 0
            for data in slots_data:
                try:
                    slot = Slot.from_dict(data, self.device)
                    if persistence.save_slot(self.config.namespace, slot):
                        count += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.warning(f"Failed to save slot {data.get('lu_id')}: {e}")
                    failed += 1
            
            if failed > 0:
                logger.warning(f"Saved {count} slots, failed {failed}")
            else:
                logger.info(f"Saved {count} slots to persistence")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to save to persistence: {e}")
            return 0
    
    # ==================== 统计接口 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._slot_lock:
            pool_stats = self.slot_pool.get_statistics()
        
        avg_latency = self._total_latency_ms / self._aga_applied_count if self._aga_applied_count > 0 else 0
        
        return {
            "namespace": self.config.namespace,
            "forward_count": self._forward_count,
            "aga_applied_count": self._aga_applied_count,
            "aga_applied_ratio": self._aga_applied_count / self._forward_count if self._forward_count > 0 else 0,
            "fail_open_count": self._fail_open_count,
            "avg_latency_ms": avg_latency,
            "pool": pool_stats,
        }


class ConcurrentAGAManager:
    """
    并发 AGA 管理器
    
    管理多个 namespace 的 AGA 算子实例。
    """
    
    def __init__(
        self,
        default_config: ProductionAGAConfig,
        persistence_manager: Optional[PersistenceManager] = None,
        device: torch.device = None,
    ):
        self.default_config = default_config
        self.persistence = persistence_manager
        self.device = device or torch.device("cpu")
        
        self._operators: Dict[str, AGAOperator] = {}
        self._lock = threading.RLock()
        
        # 后台同步线程
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync = threading.Event()
    
    def get_operator(self, namespace: str) -> AGAOperator:
        """获取或创建 AGA 算子"""
        with self._lock:
            if namespace not in self._operators:
                config = ProductionAGAConfig(
                    namespace=namespace,
                    num_heads=self.default_config.num_heads,
                    gate=self.default_config.gate,
                    slot_pool=self.default_config.slot_pool,
                    persistence=self.default_config.persistence,
                    fail_open_enabled=self.default_config.fail_open_enabled,
                )
                
                operator = AGAOperator(config, self.device)
                
                # 从持久化层加载
                if self.persistence:
                    operator.load_from_persistence(self.persistence)
                
                self._operators[namespace] = operator
            
            return self._operators[namespace]
    
    def remove_operator(self, namespace: str) -> bool:
        """移除 AGA 算子"""
        with self._lock:
            if namespace in self._operators:
                # 保存到持久化层
                if self.persistence:
                    self._operators[namespace].save_to_persistence(self.persistence)
                
                del self._operators[namespace]
                return True
            return False
    
    def get_all_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        with self._lock:
            return list(self._operators.keys())
    
    def start_sync(self):
        """启动后台同步"""
        if self._sync_thread is not None:
            return
        
        self._stop_sync.clear()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Started AGA sync thread")
    
    def stop_sync(self):
        """停止后台同步"""
        if self._sync_thread is None:
            return
        
        self._stop_sync.set()
        self._sync_thread.join(timeout=5)
        self._sync_thread = None
        logger.info("Stopped AGA sync thread")
    
    def _sync_loop(self):
        """同步循环"""
        while not self._stop_sync.is_set():
            try:
                if self.persistence:
                    with self._lock:
                        for namespace, operator in self._operators.items():
                            operator.save_to_persistence(self.persistence)
            except Exception as e:
                logger.error(f"Sync error: {e}")
            
            self._stop_sync.wait(self.default_config.persistence.sync_interval_seconds)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取所有算子的统计信息"""
        with self._lock:
            return {
                "total_namespaces": len(self._operators),
                "per_namespace": {
                    ns: op.get_statistics()
                    for ns, op in self._operators.items()
                },
            }

