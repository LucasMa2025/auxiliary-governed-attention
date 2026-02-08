"""
AGA 生产级算子

核心设计：
- AGA 是算子，不是服务
- 无状态计算，状态外置
- 每个推理实例独立持有 AGA 副本
- 线程安全 + Fail-Open

版本: v1.3

v1.3 更新:
- 集成统一监控指标系统

v1.2 更新 (大规模知识库支持):
- 集成 ANN 索引层，支持 100K+ 到百万级知识库
- 集成动态知识加载器，支持冷知识按需加载
- 两阶段路由：ANN 粗筛 O(log N) + Gate2 精筛 O(k)
- 保持向后兼容：ANN 默认关闭

v1.1 更新:
- 增强 forward 的超时保护
- 改进异步命中记录的异常处理
- 优化向量形状验证
- 添加持久化操作的错误恢复
"""
import math
import time
import queue
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ProductionAGAConfig, GateConfig, SlotPoolConfig
from .gate import GateChain, GateContext, GateResult, GateDiagnostics
from .slot_pool import SlotSubspacePool, SlotPool, Slot, LifecycleState
from .persistence import PersistenceManager

# 延迟导入 ANN 相关模块
if TYPE_CHECKING:
    from ..retrieval.ann_index import BaseANNIndex
    from ..retrieval.dynamic_loader import DynamicKnowledgeLoader

# 监控模块导入
try:
    from ..monitoring import (
        get_metrics_registry,
        track_latency,
        track_gate,
        record_slot_metrics,
    )
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    get_metrics_registry = None
    track_latency = None
    track_gate = None
    record_slot_metrics = None

logger = logging.getLogger(__name__)


@dataclass
class AGAForwardResult:
    """AGA 前向传播结果"""
    output: torch.Tensor
    diagnostics: Optional[GateDiagnostics] = None
    aga_applied: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    
    # v1.2 新增: 大规模知识库诊断
    ann_candidates: int = 0  # ANN 检索候选数
    ann_search_time_ms: float = 0.0  # ANN 检索耗时
    hot_hits: int = 0  # Hot 层命中数
    warm_hits: int = 0  # Warm 层命中数
    cold_loads: int = 0  # Cold 层加载数


class AGAOperator(nn.Module):
    """
    AGA 生产级算子
    
    🔒 不变量 1：Hot Pool 规模 ≤ 256，保证 Gate2 O(1)
    🔒 不变量 2：AGA 永远是"可绕过"的（Fail-Open）
    🔒 不变量 3：治理、学习、评估永不进入热路径
    
    v1.2 新增:
    - ANN 索引层：支持 100K+ 到百万级知识库
    - 动态加载器：支持冷知识按需加载
    - 两阶段路由：ANN 粗筛 + Gate2 精筛
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
        
        # 延迟分位数统计
        self._latency_history: List[float] = []
        self._latency_history_size = 1000

        # 命中记录线程：避免每次 forward 创建线程
        self._hit_queue: "queue.Queue[List[int]]" = queue.Queue(maxsize=2048)
        self._stop_hits = threading.Event()
        self._hit_worker = threading.Thread(target=self._hit_loop, daemon=True)
        self._hit_worker.start()
        
        # ==================== v1.2 新增: 大规模知识库支持 ====================
        
        # ANN 索引（延迟初始化）
        self.ann_index: Optional["BaseANNIndex"] = None
        if config.ann_index.enabled:
            self._init_ann_index()
        
        # 动态加载器（延迟初始化）
        self.dynamic_loader: Optional["DynamicKnowledgeLoader"] = None
        if config.dynamic_loader.enabled:
            self._init_dynamic_loader()
        
        # ANN 统计
        self._ann_search_count = 0
        self._ann_total_time_ms = 0.0
    
    def _init_ann_index(self):
        """初始化 ANN 索引"""
        try:
            from ..retrieval.ann_index import create_ann_index, ANNIndexConfig
            
            # 转换配置
            ann_config = ANNIndexConfig(
                enabled=self.config.ann_index.enabled,
                backend=self.config.ann_index.backend,
                index_type=self.config.ann_index.index_type,
                index_capacity=self.config.ann_index.index_capacity,
                retrieval_top_k=self.config.ann_index.retrieval_top_k,
                nprobe=self.config.ann_index.nprobe,
                use_gpu=self.config.ann_index.use_gpu,
            )
            
            self.ann_index = create_ann_index(
                ann_config, 
                dim=self.config.slot_pool.bottleneck_dim
            )
            logger.info(f"ANN index initialized: backend={self.config.ann_index.backend.value}")
        except ImportError as e:
            logger.warning(f"Failed to initialize ANN index: {e}")
            self.ann_index = None
    
    def _init_dynamic_loader(self):
        """初始化动态加载器"""
        try:
            from ..retrieval.dynamic_loader import DynamicKnowledgeLoader, DynamicLoaderConfig
            
            loader_config = DynamicLoaderConfig(
                enabled=self.config.dynamic_loader.enabled,
                max_cold_load_per_request=self.config.dynamic_loader.max_cold_load_per_request,
                cold_load_timeout_ms=self.config.dynamic_loader.cold_load_timeout_ms,
                batch_load_size=self.config.dynamic_loader.batch_load_size,
                warm_cache_size=self.config.dynamic_loader.warm_cache_size,
            )
            
            self.dynamic_loader = DynamicKnowledgeLoader(loader_config)
            logger.info("Dynamic knowledge loader initialized")
        except ImportError as e:
            logger.warning(f"Failed to initialize dynamic loader: {e}")
            self.dynamic_loader = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        primary_attention_output: torch.Tensor,
        context: Optional[GateContext] = None,
        logits: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        trace_id: Optional[str] = None,
        query_for_retrieval: Optional[np.ndarray] = None,  # v1.2 新增
    ) -> AGAForwardResult:
        """
        AGA 前向传播
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            primary_attention_output: [batch, seq, hidden_dim] 原始注意力输出
            context: 门控上下文
            logits: [batch, seq, vocab_size] 可选
            return_diagnostics: 是否返回诊断信息
            trace_id: 分布式追踪 ID（可选）
            query_for_retrieval: [bottleneck_dim] ANN 检索用的查询向量（可选）
        
        Returns:
            AGAForwardResult
        """
        start_time = time.perf_counter()
        self._forward_count += 1
        
        # 创建追踪 span（如果启用）
        span = self._start_trace_span("aga_forward", trace_id)
        
        # 默认上下文
        context = context or GateContext(namespace=self.config.namespace)
        
        # v1.2: ANN 检索诊断
        ann_candidates = 0
        ann_search_time_ms = 0.0
        hot_hits = 0
        warm_hits = 0
        cold_loads = 0
        
        try:
            # ==================== v1.2: ANN 预检索 ====================
            if self.ann_index is not None and query_for_retrieval is not None:
                ann_start = time.perf_counter()
                
                candidate_lu_ids, ann_scores = self.ann_index.search(
                    query_for_retrieval,
                    top_k=self.config.ann_index.retrieval_top_k
                )
                
                ann_search_time_ms = (time.perf_counter() - ann_start) * 1000
                ann_candidates = len(candidate_lu_ids)
                self._ann_search_count += 1
                self._ann_total_time_ms += ann_search_time_ms
                
                # 动态加载候选到 Hot Pool
                if self.dynamic_loader is not None and candidate_lu_ids:
                    load_result = self.dynamic_loader.load_candidates(candidate_lu_ids)
                    hot_hits = load_result.hot_hits
                    warm_hits = load_result.warm_hits
                    cold_loads = load_result.cold_loads
            
            # ==================== 原有逻辑 ====================
            
            # 检查是否有活跃槽位
            if self.slot_pool.active_count == 0:
                return self._bypass_result(primary_attention_output, start_time)
            
            # 获取槽位向量
            with self._slot_lock:
                slot_keys, slot_values, reliability = self.slot_pool.get_vectors()
            
            if slot_keys.shape[0] == 0:
                return self._bypass_result(primary_attention_output, start_time)
            
            # 查询投影
            query = self.q_proj(hidden_states)
            
            # 计算可靠性掩码
            reliability_mask = reliability.clamp_min(1e-10).log()
            
            # 门控链
            top_indices, final_gate, diagnostics = self.gate_chain(
                context=context,
                hidden_states=hidden_states,
                slot_keys=slot_keys,
                reliability_mask=reliability_mask,
                logits=logits,
                query=query,
            )
            
            # 检查是否通过门控
            if top_indices is None or final_gate is None or top_indices.numel() == 0:
                return AGAForwardResult(
                    output=primary_attention_output,
                    diagnostics=diagnostics if return_diagnostics else None,
                    aga_applied=False,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    ann_candidates=ann_candidates,
                    ann_search_time_ms=ann_search_time_ms,
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
                self._enqueue_hits(hit_indices)
            
            self._aga_applied_count += 1
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._total_latency_ms += latency_ms
            
            # 记录延迟历史（用于分位数计算）
            self._latency_history.append(latency_ms)
            if len(self._latency_history) > self._latency_history_size:
                self._latency_history.pop(0)

            if latency_ms > self.config.max_forward_timeout_ms:
                logger.warning(
                    "AGA forward slower than timeout: %.2fms > %dms",
                    latency_ms,
                    self.config.max_forward_timeout_ms,
                )
            
            # 记录 span 属性
            if span:
                span.set_attribute("namespace", self.config.namespace)
                span.set_attribute("slot_count", self.slot_pool.active_count)
                span.set_attribute("aga_applied", True)
                span.set_attribute("latency_ms", latency_ms)
                span.set_attribute("ann_candidates", ann_candidates)
            
            return AGAForwardResult(
                output=fused_output,
                diagnostics=diagnostics if return_diagnostics else None,
                aga_applied=True,
                latency_ms=latency_ms,
                ann_candidates=ann_candidates,
                ann_search_time_ms=ann_search_time_ms,
                hot_hits=hot_hits,
                warm_hits=warm_hits,
                cold_loads=cold_loads,
            )
            
        except Exception as e:
            # Fail-Open: 任何异常直接返回原输出
            self._fail_open_count += 1
            logger.warning(f"AGA fail-open: {e}")
            
            # 记录异常到 span
            if span:
                span.record_exception(e)
                span.set_attribute("aga_applied", False)
                span.set_attribute("fail_open", True)
            
            return AGAForwardResult(
                output=primary_attention_output,
                aga_applied=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )
        finally:
            # 结束 span
            if span:
                span.end()

    def _bypass_result(self, primary_attention_output: torch.Tensor, start_time: float) -> AGAForwardResult:
        return AGAForwardResult(
            output=primary_attention_output,
            aga_applied=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
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
        
        selected_keys = slot_keys.index_select(0, flat_indices)  # [batch*seq*k, bottleneck_dim]
        selected_values = slot_values.index_select(0, flat_indices)  # [batch*seq*k, hidden_dim]
        
        # 重塑
        selected_keys = selected_keys.view(batch_size, seq_len, k, -1)
        selected_values = selected_values.view(batch_size, seq_len, k, -1)
        
        # 计算注意力分数
        attn_scores = torch.einsum("bsd,bskd->bsk", query, selected_keys)
        attn_scores = attn_scores / math.sqrt(self.config.slot_pool.bottleneck_dim)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq, k]
        
        # 加权求和
        aux_output = torch.einsum("bsk,bskh->bsh", attn_weights, selected_values)
        
        return aux_output
    
    def _enqueue_hits(self, hit_indices: List[int]):
        if not hit_indices:
            return
        try:
            self._hit_queue.put_nowait(hit_indices)
        except queue.Full:
            logger.warning("hit queue full, drop %d hit indices", len(hit_indices))

    def _hit_loop(self):
        while not self._stop_hits.is_set():
            try:
                hit_indices = self._hit_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            batch = set(hit_indices)
            while True:
                try:
                    batch.update(self._hit_queue.get_nowait())
                except queue.Empty:
                    break

            try:
                with self._slot_lock:
                    self.slot_pool.record_hits(list(batch))
            except Exception as e:
                logger.warning(f"Failed to record hits: {e}")

    def shutdown(self, timeout: float = 1.0):
        self._stop_hits.set()
        if self._hit_worker.is_alive():
            self._hit_worker.join(timeout=timeout)

    def __del__(self):
        self.shutdown(timeout=0.1)
    
    # ==================== 分布式追踪支持 ====================
    
    def enable_tracing(self, service_name: str = "aga"):
        """启用 OpenTelemetry 追踪"""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(__name__)
            logger.info(f"OpenTelemetry tracing enabled for service: {service_name}")
        except ImportError:
            logger.warning("OpenTelemetry not installed, tracing disabled")
            self._tracer = None
    
    def _start_trace_span(self, name: str, trace_id: Optional[str]):
        """启动追踪 span（兼容 OpenTelemetry）"""
        if not hasattr(self, '_tracer') or self._tracer is None:
            return None
        
        try:
            from opentelemetry import trace
            
            if trace_id:
                # 从父追踪继承（简化实现）
                return self._tracer.start_span(name, attributes={"trace_id": trace_id})
            else:
                return self._tracer.start_span(name)
        except Exception as e:
            logger.debug(f"Failed to start trace span: {e}")
            return None
    
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
        """
        注入知识（带向量验证）
        
        v1.2: 同时更新 ANN 索引
        """
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
        
        if key_size != expected_key_dim:
            logger.warning(
                "inject_knowledge: key vector size mismatch for lu_id=%s, got=%d expected=%d",
                lu_id,
                key_size,
                expected_key_dim,
            )
            return None
        
        if value_size != expected_value_dim:
            logger.warning(
                "inject_knowledge: value vector size mismatch for lu_id=%s, got=%d expected=%d",
                lu_id,
                value_size,
                expected_value_dim,
            )
            return None
        
        with self._slot_lock:
            slot_idx = self.slot_pool.add_slot(
                lu_id=lu_id,
                key_vector=key_vector,
                value_vector=value_vector,
                lifecycle_state=lifecycle_state,
                condition=condition,
                decision=decision,
            )
        
        # v1.2: 更新 ANN 索引
        if slot_idx is not None and self.ann_index is not None:
            try:
                key_np = key_vector.detach().cpu().numpy().astype(np.float32)
                self.ann_index.add(lu_id, key_np)
            except Exception as e:
                logger.warning(f"Failed to add to ANN index: {e}")
        
        return slot_idx
    
    def quarantine_knowledge(self, lu_id: str) -> bool:
        """
        隔离知识
        
        v1.2: 同时从 ANN 索引移除
        """
        with self._slot_lock:
            result = self.slot_pool.quarantine_slot(lu_id)
        
        # v1.2: 从 ANN 索引移除
        if result and self.ann_index is not None:
            try:
                self.ann_index.remove(lu_id)
            except Exception as e:
                logger.warning(f"Failed to remove from ANN index: {e}")
        
        # v1.2: 使动态加载器缓存失效
        if result and self.dynamic_loader is not None:
            self.dynamic_loader.invalidate(lu_id)
        
        return result
    
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
        """获取统计信息（含延迟分位数和 ANN 统计）并更新监控指标"""
        with self._slot_lock:
            pool_stats = self.slot_pool.get_statistics()
        
        avg_latency = self._total_latency_ms / self._aga_applied_count if self._aga_applied_count > 0 else 0
        
        # 计算延迟分位数
        latency_percentiles = {}
        if self._latency_history:
            sorted_latencies = sorted(self._latency_history)
            n = len(sorted_latencies)
            latency_percentiles = {
                "p50": sorted_latencies[int(n * 0.5)] if n > 0 else 0,
                "p90": sorted_latencies[int(n * 0.9)] if n > 0 else 0,
                "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
                "p99": sorted_latencies[min(int(n * 0.99), n - 1)] if n > 0 else 0,
                "max": sorted_latencies[-1] if n > 0 else 0,
            }
        
        stats = {
            "namespace": self.config.namespace,
            "forward_count": self._forward_count,
            "aga_applied_count": self._aga_applied_count,
            "aga_applied_ratio": self._aga_applied_count / self._forward_count if self._forward_count > 0 else 0,
            "fail_open_count": self._fail_open_count,
            "fail_open_ratio": self._fail_open_count / self._forward_count if self._forward_count > 0 else 0,
            "latency": {
                "avg_ms": avg_latency,
                **latency_percentiles,
            },
            "pool": pool_stats,
        }
        
        # v1.2: ANN 统计
        if self.ann_index is not None:
            stats["ann_index"] = {
                "enabled": True,
                "search_count": self._ann_search_count,
                "avg_search_time_ms": self._ann_total_time_ms / max(1, self._ann_search_count),
                **self.ann_index.get_statistics(namespace=self.config.namespace),
            }
        else:
            stats["ann_index"] = {"enabled": False}
        
        # v1.2: 动态加载器统计
        if self.dynamic_loader is not None:
            stats["dynamic_loader"] = self.dynamic_loader.get_statistics(
                namespace=self.config.namespace
            )
        else:
            stats["dynamic_loader"] = {"enabled": False}
        
        # v1.3: 更新监控指标
        if HAS_MONITORING:
            self._update_monitoring_metrics(stats, pool_stats)
        
        return stats
    
    def _update_monitoring_metrics(self, stats: Dict[str, Any], pool_stats: Dict[str, Any]):
        """更新监控指标"""
        try:
            registry = get_metrics_registry()
            if registry is None or not registry.enabled:
                return
            
            namespace = self.config.namespace
            
            # 槽位指标
            record_slot_metrics(
                namespace=namespace,
                total_slots=pool_stats.get("total_slots", 0),
                active_slots=pool_stats.get("active_slots", 0),
                slots_by_state=pool_stats.get("by_lifecycle_state", {}),
            )
            
            # 槽位命中率
            hit_rate = stats.get("aga_applied_ratio", 0.0)
            registry.get_metric("slot_hit_rate").labels(
                namespace=namespace
            ).set(hit_rate)
            
        except Exception as e:
            logger.debug(f"Failed to update monitoring metrics: {e}")


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
                self._operators[namespace].shutdown()
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

