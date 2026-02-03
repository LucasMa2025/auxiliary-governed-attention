"""
AGA 知识写入器 (单机部署模式)

⚠️ 重要：本模块专为单机部署设计
=========================================

本模块通过 **进程内直接调用** 写入知识到本地 AGA 实例：
- 直接操作 ConcurrentAGAManager (内存中的槽位池)
- 直接操作 PersistenceManager (本地持久化)
- 使用 Python threading 实现异步写入队列

如需分离部署（API 服务与推理服务分开），请使用：
- aga.portal.PortalService: 通过 HTTP API 管理知识元数据
- aga.client.AGAClient: 外部系统访问 Portal 的 HTTP 客户端
- aga.sync: Portal 通过消息队列同步到 Runtime

架构对比：
---------
单机部署 (本模块):
    治理系统 ─── KnowledgeWriter ─── AGAOperator (同进程)
                     │
                     └─── PersistenceManager (本地)

分离部署 (aga.portal + aga.runtime):
    治理系统 ─── AGAClient ─── Portal API ─── PortalService
                                                    │
                                              Redis Pub/Sub
                                                    ▼
                                              RuntimeAgent (订阅)
                                                    │
                                              LocalCache + AGARuntime

设计原则：
- 写入操作异步，不阻塞推理
- 支持批量写入
- 支持质量评估（否决型）
- 完整的审计日志 (通过 PersistenceManager)
"""
import time
import logging
import threading
import queue
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

import torch

from .config import ProductionAGAConfig
from .slot_pool import Slot, LifecycleState
from .persistence import PersistenceManager
from .operator import AGAOperator, ConcurrentAGAManager

logger = logging.getLogger(__name__)


class WriteStatus(str, Enum):
    """写入状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"  # 被质量评估拒绝


@dataclass
class WriteRequest:
    """写入请求"""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    namespace: str = "default"
    lu_id: str = ""
    condition: str = ""
    decision: str = ""
    key_vector: Optional[torch.Tensor] = None
    value_vector: Optional[torch.Tensor] = None
    lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY
    
    # 元数据
    source: str = "manual"  # manual, self_learning, feedback
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """验证请求"""
        errors = []
        
        if not self.lu_id:
            errors.append("lu_id is required")
        
        if self.key_vector is None:
            errors.append("key_vector is required")
        
        if self.value_vector is None:
            errors.append("value_vector is required")
        
        return errors


@dataclass
class WriteResult:
    """写入结果"""
    request_id: str
    status: WriteStatus
    slot_idx: Optional[int] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    
    # 质量评估结果
    quality_score: Optional[float] = None
    quality_reason: Optional[str] = None


class QualityAssessor:
    """
    质量评估器（否决型）
    
    功能：
    - 向量范数检查
    - 重复检测
    - 安全分类检查
    
    集成了 SafetyClassifier 进行内容安全检测。
    """
    
    def __init__(
        self,
        max_key_norm: float = 100.0,
        max_value_norm: float = 100.0,
        min_key_norm: float = 0.01,
        min_value_norm: float = 0.01,
        enable_safety_check: bool = True,
        safety_classifier=None,
    ):
        self.max_key_norm = max_key_norm
        self.max_value_norm = max_value_norm
        self.min_key_norm = min_key_norm
        self.min_value_norm = min_value_norm
        self.enable_safety_check = enable_safety_check
        
        # 安全分类器
        self._safety_classifier = safety_classifier
        if self.enable_safety_check and self._safety_classifier is None:
            try:
                from .safety import RuleBasedSafetyClassifier
                self._safety_classifier = RuleBasedSafetyClassifier()
            except ImportError:
                logger.warning("Safety classifier not available")
                self.enable_safety_check = False
    
    def assess(self, request: WriteRequest) -> tuple[bool, float, str]:
        """
        评估写入请求
        
        Returns:
            (passed, score, reason)
        """
        # 1. 向量范数检查
        key_norm = request.key_vector.norm().item()
        value_norm = request.value_vector.norm().item()
        
        if key_norm > self.max_key_norm:
            return False, 0.0, f"key_norm={key_norm:.2f} exceeds max={self.max_key_norm}"
        
        if value_norm > self.max_value_norm:
            return False, 0.0, f"value_norm={value_norm:.2f} exceeds max={self.max_value_norm}"
        
        if key_norm < self.min_key_norm:
            return False, 0.0, f"key_norm={key_norm:.6f} below min={self.min_key_norm}"
        
        if value_norm < self.min_value_norm:
            return False, 0.0, f"value_norm={value_norm:.6f} below min={self.min_value_norm}"
        
        # 2. NaN/Inf 检查
        if torch.isnan(request.key_vector).any() or torch.isinf(request.key_vector).any():
            return False, 0.0, "key_vector contains NaN or Inf"
        
        if torch.isnan(request.value_vector).any() or torch.isinf(request.value_vector).any():
            return False, 0.0, "value_vector contains NaN or Inf"
        
        # 3. 安全分类检查
        if self.enable_safety_check and self._safety_classifier:
            safety_result = self._safety_classifier.classify_knowledge(
                condition=request.condition or "",
                decision=request.decision or "",
                metadata=request.metadata,
            )
            
            if safety_result.is_blocked:
                return False, 0.0, f"Safety blocked: {safety_result.suggestion}"
            
            if safety_result.needs_review:
                # 中等风险：返回通过但降低分数，标记需要审核
                base_score = 0.5
                return True, base_score * safety_result.score, f"needs_review: {safety_result.suggestion}"
        
        # 通过所有检查
        score = 1.0 - (key_norm / self.max_key_norm) * 0.2 - (value_norm / self.max_value_norm) * 0.2
        return True, score, "passed"
    
    def set_safety_classifier(self, classifier):
        """设置安全分类器"""
        self._safety_classifier = classifier
        self.enable_safety_check = classifier is not None


class KnowledgeWriter:
    """
    知识写入器
    
    提供异步写入能力，与推理路径解耦。
    """
    
    def __init__(
        self,
        aga_manager: ConcurrentAGAManager,
        persistence: Optional[PersistenceManager] = None,
        max_queue_size: int = 1000,
        num_workers: int = 2,
        enable_quality_assessment: bool = True,
    ):
        self.aga_manager = aga_manager
        self.persistence = persistence
        self.enable_quality_assessment = enable_quality_assessment
        
        # 质量评估器
        self.quality_assessor = QualityAssessor()
        
        # 写入队列
        self._queue: queue.Queue[WriteRequest] = queue.Queue(maxsize=max_queue_size)
        self._results: Dict[str, WriteResult] = {}
        self._results_lock = threading.Lock()
        
        # 工作线程
        self._workers: List[threading.Thread] = []
        self._stop_workers = threading.Event()
        self._num_workers = num_workers
        
        # 统计信息
        self._total_requests = 0
        self._success_count = 0
        self._rejected_count = 0
        self._failed_count = 0
        
        # 回调
        self._on_write_complete: Optional[Callable[[WriteResult], None]] = None
    
    def start(self):
        """启动写入器"""
        self._stop_workers.clear()
        
        for i in range(self._num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"KnowledgeWriter-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started KnowledgeWriter with {self._num_workers} workers")
    
    def stop(self):
        """停止写入器"""
        self._stop_workers.set()
        
        for worker in self._workers:
            worker.join(timeout=5)
        
        self._workers.clear()
        logger.info("Stopped KnowledgeWriter")
    
    def submit(self, request: WriteRequest) -> str:
        """
        提交写入请求（异步）
        
        Returns:
            request_id
        """
        # 验证请求
        errors = request.validate()
        if errors:
            result = WriteResult(
                request_id=request.request_id,
                status=WriteStatus.FAILED,
                error=f"Validation failed: {errors}",
            )
            self._store_result(result)
            return request.request_id
        
        # 加入队列
        try:
            self._queue.put_nowait(request)
            self._total_requests += 1
            
            # 初始化结果状态
            result = WriteResult(
                request_id=request.request_id,
                status=WriteStatus.PENDING,
            )
            self._store_result(result)
            
            return request.request_id
        except queue.Full:
            result = WriteResult(
                request_id=request.request_id,
                status=WriteStatus.FAILED,
                error="Write queue is full",
            )
            self._store_result(result)
            return request.request_id
    
    def submit_sync(self, request: WriteRequest) -> WriteResult:
        """
        同步提交写入请求
        
        Returns:
            WriteResult
        """
        return self._process_request(request)
    
    def get_result(self, request_id: str) -> Optional[WriteResult]:
        """获取写入结果"""
        with self._results_lock:
            return self._results.get(request_id)
    
    def wait_for_result(self, request_id: str, timeout: float = 30.0) -> Optional[WriteResult]:
        """等待写入结果"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.get_result(request_id)
            if result and result.status not in (WriteStatus.PENDING, WriteStatus.PROCESSING):
                return result
            time.sleep(0.1)
        
        return self.get_result(request_id)
    
    def set_on_write_complete(self, callback: Callable[[WriteResult], None]):
        """设置写入完成回调"""
        self._on_write_complete = callback
    
    def _worker_loop(self):
        """工作线程循环"""
        while not self._stop_workers.is_set():
            try:
                request = self._queue.get(timeout=1.0)
                
                # 更新状态为处理中
                self._update_result_status(request.request_id, WriteStatus.PROCESSING)
                
                # 处理请求
                result = self._process_request(request)
                
                # 存储结果
                self._store_result(result)
                
                # 触发回调
                if self._on_write_complete:
                    try:
                        self._on_write_complete(result)
                    except Exception as e:
                        logger.warning(f"Write complete callback failed: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _process_request(self, request: WriteRequest) -> WriteResult:
        """处理写入请求"""
        start_time = time.time()
        
        try:
            # 1. 质量评估
            if self.enable_quality_assessment:
                passed, score, reason = self.quality_assessor.assess(request)
                
                if not passed:
                    self._rejected_count += 1
                    return WriteResult(
                        request_id=request.request_id,
                        status=WriteStatus.REJECTED,
                        quality_score=score,
                        quality_reason=reason,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
            else:
                score = 1.0
                reason = "assessment_disabled"
            
            # 2. 获取 AGA 算子
            operator = self.aga_manager.get_operator(request.namespace)
            
            # 3. 注入知识
            slot_idx = operator.inject_knowledge(
                lu_id=request.lu_id,
                key_vector=request.key_vector,
                value_vector=request.value_vector,
                lifecycle_state=request.lifecycle_state,
                condition=request.condition,
                decision=request.decision,
            )
            
            if slot_idx is None:
                self._failed_count += 1
                return WriteResult(
                    request_id=request.request_id,
                    status=WriteStatus.FAILED,
                    error="No free slot available",
                    quality_score=score,
                    quality_reason=reason,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            
            # 4. 持久化
            if self.persistence:
                slot = Slot(
                    slot_idx=slot_idx,
                    lu_id=request.lu_id,
                    key_vector=request.key_vector,
                    value_vector=request.value_vector,
                    lifecycle_state=request.lifecycle_state,
                    condition=request.condition,
                    decision=request.decision,
                    namespace=request.namespace,
                )
                self.persistence.save_slot(request.namespace, slot)
            
            self._success_count += 1
            return WriteResult(
                request_id=request.request_id,
                status=WriteStatus.SUCCESS,
                slot_idx=slot_idx,
                quality_score=score,
                quality_reason=reason,
                latency_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            self._failed_count += 1
            return WriteResult(
                request_id=request.request_id,
                status=WriteStatus.FAILED,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    def _store_result(self, result: WriteResult):
        """存储结果"""
        with self._results_lock:
            self._results[result.request_id] = result
    
    def _update_result_status(self, request_id: str, status: WriteStatus):
        """更新结果状态"""
        with self._results_lock:
            if request_id in self._results:
                self._results[request_id].status = status
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self._total_requests,
            "success_count": self._success_count,
            "rejected_count": self._rejected_count,
            "failed_count": self._failed_count,
            "queue_size": self._queue.qsize(),
            "pending_results": len([
                r for r in self._results.values()
                if r.status in (WriteStatus.PENDING, WriteStatus.PROCESSING)
            ]),
        }
    
    # ==================== 便捷方法 ====================
    
    def write_knowledge(
        self,
        namespace: str,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
        source: str = "manual",
        sync: bool = False,
    ) -> WriteResult:
        """
        写入知识（便捷方法）
        
        Args:
            namespace: 命名空间
            lu_id: Learning Unit ID
            condition: 条件描述
            decision: 决策描述
            key_vector: Key 向量
            value_vector: Value 向量
            lifecycle_state: 生命周期状态
            source: 来源
            sync: 是否同步等待结果
        
        Returns:
            WriteResult
        """
        request = WriteRequest(
            namespace=namespace,
            lu_id=lu_id,
            condition=condition,
            decision=decision,
            key_vector=key_vector,
            value_vector=value_vector,
            lifecycle_state=lifecycle_state,
            source=source,
        )
        
        if sync:
            return self.submit_sync(request)
        else:
            request_id = self.submit(request)
            return self.wait_for_result(request_id)
    
    def quarantine_knowledge(self, namespace: str, lu_id: str) -> bool:
        """隔离知识"""
        operator = self.aga_manager.get_operator(namespace)
        success = operator.quarantine_knowledge(lu_id)
        
        if success and self.persistence:
            self.persistence.quarantine(namespace, lu_id)
        
        return success
    
    def confirm_knowledge(self, namespace: str, lu_id: str) -> bool:
        """确认知识（试用期 → 已确认）"""
        operator = self.aga_manager.get_operator(namespace)
        success = operator.update_lifecycle(lu_id, LifecycleState.CONFIRMED)
        
        if success and self.persistence:
            self.persistence.update_lifecycle(namespace, lu_id, LifecycleState.CONFIRMED.value)
        
        return success

