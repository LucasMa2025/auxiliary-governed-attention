"""
AGA 网络分区处理与一致性保证

实现分布式系统的网络分区容错和一致性保证：
1. 网络分区检测
2. 分区恢复策略
3. 一致性级别配置
4. 冲突解决机制

版本: v3.4.1

设计原则：
- CAP 定理下选择 AP（可用性 + 分区容错）
- 最终一致性模型
- 向量时钟用于因果一致性
- 冲突检测与自动解决
"""
import asyncio
import time
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Set, Callable, Tuple
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ConsistencyLevel(str, Enum):
    """一致性级别"""
    EVENTUAL = "eventual"           # 最终一致性（默认）
    STRONG = "strong"               # 强一致性（需要 quorum）
    READ_YOUR_WRITES = "ryw"        # 读己之写
    CAUSAL = "causal"               # 因果一致性


class PartitionState(str, Enum):
    """分区状态"""
    HEALTHY = "healthy"             # 正常
    SUSPECTED = "suspected"         # 疑似分区
    PARTITIONED = "partitioned"     # 确认分区
    RECOVERING = "recovering"       # 恢复中
    HEALED = "healed"               # 已恢复


class ConflictResolution(str, Enum):
    """冲突解决策略"""
    LAST_WRITE_WINS = "lww"         # 最后写入胜出
    FIRST_WRITE_WINS = "fww"        # 首次写入胜出
    MERGE = "merge"                 # 合并
    MANUAL = "manual"               # 人工解决
    HIGHER_TRUST_WINS = "htw"       # 高信任级别胜出


@dataclass
class VectorClock:
    """
    向量时钟
    
    用于追踪分布式系统中的因果关系。
    """
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, instance_id: str):
        """递增指定实例的时钟"""
        self.clocks[instance_id] = self.clocks.get(instance_id, 0) + 1
    
    def update(self, other: "VectorClock"):
        """合并另一个向量时钟"""
        for instance_id, clock in other.clocks.items():
            self.clocks[instance_id] = max(self.clocks.get(instance_id, 0), clock)
    
    def happens_before(self, other: "VectorClock") -> bool:
        """判断是否发生在另一个时钟之前"""
        if not self.clocks:
            return True
        
        for instance_id, clock in self.clocks.items():
            if clock > other.clocks.get(instance_id, 0):
                return False
        
        # 至少有一个严格小于
        for instance_id, clock in self.clocks.items():
            if clock < other.clocks.get(instance_id, 0):
                return True
        
        return False
    
    def concurrent_with(self, other: "VectorClock") -> bool:
        """判断是否与另一个时钟并发"""
        return not self.happens_before(other) and not other.happens_before(self)
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典"""
        return dict(self.clocks)
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "VectorClock":
        """从字典创建"""
        return cls(clocks=dict(data))
    
    def __str__(self) -> str:
        return json.dumps(self.clocks)


@dataclass
class VersionedValue:
    """
    带版本的值
    
    用于追踪值的版本和来源。
    """
    value: Any
    vector_clock: VectorClock
    timestamp: float = field(default_factory=time.time)
    source_instance: str = ""
    trust_tier: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "value": self.value,
            "vector_clock": self.vector_clock.to_dict(),
            "timestamp": self.timestamp,
            "source_instance": self.source_instance,
            "trust_tier": self.trust_tier,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionedValue":
        """从字典创建"""
        return cls(
            value=data["value"],
            vector_clock=VectorClock.from_dict(data["vector_clock"]),
            timestamp=data.get("timestamp", time.time()),
            source_instance=data.get("source_instance", ""),
            trust_tier=data.get("trust_tier"),
        )


@dataclass
class PartitionEvent:
    """分区事件"""
    event_type: str  # detected, suspected, healed
    affected_instances: List[str]
    timestamp: float = field(default_factory=time.time)
    details: Optional[Dict[str, Any]] = None


class PartitionDetector:
    """
    网络分区检测器
    
    通过心跳和 gossip 协议检测网络分区。
    """
    
    def __init__(
        self,
        instance_id: str,
        heartbeat_interval: float = 5.0,
        suspect_threshold: float = 15.0,
        partition_threshold: float = 30.0,
        min_healthy_ratio: float = 0.5,
    ):
        """
        初始化分区检测器
        
        Args:
            instance_id: 本实例 ID
            heartbeat_interval: 心跳间隔（秒）
            suspect_threshold: 疑似分区阈值（秒）
            partition_threshold: 确认分区阈值（秒）
            min_healthy_ratio: 最小健康实例比例
        """
        self.instance_id = instance_id
        self.heartbeat_interval = heartbeat_interval
        self.suspect_threshold = suspect_threshold
        self.partition_threshold = partition_threshold
        self.min_healthy_ratio = min_healthy_ratio
        
        # 实例状态
        self._instances: Dict[str, Dict[str, Any]] = {}
        self._last_heartbeats: Dict[str, float] = {}
        self._partition_state = PartitionState.HEALTHY
        
        # 事件回调
        self._on_partition_detected: Optional[Callable] = None
        self._on_partition_healed: Optional[Callable] = None
        
        # 统计
        self._stats = {
            "partitions_detected": 0,
            "partitions_healed": 0,
            "false_positives": 0,
        }
        
        # 运行状态
        self._running = False
        self._detection_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动分区检测"""
        if self._running:
            return
        
        self._running = True
        self._detection_task = asyncio.create_task(self._detection_loop())
        logger.info(f"Partition detector started: instance={self.instance_id}")
    
    async def stop(self):
        """停止分区检测"""
        self._running = False
        
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Partition detector stopped: instance={self.instance_id}")
    
    def register_instance(self, instance_id: str, metadata: Optional[Dict] = None):
        """注册实例"""
        self._instances[instance_id] = {
            "metadata": metadata or {},
            "state": PartitionState.HEALTHY,
            "registered_at": time.time(),
        }
        self._last_heartbeats[instance_id] = time.time()
    
    def unregister_instance(self, instance_id: str):
        """注销实例"""
        self._instances.pop(instance_id, None)
        self._last_heartbeats.pop(instance_id, None)
    
    def record_heartbeat(self, instance_id: str):
        """记录心跳"""
        self._last_heartbeats[instance_id] = time.time()
        
        if instance_id in self._instances:
            old_state = self._instances[instance_id].get("state")
            self._instances[instance_id]["state"] = PartitionState.HEALTHY
            
            # 如果从分区状态恢复
            if old_state in (PartitionState.SUSPECTED, PartitionState.PARTITIONED):
                logger.info(f"Instance {instance_id} recovered from partition")
    
    async def _detection_loop(self):
        """检测循环"""
        while self._running:
            try:
                await self._check_partitions()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Partition detection error: {e}")
                await asyncio.sleep(1)
    
    async def _check_partitions(self):
        """检查分区状态"""
        now = time.time()
        suspected = []
        partitioned = []
        healthy = []
        
        for instance_id in list(self._instances.keys()):
            if instance_id == self.instance_id:
                continue
            
            last_heartbeat = self._last_heartbeats.get(instance_id, 0)
            elapsed = now - last_heartbeat
            
            if elapsed > self.partition_threshold:
                partitioned.append(instance_id)
                self._instances[instance_id]["state"] = PartitionState.PARTITIONED
            elif elapsed > self.suspect_threshold:
                suspected.append(instance_id)
                self._instances[instance_id]["state"] = PartitionState.SUSPECTED
            else:
                healthy.append(instance_id)
                self._instances[instance_id]["state"] = PartitionState.HEALTHY
        
        # 计算健康比例
        total = len(self._instances) - 1  # 排除自己
        if total > 0:
            healthy_ratio = len(healthy) / total
        else:
            healthy_ratio = 1.0
        
        # 更新分区状态
        old_state = self._partition_state
        
        if partitioned:
            self._partition_state = PartitionState.PARTITIONED
            
            if old_state != PartitionState.PARTITIONED:
                self._stats["partitions_detected"] += 1
                logger.warning(f"Network partition detected: {partitioned}")
                
                if self._on_partition_detected:
                    event = PartitionEvent(
                        event_type="detected",
                        affected_instances=partitioned,
                        details={"healthy_ratio": healthy_ratio},
                    )
                    await self._on_partition_detected(event)
        
        elif suspected:
            self._partition_state = PartitionState.SUSPECTED
        
        else:
            if old_state in (PartitionState.PARTITIONED, PartitionState.SUSPECTED):
                self._partition_state = PartitionState.HEALED
                self._stats["partitions_healed"] += 1
                logger.info("Network partition healed")
                
                if self._on_partition_healed:
                    event = PartitionEvent(
                        event_type="healed",
                        affected_instances=[],
                    )
                    await self._on_partition_healed(event)
            else:
                self._partition_state = PartitionState.HEALTHY
    
    def get_partition_state(self) -> PartitionState:
        """获取当前分区状态"""
        return self._partition_state
    
    def get_healthy_instances(self) -> List[str]:
        """获取健康实例列表"""
        return [
            instance_id for instance_id, info in self._instances.items()
            if info.get("state") == PartitionState.HEALTHY
        ]
    
    def get_partitioned_instances(self) -> List[str]:
        """获取分区实例列表"""
        return [
            instance_id for instance_id, info in self._instances.items()
            if info.get("state") == PartitionState.PARTITIONED
        ]
    
    def on_partition_detected(self, callback: Callable):
        """注册分区检测回调"""
        self._on_partition_detected = callback
    
    def on_partition_healed(self, callback: Callable):
        """注册分区恢复回调"""
        self._on_partition_healed = callback
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "current_state": self._partition_state.value,
            "total_instances": len(self._instances),
            "healthy_instances": len(self.get_healthy_instances()),
            "partitioned_instances": len(self.get_partitioned_instances()),
        }


class ConsistencyManager:
    """
    一致性管理器
    
    管理分布式系统的一致性保证。
    """
    
    def __init__(
        self,
        instance_id: str,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
        conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
        quorum_size: int = 2,
    ):
        """
        初始化一致性管理器
        
        Args:
            instance_id: 本实例 ID
            consistency_level: 一致性级别
            conflict_resolution: 冲突解决策略
            quorum_size: quorum 大小
        """
        self.instance_id = instance_id
        self.consistency_level = consistency_level
        self.conflict_resolution = conflict_resolution
        self.quorum_size = quorum_size
        
        # 本地向量时钟
        self._vector_clock = VectorClock()
        
        # 版本存储
        self._versions: Dict[str, List[VersionedValue]] = defaultdict(list)
        
        # 待解决冲突
        self._pending_conflicts: Dict[str, List[VersionedValue]] = {}
        
        # 写入缓冲（用于 read-your-writes）
        self._write_buffer: Dict[str, VersionedValue] = {}
        self._write_buffer_ttl = 30.0  # 秒
        
        # 统计
        self._stats = {
            "writes": 0,
            "reads": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "conflicts_manual": 0,
        }
    
    def write(
        self,
        key: str,
        value: Any,
        trust_tier: Optional[str] = None,
    ) -> VersionedValue:
        """
        写入值
        
        Args:
            key: 键
            value: 值
            trust_tier: 信任层级
        
        Returns:
            带版本的值
        """
        self._stats["writes"] += 1
        
        # 递增向量时钟
        self._vector_clock.increment(self.instance_id)
        
        # 创建版本化值
        versioned = VersionedValue(
            value=value,
            vector_clock=VectorClock(clocks=dict(self._vector_clock.clocks)),
            source_instance=self.instance_id,
            trust_tier=trust_tier,
        )
        
        # 存储
        self._versions[key].append(versioned)
        
        # 更新写入缓冲
        self._write_buffer[key] = versioned
        
        return versioned
    
    def read(
        self,
        key: str,
        consistency: Optional[ConsistencyLevel] = None,
    ) -> Optional[Any]:
        """
        读取值
        
        Args:
            key: 键
            consistency: 一致性级别（覆盖默认）
        
        Returns:
            值或 None
        """
        self._stats["reads"] += 1
        
        level = consistency or self.consistency_level
        
        # Read-your-writes: 优先返回本地写入
        if level == ConsistencyLevel.READ_YOUR_WRITES:
            if key in self._write_buffer:
                buffered = self._write_buffer[key]
                if time.time() - buffered.timestamp < self._write_buffer_ttl:
                    return buffered.value
        
        # 获取所有版本
        versions = self._versions.get(key, [])
        if not versions:
            return None
        
        # 解决冲突
        resolved = self._resolve_versions(key, versions)
        
        return resolved.value if resolved else None
    
    def merge_remote(
        self,
        key: str,
        remote_value: VersionedValue,
    ) -> Tuple[bool, Optional[VersionedValue]]:
        """
        合并远程值
        
        Args:
            key: 键
            remote_value: 远程版本化值
        
        Returns:
            (是否有冲突, 解决后的值)
        """
        # 更新向量时钟
        self._vector_clock.update(remote_value.vector_clock)
        
        # 获取本地版本
        local_versions = self._versions.get(key, [])
        
        # 检查是否有冲突
        has_conflict = False
        for local in local_versions:
            if local.vector_clock.concurrent_with(remote_value.vector_clock):
                has_conflict = True
                self._stats["conflicts_detected"] += 1
                break
        
        # 添加远程版本
        self._versions[key].append(remote_value)
        
        # 解决冲突
        resolved = self._resolve_versions(key, self._versions[key])
        
        return has_conflict, resolved
    
    def _resolve_versions(
        self,
        key: str,
        versions: List[VersionedValue],
    ) -> Optional[VersionedValue]:
        """解决版本冲突"""
        if not versions:
            return None
        
        if len(versions) == 1:
            return versions[0]
        
        # 找出并发版本
        concurrent = []
        latest = versions[0]
        
        for v in versions[1:]:
            if v.vector_clock.happens_before(latest.vector_clock):
                continue
            elif latest.vector_clock.happens_before(v.vector_clock):
                latest = v
            else:
                # 并发
                concurrent.append(v)
        
        if not concurrent:
            return latest
        
        concurrent.append(latest)
        
        # 根据策略解决冲突
        if self.conflict_resolution == ConflictResolution.LAST_WRITE_WINS:
            resolved = max(concurrent, key=lambda v: v.timestamp)
        
        elif self.conflict_resolution == ConflictResolution.FIRST_WRITE_WINS:
            resolved = min(concurrent, key=lambda v: v.timestamp)
        
        elif self.conflict_resolution == ConflictResolution.HIGHER_TRUST_WINS:
            # 信任层级优先级
            trust_priority = {
                "s3_immutable": 4,
                "s2_policy": 3,
                "s1_experience": 2,
                "s0_acceleration": 1,
                None: 0,
            }
            resolved = max(concurrent, key=lambda v: (
                trust_priority.get(v.trust_tier, 0),
                v.timestamp
            ))
        
        elif self.conflict_resolution == ConflictResolution.MERGE:
            # 合并策略：保留所有值（需要应用层处理）
            self._pending_conflicts[key] = concurrent
            resolved = concurrent[0]  # 临时返回第一个
        
        else:  # MANUAL
            self._pending_conflicts[key] = concurrent
            self._stats["conflicts_manual"] += 1
            return None
        
        self._stats["conflicts_resolved"] += 1
        
        # 清理旧版本
        self._versions[key] = [resolved]
        
        return resolved
    
    def get_pending_conflicts(self) -> Dict[str, List[VersionedValue]]:
        """获取待解决冲突"""
        return dict(self._pending_conflicts)
    
    def resolve_conflict(
        self,
        key: str,
        chosen_value: VersionedValue,
    ):
        """手动解决冲突"""
        if key in self._pending_conflicts:
            del self._pending_conflicts[key]
            self._versions[key] = [chosen_value]
            self._stats["conflicts_resolved"] += 1
    
    def get_vector_clock(self) -> VectorClock:
        """获取当前向量时钟"""
        return self._vector_clock
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "consistency_level": self.consistency_level.value,
            "conflict_resolution": self.conflict_resolution.value,
            "pending_conflicts": len(self._pending_conflicts),
            "vector_clock": self._vector_clock.to_dict(),
        }


class PartitionRecoveryManager:
    """
    分区恢复管理器
    
    处理网络分区恢复后的数据同步。
    """
    
    def __init__(
        self,
        instance_id: str,
        consistency_manager: ConsistencyManager,
        max_sync_batch: int = 100,
    ):
        """
        初始化恢复管理器
        
        Args:
            instance_id: 本实例 ID
            consistency_manager: 一致性管理器
            max_sync_batch: 最大同步批次大小
        """
        self.instance_id = instance_id
        self.consistency_manager = consistency_manager
        self.max_sync_batch = max_sync_batch
        
        # 恢复队列
        self._recovery_queue: List[Dict[str, Any]] = []
        
        # 同步状态
        self._sync_in_progress = False
        self._last_sync_time: Optional[float] = None
        
        # 统计
        self._stats = {
            "recoveries_initiated": 0,
            "items_synced": 0,
            "sync_failures": 0,
        }
    
    async def initiate_recovery(
        self,
        remote_instance: str,
        remote_clock: VectorClock,
        fetch_callback: Callable,
    ):
        """
        发起恢复
        
        Args:
            remote_instance: 远程实例 ID
            remote_clock: 远程向量时钟
            fetch_callback: 获取远程数据的回调
        """
        if self._sync_in_progress:
            logger.warning("Recovery already in progress")
            return
        
        self._sync_in_progress = True
        self._stats["recoveries_initiated"] += 1
        
        try:
            # 计算需要同步的数据
            local_clock = self.consistency_manager.get_vector_clock()
            
            # 获取远程数据
            remote_data = await fetch_callback(
                self.instance_id,
                local_clock.to_dict(),
                self.max_sync_batch,
            )
            
            # 合并数据
            for item in remote_data:
                key = item["key"]
                versioned = VersionedValue.from_dict(item["versioned"])
                
                has_conflict, resolved = self.consistency_manager.merge_remote(key, versioned)
                
                if has_conflict:
                    logger.info(f"Conflict detected during recovery: key={key}")
                
                self._stats["items_synced"] += 1
            
            self._last_sync_time = time.time()
            logger.info(f"Recovery completed: synced {len(remote_data)} items")
            
        except Exception as e:
            self._stats["sync_failures"] += 1
            logger.error(f"Recovery failed: {e}")
            raise
        
        finally:
            self._sync_in_progress = False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "sync_in_progress": self._sync_in_progress,
            "last_sync_time": self._last_sync_time,
        }


# 导出
__all__ = [
    "ConsistencyLevel",
    "PartitionState",
    "ConflictResolution",
    "VectorClock",
    "VersionedValue",
    "PartitionEvent",
    "PartitionDetector",
    "ConsistencyManager",
    "PartitionRecoveryManager",
]
