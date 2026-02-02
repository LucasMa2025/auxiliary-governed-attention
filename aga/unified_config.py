"""
AGA 统一配置模块

合并 core.AGAConfig 和 production.ProductionAGAConfig，
提供完整的产品级配置支持。

版本: v3.0
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from .types import UncertaintySource, EvictionPolicy, DecayStrategy, PersistenceAdapter


# ==================== 子配置类 ====================

@dataclass
class GateConfig:
    """
    门控配置
    
    三段式门控架构：
    - Gate-0: 先验门控（零成本，基于 namespace/app_id/route）
    - Gate-1: 置信门控（轻量，基于 hidden state 不确定性）
    - Gate-2: Top-k 路由（只在 Gate-1 通过时执行）
    """
    # Gate-0: 先验门控
    gate0_enabled: bool = True
    gate0_disabled_namespaces: List[str] = field(default_factory=list)
    gate0_required_namespaces: List[str] = field(default_factory=list)
    
    # Gate-1: 置信门控
    gate1_enabled: bool = True
    gate1_threshold: float = 0.1  # confidence < threshold 时 bypass
    gate1_uncertainty_source: UncertaintySource = UncertaintySource.HIDDEN_VARIANCE
    
    # Gate-2: Top-k 路由
    gate2_top_k: int = 8
    gate2_chunk_size: int = 64
    
    # 熵门控参数
    tau_low: float = 0.5   # 熵否决下限
    tau_high: float = 2.0  # 熵否决上限
    max_gate: float = 0.8  # 最大门控值
    
    # Early Exit
    early_exit_enabled: bool = True
    early_exit_threshold: float = 0.05


@dataclass
class SlotPoolConfig:
    """
    槽位池配置
    
    核心约束：max_slots 保证 O(1) 推理复杂度
    """
    # 容量配置
    max_slots: int = 128  # 🔒 硬上限，保证 O(1)
    initial_slots: int = 64
    
    # 向量维度
    hidden_dim: int = 4096
    bottleneck_dim: int = 64
    
    # 范数控制
    key_norm_target: float = 5.0
    value_norm_target: float = 3.0
    enable_norm_clipping: bool = True
    
    # Value Projection (delta subspace)
    use_value_projection: bool = True
    value_bottleneck_dim: int = 256
    
    # 淘汰策略
    eviction_enabled: bool = True
    eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID
    eviction_trigger_ratio: float = 0.9
    eviction_target_ratio: float = 0.7
    eviction_min_hit_count: int = 5
    eviction_max_age_days: int = 30
    
    # 自动降级
    auto_deprecate_enabled: bool = False
    auto_deprecate_threshold: int = 100  # 连续未命中次数阈值


@dataclass
class PersistenceConfig:
    """
    持久化配置
    
    支持多适配器：SQLite (开发) / Redis (热缓存) / PostgreSQL (冷存储)
    """
    # 适配器类型
    adapter_type: PersistenceAdapter = PersistenceAdapter.SQLITE
    
    # SQLite 配置
    sqlite_path: str = "aga_data.db"
    
    # Redis 配置
    redis_enabled: bool = False
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    redis_password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    redis_key_prefix: str = "aga"
    redis_slot_ttl_days: int = 7
    redis_pool_size: int = 10
    
    # PostgreSQL 配置
    postgres_enabled: bool = False
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "aga"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "aga"))
    postgres_password: Optional[str] = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD"))
    postgres_pool_size: int = 5
    postgres_max_overflow: int = 10
    
    # 同步配置
    sync_interval_seconds: int = 5
    sync_batch_size: int = 100


@dataclass
class DecayConfig:
    """
    持久化衰减配置
    
    防止 AGA 在多层中过度干预。
    """
    enabled: bool = True
    strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    gamma: float = 0.9  # 衰减系数
    hard_reset_threshold: float = 3.0  # 累积 gate 超过此值时硬重置
    min_effective_gate: float = 0.01  # 最小有效 gate


@dataclass
class DistributedConfig:
    """
    分布式配置
    
    支持多实例部署的同步和协调。
    """
    enabled: bool = False
    
    # 实例标识
    instance_id: str = field(default_factory=lambda: os.getenv("AGA_INSTANCE_ID", "default"))
    
    # Kafka 配置 (事件同步)
    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    kafka_topic_prefix: str = "aga"
    kafka_consumer_group: str = "aga-sync"
    
    # 分布式锁配置
    lock_timeout_seconds: int = 30
    lock_retry_interval_ms: int = 100
    
    # 心跳配置
    heartbeat_interval_seconds: int = 10
    instance_timeout_seconds: int = 30


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    metrics_prefix: str = "aga"
    log_level: str = "INFO"
    enable_diagnostics: bool = True
    
    # 性能监控
    latency_histogram_buckets: List[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )


# ==================== 主配置类 ====================

@dataclass
class AGAConfig:
    """
    AGA 完整配置
    
    统一的产品级配置，支持：
    - 多适配器持久化
    - 三段式门控
    - 持久化衰减
    - 分布式部署
    - 监控和诊断
    """
    # 基础配置
    namespace: str = "default"
    num_heads: int = 32
    
    # 子配置
    gate: GateConfig = field(default_factory=GateConfig)
    slot_pool: SlotPoolConfig = field(default_factory=SlotPoolConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 运行时配置
    fail_open_enabled: bool = True  # 🔒 必须为 True（安全约束）
    max_forward_timeout_ms: int = 50
    
    def validate(self) -> List[str]:
        """
        验证配置合法性
        
        Returns:
            错误列表，空列表表示配置有效
        """
        errors = []
        
        # 不变量检查
        if self.slot_pool.max_slots > 256:
            errors.append(
                f"max_slots={self.slot_pool.max_slots} > 256, violates O(1) invariant"
            )
        
        if not self.fail_open_enabled:
            errors.append("fail_open_enabled must be True for production safety")
        
        if self.gate.gate2_top_k > self.slot_pool.max_slots:
            errors.append(
                f"gate2_top_k={self.gate.gate2_top_k} > max_slots={self.slot_pool.max_slots}"
            )
        
        if self.decay.gamma <= 0 or self.decay.gamma > 1:
            errors.append(f"decay.gamma={self.decay.gamma} must be in (0, 1]")
        
        if self.gate.tau_low >= self.gate.tau_high:
            errors.append(
                f"tau_low={self.gate.tau_low} must be < tau_high={self.gate.tau_high}"
            )
        
        return errors
    
    @classmethod
    def from_env(cls) -> "AGAConfig":
        """从环境变量加载配置"""
        config = cls()
        
        # 基础配置
        if os.getenv("AGA_NAMESPACE"):
            config.namespace = os.getenv("AGA_NAMESPACE")
        
        # 槽位池配置
        if os.getenv("AGA_HIDDEN_DIM"):
            config.slot_pool.hidden_dim = int(os.getenv("AGA_HIDDEN_DIM"))
        
        if os.getenv("AGA_MAX_SLOTS"):
            config.slot_pool.max_slots = int(os.getenv("AGA_MAX_SLOTS"))
        
        if os.getenv("AGA_BOTTLENECK_DIM"):
            config.slot_pool.bottleneck_dim = int(os.getenv("AGA_BOTTLENECK_DIM"))
        
        # 门控配置
        if os.getenv("AGA_TOP_K"):
            config.gate.gate2_top_k = int(os.getenv("AGA_TOP_K"))
        
        if os.getenv("AGA_EARLY_EXIT_THRESHOLD"):
            config.gate.early_exit_threshold = float(os.getenv("AGA_EARLY_EXIT_THRESHOLD"))
        
        # 持久化配置
        if os.getenv("AGA_PERSISTENCE_ADAPTER"):
            config.persistence.adapter_type = PersistenceAdapter(
                os.getenv("AGA_PERSISTENCE_ADAPTER")
            )
        
        if os.getenv("AGA_SQLITE_PATH"):
            config.persistence.sqlite_path = os.getenv("AGA_SQLITE_PATH")
        
        # 分布式配置
        if os.getenv("AGA_DISTRIBUTED_ENABLED"):
            config.distributed.enabled = os.getenv("AGA_DISTRIBUTED_ENABLED").lower() == "true"
        
        return config
    
    @classmethod
    def for_development(cls) -> "AGAConfig":
        """开发环境配置"""
        config = cls()
        config.persistence.adapter_type = PersistenceAdapter.SQLITE
        config.distributed.enabled = False
        config.monitoring.log_level = "DEBUG"
        return config
    
    @classmethod
    def for_production(cls) -> "AGAConfig":
        """生产环境配置"""
        config = cls()
        config.persistence.adapter_type = PersistenceAdapter.COMPOSITE
        config.persistence.redis_enabled = True
        config.persistence.postgres_enabled = True
        config.distributed.enabled = True
        config.monitoring.log_level = "INFO"
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "namespace": self.namespace,
            "num_heads": self.num_heads,
            "gate": {
                "gate0_enabled": self.gate.gate0_enabled,
                "gate1_enabled": self.gate.gate1_enabled,
                "gate1_threshold": self.gate.gate1_threshold,
                "gate1_uncertainty_source": self.gate.gate1_uncertainty_source.value,
                "gate2_top_k": self.gate.gate2_top_k,
                "tau_low": self.gate.tau_low,
                "tau_high": self.gate.tau_high,
                "early_exit_enabled": self.gate.early_exit_enabled,
                "early_exit_threshold": self.gate.early_exit_threshold,
            },
            "slot_pool": {
                "max_slots": self.slot_pool.max_slots,
                "hidden_dim": self.slot_pool.hidden_dim,
                "bottleneck_dim": self.slot_pool.bottleneck_dim,
                "use_value_projection": self.slot_pool.use_value_projection,
                "eviction_enabled": self.slot_pool.eviction_enabled,
                "eviction_policy": self.slot_pool.eviction_policy.value,
            },
            "persistence": {
                "adapter_type": self.persistence.adapter_type.value,
                "redis_enabled": self.persistence.redis_enabled,
                "postgres_enabled": self.persistence.postgres_enabled,
            },
            "decay": {
                "enabled": self.decay.enabled,
                "strategy": self.decay.strategy.value,
                "gamma": self.decay.gamma,
            },
            "distributed": {
                "enabled": self.distributed.enabled,
                "instance_id": self.distributed.instance_id,
            },
            "fail_open_enabled": self.fail_open_enabled,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AGAConfig":
        """从字典创建配置"""
        config = cls()
        
        # 基础配置
        config.namespace = data.get("namespace", config.namespace)
        config.num_heads = data.get("num_heads", config.num_heads)
        
        # 门控配置
        if "gate" in data:
            gate_data = data["gate"]
            config.gate.gate0_enabled = gate_data.get("gate0_enabled", config.gate.gate0_enabled)
            config.gate.gate1_enabled = gate_data.get("gate1_enabled", config.gate.gate1_enabled)
            config.gate.gate1_threshold = gate_data.get("gate1_threshold", config.gate.gate1_threshold)
            if "gate1_uncertainty_source" in gate_data:
                config.gate.gate1_uncertainty_source = UncertaintySource(
                    gate_data["gate1_uncertainty_source"]
                )
            config.gate.gate2_top_k = gate_data.get("gate2_top_k", config.gate.gate2_top_k)
            config.gate.tau_low = gate_data.get("tau_low", config.gate.tau_low)
            config.gate.tau_high = gate_data.get("tau_high", config.gate.tau_high)
            config.gate.early_exit_enabled = gate_data.get(
                "early_exit_enabled", config.gate.early_exit_enabled
            )
            config.gate.early_exit_threshold = gate_data.get(
                "early_exit_threshold", config.gate.early_exit_threshold
            )
        
        # 槽位池配置
        if "slot_pool" in data:
            pool_data = data["slot_pool"]
            config.slot_pool.max_slots = pool_data.get("max_slots", config.slot_pool.max_slots)
            config.slot_pool.hidden_dim = pool_data.get("hidden_dim", config.slot_pool.hidden_dim)
            config.slot_pool.bottleneck_dim = pool_data.get(
                "bottleneck_dim", config.slot_pool.bottleneck_dim
            )
            config.slot_pool.use_value_projection = pool_data.get(
                "use_value_projection", config.slot_pool.use_value_projection
            )
            config.slot_pool.eviction_enabled = pool_data.get(
                "eviction_enabled", config.slot_pool.eviction_enabled
            )
            if "eviction_policy" in pool_data:
                config.slot_pool.eviction_policy = EvictionPolicy(pool_data["eviction_policy"])
        
        # 持久化配置
        if "persistence" in data:
            persist_data = data["persistence"]
            if "adapter_type" in persist_data:
                config.persistence.adapter_type = PersistenceAdapter(persist_data["adapter_type"])
            config.persistence.redis_enabled = persist_data.get(
                "redis_enabled", config.persistence.redis_enabled
            )
            config.persistence.postgres_enabled = persist_data.get(
                "postgres_enabled", config.persistence.postgres_enabled
            )
        
        # 衰减配置
        if "decay" in data:
            decay_data = data["decay"]
            config.decay.enabled = decay_data.get("enabled", config.decay.enabled)
            if "strategy" in decay_data:
                config.decay.strategy = DecayStrategy(decay_data["strategy"])
            config.decay.gamma = decay_data.get("gamma", config.decay.gamma)
        
        # 分布式配置
        if "distributed" in data:
            dist_data = data["distributed"]
            config.distributed.enabled = dist_data.get("enabled", config.distributed.enabled)
            config.distributed.instance_id = dist_data.get(
                "instance_id", config.distributed.instance_id
            )
        
        config.fail_open_enabled = data.get("fail_open_enabled", config.fail_open_enabled)
        
        return config


# ==================== 兼容性别名 ====================

# 为了向后兼容，保留旧的配置类名
ProductionAGAConfig = AGAConfig
