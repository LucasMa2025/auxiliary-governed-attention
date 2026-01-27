"""
AGA Production 配置模块

所有配置项都有合理默认值，支持环境变量覆盖。
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class UncertaintySource(str, Enum):
    """不确定性信号来源"""
    HIDDEN_VARIANCE = "hidden_variance"
    LOGITS_ENTROPY = "logits_entropy"
    LEARNED_PROJECTION = "learned_projection"
    CONSTANT = "constant"


@dataclass
class GateConfig:
    """
    三段式门控配置
    
    Gate-0: 先验门控（零成本，基于 namespace/app_id/route）
    Gate-1: 置信门控（轻量，基于 hidden state）
    Gate-2: Top-k 路由（只在 Gate-1 通过时执行）
    """
    # Gate-0 配置
    gate0_enabled: bool = True
    gate0_disabled_namespaces: List[str] = field(default_factory=list)  # 完全禁用的 namespace
    gate0_required_namespaces: List[str] = field(default_factory=list)  # 强制启用的 namespace
    
    # Gate-1 配置
    gate1_enabled: bool = True
    gate1_threshold: float = 0.1  # confidence < threshold 时 bypass
    gate1_uncertainty_source: UncertaintySource = UncertaintySource.HIDDEN_VARIANCE
    
    # Gate-2 配置
    gate2_top_k: int = 8  # 每次推理最多路由的槽位数
    gate2_chunk_size: int = 64  # 分块计算大小
    
    # Early Exit 配置
    early_exit_threshold: float = 0.05  # gate < threshold 完全跳过
    early_exit_enabled: bool = True
    
    # 融合配置
    tau_low: float = 0.5  # 熵否决下限
    tau_high: float = 2.0  # 熵否决上限


@dataclass
class SlotPoolConfig:
    """
    槽位池配置
    
    核心约束：max_slots_per_namespace 保证 O(1) 推理复杂度
    """
    # 容量配置
    max_slots_per_namespace: int = 128  # 🔒 硬上限，保证 O(1)
    initial_slots_per_namespace: int = 64
    
    # 向量维度
    hidden_dim: int = 4096
    bottleneck_dim: int = 64
    
    # 淘汰策略
    eviction_enabled: bool = True
    eviction_trigger_ratio: float = 0.9  # 占用率超过此值触发淘汰
    eviction_target_ratio: float = 0.7  # 淘汰后目标占用率
    eviction_min_hit_count: int = 5  # hit_count 低于此值优先淘汰
    eviction_max_age_days: int = 30  # 超过此天数优先淘汰
    
    # 范数控制
    key_norm_target: float = 5.0
    value_norm_target: float = 3.0
    enable_norm_clipping: bool = True
    
    # Value Projection
    use_value_projection: bool = True
    value_bottleneck_dim: int = 256


@dataclass
class PersistenceConfig:
    """
    持久化配置
    
    Redis: 热槽位缓存
    PostgreSQL: 冷存储 + 审计日志
    """
    # Redis 配置
    redis_enabled: bool = True
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    redis_password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    redis_key_prefix: str = "aga"
    redis_slot_ttl_days: int = 7  # 热槽位 TTL
    redis_pool_size: int = 10
    
    # PostgreSQL 配置
    postgres_enabled: bool = True
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "aga"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "aga"))
    postgres_password: Optional[str] = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD"))
    postgres_pool_size: int = 5
    postgres_max_overflow: int = 10
    
    # 同步配置
    sync_interval_seconds: int = 5  # 后台同步间隔
    sync_batch_size: int = 100  # 批量同步大小


@dataclass
class ProductionAGAConfig:
    """
    AGA 生产级完整配置
    """
    # 基础配置
    namespace: str = "default"
    num_heads: int = 32
    
    # 子配置
    gate: GateConfig = field(default_factory=GateConfig)
    slot_pool: SlotPoolConfig = field(default_factory=SlotPoolConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    
    # 运行时配置
    fail_open_enabled: bool = True  # 🔒 必须为 True
    max_forward_timeout_ms: int = 50  # 前向传播超时
    enable_diagnostics: bool = True
    log_level: str = "INFO"
    
    # 监控配置
    metrics_enabled: bool = True
    metrics_prefix: str = "aga"
    
    def validate(self) -> List[str]:
        """验证配置合法性"""
        errors = []
        
        # 不变量检查
        if self.slot_pool.max_slots_per_namespace > 256:
            errors.append(f"max_slots_per_namespace={self.slot_pool.max_slots_per_namespace} > 256, "
                         "violates O(1) invariant")
        
        if not self.fail_open_enabled:
            errors.append("fail_open_enabled must be True for production safety")
        
        if self.gate.gate2_top_k > self.slot_pool.max_slots_per_namespace:
            errors.append(f"gate2_top_k={self.gate.gate2_top_k} > max_slots={self.slot_pool.max_slots_per_namespace}")
        
        return errors
    
    @classmethod
    def from_env(cls) -> "ProductionAGAConfig":
        """从环境变量加载配置"""
        config = cls()
        
        # 覆盖基础配置
        if os.getenv("AGA_NAMESPACE"):
            config.namespace = os.getenv("AGA_NAMESPACE")
        
        if os.getenv("AGA_HIDDEN_DIM"):
            config.slot_pool.hidden_dim = int(os.getenv("AGA_HIDDEN_DIM"))
        
        if os.getenv("AGA_MAX_SLOTS"):
            config.slot_pool.max_slots_per_namespace = int(os.getenv("AGA_MAX_SLOTS"))
        
        if os.getenv("AGA_TOP_K"):
            config.gate.gate2_top_k = int(os.getenv("AGA_TOP_K"))
        
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
                "gate2_top_k": self.gate.gate2_top_k,
                "early_exit_threshold": self.gate.early_exit_threshold,
            },
            "slot_pool": {
                "max_slots": self.slot_pool.max_slots_per_namespace,
                "hidden_dim": self.slot_pool.hidden_dim,
                "bottleneck_dim": self.slot_pool.bottleneck_dim,
            },
            "persistence": {
                "redis_enabled": self.persistence.redis_enabled,
                "postgres_enabled": self.persistence.postgres_enabled,
            },
            "fail_open_enabled": self.fail_open_enabled,
        }

