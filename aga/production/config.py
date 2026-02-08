"""
AGA Production 配置模块

所有配置项都有合理默认值，支持环境变量覆盖。

v1.1 新增:
- ANNIndexConfig: ANN 索引配置，支持大规模知识库
- DynamicLoaderConfig: 动态加载器配置
- 重新定义不变量：hot_pool_size ≤ 256 (而非 max_slots)
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


class ANNIndexBackend(str, Enum):
    """ANN 索引后端"""
    FAISS = "faiss"
    HNSW = "hnsw"
    FLAT = "flat"  # 精确搜索，用于小规模或测试


@dataclass
class ANNIndexConfig:
    """
    ANN 索引配置（v1.1 新增）
    
    用于大规模知识库 (100K+ Slots) 的近似最近邻检索。
    默认关闭，向后兼容。
    """
    # 基础配置
    enabled: bool = False  # 🔒 默认关闭，向后兼容
    backend: ANNIndexBackend = ANNIndexBackend.FAISS
    
    # 索引类型 (FAISS 专用)
    index_type: str = "IVF4096,PQ64"  # IVF + PQ 组合
    
    # 容量配置
    index_capacity: int = 1_000_000  # 索引容量上限
    
    # 检索配置
    retrieval_top_k: int = 200  # ANN 返回候选数
    nprobe: int = 64  # IVF 探测数（精度 vs 延迟）
    ef_search: int = 128  # HNSW 搜索参数
    
    # 性能配置
    use_gpu: bool = True  # 使用 GPU 加速
    gpu_device_id: int = 0  # GPU 设备 ID
    search_timeout_ms: float = 10.0  # 检索超时
    
    # 更新配置
    incremental_update: bool = True  # 增量更新
    rebuild_interval_hours: int = 24  # 全量重建间隔
    rebuild_threshold: float = 0.3  # 删除比例超过此值触发重建
    
    # HNSW 专用配置
    hnsw_m: int = 32  # 连接数
    hnsw_ef_construction: int = 200  # 构建时的 ef 参数


@dataclass
class DynamicLoaderConfig:
    """
    动态加载器配置（v1.1 新增）
    
    用于从分层存储 (Hot/Warm/Cold) 动态加载知识到 Hot Pool。
    """
    # 基础配置
    enabled: bool = True
    
    # 加载配置
    max_cold_load_per_request: int = 50  # 单次最大冷加载数
    cold_load_timeout_ms: float = 10.0  # 冷加载超时
    batch_load_size: int = 20  # 批量加载大小
    
    # 预取配置
    prefetch_enabled: bool = True
    prefetch_threshold: int = 3  # 访问次数阈值触发预取
    prefetch_batch_size: int = 10  # 预取批量大小
    
    # Warm 缓存配置
    warm_cache_size: int = 2000  # Warm 层容量
    warm_cache_ttl_seconds: float = 3600.0  # Warm 缓存 TTL


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
    
    核心约束：
    - hot_pool_size ≤ 256: 保证 Gate2 O(1) 推理复杂度
    - max_slots_per_namespace: 兼容旧配置，等同于 hot_pool_size
    
    v1.1 变更：
    - 新增 hot_pool_size 作为主要配置
    - max_slots_per_namespace 保留用于向后兼容
    """
    # Hot Pool 配置（v1.1 新增）
    hot_pool_size: int = 256  # 🔒 硬上限，保证 Gate2 O(1)
    hot_pool_min_size: int = 16  # 动态扩容下限
    
    # 容量配置（向后兼容）
    max_slots_per_namespace: int = 128  # 兼容旧配置
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
    
    v1.1 新增:
    - ann_index: ANN 索引配置，支持大规模知识库
    - dynamic_loader: 动态加载器配置
    """
    # 基础配置
    namespace: str = "default"
    num_heads: int = 32
    
    # 子配置
    gate: GateConfig = field(default_factory=GateConfig)
    slot_pool: SlotPoolConfig = field(default_factory=SlotPoolConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    
    # v1.1 新增: 大规模知识库支持
    ann_index: ANNIndexConfig = field(default_factory=ANNIndexConfig)
    dynamic_loader: DynamicLoaderConfig = field(default_factory=DynamicLoaderConfig)
    
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
        
        # 🔒 不变量 1: Hot Pool 上限（v1.1 重新定义）
        # 当启用 ANN 时，使用 hot_pool_size；否则使用 max_slots_per_namespace
        effective_hot_pool_size = (
            self.slot_pool.hot_pool_size 
            if self.ann_index.enabled 
            else self.slot_pool.max_slots_per_namespace
        )
        
        if effective_hot_pool_size > 256:
            errors.append(
                f"hot_pool_size={effective_hot_pool_size} > 256, "
                "violates O(1) invariant for Gate2"
            )
        
        # 🔒 不变量 2: Fail-Open 必须启用
        if not self.fail_open_enabled:
            errors.append("fail_open_enabled must be True for production safety")
        
        # 🔒 不变量 3: Top-k ≤ Hot Pool
        if self.gate.gate2_top_k > effective_hot_pool_size:
            errors.append(
                f"gate2_top_k={self.gate.gate2_top_k} > "
                f"hot_pool_size={effective_hot_pool_size}"
            )
        
        # ANN 配置验证
        if self.ann_index.enabled:
            # retrieval_top_k 应该合理
            if self.ann_index.retrieval_top_k > self.slot_pool.hot_pool_size * 2:
                errors.append(
                    f"ann_index.retrieval_top_k={self.ann_index.retrieval_top_k} "
                    f"is too large compared to hot_pool_size={self.slot_pool.hot_pool_size}"
                )
        
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
                "hot_pool_size": self.slot_pool.hot_pool_size,
                "hidden_dim": self.slot_pool.hidden_dim,
                "bottleneck_dim": self.slot_pool.bottleneck_dim,
            },
            "persistence": {
                "redis_enabled": self.persistence.redis_enabled,
                "postgres_enabled": self.persistence.postgres_enabled,
            },
            "ann_index": {
                "enabled": self.ann_index.enabled,
                "backend": self.ann_index.backend.value,
                "index_capacity": self.ann_index.index_capacity,
                "retrieval_top_k": self.ann_index.retrieval_top_k,
            },
            "dynamic_loader": {
                "enabled": self.dynamic_loader.enabled,
                "warm_cache_size": self.dynamic_loader.warm_cache_size,
            },
            "fail_open_enabled": self.fail_open_enabled,
        }
    
    @classmethod
    def for_large_scale(
        cls,
        namespace: str = "default",
        index_capacity: int = 1_000_000,
        hot_pool_size: int = 256,
        use_gpu: bool = True,
    ) -> "ProductionAGAConfig":
        """
        创建大规模知识库配置
        
        Args:
            namespace: 命名空间
            index_capacity: 索引容量（支持的最大知识数）
            hot_pool_size: Hot Pool 大小（≤256）
            use_gpu: 是否使用 GPU 加速 ANN
            
        Returns:
            配置好的 ProductionAGAConfig
        """
        config = cls(namespace=namespace)
        
        # 启用 ANN 索引
        config.ann_index.enabled = True
        config.ann_index.index_capacity = index_capacity
        config.ann_index.use_gpu = use_gpu
        
        # 设置 Hot Pool
        config.slot_pool.hot_pool_size = min(hot_pool_size, 256)
        
        # 启用动态加载
        config.dynamic_loader.enabled = True
        
        return config

