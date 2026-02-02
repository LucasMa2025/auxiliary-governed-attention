"""
AGA (Auxiliary Governed Attention) - 辅助治理注意力

热插拔式知识注入系统，无需向量化训练。

版本: v3.1

主要特性：
- 零训练注入：知识直接写入 buffer，无需梯度计算
- 热插拔设计：运行时动态添加/移除知识
- 治理控制：生命周期状态、熵门控、可追溯性
- 即时隔离：问题知识可立即移除影响
- 多适配器持久化：SQLite/Redis/PostgreSQL
- 分布式支持：多实例同步
- REST API：FastAPI 接口，支持外部治理系统集成

使用示例：

```python
from aga import AGAConfig, AGAOperator, AGAManager
from aga.persistence import SQLiteAdapter, PersistenceManager

# 创建配置
config = AGAConfig.for_production()

# 创建 AGA 管理器
manager = AGAManager(config)

# 挂载到模型
aga_modules = manager.attach_to_model(
    model=your_model,
    layer_indices=[-1, -2, -3],  # 最后三层
)

# 注入知识
manager.inject_knowledge_to_all(
    key_vector=key_tensor,
    value_vector=value_tensor,
    lu_id="knowledge_001",
    condition="当用户询问...",
    decision="应该回答...",
)

# 持久化
adapter = SQLiteAdapter("aga_data.db")
persistence = PersistenceManager(adapter, namespace="default")
await persistence.connect()
await persistence.save_aga_state(aga_modules[-1])
```
"""

# 版本信息
__version__ = "3.2.0"
__author__ = "AGA Team"

# ==================== 类型定义 ====================
from .types import (
    # 枚举
    LifecycleState,
    UncertaintySource,
    GateResult,
    EvictionPolicy,
    DecayStrategy,
    PersistenceAdapter as PersistenceAdapterType,
    # 常量
    LIFECYCLE_RELIABILITY,
    # 数据类
    Slot,
    GateContext,
    KnowledgeSlotInfo,
    AGADiagnostics,
    DecayContext,
    AGAForwardResult,
)

# ==================== 配置 ====================
from .unified_config import (
    AGAConfig,
    GateConfig,
    SlotPoolConfig,
    PersistenceConfig,
    DecayConfig,
    DistributedConfig,
    MonitoringConfig,
    # 兼容性别名
    ProductionAGAConfig,
)

# ==================== 算子层 ====================
from .operator import (
    AGAOperator,
    AGAManager,
    AGAAugmentedTransformerLayer,
)

# ==================== 持久化层 ====================
from .persistence import (
    # 基类
    PersistenceAdapter,
    KnowledgeRecord,
    PersistenceError,
    # 适配器
    SQLiteAdapter,
    MemoryAdapter,
    CompositeAdapter,
    # 管理器
    PersistenceManager,
    # 工厂
    create_adapter,
)

# ==================== 核心模块（向后兼容） ====================
from .core import (
    AuxiliaryGovernedAttention,
    AGAAugmentedTransformerLayer as LegacyAGAAugmentedTransformerLayer,
    AGAManager as LegacyAGAManager,
    AGAConfig as LegacyAGAConfig,
    UncertaintyEstimator,
    SlotRouter,
)

# ==================== 异常处理 ====================
from .exceptions import (
    AGAException,
    AGAConfigError,
    AGAInjectionError,
    AGARoutingError,
    AGAPersistenceError,
    AGAGateError,
    StructuredLogger,
    RetryPolicy,
    with_retry,
)

# ==================== 衰减模块 ====================
from .decay import (
    PersistenceDecay,
    DecayAwareAGAManager,
)

# ==================== 熵门控 ====================
from .entropy_gate import (
    EntropyGate,
    EntropyGateWithDecay,
    EntropyCalculator,
)

# ==================== REST API ====================
# API 模块需要额外依赖（fastapi, uvicorn）
try:
    from .api import (
        create_api_app,
        AGAAPIService,
        AGAClient,
        AsyncAGAClient,
    )
    _HAS_API = True
except ImportError:
    _HAS_API = False
    create_api_app = None
    AGAAPIService = None
    AGAClient = None
    AsyncAGAClient = None

# ==================== Portal (分离部署) ====================
try:
    from .portal import (
        create_portal_app,
        PortalService,
    )
    _HAS_PORTAL = True
except ImportError:
    _HAS_PORTAL = False
    create_portal_app = None
    PortalService = None

# ==================== Runtime (分离部署) ====================
try:
    from .runtime import (
        RuntimeAgent,
        AGARuntime,
        LocalCache,
    )
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False
    RuntimeAgent = None
    AGARuntime = None
    LocalCache = None

# ==================== 配置 (分离部署) ====================
try:
    from .config import (
        PortalConfig,
        RuntimeConfig,
        SyncConfig,
        load_config,
    )
except ImportError:
    PortalConfig = None
    RuntimeConfig = None
    SyncConfig = None
    load_config = None

# ==================== 同步协议 ====================
try:
    from .sync import (
        SyncMessage,
        SyncPublisher,
        SyncSubscriber,
    )
except ImportError:
    SyncMessage = None
    SyncPublisher = None
    SyncSubscriber = None

# ==================== 客户端 ====================
try:
    from .client import (
        AGAClient as PortalClient,
        AsyncAGAClient as AsyncPortalClient,
    )
except ImportError:
    PortalClient = None
    AsyncPortalClient = None

# ==================== 导出列表 ====================
__all__ = [
    # 版本
    "__version__",
    
    # 类型
    "LifecycleState",
    "UncertaintySource",
    "GateResult",
    "EvictionPolicy",
    "DecayStrategy",
    "PersistenceAdapterType",
    "LIFECYCLE_RELIABILITY",
    "Slot",
    "GateContext",
    "KnowledgeSlotInfo",
    "AGADiagnostics",
    "DecayContext",
    "AGAForwardResult",
    
    # 配置
    "AGAConfig",
    "GateConfig",
    "SlotPoolConfig",
    "PersistenceConfig",
    "DecayConfig",
    "DistributedConfig",
    "MonitoringConfig",
    "ProductionAGAConfig",
    
    # 算子
    "AGAOperator",
    "AGAManager",
    "AGAAugmentedTransformerLayer",
    
    # 持久化
    "PersistenceAdapter",
    "KnowledgeRecord",
    "PersistenceError",
    "SQLiteAdapter",
    "MemoryAdapter",
    "CompositeAdapter",
    "PersistenceManager",
    "create_adapter",
    
    # 核心（向后兼容）
    "AuxiliaryGovernedAttention",
    "UncertaintyEstimator",
    "SlotRouter",
    
    # 异常
    "AGAException",
    "AGAConfigError",
    "AGAInjectionError",
    "AGARoutingError",
    "AGAPersistenceError",
    "AGAGateError",
    "StructuredLogger",
    "RetryPolicy",
    "with_retry",
    
    # 衰减
    "PersistenceDecay",
    "DecayAwareAGAManager",
    
    # 熵门控
    "EntropyGate",
    "EntropyGateWithDecay",
    "EntropyCalculator",
    
    # REST API
    "create_api_app",
    "AGAAPIService",
    "AGAClient",
    "AsyncAGAClient",
    
    # Portal (分离部署)
    "create_portal_app",
    "PortalService",
    
    # Runtime (分离部署)
    "RuntimeAgent",
    "AGARuntime",
    "LocalCache",
    
    # 配置
    "PortalConfig",
    "RuntimeConfig",
    "SyncConfig",
    "load_config",
    
    # 同步
    "SyncMessage",
    "SyncPublisher",
    "SyncSubscriber",
    
    # 客户端
    "PortalClient",
    "AsyncPortalClient",
]


# ==================== 便捷函数 ====================

def create_aga_system(
    model,
    layer_indices: list,
    config: AGAConfig = None,
    persistence_adapter: str = "sqlite",
    persistence_path: str = "aga_data.db",
    namespace: str = "default",
) -> tuple:
    """
    便捷函数：创建完整的 AGA 系统
    
    Args:
        model: Transformer 模型
        layer_indices: 要挂载的层索引
        config: AGA 配置
        persistence_adapter: 持久化适配器类型
        persistence_path: 持久化路径
        namespace: 命名空间
    
    Returns:
        (manager, persistence_manager) 元组
    """
    config = config or AGAConfig()
    
    # 创建管理器
    manager = AGAManager(config)
    manager.attach_to_model(model, layer_indices)
    
    # 创建持久化
    adapter = create_adapter(persistence_adapter, db_path=persistence_path)
    persistence = PersistenceManager(adapter, namespace=namespace)
    
    return manager, persistence
