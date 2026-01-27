# AGA Production Runtime v1.0

## 概述

AGA (Auxiliary Governed Attention) 生产级运行时，为持续学习系统提供知识注入和路由能力。

### 核心定位

```
AGA = 持续学习系统的注入器 + 新知识的路由器
```

### 设计原则（不变量）

```
🔒 不变量 1：推理可见知识规模 = O(1)
   - 单次推理最多接触 ≤128 条 AGA 槽
   - 全局知识规模 ∞ ≠ 推理复杂度增长

🔒 不变量 2：AGA 永远是"可绕过"的
   - 任何异常情况：AGA = NO-OP
   - 不允许"必须依赖 AGA 才能正确回答"

🔒 不变量 3：治理、学习、评估永不进入热路径
   - 推理线程只做：gate → top-k → attention bias
   - 其余全部异步
```

## 架构

```
                                    ┌─────────────────────────┐
                                    │   Knowledge Write API   │
                                    │   (独立模块/异步)        │
                                    └───────────┬─────────────┘
                                                │ 异步写入
                                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        AGA Production Runtime                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  │
│  │   Gate-0    │───▶│   Gate-1    │───▶│        Gate-2 (top-k)       │  │
│  │ (先验门控)   │    │ (置信门控)   │    │       + Attention Bias      │  │
│  │ namespace   │    │ confidence  │    │                             │  │
│  │ app_id      │    │ threshold   │    │  ┌─────────────────────────┐│  │
│  │ route       │    │             │    │  │   Slot Subspace Pool    ││  │
│  └─────────────┘    └─────────────┘    │  │   (per-namespace)       ││  │
│        │                   │           │  │   max_slots=128         ││  │
│        │ DISABLED          │ bypass    │  └─────────────────────────┘│  │
│        ▼                   ▼           └─────────────────────────────────┘
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Fail-Open Fallback                           │   │
│   │                 (异常时直接返回原模型输出)                         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                        Persistence Layer                                 │
│  ┌─────────────────────┐         ┌─────────────────────────────────┐    │
│  │   Redis (Hot Cache) │◀───────▶│   PostgreSQL (Cold Storage)     │    │
│  │   - 热槽位缓存        │         │   - 全量知识                      │    │
│  │   - per-namespace   │         │   - 审计日志                      │    │
│  │   - TTL 7天         │         │   - 版本控制                      │    │
│  └─────────────────────┘         └─────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 模块说明

### 1. Gate（三段式门控）

| Gate | 功能 | 成本 | 目标 |
|------|------|------|------|
| Gate-0 | 先验门控 | 零成本 | 挡掉 80-90% 请求 |
| Gate-1 | 置信门控 | 轻量 | 低置信度时 bypass |
| Gate-2 | Top-k 路由 | 中等 | 选择最相关槽位 |

### 2. Slot Pool（槽位池）

- **物理隔离**：每个 namespace 独立槽池
- **硬上限**：max_slots_per_namespace = 128
- **淘汰策略**：LRU + hit_count 混合

### 3. Persistence（持久化）

- **Redis**：热槽位缓存，TTL 7 天
- **PostgreSQL**：冷存储 + 审计日志
- **混合模式**：写入双写，读取 Redis 优先

### 4. Knowledge Writer（知识写入）

- **异步写入**：不阻塞推理路径
- **质量评估**：否决型检查
- **批量支持**：支持批量写入

## 使用示例

### 基础使用

```python
from aga.production import (
    ProductionAGAConfig,
    AGAOperator,
    GateContext,
    LifecycleState,
)

# 1. 创建配置
config = ProductionAGAConfig(
    namespace="my_app",
    slot_pool=SlotPoolConfig(
        max_slots_per_namespace=64,
        hidden_dim=4096,
        bottleneck_dim=64,
    ),
)

# 2. 创建算子
operator = AGAOperator(config)

# 3. 注入知识
operator.inject_knowledge(
    lu_id="LU_001",
    key_vector=key_vec,
    value_vector=value_vec,
    lifecycle_state=LifecycleState.PROBATIONARY,
)

# 4. 前向传播
result = operator.forward(
    hidden_states=hidden,
    primary_attention_output=primary,
    context=GateContext(namespace="my_app"),
)

# 5. 使用结果
if result.aga_applied:
    output = result.output
else:
    output = primary  # Fail-open
```

### 并发管理

```python
from aga.production import ConcurrentAGAManager

# 创建管理器
manager = ConcurrentAGAManager(default_config)

# 获取不同 namespace 的算子
operator_a = manager.get_operator("tenant_a")
operator_b = manager.get_operator("tenant_b")

# 并发安全
```

### 知识写入

```python
from aga.production import KnowledgeWriter

# 创建写入器
writer = KnowledgeWriter(aga_manager, persistence)
writer.start()

# 同步写入
result = writer.write_knowledge(
    namespace="my_app",
    lu_id="LU_001",
    condition="用户询问天气",
    decision="查询天气API",
    key_vector=key_vec,
    value_vector=value_vec,
    sync=True,
)

# 异步写入
request_id = writer.submit(write_request)
result = writer.wait_for_result(request_id)
```

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `REDIS_HOST` | Redis 主机 | localhost |
| `REDIS_PORT` | Redis 端口 | 6379 |
| `REDIS_PASSWORD` | Redis 密码 | - |
| `POSTGRES_HOST` | PostgreSQL 主机 | localhost |
| `POSTGRES_PORT` | PostgreSQL 端口 | 5432 |
| `POSTGRES_DB` | 数据库名 | aga |
| `POSTGRES_USER` | 用户名 | aga |
| `POSTGRES_PASSWORD` | 密码 | - |
| `AGA_NAMESPACE` | 默认 namespace | default |
| `AGA_MAX_SLOTS` | 最大槽位数 | 128 |
| `AGA_TOP_K` | Top-k 路由数 | 8 |

## 监控指标

| 指标 | 说明 |
|------|------|
| `aga_forward_count` | 前向传播总次数 |
| `aga_applied_count` | AGA 实际应用次数 |
| `aga_applied_ratio` | AGA 应用比例 |
| `aga_fail_open_count` | Fail-open 次数 |
| `aga_avg_latency_ms` | 平均延迟 |
| `aga_slot_occupancy` | 槽位占用率 |
| `aga_hit_rate` | 命中率 |

## Phase-1 验收标准

- ✅ 任意关闭 AGA → 系统行为稳定
- ✅ 知识规模 ×10 → 推理延迟几乎不变
- ✅ 私域请求 → 高命中
- ✅ 公域请求 → AGA 几乎不触发
- ✅ 单条坏知识 → 可 1 分钟内隔离

## 后续规划

### Phase-2（6-12 个月）
- 分布式一致性（Kafka 广播）
- 自动化 lifecycle 迁移
- A/B 测试 + 监控仪表盘

### Phase-3（12-24 个月）
- 知识 → LoRA 定期合并
- 动态扩容 memory bank

