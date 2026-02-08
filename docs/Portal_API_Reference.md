# AGA Portal API 参考文档

> **版本**: 3.4.0  
> **Base URL**: `http://localhost:8081` (默认)  
> **协议**: HTTP/HTTPS REST API  
> **数据格式**: JSON

## 目录

-   [概述](#概述)
-   [认证](#认证)
-   [通用响应格式](#通用响应格式)
-   [错误处理](#错误处理)
-   [API 端点](#api-端点)
    -   [健康检查](#健康检查)
    -   [知识管理](#知识管理)
    -   [生命周期管理](#生命周期管理)
    -   [槽位管理](#槽位管理)
    -   [统计信息](#统计信息)
    -   [审计日志](#审计日志)
    -   [命名空间](#命名空间)
    -   [Runtime 管理](#runtime-管理)
    -   [版本控制](#版本控制)
-   [数据模型](#数据模型)
-   [客户端示例](#客户端示例)

---

## 概述

AGA Portal 是 AGA (Auxiliary Governed Attention) 系统的 API 网关，提供：

-   **知识管理**: 注入、查询、删除知识
-   **生命周期治理**: 状态转换、隔离、确认
-   **多租户隔离**: 基于命名空间的租户隔离
-   **动态槽位管理**: 运行时容量调整
-   **审计追溯**: 完整的操作日志
-   **分布式同步**: 通过消息队列同步到 Runtime

### 架构位置

```
┌─────────────┐     HTTP      ┌──────────────┐    Pub/Sub    ┌─────────────┐
│ 治理系统     │ ───────────▶ │  AGA Portal  │ ────────────▶ │ AGA Runtime │
│ AGAClient   │               │  (本 API)    │               │ (GPU 推理)  │
└─────────────┘               └──────────────┘               └─────────────┘
```

---

## 认证

Portal 支持可选的 API Key 认证：

```http
Authorization: Bearer <api_key>
```

未配置认证时，所有端点均可公开访问。

---

## 通用响应格式

所有 API 响应遵循统一格式：

```json
{
  "success": true,
  "message": "操作成功",
  "data": { ... },
  "timestamp": "2026-02-06T10:30:00.000Z"
}
```

### 响应字段

| 字段        | 类型    | 描述                   |
| ----------- | ------- | ---------------------- |
| `success`   | boolean | 操作是否成功           |
| `message`   | string  | 可选的消息说明         |
| `data`      | object  | 响应数据，根据端点不同 |
| `timestamp` | string  | ISO 8601 时间戳        |

---

## 错误处理

### HTTP 状态码

| 状态码 | 描述               |
| ------ | ------------------ |
| `200`  | 成功               |
| `400`  | 请求参数错误       |
| `401`  | 未授权             |
| `404`  | 资源不存在         |
| `409`  | 冲突（如重复注入） |
| `500`  | 服务器内部错误     |
| `503`  | 服务不可用         |

### 错误响应格式

```json
{
    "success": false,
    "detail": "错误详情描述",
    "error_code": "KNOWLEDGE_NOT_FOUND",
    "timestamp": "2026-02-06T10:30:00.000Z"
}
```

### 错误码

| 错误码                         | 描述           |
| ------------------------------ | -------------- |
| `KNOWLEDGE_NOT_FOUND`          | 知识不存在     |
| `NAMESPACE_NOT_FOUND`          | 命名空间不存在 |
| `INVALID_LIFECYCLE_TRANSITION` | 无效的状态转换 |
| `DUPLICATE_KNOWLEDGE`          | 重复的知识 ID  |
| `VECTOR_DIMENSION_MISMATCH`    | 向量维度不匹配 |
| `PERSISTENCE_ERROR`            | 持久化层错误   |
| `CAPACITY_EXCEEDED`            | 容量超限       |
| `VERSION_CONFLICT`             | 版本冲突       |

---

## API 端点

### 健康检查

#### `GET /health`

检查服务健康状态。

**响应**

```json
{
    "status": "healthy",
    "version": "3.4.0",
    "uptime_seconds": 3600,
    "persistence": {
        "status": "connected",
        "type": "postgresql"
    },
    "sync": {
        "status": "connected",
        "backend": "redis"
    },
    "features": {
        "dynamic_slots": true,
        "compression": true,
        "versioning": true
    }
}
```

#### `GET /health/ready`

就绪探针（用于 Kubernetes）。

**响应**

-   `200`: 服务就绪
-   `503`: 服务未就绪

---

### 知识管理

#### `POST /knowledge/inject`

注入单条知识。

**请求体**

```json
{
  "lu_id": "knowledge_001",
  "namespace": "default",
  "condition": "当用户询问产品价格时",
  "decision": "查询最新价格表并回答",
  "key_vector": [0.1, 0.2, 0.3, ...],
  "value_vector": [0.4, 0.5, 0.6, ...],
  "lifecycle_state": "probationary",
  "trust_tier": "s1_experience",
  "metadata": {
    "source": "governance_system",
    "created_by": "admin"
  }
}
```

**请求参数**

| 参数              | 类型    | 必填 | 描述                          |
| ----------------- | ------- | ---- | ----------------------------- |
| `lu_id`           | string  | ✅   | Learning Unit 唯一标识        |
| `namespace`       | string  |      | 命名空间，默认 `default`      |
| `condition`       | string  | ✅   | 触发条件描述                  |
| `decision`        | string  | ✅   | 决策/动作描述                 |
| `key_vector`      | float[] | ✅   | 条件编码向量                  |
| `value_vector`    | float[] | ✅   | 决策编码向量                  |
| `lifecycle_state` | string  |      | 初始状态，默认 `probationary` |
| `trust_tier`      | string  |      | 信任层级                      |
| `metadata`        | object  |      | 扩展元数据                    |

**响应**

```json
{
    "success": true,
    "data": {
        "lu_id": "knowledge_001",
        "namespace": "default",
        "lifecycle_state": "probationary",
        "slot_idx": 42,
        "created_at": "2026-02-06T10:30:00.000Z"
    }
}
```

---

#### `POST /knowledge/batch`

批量注入知识。

**请求体**

```json
{
  "items": [
    {
      "lu_id": "knowledge_001",
      "condition": "...",
      "decision": "...",
      "key_vector": [...],
      "value_vector": [...]
    },
    {
      "lu_id": "knowledge_002",
      "condition": "...",
      "decision": "...",
      "key_vector": [...],
      "value_vector": [...]
    }
  ],
  "namespace": "default",
  "skip_duplicates": true
}
```

**请求参数**

| 参数              | 类型            | 必填 | 描述                  |
| ----------------- | --------------- | ---- | --------------------- |
| `items`           | InjectRequest[] | ✅   | 知识列表              |
| `namespace`       | string          |      | 默认命名空间          |
| `skip_duplicates` | boolean         |      | 跳过重复，默认 `true` |

**响应**

```json
{
    "success": true,
    "data": {
        "total": 10,
        "success_count": 9,
        "failed_count": 1,
        "skipped_count": 0,
        "results": [
            { "lu_id": "knowledge_001", "success": true, "slot_idx": 42 },
            { "lu_id": "knowledge_002", "success": true, "slot_idx": 43 },
            {
                "lu_id": "knowledge_003",
                "success": false,
                "error": "Vector dimension mismatch"
            }
        ]
    }
}
```

---

#### `GET /knowledge/{namespace}/{lu_id}`

获取单条知识详情。

**路径参数**

| 参数        | 类型   | 描述             |
| ----------- | ------ | ---------------- |
| `namespace` | string | 命名空间         |
| `lu_id`     | string | Learning Unit ID |

**查询参数**

| 参数              | 类型    | 默认    | 描述             |
| ----------------- | ------- | ------- | ---------------- |
| `include_vectors` | boolean | `false` | 是否包含向量数据 |
| `include_history` | boolean | `false` | 是否包含版本历史 |

**响应**

```json
{
  "success": true,
  "data": {
    "lu_id": "knowledge_001",
    "namespace": "default",
    "condition": "当用户询问产品价格时",
    "decision": "查询最新价格表并回答",
    "lifecycle_state": "confirmed",
    "trust_tier": "s1_experience",
    "hit_count": 150,
    "reliability": 1.0,
    "version": 3,
    "created_at": "2026-01-15T10:00:00.000Z",
    "updated_at": "2026-02-01T14:30:00.000Z",
    "key_vector": [0.1, 0.2, ...],
    "value_vector": [0.4, 0.5, ...]
  }
}
```

---

#### `GET /knowledge/{namespace}`

查询知识列表。

**路径参数**

| 参数        | 类型   | 描述     |
| ----------- | ------ | -------- |
| `namespace` | string | 命名空间 |

**查询参数**

| 参数               | 类型    | 默认    | 描述               |
| ------------------ | ------- | ------- | ------------------ |
| `lifecycle_states` | string  |         | 状态过滤，逗号分隔 |
| `trust_tiers`      | string  |         | 层级过滤，逗号分隔 |
| `limit`            | int     | 100     | 返回数量 (1-1000)  |
| `offset`           | int     | 0       | 偏移量             |
| `include_vectors`  | boolean | `false` | 是否包含向量       |

**示例请求**

```
GET /knowledge/default?lifecycle_states=confirmed,probationary&limit=50
```

**响应**

```json
{
  "success": true,
  "data": {
    "items": [
      { "lu_id": "knowledge_001", "lifecycle_state": "confirmed", ... },
      { "lu_id": "knowledge_002", "lifecycle_state": "probationary", ... }
    ],
    "total": 150,
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

---

#### `DELETE /knowledge/{namespace}/{lu_id}`

删除知识。

**路径参数**

| 参数        | 类型   | 描述             |
| ----------- | ------ | ---------------- |
| `namespace` | string | 命名空间         |
| `lu_id`     | string | Learning Unit ID |

**查询参数**

| 参数     | 类型   | 描述                       |
| -------- | ------ | -------------------------- |
| `reason` | string | 删除原因（记录到审计日志） |

**响应**

```json
{
    "success": true,
    "data": {
        "lu_id": "knowledge_001",
        "deleted": true,
        "deleted_at": "2026-02-06T10:30:00.000Z"
    }
}
```

---

### 生命周期管理

#### 生命周期状态说明

| 状态           | 可靠性 (r) | 说明                 |
| -------------- | ---------- | -------------------- |
| `probationary` | 0.3        | 试用期，新注入的知识 |
| `confirmed`    | 1.0        | 已确认，验证通过     |
| `deprecated`   | 0.1        | 已弃用，准备下线     |
| `quarantined`  | 0.0        | 已隔离，不参与推理   |

#### 状态转换规则

```
             ┌────────────────────────────┐
             │                            │
             ▼                            │
    ┌──────────────┐                      │
    │ probationary │────────┐             │
    │   (r=0.3)    │        │             │
    └──────┬───────┘        │             │
           │ confirm        │ quarantine  │
           ▼                ▼             │
    ┌──────────────┐  ┌──────────────┐   │
    │  confirmed   │  │ quarantined  │◀──┘
    │   (r=1.0)    │  │   (r=0.0)    │
    └──────┬───────┘  └──────────────┘
           │ deprecate        ▲
           ▼                  │
    ┌──────────────┐          │
    │  deprecated  │──────────┘
    │   (r=0.1)    │  quarantine
    └──────────────┘
```

---

#### `PUT /lifecycle/update`

更新生命周期状态。

**请求体**

```json
{
    "lu_id": "knowledge_001",
    "namespace": "default",
    "new_state": "confirmed",
    "reason": "经人工验证，知识准确有效"
}
```

**请求参数**

| 参数        | 类型   | 必填 | 描述                     |
| ----------- | ------ | ---- | ------------------------ |
| `lu_id`     | string | ✅   | Learning Unit ID         |
| `namespace` | string |      | 命名空间，默认 `default` |
| `new_state` | string | ✅   | 新状态                   |
| `reason`    | string |      | 变更原因                 |

**响应**

```json
{
    "success": true,
    "data": {
        "lu_id": "knowledge_001",
        "old_state": "probationary",
        "new_state": "confirmed",
        "reliability": {
            "old": 0.3,
            "new": 1.0
        },
        "updated_at": "2026-02-06T10:30:00.000Z"
    }
}
```

---

#### `POST /lifecycle/quarantine`

紧急隔离知识。

**请求体**

```json
{
    "lu_id": "knowledge_001",
    "namespace": "default",
    "reason": "检测到输出异常，紧急隔离"
}
```

**请求参数**

| 参数        | 类型   | 必填 | 描述             |
| ----------- | ------ | ---- | ---------------- |
| `lu_id`     | string | ✅   | Learning Unit ID |
| `namespace` | string |      | 命名空间         |
| `reason`    | string | ✅   | 隔离原因（必填） |

**响应**

```json
{
    "success": true,
    "data": {
        "lu_id": "knowledge_001",
        "old_state": "confirmed",
        "new_state": "quarantined",
        "quarantined_at": "2026-02-06T10:30:00.000Z",
        "reason": "检测到输出异常，紧急隔离"
    }
}
```

---

### 槽位管理

#### `GET /slots/{namespace}/status`

获取槽位池状态。

**路径参数**

| 参数        | 类型   | 描述     |
| ----------- | ------ | -------- |
| `namespace` | string | 命名空间 |

**响应**

```json
{
    "success": true,
    "data": {
        "namespace": "default",
        "current_capacity": 128,
        "active_slots": 95,
        "free_slots": 33,
        "occupancy_ratio": 0.742,
        "scaling_policy": "auto_scale",
        "min_capacity": 16,
        "max_capacity": 512,
        "tiered_storage": {
            "hot_count": 64,
            "warm_count": 31,
            "cold_count": 0,
            "hot_capacity": 128,
            "warm_capacity": 512
        },
        "state_distribution": {
            "probationary": 20,
            "confirmed": 70,
            "deprecated": 5,
            "quarantined": 0
        }
    }
}
```

---

#### `POST /slots/{namespace}/resize`

手动调整槽位池容量。

**路径参数**

| 参数        | 类型   | 描述     |
| ----------- | ------ | -------- |
| `namespace` | string | 命名空间 |

**请求体**

```json
{
    "new_capacity": 256,
    "reason": "预期流量增加，提前扩容"
}
```

**响应**

```json
{
    "success": true,
    "data": {
        "namespace": "default",
        "old_capacity": 128,
        "new_capacity": 256,
        "resized_at": "2026-02-06T10:30:00.000Z"
    }
}
```

---

#### `GET /slots/{namespace}/scaling-history`

获取扩缩容历史。

**路径参数**

| 参数        | 类型   | 描述     |
| ----------- | ------ | -------- |
| `namespace` | string | 命名空间 |

**查询参数**

| 参数    | 类型 | 默认 | 描述     |
| ------- | ---- | ---- | -------- |
| `limit` | int  | 100  | 返回数量 |

**响应**

```json
{
    "success": true,
    "data": {
        "events": [
            {
                "timestamp": "2026-02-06T10:00:00.000Z",
                "event_type": "expand",
                "old_capacity": 64,
                "new_capacity": 128,
                "trigger_reason": "occupancy=0.87",
                "namespace": "default"
            },
            {
                "timestamp": "2026-02-05T14:30:00.000Z",
                "event_type": "shrink",
                "old_capacity": 128,
                "new_capacity": 64,
                "trigger_reason": "occupancy=0.25",
                "namespace": "default"
            }
        ],
        "total": 15
    }
}
```

---

#### `PUT /slots/{namespace}/config`

更新槽位池配置。

**路径参数**

| 参数        | 类型   | 描述     |
| ----------- | ------ | -------- |
| `namespace` | string | 命名空间 |

**请求体**

```json
{
    "scaling_policy": "auto_scale",
    "expand_threshold": 0.85,
    "shrink_threshold": 0.3,
    "min_capacity": 32,
    "max_capacity": 256
}
```

**响应**

```json
{
    "success": true,
    "data": {
        "namespace": "default",
        "config_updated": true,
        "updated_at": "2026-02-06T10:30:00.000Z"
    }
}
```

---

### 统计信息

#### `GET /statistics`

获取所有命名空间的统计摘要。

**响应**

```json
{
    "success": true,
    "data": {
        "total_namespaces": 3,
        "total_knowledge": 500,
        "total_capacity": 384,
        "overall_occupancy": 0.65,
        "by_namespace": {
            "default": {
                "count": 200,
                "capacity": 128,
                "confirmed": 150,
                "probationary": 50
            },
            "product": {
                "count": 180,
                "capacity": 128,
                "confirmed": 100,
                "probationary": 80
            },
            "support": {
                "count": 120,
                "capacity": 128,
                "confirmed": 80,
                "probationary": 40
            }
        },
        "by_state": {
            "probationary": 170,
            "confirmed": 330,
            "deprecated": 0,
            "quarantined": 0
        },
        "compression": {
            "enabled": true,
            "algorithm": "zstd",
            "compression_ratio": 6.2
        }
    }
}
```

---

#### `GET /statistics/{namespace}`

获取指定命名空间的详细统计。

**路径参数**

| 参数        | 类型   | 描述     |
| ----------- | ------ | -------- |
| `namespace` | string | 命名空间 |

**响应**

```json
{
    "success": true,
    "data": {
        "namespace": "default",
        "total_count": 200,
        "current_capacity": 128,
        "occupancy_ratio": 0.78,
        "state_distribution": {
            "probationary": 50,
            "confirmed": 150,
            "deprecated": 0,
            "quarantined": 0
        },
        "trust_distribution": {
            "s0_acceleration": 20,
            "s1_experience": 100,
            "s2_policy": 50,
            "s3_immutable": 30
        },
        "hit_statistics": {
            "total_hits": 15000,
            "avg_hit_count": 75,
            "max_hit_count": 500
        },
        "tiered_storage": {
            "hot_count": 64,
            "warm_count": 36,
            "cold_count": 0,
            "promotions": 120,
            "demotions": 45
        },
        "recent_activity": {
            "injected_24h": 10,
            "updated_24h": 5,
            "quarantined_24h": 0
        }
    }
}
```

---

### 审计日志

#### `GET /audit`

获取审计日志。

**查询参数**

| 参数        | 类型   | 默认 | 描述           |
| ----------- | ------ | ---- | -------------- |
| `namespace` | string |      | 按命名空间过滤 |
| `lu_id`     | string |      | 按 LU ID 过滤  |
| `limit`     | int    | 100  | 返回数量       |
| `offset`    | int    | 0    | 偏移量         |

**响应**

```json
{
    "success": true,
    "data": {
        "items": [
            {
                "id": "audit_001",
                "namespace": "default",
                "lu_id": "knowledge_001",
                "action": "lifecycle_update",
                "old_state": "probationary",
                "new_state": "confirmed",
                "reason": "验证通过",
                "operator": "admin",
                "timestamp": "2026-02-06T10:30:00.000Z"
            },
            {
                "id": "audit_002",
                "namespace": "default",
                "lu_id": "knowledge_002",
                "action": "inject",
                "new_state": "probationary",
                "operator": "governance_system",
                "timestamp": "2026-02-06T10:25:00.000Z"
            }
        ],
        "total": 500,
        "limit": 100,
        "offset": 0
    }
}
```

---

### 命名空间

#### `GET /namespaces`

列出所有命名空间。

**响应**

```json
{
    "success": true,
    "data": {
        "namespaces": [
            {
                "name": "default",
                "knowledge_count": 200,
                "capacity": 128,
                "created_at": "2026-01-01T00:00:00.000Z"
            },
            {
                "name": "product",
                "knowledge_count": 180,
                "capacity": 128,
                "created_at": "2026-01-15T00:00:00.000Z"
            }
        ]
    }
}
```

---

### Runtime 管理

#### `GET /runtimes`

列出已注册的 Runtime 实例。

**响应**

```json
{
    "success": true,
    "data": {
        "runtimes": [
            {
                "runtime_id": "runtime_gpu01",
                "namespaces": ["default", "product"],
                "status": "active",
                "last_heartbeat": "2026-02-06T10:29:55.000Z",
                "registered_at": "2026-02-01T08:00:00.000Z",
                "stats": {
                    "cached_knowledge": 150,
                    "sync_lag_ms": 50,
                    "gpu_memory_mb": 2048
                }
            },
            {
                "runtime_id": "runtime_gpu02",
                "namespaces": ["support"],
                "status": "active",
                "last_heartbeat": "2026-02-06T10:29:58.000Z"
            }
        ]
    }
}
```

---

### 版本控制

#### `GET /knowledge/{namespace}/{lu_id}/versions`

获取知识版本历史。

**路径参数**

| 参数        | 类型   | 描述             |
| ----------- | ------ | ---------------- |
| `namespace` | string | 命名空间         |
| `lu_id`     | string | Learning Unit ID |

**响应**

```json
{
    "success": true,
    "data": {
        "lu_id": "knowledge_001",
        "current_version": 3,
        "versions": [
            {
                "version": 3,
                "created_at": "2026-02-06T10:00:00.000Z",
                "changes": ["decision updated"],
                "author": "admin"
            },
            {
                "version": 2,
                "created_at": "2026-02-01T14:00:00.000Z",
                "changes": ["lifecycle_state: probationary -> confirmed"],
                "author": "governance_system"
            },
            {
                "version": 1,
                "created_at": "2026-01-15T10:00:00.000Z",
                "changes": ["initial creation"],
                "author": "governance_system"
            }
        ]
    }
}
```

---

#### `POST /knowledge/{namespace}/{lu_id}/rollback`

回滚到指定版本。

**路径参数**

| 参数        | 类型   | 描述             |
| ----------- | ------ | ---------------- |
| `namespace` | string | 命名空间         |
| `lu_id`     | string | Learning Unit ID |

**请求体**

```json
{
    "target_version": 2,
    "reason": "版本 3 存在问题，回滚到版本 2"
}
```

**响应**

```json
{
    "success": true,
    "data": {
        "lu_id": "knowledge_001",
        "old_version": 3,
        "new_version": 4,
        "rolled_back_to": 2,
        "rolled_back_at": "2026-02-06T10:30:00.000Z"
    }
}
```

---

## 数据模型

### LifecycleState (生命周期状态)

```typescript
enum LifecycleState {
  PROBATIONARY = "probationary"  // 试用期
  CONFIRMED = "confirmed"         // 已确认
  DEPRECATED = "deprecated"       // 已弃用
  QUARANTINED = "quarantined"     // 已隔离
}
```

### TrustTier (信任层级)

```typescript
enum TrustTier {
  S0_ACCELERATION = "s0_acceleration"  // 可丢失的缓存知识
  S1_EXPERIENCE = "s1_experience"      // 可回滚的经验知识
  S2_POLICY = "s2_policy"              // 需审批的策略知识
  S3_IMMUTABLE = "s3_immutable"        // 只读的核心知识
}
```

### ScalingPolicy (扩缩容策略)

```typescript
enum ScalingPolicy {
  FIXED = "fixed"                // 固定容量
  AUTO_SCALE = "auto_scale"      // 自动扩缩容
  MEMORY_BUDGET = "memory_budget" // 基于内存预算
}
```

### KnowledgeRecord (知识记录)

```typescript
interface KnowledgeRecord {
    lu_id: string; // Learning Unit ID
    namespace: string; // 命名空间
    condition: string; // 触发条件
    decision: string; // 决策/动作
    key_vector?: number[]; // 条件编码向量
    value_vector?: number[]; // 决策编码向量
    lifecycle_state: LifecycleState;
    trust_tier?: TrustTier;
    reliability: number; // 可靠性 (0.0-1.0)
    hit_count: number; // 命中次数
    version: number; // 版本号
    created_at: string; // ISO 8601
    updated_at: string; // ISO 8601
    metadata?: Record<string, any>;
}
```

### SlotPoolStatus (槽位池状态)

```typescript
interface SlotPoolStatus {
    namespace: string;
    current_capacity: number;
    active_slots: number;
    free_slots: number;
    occupancy_ratio: number;
    scaling_policy: ScalingPolicy;
    min_capacity: number;
    max_capacity: number;
    tiered_storage: {
        hot_count: number;
        warm_count: number;
        cold_count: number;
    };
}
```

### AuditLogEntry (审计日志条目)

```typescript
interface AuditLogEntry {
    id: string;
    namespace: string;
    lu_id?: string;
    action:
        | "inject"
        | "update"
        | "delete"
        | "lifecycle_update"
        | "quarantine"
        | "resize";
    old_state?: string;
    new_state?: string;
    reason?: string;
    operator?: string;
    timestamp: string; // ISO 8601
    details?: Record<string, any>;
}
```

---

## 客户端示例

### Python (使用 aga.client)

```python
from aga.client import AGAClient, AsyncAGAClient

# 同步客户端
with AGAClient("http://portal:8081", api_key="your_api_key") as client:
    # 注入知识
    result = client.inject_knowledge(
        lu_id="knowledge_001",
        condition="当用户询问产品价格时",
        decision="查询最新价格表并回答",
        key_vector=[0.1, 0.2, 0.3] * 20,  # 64 维
        value_vector=[0.4, 0.5, 0.6] * 1366,  # 4096 维
        namespace="product",
        lifecycle_state="probationary",
    )
    print(f"注入成功: {result}")

    # 确认知识
    client.confirm("knowledge_001", namespace="product", reason="验证通过")

    # 查询槽位状态
    status = client.get_slot_status(namespace="product")
    print(f"槽位占用率: {status['occupancy_ratio']:.1%}")

    # 手动扩容
    client.resize_slots(namespace="product", new_capacity=256)

    # 查询知识
    items = client.query_knowledge(
        namespace="product",
        lifecycle_states=["confirmed"],
        limit=50,
    )
    print(f"已确认知识: {items['data']['total']} 条")

# 异步客户端
import asyncio

async def main():
    async with AsyncAGAClient("http://portal:8081") as client:
        # 并发注入
        tasks = [
            client.inject_knowledge(
                lu_id=f"batch_{i}",
                condition=f"条件 {i}",
                decision=f"决策 {i}",
                key_vector=[0.1] * 64,
                value_vector=[0.1] * 4096,
            )
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        print(f"批量注入完成: {len(results)} 条")

asyncio.run(main())
```

### cURL

```bash
# 健康检查
curl http://portal:8081/health

# 注入知识
curl -X POST http://portal:8081/knowledge/inject \
  -H "Content-Type: application/json" \
  -d '{
    "lu_id": "knowledge_001",
    "condition": "当用户询问产品价格时",
    "decision": "查询最新价格表并回答",
    "key_vector": [0.1, 0.2, 0.3, ...],
    "value_vector": [0.4, 0.5, 0.6, ...],
    "namespace": "default"
  }'

# 确认知识
curl -X PUT http://portal:8081/lifecycle/update \
  -H "Content-Type: application/json" \
  -d '{
    "lu_id": "knowledge_001",
    "new_state": "confirmed",
    "namespace": "default",
    "reason": "验证通过"
  }'

# 查询槽位状态
curl "http://portal:8081/slots/default/status"

# 手动扩容
curl -X POST http://portal:8081/slots/default/resize \
  -H "Content-Type: application/json" \
  -d '{
    "new_capacity": 256,
    "reason": "预期流量增加"
  }'

# 查询知识
curl "http://portal:8081/knowledge/default?lifecycle_states=confirmed&limit=50"

# 隔离知识
curl -X POST http://portal:8081/lifecycle/quarantine \
  -H "Content-Type: application/json" \
  -d '{
    "lu_id": "knowledge_001",
    "namespace": "default",
    "reason": "检测到异常输出"
  }'

# 获取版本历史
curl "http://portal:8081/knowledge/default/knowledge_001/versions"

# 回滚版本
curl -X POST http://portal:8081/knowledge/default/knowledge_001/rollback \
  -H "Content-Type: application/json" \
  -d '{
    "target_version": 2,
    "reason": "版本 3 存在问题"
  }'
```

---

## API 一致性说明

### `aga.api` vs `aga.portal` 差异

| 端点         | `aga.api` (单机)               | `aga.portal` (分离部署)             |
| ------------ | ------------------------------ | ----------------------------------- |
| 批量注入     | `POST /knowledge/inject/batch` | `POST /knowledge/batch`             |
| 查询知识     | `POST /knowledge/query` (Body) | `GET /knowledge/{ns}` (Query)       |
| 更新生命周期 | `POST /lifecycle/update`       | `PUT /lifecycle/update`             |
| 审计日志     | `GET /audit/{namespace}`       | `GET /audit` (通用)                 |
| 槽位管理     | `GET /slots/{ns}/free`         | `GET /slots/{ns}/status`            |
| 动态扩容     | ❌ 不支持                      | `POST /slots/{ns}/resize`           |
| 版本控制     | ❌ 不支持                      | `GET /knowledge/{ns}/{id}/versions` |
| Runtime 管理 | ❌ 不支持                      | `GET /runtimes`                     |

**推荐**: 使用 `aga.client.AGAClient`，它兼容两种 API。

---

## 版本历史

| 版本  | 日期       | 变更                            |
| ----- | ---------- | ------------------------------- |
| 3.4.0 | 2026-02-06 | 新增动态槽位管理、版本控制 API  |
| 3.3.0 | 2026-02-04 | 新增 KV 压缩、分层存储支持      |
| 3.2.0 | 2026-02-02 | 分离部署架构，新增 Runtime 管理 |
| 3.1.0 | 2026-01-15 | 新增信任层级，优化批量操作      |
| 3.0.0 | 2026-01-01 | Portal/Runtime 分离架构         |
