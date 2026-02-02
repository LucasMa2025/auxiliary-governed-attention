# AGA REST API

为外部治理系统提供的知识管理 REST API 接口。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    外部治理系统                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              AGAClient / AsyncAGAClient                │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │ HTTP
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      AGA API 服务                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   │
│  │   路由层     │ → │   服务层     │ → │   持久化层    │   │
│  │ (routes.py)  │   │ (service.py) │   │  (adapter)    │   │
│  └──────────────┘   └──────────────┘   └───────────────┘   │
│                            │                                │
│                            ▼                                │
│                   ┌──────────────────┐                     │
│                   │  KnowledgeWriter │                     │
│                   │  (异步写入器)    │                     │
│                   └──────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 模块说明

| 模块     | 文件         | 职责                             |
| -------- | ------------ | -------------------------------- |
| 应用入口 | `app.py`     | FastAPI 应用创建、配置、生命周期 |
| 路由层   | `routes.py`  | HTTP 协议转换，参数验证          |
| 服务层   | `service.py` | 核心业务逻辑                     |
| 数据模型 | `models.py`  | Pydantic 请求/响应模型           |
| 客户端库 | `client.py`  | 供外部系统使用的 HTTP 客户端     |

## 快速开始

### 启动服务

```bash
# 默认配置启动
python -m aga.api

# 指定端口
python -m aga.api --port 8082

# 自定义维度
python -m aga.api --hidden-dim 2048 --bottleneck-dim 32 --num-slots 200

# 开发模式（自动重载）
python -m aga.api --reload

# 禁用持久化
python -m aga.api --no-persistence
```

### 访问文档

启动后访问：

-   Swagger UI: http://localhost:8081/docs
-   ReDoc: http://localhost:8081/redoc
-   OpenAPI JSON: http://localhost:8081/openapi.json

## API 端点

### 健康检查

```
GET /health
```

返回服务状态、版本、运行时间等信息。

### 命名空间

```
GET  /namespaces              # 列出所有命名空间
DELETE /namespaces/{namespace}  # 删除命名空间
```

### 知识管理

```
POST /knowledge/inject         # 注入单条知识
POST /knowledge/inject/batch   # 批量注入
GET  /knowledge/{namespace}/{lu_id}  # 获取知识
POST /knowledge/query          # 查询知识列表
DELETE /knowledge/{namespace}/{lu_id}  # 删除知识
```

### 生命周期管理

```
POST /lifecycle/update         # 更新生命周期状态
POST /lifecycle/update/batch   # 批量更新
POST /lifecycle/quarantine     # 隔离知识
```

### 槽位管理

```
GET /slots/{namespace}/free           # 查找空闲槽位
GET /slots/{namespace}/{slot_idx}     # 获取槽位信息
```

### 统计

```
GET /statistics                # 所有命名空间统计
GET /statistics/{namespace}    # 单个命名空间统计
GET /statistics/writer         # 写入器统计
```

### 审计日志

```
GET /audit/{namespace}         # 获取审计日志
```

## 数据模型

### 生命周期状态

| 状态   | 值             | 可靠性 | 说明         |
| ------ | -------------- | ------ | ------------ |
| 试用期 | `probationary` | 0.3    | 新注入的知识 |
| 已确认 | `confirmed`    | 1.0    | 验证通过     |
| 已弃用 | `deprecated`   | 0.1    | 准备下线     |
| 已隔离 | `quarantined`  | 0.0    | 不参与推理   |

状态转换：

```
probationary ─→ confirmed ─→ deprecated ─→ quarantined
      │              │
      └──────────────┴─────────────────→ quarantined (紧急)
```

### 信任层级

| 层级    | 值                | 说明             |
| ------- | ----------------- | ---------------- |
| S0 加速 | `s0_acceleration` | 可丢失的缓存知识 |
| S1 经验 | `s1_experience`   | 可回滚的经验知识 |
| S2 策略 | `s2_policy`       | 需审批的策略知识 |
| S3 禁止 | `s3_immutable`    | 只读的核心知识   |

## 客户端使用

### Python 同步客户端

```python
from aga.api import AGAClient

# 创建客户端
client = AGAClient("http://localhost:8081")

# 健康检查
health = client.health_check()
print(f"Status: {health['status']}")

# 注入知识
result = client.inject_knowledge(
    lu_id="LU_001",
    condition="capital of France",
    decision="Paris",
    key_vector=[0.1] * 64,      # bottleneck_dim 维
    value_vector=[0.1] * 4096,  # hidden_dim 维
    namespace="default",
    lifecycle_state="probationary",
)
print(f"Injected to slot: {result['slot_idx']}")

# 确认知识
client.update_lifecycle("LU_001", "confirmed")

# 查询知识
knowledge_list = client.query_knowledge(
    namespace="default",
    lifecycle_states=["confirmed"],
    limit=10,
)

# 隔离知识
client.quarantine_knowledge(
    lu_id="LU_001",
    reason="检测到异常输出",
)

# 关闭客户端
client.close()
```

### Python 异步客户端

```python
from aga.api import AsyncAGAClient
import asyncio

async def main():
    async with AsyncAGAClient("http://localhost:8081") as client:
        # 健康检查
        health = await client.health_check()

        # 批量注入
        items = [
            {
                "lu_id": f"LU_{i:03d}",
                "condition": f"condition {i}",
                "decision": f"decision {i}",
                "key_vector": [0.1] * 64,
                "value_vector": [0.1] * 4096,
            }
            for i in range(10)
        ]
        result = await client.batch_inject(items)
        print(f"Injected: {result['success_count']}/{result['total']}")

asyncio.run(main())
```

### cURL 示例

```bash
# 健康检查
curl http://localhost:8081/health

# 注入知识
curl -X POST http://localhost:8081/knowledge/inject \
  -H "Content-Type: application/json" \
  -d '{
    "lu_id": "LU_001",
    "namespace": "default",
    "condition": "capital of France",
    "decision": "Paris",
    "key_vector": [0.1, 0.2, ...],
    "value_vector": [0.1, 0.2, ...],
    "lifecycle_state": "probationary"
  }'

# 查询知识
curl -X POST http://localhost:8081/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "default",
    "lifecycle_states": ["confirmed"],
    "limit": 10
  }'

# 更新生命周期
curl -X POST http://localhost:8081/lifecycle/update \
  -H "Content-Type: application/json" \
  -d '{
    "lu_id": "LU_001",
    "namespace": "default",
    "new_state": "confirmed",
    "reason": "验证通过"
  }'

# 隔离知识
curl -X POST http://localhost:8081/lifecycle/quarantine \
  -H "Content-Type: application/json" \
  -d '{
    "lu_id": "LU_001",
    "namespace": "default",
    "reason": "检测到异常"
  }'
```

## 配置选项

### 命令行参数

| 参数                 | 默认值            | 说明             |
| -------------------- | ----------------- | ---------------- |
| `--host`             | 0.0.0.0           | 绑定地址         |
| `--port`             | 8081              | 绑定端口         |
| `--reload`           | false             | 开发模式自动重载 |
| `--workers`          | 1                 | 工作进程数       |
| `--hidden-dim`       | 4096              | 隐藏层维度       |
| `--bottleneck-dim`   | 64                | 瓶颈层维度       |
| `--num-slots`        | 100               | 每命名空间槽位数 |
| `--num-heads`        | 32                | 注意力头数       |
| `--no-persistence`   | false             | 禁用持久化       |
| `--persistence-type` | sqlite            | 持久化类型       |
| `--db-path`          | ./aga_api_data.db | 数据库路径       |
| `--no-writer`        | false             | 禁用写入器       |
| `--no-quality-check` | false             | 禁用质量评估     |
| `--no-cors`          | false             | 禁用 CORS        |

### 代码配置

```python
from aga.api import create_api_app, ServiceConfig

# 使用 ServiceConfig
config = ServiceConfig(
    hidden_dim=2048,
    bottleneck_dim=32,
    num_slots=200,
    persistence_enabled=True,
    persistence_type="postgres",
    persistence_path="postgresql://user:pass@localhost/aga",
    writer_enabled=True,
    enable_quality_assessment=True,
)

# 创建应用
app = create_api_app(
    hidden_dim=config.hidden_dim,
    bottleneck_dim=config.bottleneck_dim,
    # ... 其他参数
)
```

## 错误处理

### HTTP 状态码

| 状态码 | 说明                       |
| ------ | -------------------------- |
| 200    | 成功                       |
| 400    | 请求参数错误               |
| 404    | 资源未找到                 |
| 500    | 服务器内部错误             |
| 503    | 服务不可用（如无空闲槽位） |

### 错误响应格式

```json
{
    "detail": "错误描述"
}
```

## 与外部治理系统集成

### 持续学习系统集成

```python
from aga.api import AGAClient

class ContinuousLearningSystem:
    def __init__(self, aga_url: str):
        self.aga_client = AGAClient(aga_url)

    def process_learning_unit(self, lu):
        """处理 Learning Unit"""
        # 1. 编码为向量
        key_vector = self.encode_condition(lu.condition)
        value_vector = self.encode_decision(lu.decision)

        # 2. 注入到 AGA
        result = self.aga_client.inject_knowledge(
            lu_id=lu.id,
            condition=lu.condition,
            decision=lu.decision,
            key_vector=key_vector,
            value_vector=value_vector,
            lifecycle_state="probationary",
        )

        return result

    def confirm_knowledge(self, lu_id: str):
        """确认知识"""
        return self.aga_client.update_lifecycle(
            lu_id=lu_id,
            new_state="confirmed",
            reason="验证通过",
        )

    def handle_error_feedback(self, lu_id: str, error_reason: str):
        """处理错误反馈"""
        return self.aga_client.quarantine_knowledge(
            lu_id=lu_id,
            reason=error_reason,
        )
```

### 监控系统集成

```python
from aga.api import AGAClient

def collect_aga_metrics(aga_url: str):
    """收集 AGA 指标"""
    client = AGAClient(aga_url)

    # 健康状态
    health = client.health_check()

    # 统计信息
    stats = client.get_all_statistics()

    return {
        "status": health["status"],
        "uptime": health["uptime_seconds"],
        "total_namespaces": stats["total_namespaces"],
        "total_knowledge": stats["total_knowledge"],
        "namespaces": {
            ns: {
                "active_slots": ns_stats["active_slots"],
                "total_hits": ns_stats["total_hits"],
            }
            for ns, ns_stats in stats["namespaces"].items()
        }
    }
```

## 性能建议

1. **批量操作**: 使用 `/inject/batch` 和 `/lifecycle/update/batch` 提高效率
2. **连接复用**: 使用客户端上下文管理器或保持客户端实例
3. **异步客户端**: 在异步应用中使用 `AsyncAGAClient`
4. **持久化选择**: 生产环境使用 PostgreSQL，测试使用 SQLite
5. **向量缓存**: 避免重复编码相同的条件/决策

## 安全建议

1. **网络隔离**: 在内网部署，不暴露到公网
2. **认证**: 实现 API Key 认证（可通过 `api_key` 参数）
3. **HTTPS**: 生产环境使用 HTTPS
4. **审计**: 定期检查审计日志
