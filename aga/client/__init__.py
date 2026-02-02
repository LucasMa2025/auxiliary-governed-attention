"""
AGA 客户端库

为外部系统（治理系统、自学习系统）提供与 AGA Portal 的集成。

架构位置：
```
可控自学习系统  →  治理系统  →  [Bridge/Client]  →  AGA Portal  →  AGA Runtime
      ↑                                                              ↓
      └──────────────────────── 反馈 ←────────────────────────────────┘
```

使用场景：
1. 治理系统注入知识
2. 监控系统查询统计
3. 运维系统管理生命周期
4. 开发测试工具

示例：

```python
from aga.client import AGAClient, AsyncAGAClient

# 同步客户端
client = AGAClient("http://portal:8081")
result = client.inject_knowledge(
    lu_id="knowledge_001",
    condition="当用户询问...",
    decision="应该回答...",
    key_vector=[...],
    value_vector=[...],
)

# 异步客户端
async with AsyncAGAClient("http://portal:8081") as client:
    result = await client.inject_knowledge(...)
```
"""

from .portal_client import AGAClient, AsyncAGAClient

__all__ = [
    "AGAClient",
    "AsyncAGAClient",
]
