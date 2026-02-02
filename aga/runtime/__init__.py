"""
AGA Runtime - 与 LLM 同部署的执行模块

Runtime 是 AGA 系统的推理执行组件，与 LLM 模型同机部署。

主要职责：
- 运行时知识融合（推理增强）
- 从 Portal 同步知识
- 状态上报

架构特点：
- 与 LLM 同部署（需要 GPU）
- 订阅 Portal 同步消息
- 本地内存缓存知识槽位
- 支持多实例水平扩展

使用示例：

```python
from aga.runtime import RuntimeAgent, AGARuntime
from aga.config import RuntimeConfig

# 创建 Runtime
config = RuntimeConfig.for_production(
    instance_id="runtime-001",
    portal_url="http://portal:8081",
    redis_host="localhost",
)
agent = RuntimeAgent(config)

# 初始化并附加到模型
await agent.initialize()
aga_module = agent.attach_to_layer(transformer_layer)

# 启动同步
await agent.start()

# 使用
output = aga_module.forward(hidden_states, attention_mask)
```
"""

from .agent import RuntimeAgent
from .aga_runtime import AGARuntime
from .cache import LocalCache

__all__ = [
    "RuntimeAgent",
    "AGARuntime",
    "LocalCache",
]
