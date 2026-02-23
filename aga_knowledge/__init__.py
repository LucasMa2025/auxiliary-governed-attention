"""
aga-knowledge — AGA 知识管理包（明文 KV 版本）

独立于 aga-core 的知识管理系统，负责：
- 知识的持久化存储（SQLite / PostgreSQL / Redis / Memory）
- Portal API（FastAPI，无 GPU 依赖）
- 消息同步（Redis / Memory）
- 配置适配器（YAML / 环境变量 / 字典）
- 知识编码器（SentenceTransformer / SimpleHash）
- 知识分片器（FixedSize / Sentence / Semantic / SlidingWindow）
- 知识召回器（BaseRetriever 协议适配器）
- 版本控制与文本压缩

明文 KV 版本说明：
    Portal 只存储和同步明文 condition/decision 文本对。
    向量化编码由 encoder 模块或 aga-core Runtime 在推理时按需处理。
    这确保了编码器一致性和系统解耦。

安装:
    pip install aga-knowledge

使用:
    from aga_knowledge import KnowledgeManager
    from aga_knowledge.portal import create_portal_app
    from aga_knowledge.config import PortalConfig
"""

__version__ = "0.3.0"

from .manager.knowledge_manager import KnowledgeManager
from .alignment import AGACoreAlignment

__all__ = [
    "KnowledgeManager",
    "AGACoreAlignment",
    "__version__",
]
