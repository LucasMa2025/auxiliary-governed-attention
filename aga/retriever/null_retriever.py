"""
aga/retriever/null_retriever.py — 空召回器（默认）

当用户未配置外部召回器时使用。不执行任何检索操作，
AGA 仅使用 KVStore 中已有的知识。

这是 aga-core 的默认行为：
  - 研发场景：手动 register() 知识到 KVStore
  - 生产场景：通过 aga-knowledge 同步知识到 KVStore
  - 两种场景下 NullRetriever 都是合理的默认值
"""
import logging
from typing import List, Dict, Any

from .base import BaseRetriever, RetrievalQuery, RetrievalResult

logger = logging.getLogger(__name__)


class NullRetriever(BaseRetriever):
    """
    空召回器 — 不执行任何检索

    作为 aga-core 的默认召回器，确保在没有外部知识源时
    AGA 仍然可以正常工作（仅使用 KVStore 已有知识）。
    """

    def __init__(self):
        self._call_count = 0

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """不执行检索，返回空列表"""
        self._call_count += 1
        return []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "NullRetriever",
            "call_count": self._call_count,
            "description": "No external retrieval configured",
        }

    def __repr__(self) -> str:
        return "NullRetriever()"
