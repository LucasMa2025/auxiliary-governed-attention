"""
Condition 生成器

从分片文本中生成高质量的 condition，用于检索匹配。

condition 是检索的关键 — 它决定了知识能否被 aga-core 正确召回。
好的 condition 应该:
  1. 包含关键语义信息（能被编码器正确编码）
  2. 包含文档结构上下文（标题、章节）
  3. 足够简洁（避免稀释语义密度）

策略:
  - first_sentence: 取第一句话（默认，简单快速）
  - summary: 取前 N 个词作为摘要
  - title_context: 使用文档标题 + 章节标题 + 首句（推荐）
  - keyword: 提取关键词

配置:
  chunker:
    condition_mode: "title_context"
"""

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ConditionGenerator:
    """
    condition 生成器

    从分片文本中生成高质量的 condition，用于检索匹配。

    Args:
        mode: 生成策略
            - "first_sentence": 取第一句话
            - "summary": 取前 N 个词
            - "title_context": 文档标题 + 章节标题 + 首句（推荐）
            - "keyword": 提取关键词
    """

    def __init__(self, mode: str = "title_context"):
        valid_modes = {"first_sentence", "summary", "title_context", "keyword"}
        if mode not in valid_modes:
            logger.warning(
                f"未知的 condition 生成模式: {mode}，"
                f"回退到 first_sentence。支持: {valid_modes}"
            )
            mode = "first_sentence"
        self.mode = mode

    def generate(
        self,
        text: str,
        title: str = "",
        section: str = "",
        chunk_index: int = 0,
    ) -> str:
        """
        生成 condition

        Args:
            text: 分片文本（decision 内容）
            title: 文档标题
            section: 章节标题
            chunk_index: 分片序号

        Returns:
            生成的 condition 文本
        """
        if not text:
            return title or section or ""

        if self.mode == "first_sentence":
            return self._first_sentence(text)
        elif self.mode == "title_context":
            return self._title_context(text, title, section)
        elif self.mode == "keyword":
            return self._keyword(text, title)
        elif self.mode == "summary":
            return self._summary(text)
        else:
            return self._first_sentence(text)

    def _title_context(self, text: str, title: str, section: str) -> str:
        """
        标题 + 章节 + 首句

        格式: "文档标题 > 章节标题 > 首句"
        这是推荐的生成模式，因为它:
        1. 提供了文档级上下文
        2. 提供了章节级上下文
        3. 保留了内容级语义

        Example:
            title="心脏解剖学", section="心脏结构"
            text="心脏由四个腔室组成..."
            → "心脏解剖学 > 心脏结构 > 心脏由四个腔室组成。"
        """
        parts = []
        if title:
            parts.append(title.strip())
        if section:
            parts.append(section.strip())
        first = self._first_sentence(text)
        if first:
            parts.append(first)

        if parts:
            return " > ".join(parts)

        # 回退
        return text[:80].strip()

    def _keyword(self, text: str, title: str = "") -> str:
        """
        TF-IDF 关键词提取

        简单的关键词提取（不依赖外部 NLP 库）。
        对中英文混合文本进行词频统计，取 top-5 关键词。

        Example:
            title="深度学习"
            text="Transformer 模型在自然语言处理中广泛应用..."
            → "深度学习: transformer 自然 语言 处理 模型"
        """
        words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]{2,}", text)

        # 词频统计
        freq: Dict[str, int] = {}
        for w in words:
            w_lower = w.lower()
            freq[w_lower] = freq.get(w_lower, 0) + 1

        # 过滤停用词（简单版）
        stop_words = {
            "the", "is", "at", "which", "on", "in", "of", "to", "and",
            "for", "with", "that", "this", "from", "by", "are", "was",
            "be", "an", "as", "it", "or", "but", "not", "no", "if",
            "its", "has", "had", "have", "can", "may", "will",
            "的", "了", "在", "是", "我", "有", "和", "就", "不",
            "人", "都", "一", "一个", "上", "也", "很", "到", "说",
        }
        for sw in stop_words:
            freq.pop(sw, None)

        # 取 top-5 关键词
        top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = " ".join(w for w, _ in top_words)

        if title:
            return f"{title}: {keywords}"
        return keywords if keywords else text[:80].strip()

    def _summary(self, text: str) -> str:
        """
        前 N 个词作为摘要

        取前 15 个词（中英文混合按空格和字符分割）。

        Example:
            text="心脏是人体最重要的器官之一，负责将血液..."
            → "心脏是人体最重要的器官之一，负责将血液"
        """
        # 对中文文本按字符计算
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        total_chars = len(text)

        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            # 中文为主: 取前 30 个字符
            return text[:30].strip()
        else:
            # 英文为主: 取前 15 个词
            words = text.split()[:15]
            return " ".join(words)

    @staticmethod
    def _first_sentence(text: str) -> str:
        """
        提取第一句话

        支持中英文句子结束符。
        如果第一句话过长（超过文本 50%），取前 80 个字符。

        Example:
            text="心脏由四个腔室组成。左心室负责..."
            → "心脏由四个腔室组成。"
        """
        if not text:
            return ""

        match = re.search(r"[。！？.!?]\s*", text)
        if match and match.end() < len(text) * 0.5:
            return text[: match.end()].strip()

        # 没有找到句子结束符，取前 80 个字符
        if len(text) > 80:
            truncated = text[:80]
            last_space = truncated.rfind(" ")
            if last_space > 40:
                return truncated[:last_space] + "..."
            return truncated + "..."

        return text.strip()

    def get_stats(self) -> dict:
        """获取生成器统计"""
        return {
            "type": "ConditionGenerator",
            "mode": self.mode,
        }

    def __repr__(self) -> str:
        return f"ConditionGenerator(mode={self.mode!r})"
