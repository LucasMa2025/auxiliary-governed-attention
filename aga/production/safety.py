"""
AGA 安全分类器

提供知识注入时的安全检查能力：
- 内容安全检测
- 敏感信息过滤
- 风险等级评估

设计原则：
- 可插拔架构，支持多种安全检测后端
- 默认使用基于规则的快速检测
- 可扩展支持 LLM 辅助检测
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    """安全等级"""
    SAFE = "safe"               # 安全，可直接注入
    LOW_RISK = "low_risk"       # 低风险，可注入但需监控
    MEDIUM_RISK = "medium_risk" # 中风险，需人工审核
    HIGH_RISK = "high_risk"     # 高风险，建议拒绝
    BLOCKED = "blocked"         # 阻止，直接拒绝


class SafetyCategory(str, Enum):
    """安全类别"""
    CLEAN = "clean"                     # 无问题
    SENSITIVE_INFO = "sensitive_info"   # 敏感信息（PII、密码等）
    HARMFUL_CONTENT = "harmful_content" # 有害内容
    BIAS = "bias"                       # 偏见内容
    MISINFORMATION = "misinformation"   # 错误信息
    PROMPT_INJECTION = "prompt_injection" # 提示注入风险
    UNKNOWN = "unknown"                 # 未知


@dataclass
class SafetyResult:
    """安全检查结果"""
    level: SafetyLevel
    categories: List[SafetyCategory] = field(default_factory=list)
    score: float = 1.0  # 安全分数 0-1，1 表示完全安全
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestion: str = ""
    
    @property
    def is_safe(self) -> bool:
        """是否安全"""
        return self.level in (SafetyLevel.SAFE, SafetyLevel.LOW_RISK)
    
    @property
    def needs_review(self) -> bool:
        """是否需要人工审核"""
        return self.level == SafetyLevel.MEDIUM_RISK
    
    @property
    def is_blocked(self) -> bool:
        """是否被阻止"""
        return self.level in (SafetyLevel.HIGH_RISK, SafetyLevel.BLOCKED)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "categories": [c.value for c in self.categories],
            "score": self.score,
            "issues": self.issues,
            "suggestion": self.suggestion,
            "is_safe": self.is_safe,
            "needs_review": self.needs_review,
            "is_blocked": self.is_blocked,
        }


class BaseSafetyClassifier(ABC):
    """安全分类器基类"""
    
    @abstractmethod
    def classify(self, text: str) -> SafetyResult:
        """
        对文本进行安全分类
        
        Args:
            text: 待检查的文本
        
        Returns:
            SafetyResult
        """
        pass
    
    @abstractmethod
    def classify_knowledge(
        self, 
        condition: str, 
        decision: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> SafetyResult:
        """
        对知识条目进行安全分类
        
        Args:
            condition: 条件描述
            decision: 决策描述
            metadata: 元数据
        
        Returns:
            SafetyResult
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取分类器信息"""
        return {
            "name": self.__class__.__name__,
            "type": "base",
        }


class RuleBasedSafetyClassifier(BaseSafetyClassifier):
    """
    基于规则的安全分类器
    
    使用正则表达式和关键词匹配进行快速安全检测。
    适用于：
    - 敏感信息检测（PII、密码、API Key 等）
    - 明显有害内容过滤
    - 提示注入检测
    
    优点：速度快、无外部依赖
    缺点：覆盖面有限，需要持续更新规则
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 敏感信息模式
        self._sensitive_patterns = [
            # 邮箱
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
             "email", SafetyLevel.LOW_RISK),
            # 手机号（中国）
            (r'\b1[3-9]\d{9}\b', 
             "phone_cn", SafetyLevel.MEDIUM_RISK),
            # 身份证号
            (r'\b\d{17}[\dXx]\b', 
             "id_card_cn", SafetyLevel.HIGH_RISK),
            # API Key 模式
            (r'\b(sk-|pk_|api_|key_|secret_)[a-zA-Z0-9]{20,}\b', 
             "api_key", SafetyLevel.BLOCKED),
            # 密码模式
            (r'(?i)(password|passwd|pwd|密码)\s*[:=]\s*\S+', 
             "password", SafetyLevel.HIGH_RISK),
            # 银行卡号
            (r'\b\d{16,19}\b', 
             "bank_card", SafetyLevel.MEDIUM_RISK),
            # IP 地址（内网）
            (r'\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b',
             "internal_ip", SafetyLevel.LOW_RISK),
        ]
        
        # 有害内容关键词（需要根据实际需求扩展）
        self._harmful_keywords = {
            SafetyLevel.HIGH_RISK: [
                # 危险行为指导
                "如何制造", "怎么制作", "制造方法", 
                "如何攻击", "黑客教程",
            ],
            SafetyLevel.MEDIUM_RISK: [
                # 可能有问题的内容
                "绕过限制", "破解", "无视安全",
            ],
        }
        
        # 提示注入模式
        self._injection_patterns = [
            # 角色扮演注入
            (r'(?i)(ignore|forget|disregard)\s+(previous|all|above)\s+(instructions?|rules?|constraints?)', 
             "role_override"),
            # 系统提示覆盖
            (r'(?i)you\s+are\s+(now|actually|really)\s+a', 
             "persona_injection"),
            # 越狱尝试
            (r'(?i)(jailbreak|DAN|do anything now)', 
             "jailbreak"),
            # 忽略指令
            (r'(?i)忽略(之前|上面|以上|所有)(的)?(指令|规则|限制)', 
             "ignore_instructions_cn"),
            # 角色扮演
            (r'(?i)你(现在|其实)是一个', 
             "persona_injection_cn"),
        ]
        
        # 编译正则表达式
        self._compiled_sensitive = [
            (re.compile(p), n, l) for p, n, l in self._sensitive_patterns
        ]
        self._compiled_injection = [
            (re.compile(p), n) for p, n in self._injection_patterns
        ]
    
    def classify(self, text: str) -> SafetyResult:
        """对文本进行安全分类"""
        issues = []
        categories = []
        max_level = SafetyLevel.SAFE
        
        # 1. 敏感信息检测
        for pattern, name, level in self._compiled_sensitive:
            matches = pattern.findall(text)
            if matches:
                issues.append({
                    "type": SafetyCategory.SENSITIVE_INFO.value,
                    "pattern": name,
                    "count": len(matches),
                    "level": level.value,
                })
                if SafetyCategory.SENSITIVE_INFO not in categories:
                    categories.append(SafetyCategory.SENSITIVE_INFO)
                max_level = self._higher_level(max_level, level)
        
        # 2. 有害内容检测
        for level, keywords in self._harmful_keywords.items():
            for kw in keywords:
                if kw.lower() in text.lower():
                    issues.append({
                        "type": SafetyCategory.HARMFUL_CONTENT.value,
                        "keyword": kw,
                        "level": level.value,
                    })
                    if SafetyCategory.HARMFUL_CONTENT not in categories:
                        categories.append(SafetyCategory.HARMFUL_CONTENT)
                    max_level = self._higher_level(max_level, level)
        
        # 3. 提示注入检测
        for pattern, name in self._compiled_injection:
            if pattern.search(text):
                issues.append({
                    "type": SafetyCategory.PROMPT_INJECTION.value,
                    "pattern": name,
                    "level": SafetyLevel.HIGH_RISK.value,
                })
                if SafetyCategory.PROMPT_INJECTION not in categories:
                    categories.append(SafetyCategory.PROMPT_INJECTION)
                max_level = self._higher_level(max_level, SafetyLevel.HIGH_RISK)
        
        # 计算安全分数
        if not issues:
            score = 1.0
            categories = [SafetyCategory.CLEAN]
        else:
            # 根据问题严重程度计算分数
            level_scores = {
                SafetyLevel.SAFE: 1.0,
                SafetyLevel.LOW_RISK: 0.8,
                SafetyLevel.MEDIUM_RISK: 0.5,
                SafetyLevel.HIGH_RISK: 0.2,
                SafetyLevel.BLOCKED: 0.0,
            }
            score = level_scores.get(max_level, 0.5)
        
        # 生成建议
        suggestion = self._generate_suggestion(max_level, categories)
        
        return SafetyResult(
            level=max_level,
            categories=categories,
            score=score,
            issues=issues,
            suggestion=suggestion,
        )
    
    def classify_knowledge(
        self, 
        condition: str, 
        decision: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> SafetyResult:
        """对知识条目进行安全分类"""
        # 合并检查条件和决策
        combined_text = f"{condition}\n{decision}"
        result = self.classify(combined_text)
        
        # 额外检查：决策中的特殊风险
        decision_issues = self._check_decision_risks(decision)
        if decision_issues:
            result.issues.extend(decision_issues)
            result.level = self._higher_level(result.level, SafetyLevel.MEDIUM_RISK)
        
        return result
    
    def _check_decision_risks(self, decision: str) -> List[Dict[str, Any]]:
        """检查决策中的特殊风险"""
        issues = []
        
        # 检查是否包含代码执行
        code_patterns = [
            r'`.*`',       # 内联代码
            r'```[\s\S]*```',  # 代码块
            r'eval\s*\(',  # eval 调用
            r'exec\s*\(',  # exec 调用
        ]
        for pattern in code_patterns:
            if re.search(pattern, decision):
                issues.append({
                    "type": "code_execution_risk",
                    "level": SafetyLevel.LOW_RISK.value,
                    "message": "Decision contains code that may be executed",
                })
                break
        
        # 检查是否包含 URL
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        if re.search(url_pattern, decision):
            issues.append({
                "type": "external_reference",
                "level": SafetyLevel.LOW_RISK.value,
                "message": "Decision contains external URL reference",
            })
        
        return issues
    
    def _higher_level(self, a: SafetyLevel, b: SafetyLevel) -> SafetyLevel:
        """返回更高的风险等级"""
        order = [
            SafetyLevel.SAFE,
            SafetyLevel.LOW_RISK,
            SafetyLevel.MEDIUM_RISK,
            SafetyLevel.HIGH_RISK,
            SafetyLevel.BLOCKED,
        ]
        return order[max(order.index(a), order.index(b))]
    
    def _generate_suggestion(
        self, 
        level: SafetyLevel, 
        categories: List[SafetyCategory]
    ) -> str:
        """生成处理建议"""
        if level == SafetyLevel.SAFE:
            return "内容安全，可以注入"
        elif level == SafetyLevel.LOW_RISK:
            return "存在轻微风险，建议记录后注入"
        elif level == SafetyLevel.MEDIUM_RISK:
            if SafetyCategory.SENSITIVE_INFO in categories:
                return "包含敏感信息，建议脱敏处理或人工审核"
            return "存在中等风险，建议人工审核后再注入"
        elif level == SafetyLevel.HIGH_RISK:
            if SafetyCategory.PROMPT_INJECTION in categories:
                return "检测到提示注入风险，强烈建议拒绝"
            return "高风险内容，建议拒绝注入"
        else:
            return "检测到严重安全问题，已自动阻止"
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "RuleBasedSafetyClassifier",
            "type": "rule_based",
            "patterns": {
                "sensitive": len(self._sensitive_patterns),
                "harmful_keywords": sum(len(v) for v in self._harmful_keywords.values()),
                "injection": len(self._injection_patterns),
            },
        }


class LLMSafetyClassifier(BaseSafetyClassifier):
    """
    基于 LLM 的安全分类器
    
    使用 LLM 进行更智能的安全检测：
    - 语义级别的有害内容检测
    - 上下文相关的风险评估
    - 细粒度的分类
    
    需要配置 LLM 客户端。
    """
    
    def __init__(
        self, 
        llm_client=None,
        model: str = "gpt-4",
        timeout: int = 30,
        fallback_to_rules: bool = True,
    ):
        self.llm_client = llm_client
        self.model = model
        self.timeout = timeout
        self.fallback_to_rules = fallback_to_rules
        
        # 规则分类器作为后备
        self._rule_classifier = RuleBasedSafetyClassifier() if fallback_to_rules else None
    
    def classify(self, text: str) -> SafetyResult:
        """使用 LLM 进行安全分类"""
        if self.llm_client is None:
            if self._rule_classifier:
                return self._rule_classifier.classify(text)
            return SafetyResult(level=SafetyLevel.SAFE)
        
        try:
            # 构建 prompt
            prompt = self._build_safety_prompt(text)
            
            # 调用 LLM
            response = self._call_llm(prompt)
            
            # 解析结果
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.warning(f"LLM safety check failed: {e}")
            if self._rule_classifier:
                return self._rule_classifier.classify(text)
            return SafetyResult(
                level=SafetyLevel.MEDIUM_RISK,
                suggestion="LLM 安全检查失败，建议人工审核"
            )
    
    def classify_knowledge(
        self, 
        condition: str, 
        decision: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> SafetyResult:
        """对知识条目进行安全分类"""
        combined = f"条件: {condition}\n决策: {decision}"
        return self.classify(combined)
    
    def _build_safety_prompt(self, text: str) -> str:
        """构建安全检查 prompt"""
        return f"""请对以下文本进行安全评估，返回 JSON 格式结果：

文本:
{text}

请评估以下方面：
1. 是否包含敏感个人信息（PII）
2. 是否包含有害或危险内容
3. 是否存在偏见或歧视
4. 是否可能是提示注入攻击
5. 信息是否可能是错误的

返回格式（JSON）：
{{
    "level": "safe|low_risk|medium_risk|high_risk|blocked",
    "categories": ["clean|sensitive_info|harmful_content|bias|misinformation|prompt_injection"],
    "score": 0.0-1.0,
    "issues": [
        {{"type": "...", "description": "...", "severity": "..."}}
    ],
    "suggestion": "处理建议"
}}"""
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if hasattr(self.llm_client, 'chat'):
            return self.llm_client.chat([
                {"role": "system", "content": "你是一个安全内容审核助手。"},
                {"role": "user", "content": prompt}
            ])
        elif hasattr(self.llm_client, 'complete'):
            return self.llm_client.complete(prompt)
        else:
            raise ValueError("Unsupported LLM client type")
    
    def _parse_llm_response(self, response: str) -> SafetyResult:
        """解析 LLM 响应"""
        import json
        
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                level = SafetyLevel(data.get("level", "safe"))
                categories = [SafetyCategory(c) for c in data.get("categories", [])]
                
                return SafetyResult(
                    level=level,
                    categories=categories,
                    score=data.get("score", 0.5),
                    issues=data.get("issues", []),
                    suggestion=data.get("suggestion", ""),
                )
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        # 解析失败，返回需要审核
        return SafetyResult(
            level=SafetyLevel.MEDIUM_RISK,
            suggestion="无法解析 LLM 响应，建议人工审核"
        )
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "LLMSafetyClassifier",
            "type": "llm",
            "model": self.model,
            "fallback_to_rules": self.fallback_to_rules,
        }


class CompositeSafetyClassifier(BaseSafetyClassifier):
    """
    组合安全分类器
    
    组合多个分类器，取最严格的结果。
    """
    
    def __init__(self, classifiers: List[BaseSafetyClassifier]):
        self.classifiers = classifiers
    
    def classify(self, text: str) -> SafetyResult:
        """组合多个分类器的结果"""
        results = [c.classify(text) for c in self.classifiers]
        return self._merge_results(results)
    
    def classify_knowledge(
        self, 
        condition: str, 
        decision: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> SafetyResult:
        """组合多个分类器的知识分类结果"""
        results = [
            c.classify_knowledge(condition, decision, metadata) 
            for c in self.classifiers
        ]
        return self._merge_results(results)
    
    def _merge_results(self, results: List[SafetyResult]) -> SafetyResult:
        """合并多个结果，取最严格的"""
        if not results:
            return SafetyResult(level=SafetyLevel.SAFE)
        
        level_order = [
            SafetyLevel.SAFE,
            SafetyLevel.LOW_RISK,
            SafetyLevel.MEDIUM_RISK,
            SafetyLevel.HIGH_RISK,
            SafetyLevel.BLOCKED,
        ]
        
        # 取最高风险等级
        max_level = max(results, key=lambda r: level_order.index(r.level)).level
        
        # 合并所有分类
        all_categories = []
        for r in results:
            for c in r.categories:
                if c not in all_categories:
                    all_categories.append(c)
        
        # 合并所有问题
        all_issues = []
        for r in results:
            all_issues.extend(r.issues)
        
        # 取最低分数
        min_score = min(r.score for r in results)
        
        # 取最严重的建议
        suggestions = [r.suggestion for r in results if r.suggestion]
        suggestion = suggestions[0] if suggestions else ""
        
        return SafetyResult(
            level=max_level,
            categories=all_categories,
            score=min_score,
            issues=all_issues,
            suggestion=suggestion,
        )
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "CompositeSafetyClassifier",
            "type": "composite",
            "classifiers": [c.get_info() for c in self.classifiers],
        }


def create_safety_classifier(
    mode: str = "rule_based",
    llm_client=None,
    **kwargs
) -> BaseSafetyClassifier:
    """
    创建安全分类器工厂函数
    
    Args:
        mode: 模式 - "rule_based", "llm", "composite"
        llm_client: LLM 客户端（用于 llm 模式）
        **kwargs: 其他参数
    
    Returns:
        安全分类器实例
    """
    if mode == "rule_based":
        return RuleBasedSafetyClassifier(kwargs.get("config"))
    elif mode == "llm":
        return LLMSafetyClassifier(
            llm_client=llm_client,
            model=kwargs.get("model", "gpt-4"),
            fallback_to_rules=kwargs.get("fallback_to_rules", True),
        )
    elif mode == "composite":
        classifiers = [RuleBasedSafetyClassifier()]
        if llm_client:
            classifiers.append(LLMSafetyClassifier(llm_client=llm_client))
        return CompositeSafetyClassifier(classifiers)
    else:
        raise ValueError(f"Unknown safety classifier mode: {mode}")
