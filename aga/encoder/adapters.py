"""
编码器适配器实现

支持多种编码策略：
- Hash: 哈希编码（测试用）
- EmbeddingLayer: 从 LLM 嵌入层提取
- OpenAI: OpenAI text-embedding
- OpenAICompatible: OpenAI 兼容 API（DeepSeek/Qwen/智谱 等）
- SentenceTransformers: HuggingFace 本地模型
- Ollama: Ollama 本地模型
- vLLM: vLLM 本地部署
"""

from typing import List, Dict, Any, Optional
import hashlib
import logging

from .base import BaseEncoder, EncoderType, EncoderConfig

logger = logging.getLogger(__name__)


class HashEncoder(BaseEncoder):
    """
    哈希编码器
    
    将文本编码为确定性哈希向量。
    
    特点:
    - 不需要外部依赖
    - 确定性输出（相同输入产生相同输出）
    - 无语义理解能力
    
    适用场景:
    - 开发和测试
    - 离线环境
    - 简单的精确匹配场景
    """
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.HASH
    
    @property
    def native_dim(self) -> int:
        # 哈希编码可以输出任意维度，使用较大的值
        return max(self.config.key_dim, self.config.value_dim, 1024)
    
    @property
    def model_name(self) -> str:
        return "shake256-hash"
    
    def encode(self, text: str) -> List[float]:
        """使用 SHAKE256 哈希编码文本"""
        import struct
        
        if text is None:
            raise ValueError("text must not be None")
        
        # 计算哈希
        seed = hashlib.sha256(text.encode()).digest()
        
        # 扩展到目标维度
        dim = self.native_dim
        extended = hashlib.shake_256(seed).digest(dim * 4)
        
        # 转换为 float 列表（范围 [-0.1, 0.1]）
        vector = []
        for i in range(dim):
            val = struct.unpack('<I', extended[i*4:(i+1)*4])[0]
            normalized = (val / 0xFFFFFFFF - 0.5) * 0.2
            vector.append(normalized)
        
        return vector


class EmbeddingLayerEncoder(BaseEncoder):
    """
    嵌入层编码器
    
    从 LLM 的嵌入层提取向量，使用 mean pooling。
    需要访问模型权重。
    
    支持模型架构:
    - LLaMA / Qwen / DeepSeek: model.embed_tokens
    - GPT-2 / GPT-J: transformer.wte
    - BERT: embeddings.word_embeddings
    
    注意：此编码器需要模型加载到 GPU/CPU，适合本地部署场景。
    """
    
    def __init__(self, config: EncoderConfig, model=None, tokenizer=None):
        super().__init__(config)
        self._model = model
        self._tokenizer = tokenizer
        self._embed_layer = None
        self._native_dim = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.EMBEDDING_LAYER
    
    @property
    def native_dim(self) -> int:
        if self._native_dim:
            return self._native_dim
        if self.config.native_dim:
            return self.config.native_dim
        return 4096  # 默认值
    
    @property
    def is_available(self) -> bool:
        return self._model is not None and self._tokenizer is not None
    
    def set_model(self, model, tokenizer):
        """设置模型和 tokenizer"""
        self._model = model
        self._tokenizer = tokenizer
        self._embed_layer = None
        self._native_dim = None
    
    def initialize(self) -> bool:
        """初始化嵌入层"""
        if not self.is_available:
            logger.warning("Model or tokenizer not set")
            return False
        
        # 查找嵌入层
        model = self._model
        
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # LLaMA / Qwen / DeepSeek
            self._embed_layer = model.model.embed_tokens
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            # GPT-2 / GPT-J
            self._embed_layer = model.transformer.wte
        elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
            # BERT
            self._embed_layer = model.embeddings.word_embeddings
        elif hasattr(model, 'get_input_embeddings'):
            # 通用方法
            self._embed_layer = model.get_input_embeddings()
        else:
            logger.error("Could not find embedding layer in model")
            return False
        
        # 获取嵌入维度
        if hasattr(self._embed_layer, 'embedding_dim'):
            self._native_dim = self._embed_layer.embedding_dim
        elif hasattr(self._embed_layer, 'weight'):
            self._native_dim = self._embed_layer.weight.shape[1]
        
        self._initialized = True
        logger.info(f"EmbeddingLayerEncoder initialized with dim={self._native_dim}")
        return True
    
    def encode(self, text: str) -> List[float]:
        """使用嵌入层编码文本"""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Encoder not initialized")
        
        import torch
        
        # Tokenize
        device = next(self._embed_layer.parameters()).device
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens.input_ids.to(device)
        
        # 获取嵌入
        with torch.no_grad():
            embeddings = self._embed_layer(input_ids)  # [1, seq_len, hidden_dim]
            
            # Mean pooling
            attention_mask = tokens.attention_mask.to(device) if 'attention_mask' in tokens else None
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = embeddings.mean(dim=1)
            
            vector = pooled[0].cpu().numpy().tolist()
        
        return vector


class OpenAIEncoder(BaseEncoder):
    """
    OpenAI Embeddings 编码器
    
    支持模型:
    - text-embedding-3-small (1536 维，推荐)
    - text-embedding-3-large (3072 维)
    - text-embedding-ada-002 (1536 维，旧版)
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.OPENAI
    
    @property
    def native_dim(self) -> int:
        if self.config.native_dim:
            return self.config.native_dim
        model = self.config.model or self.DEFAULT_MODEL
        return self.MODEL_DIMS.get(model, 1536)
    
    @property
    def model_name(self) -> str:
        return self.config.model or self.DEFAULT_MODEL
    
    @property
    def is_available(self) -> bool:
        return bool(self.config.api_key)
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def encode(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("OpenAI API key not configured")
        
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if not self.is_available:
            raise ValueError("OpenAI API key not configured")
        
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]


class OpenAICompatibleEncoder(BaseEncoder):
    """
    OpenAI 兼容 API 编码器
    
    支持所有提供 OpenAI 兼容 Embedding API 的服务商：
    - DeepSeek: deepseek-embedding (1024 维)
    - 通义千问: text-embedding-v2 (1536 维)
    - 智谱 GLM: embedding-3 (2048 维)
    - Moonshot: moonshot-embedding (1024 维)
    - 百川: baichuan-embedding (1024 维)
    - 零一万物: yi-large-embedding (2048 维)
    - SiliconFlow: 多种模型
    """
    
    PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-embedding",
            "dim": 1024,
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "text-embedding-v2",
            "dim": 1536,
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "embedding-3",
            "dim": 2048,
        },
        "moonshot": {
            "base_url": "https://api.moonshot.cn/v1",
            "model": "moonshot-embedding",
            "dim": 1024,
        },
        "baichuan": {
            "base_url": "https://api.baichuan-ai.com/v1",
            "model": "Baichuan-Text-Embedding",
            "dim": 1024,
        },
        "yi": {
            "base_url": "https://api.01.ai/v1",
            "model": "yi-large-embedding",
            "dim": 2048,
        },
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "BAAI/bge-large-zh-v1.5",
            "dim": 1024,
        },
    }
    
    MODEL_DIMS = {
        "deepseek-embedding": 1024,
        "text-embedding-v1": 1536,
        "text-embedding-v2": 1536,
        "text-embedding-v3": 1024,
        "embedding-2": 1024,
        "embedding-3": 2048,
        "moonshot-embedding": 1024,
        "Baichuan-Text-Embedding": 1024,
        "baichuan-embedding": 1024,
        "yi-large-embedding": 2048,
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-m3": 1024,
    }
    
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self._client = None
        self.provider = config.provider
        
        # 如果指定了 provider，使用预设配置
        if self.provider and self.provider in self.PROVIDERS:
            preset = self.PROVIDERS[self.provider]
            if not config.base_url:
                config.base_url = preset["base_url"]
            if not config.model:
                config.model = preset["model"]
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.OPENAI_COMPATIBLE
    
    @property
    def native_dim(self) -> int:
        if self.config.native_dim:
            return self.config.native_dim
        
        model = self.config.model
        if model and model in self.MODEL_DIMS:
            return self.MODEL_DIMS[model]
        
        if self.provider and self.provider in self.PROVIDERS:
            return self.PROVIDERS[self.provider]["dim"]
        
        return 1024
    
    @property
    def model_name(self) -> str:
        return self.config.model or "text-embedding-v2"
    
    @property
    def is_available(self) -> bool:
        return bool(self.config.api_key and self.config.base_url)
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def encode(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("OpenAI compatible API not configured")
        
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if not self.is_available:
            raise ValueError("OpenAI compatible API not configured")
        
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]
    
    @classmethod
    def from_provider(cls, provider: str, api_key: str, **kwargs) -> "OpenAICompatibleEncoder":
        """从服务商名称创建编码器"""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls.PROVIDERS.keys())}")
        
        preset = cls.PROVIDERS[provider]
        config = EncoderConfig(
            encoder_type=EncoderType.OPENAI_COMPATIBLE,
            base_url=kwargs.get("base_url", preset["base_url"]),
            api_key=api_key,
            model=kwargs.get("model", preset["model"]),
            provider=provider,
            key_dim=kwargs.get("key_dim", 64),
            value_dim=kwargs.get("value_dim", 4096),
        )
        return cls(config)


class SentenceTransformersEncoder(BaseEncoder):
    """
    HuggingFace Sentence Transformers 编码器
    
    使用本地运行的 Sentence Transformers 模型。
    
    推荐模型:
    - all-MiniLM-L6-v2 (384 维，快速)
    - all-mpnet-base-v2 (768 维，高质量)
    - paraphrase-multilingual-MiniLM-L12-v2 (384 维，多语言)
    - bge-small-zh-v1.5 (512 维，中文)
    - bge-large-zh-v1.5 (1024 维，中文高质量)
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    MODEL_DIMS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "bge-small-zh-v1.5": 512,
        "bge-large-zh-v1.5": 1024,
        "text2vec-base-chinese": 768,
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-m3": 1024,
    }
    
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self._model = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.SENTENCE_TRANSFORMERS
    
    @property
    def native_dim(self) -> int:
        if self.config.native_dim:
            return self.config.native_dim
        model = self.config.model or self.DEFAULT_MODEL
        return self.MODEL_DIMS.get(model, 384)
    
    @property
    def model_name(self) -> str:
        return self.config.model or self.DEFAULT_MODEL
    
    @property
    def is_available(self) -> bool:
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
    
    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.config.device)
            except ImportError:
                raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        return self._model
    
    def encode(self, text: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class OllamaEncoder(BaseEncoder):
    """
    Ollama 本地 Embedding 编码器
    
    支持模型:
    - nomic-embed-text (768 维)
    - mxbai-embed-large (1024 维)
    - all-minilm (384 维)
    """
    
    DEFAULT_MODEL = "nomic-embed-text"
    MODEL_DIMS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }
    
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.OLLAMA
    
    @property
    def native_dim(self) -> int:
        if self.config.native_dim:
            return self.config.native_dim
        model = self.config.model or self.DEFAULT_MODEL
        return self.MODEL_DIMS.get(model, 768)
    
    @property
    def model_name(self) -> str:
        return self.config.model or self.DEFAULT_MODEL
    
    @property
    def is_available(self) -> bool:
        try:
            import httpx
            base_url = (self.config.base_url or "http://localhost:11434").rstrip('/')
            response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def encode(self, text: str) -> List[float]:
        import httpx
        
        base_url = (self.config.base_url or "http://localhost:11434").rstrip('/')
        response = httpx.post(
            f"{base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        if "embedding" not in data:
            raise RuntimeError("Ollama embedding response missing 'embedding' field")
        return data["embedding"]


class VLLMEncoder(BaseEncoder):
    """
    vLLM Embedding 编码器
    
    使用 vLLM 部署的本地模型生成 embedding。
    
    推荐模型:
    - BAAI/bge-large-zh-v1.5 (1024 维，中文)
    - BAAI/bge-m3 (1024 维，多语言)
    - intfloat/multilingual-e5-large (1024 维)
    
    vLLM 启动方式:
    ```bash
    python -m vllm.entrypoints.openai.api_server \\
        --model BAAI/bge-large-zh-v1.5 \\
        --host 0.0.0.0 --port 8000 \\
        --task embed
    ```
    """
    
    DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"
    MODEL_DIMS = {
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-m3": 1024,
        "BAAI/bge-small-zh-v1.5": 512,
        "intfloat/multilingual-e5-large": 1024,
        "intfloat/e5-large-v2": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }
    
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.VLLM
    
    @property
    def native_dim(self) -> int:
        if self.config.native_dim:
            return self.config.native_dim
        model = self.config.model or self.DEFAULT_MODEL
        return self.MODEL_DIMS.get(model, 1024)
    
    @property
    def model_name(self) -> str:
        return self.config.model or self.DEFAULT_MODEL
    
    @property
    def is_available(self) -> bool:
        try:
            import httpx
            base_url = (self.config.base_url or "http://localhost:8000").rstrip('/')
            response = httpx.get(f"{base_url}/v1/models", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                base_url = (self.config.base_url or "http://localhost:8000").rstrip('/')
                self._client = OpenAI(
                    api_key="EMPTY",  # vLLM 不需要 API key
                    base_url=f"{base_url}/v1",
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def encode(self, text: str) -> List[float]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        if not response.data:
            raise RuntimeError("vLLM embedding response is empty")
        return response.data[0].embedding
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        client = self._get_client()
        
        # vLLM 支持大批量
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            if not response.data:
                raise RuntimeError("vLLM embedding response is empty")
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        
        return all_embeddings
