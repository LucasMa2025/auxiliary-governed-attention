"""
aga/adapter/vllm.py — vLLM 推理框架适配器

设计要点:
  - 继承 LLMAdapter 抽象基类，遵循 AGA 统一适配器接口
  - 通过 register_forward_hook 注入 AGA，与 HuggingFaceAdapter 同源
  - 不需要 fork vLLM — 通过访问 vLLM 内部模型对象实现
  - 提供两种集成模式:
    Mode A: 直接访问 vLLM 的内部 model（标准模式）
    Mode B: 兼容 IBM vLLM-Hook 插件系统（可选）

技术依据:
  - vLLM 的 LLM 对象内部持有一个标准的 nn.Module 模型
  - 该模型的 Transformer 层结构与 HuggingFace 模型一致
  - register_forward_hook 在 vLLM 的 PyTorch 执行路径上仍然有效
  - AGA 的注入发生在 attention 输出之后（不修改 KV Cache），
    因此不与 PagedAttention 冲突

不需要 fork vLLM 的原因:
  1. AGA 通过 PyTorch 标准的 register_forward_hook 注入
  2. vLLM 内部的模型仍然是标准的 nn.Module
  3. IBM vLLM-Hook (https://github.com/IBM/vLLM-Hook) 提供了
     更高级的插件系统，可以作为可选的集成路径
  4. AGA 的注入点在 attention 输出之后，不影响 PagedAttention

限制与注意事项:
  1. CUDA Graph: vLLM 使用 CUDA Graph 加速推理，AGA 的条件分支
     可能导致 CUDA Graph capture 失败。解决方案:
     - 使用 torch.where 替代 if/else（已在 EntropyGateSystem 中实现）
     - 或在 vLLM 配置中禁用 CUDA Graph（enforce_eager=True）
     - wrap_layer 中会自动检测并发出警告
  2. 连续批处理: vLLM 将多个请求的 token 打包处理。
     当前 AGA 使用全局 KVStore，所有请求共享同一知识库。
     这在"全局知识注入"场景下是正确的（AGA 的核心设计目标）。
     如需 per-request 知识隔离，应通过 aga-knowledge 的
     namespace 机制在请求前后动态注册/注销知识。
  3. Tensor Parallelism: 多 GPU 场景下，AGA 的 KVStore 需要
     在所有 TP rank 上保持一致（当前版本不支持，需要手动同步）
  4. 模型访问: 需要通过 vLLM 内部 API 访问模型对象，
     这些 API 可能在 vLLM 版本更新时变化

变更记录:
  v4.3.0 — 初始实现
  v4.3.1 — 修复:
    - hook_fn 输出处理更健壮（支持空 tuple、None 元素等）
    - hook_fn Fail-Open 增加 exc_info 日志
    - wrap_layer 增加 CUDA Graph 自动检测与警告
    - 增加 Continuous Batching 设计说明与文档
"""
import logging
from typing import List, Callable, Any, Optional

import torch
import torch.nn as nn

from .base import LLMAdapter
from ..exceptions import AdapterError

logger = logging.getLogger(__name__)


class VLLMAdapter(LLMAdapter):
    """
    vLLM 推理框架适配器

    通过访问 vLLM 内部的 nn.Module 模型对象，使用标准的
    register_forward_hook 注入 AGA。不需要 fork vLLM。

    支持的 vLLM 模型架构:
      - LLaMA / LLaMA-2 / LLaMA-3 (LlamaForCausalLM)
      - Qwen / Qwen-2 (QWenLMHeadModel / Qwen2ForCausalLM)
      - Mistral / Mixtral (MistralForCausalLM)
      - GPT-NeoX (GPTNeoXForCausalLM)
      - Phi / Phi-2 / Phi-3 (PhiForCausalLM)
      - Gemma (GemmaForCausalLM)
      - ChatGLM (ChatGLMForConditionalGeneration)
      - Yi (YiForCausalLM — 基于 LLaMA 架构)
      - Baichuan (BaichuanForCausalLM)
      - InternLM (InternLMForCausalLM)

    使用方式 (Mode A — 直接模型访问):
        from vllm import LLM
        from aga import AGAPlugin
        from aga.adapter.vllm import VLLMAdapter

        # 1. 创建 vLLM 引擎
        llm = LLM(model="meta-llama/Llama-3-8B", enforce_eager=True)

        # 2. 提取内部模型
        model = VLLMAdapter.extract_model(llm)

        # 3. 创建 AGA 插件并挂载
        plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
        plugin.attach(model, adapter=VLLMAdapter())

        # 4. 正常使用 vLLM 推理 — AGA 自动介入
        outputs = llm.generate(prompts, sampling_params)

    使用方式 (Mode B — vLLM-Hook 插件):
        参见 VLLMHookWorker 类

    Continuous Batching 说明:
        vLLM 默认启用连续批处理，一个 batch 中可能包含多个独立请求。
        AGA 的设计目标是"全局知识注入"——所有请求共享同一知识库，
        这与 vLLM 的连续批处理天然兼容:
        - KVStore 中的知识是全局的（如医学知识、法律条文）
        - 熵门控在 [batch, seq] 维度上逐 token 独立评估
        - 每个 token 位置根据自身的不确定性决定是否注入

        如需 per-request 知识隔离（不同请求使用不同知识集），
        应在请求级别通过 namespace 机制管理:
        - 请求前: plugin.register(..., metadata={"namespace": request_id})
        - 请求后: plugin.clear(namespace=request_id)
        或使用 aga-knowledge 的 Portal API 进行请求级知识管理。
    """

    # vLLM 内部模型的 Transformer 层路径
    # vLLM 的模型结构通常是: model.model.layers (与 HF 类似)
    _LAYER_PATHS = [
        # LLaMA / Qwen / Mistral / Gemma / Yi / InternLM / Baichuan
        ("model", "layers"),
        # GPT-NeoX
        ("gpt_neox", "layers"),
        # ChatGLM
        ("transformer", "encoder", "layers"),
        # Phi
        ("model", "layers"),
        # Falcon
        ("transformer", "h"),
    ]

    # vLLM 特有的三层嵌套路径 (LLM → model_executor → model → ...)
    _VLLM_MODEL_PATHS = [
        # vLLM >= 0.4.x
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        # vLLM >= 0.5.x (简化路径)
        ("llm_engine", "model_executor", "model"),
        # vLLM AsyncLLMEngine
        ("engine", "model_executor", "driver_worker", "model_runner", "model"),
    ]

    def __init__(self, enforce_eager: bool = True):
        """
        初始化 vLLM 适配器

        Args:
            enforce_eager: 是否建议使用 eager 模式（禁用 CUDA Graph）。
                          当 AGA 的条件门控可能导致 CUDA Graph 失效时，
                          建议设为 True。默认 True。
        """
        self._enforce_eager = enforce_eager
        # 缓存 vLLM 引擎引用，用于 CUDA Graph 检测
        self._llm_engine_ref = None

    @staticmethod
    def extract_model(llm_engine) -> nn.Module:
        """
        从 vLLM 的 LLM/LLMEngine 对象中提取内部 nn.Module 模型

        vLLM 的对象层次结构:
          LLM → LLMEngine → ModelExecutor → Worker → ModelRunner → Model

        Args:
            llm_engine: vLLM 的 LLM 或 LLMEngine 实例

        Returns:
            内部的 nn.Module 模型对象

        Raises:
            AdapterError: 无法提取模型
        """
        # 尝试多种路径
        for path in VLLMAdapter._VLLM_MODEL_PATHS:
            obj = llm_engine
            found = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    found = False
                    break
            if found and isinstance(obj, nn.Module):
                logger.info(
                    f"从 vLLM 提取模型成功: {type(obj).__name__} "
                    f"(路径: {'.'.join(path)})"
                )
                return obj

        # 尝试直接属性访问
        for attr in ["model", "model_runner"]:
            obj = getattr(llm_engine, attr, None)
            if obj is not None:
                if isinstance(obj, nn.Module):
                    logger.info(f"从 vLLM 提取模型成功: {type(obj).__name__} (直接访问)")
                    return obj
                # model_runner.model
                inner = getattr(obj, "model", None)
                if inner is not None and isinstance(inner, nn.Module):
                    logger.info(f"从 vLLM 提取模型成功: {type(inner).__name__}")
                    return inner

        raise AdapterError(
            "无法从 vLLM 引擎中提取模型。"
            "请确保 vLLM 版本 >= 0.4.0，或手动传入模型对象。"
            f"尝试的路径: {VLLMAdapter._VLLM_MODEL_PATHS}"
        )

    def get_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        获取 Transformer 层列表

        vLLM 的模型结构与 HuggingFace 类似，但可能有额外的包装层。
        """
        # 尝试标准路径
        for path in self._LAYER_PATHS:
            obj = model
            found = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    found = False
                    break
            if found and hasattr(obj, '__len__'):
                layers = list(obj)
                if layers:
                    logger.debug(
                        f"找到 {len(layers)} 个 Transformer 层 "
                        f"(路径: {'.'.join(path)})"
                    )
                    return layers

        # 尝试直接访问
        if hasattr(model, "layers"):
            return list(model.layers)

        # 递归搜索 ModuleList
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 1:
                # 检查是否像 Transformer 层（有 self_attn 或 attention）
                first = module[0]
                if hasattr(first, "self_attn") or hasattr(first, "attention"):
                    logger.debug(
                        f"通过递归搜索找到 {len(module)} 个 Transformer 层 "
                        f"(路径: {name})"
                    )
                    return list(module)

        raise AdapterError(
            f"无法在 vLLM 模型中找到 Transformer 层: {type(model).__name__}。"
            f"请实现自定义 LLMAdapter 或检查模型结构。"
            f"已尝试的路径: {self._LAYER_PATHS}"
        )

    def get_hidden_dim(self, model: nn.Module) -> int:
        """
        获取隐藏维度

        vLLM 模型通常有 config 属性，与 HuggingFace 一致。
        """
        # 尝试从 config 获取
        config = getattr(model, "config", None)
        if config is not None:
            for attr in ["hidden_size", "d_model", "n_embd", "dim"]:
                val = getattr(config, attr, None)
                if val is not None:
                    return val

        # 尝试从第一层推断
        try:
            layers = self.get_layers(model)
            if layers:
                first_layer = layers[0]
                # 从 self_attn 的权重推断
                attn = getattr(first_layer, "self_attn", None)
                if attn is not None:
                    for param_name in ["q_proj", "qkv_proj", "query_key_value"]:
                        proj = getattr(attn, param_name, None)
                        if proj is not None and hasattr(proj, "weight"):
                            # q_proj.weight shape: [num_heads * head_dim, hidden_dim]
                            return proj.weight.shape[-1]
        except Exception:
            pass

        logger.warning("无法自动检测 hidden_dim，使用默认值 4096")
        return 4096

    def _detect_cuda_graph_risk(self, model: nn.Module) -> bool:
        """
        检测 vLLM 是否启用了 CUDA Graph（可能与 AGA 动态门控冲突）

        Returns:
            True 如果检测到 CUDA Graph 风险
        """
        # 尝试从缓存的引擎引用检测
        if self._llm_engine_ref is not None:
            engine = getattr(self._llm_engine_ref, "llm_engine", None) or self._llm_engine_ref
            model_config = getattr(engine, "model_config", None)
            if model_config is not None:
                enforce_eager = getattr(model_config, "enforce_eager", None)
                if enforce_eager is False:
                    return True

        # 尝试从模型的 config 推断
        config = getattr(model, "config", None)
        if config is not None:
            # 某些 vLLM 版本会在 config 中标记
            if getattr(config, "_vllm_cuda_graph", False):
                return True

        return False

    def wrap_layer(
        self,
        model: nn.Module,
        layer_idx: int,
        aga_forward: Callable,
    ) -> Any:
        """
        通过 register_forward_hook 注入 AGA

        与 HuggingFaceAdapter 的核心区别:
        1. vLLM 的 attention 输出格式可能不同（取决于 vLLM 版本）
        2. 需要处理 vLLM 的连续批处理（多请求 token 打包）
        3. 使用 torch.where 替代 if/else 以兼容 CUDA Graph
        4. 自动检测 CUDA Graph 风险并发出警告

        Args:
            model: vLLM 内部模型（nn.Module）
            layer_idx: 层索引
            aga_forward: AGA forward 函数

        Returns:
            hook handle
        """
        layers = self.get_layers(model)

        if layer_idx < 0 or layer_idx >= len(layers):
            raise AdapterError(
                f"层索引 {layer_idx} 超出范围 [0, {len(layers)})"
            )

        # CUDA Graph 风险检测（仅在第一层挂载时检测一次）
        if layer_idx == layers[0] if not isinstance(layers[0], int) else 0:
            if not self._enforce_eager and self._detect_cuda_graph_risk(model):
                logger.warning(
                    "检测到 vLLM 可能启用了 CUDA Graph，但 enforce_eager=False。"
                    "AGA 的动态熵门控可能导致 CUDA Graph 重新捕获，"
                    "建议创建 VLLMAdapter(enforce_eager=True) 或在 vLLM 中设置 "
                    "enforce_eager=True 以确保正确性。"
                )

        layer = layers[layer_idx]

        def hook_fn(module, input, output):
            """
            Forward hook: 在 attention 输出后注入 AGA

            vLLM 的 attention 层输出格式:
            - 大多数模型: (attn_output,) 或 attn_output
            - 某些模型: (attn_output, residual) 或 (attn_output, attn_weights, ...)
            - 某些版本: (attn_output, None, attn_weights)

            AGA 注入策略:
            - 提取 hidden_states（output 的第一个元素）
            - 调用 aga_forward 计算融合结果
            - 重新组装输出（保留其余元素不变）
            """
            # ---- 提取 output hidden_states ----
            if isinstance(output, tuple):
                if len(output) == 0:
                    return output
                hidden_states = output[0]
                other = output[1:]
            elif isinstance(output, torch.Tensor):
                hidden_states = output
                other = ()
            else:
                # 未知格式，Fail-Open
                return output

            # hidden_states 必须是 Tensor
            if not isinstance(hidden_states, torch.Tensor):
                return output

            # ---- 提取 input hidden_states（用于熵计算）----
            if isinstance(input, tuple) and len(input) > 0:
                input_hidden = input[0]
            elif isinstance(input, torch.Tensor):
                input_hidden = input
            else:
                input_hidden = hidden_states

            # input_hidden 也必须是 Tensor
            if not isinstance(input_hidden, torch.Tensor):
                input_hidden = hidden_states

            # ---- 调用 AGA forward ----
            try:
                fused = aga_forward(
                    hidden_states=input_hidden,
                    primary_attention_output=hidden_states,
                )
            except Exception as e:
                # Fail-Open: 出错时返回原始输出，记录完整异常信息
                logger.warning(
                    f"AGA forward hook 异常 (Fail-Open 回退): {e}",
                    exc_info=True,
                )
                return output

            # ---- 重新组装输出 ----
            if other:
                return (fused,) + other
            elif isinstance(output, tuple):
                return (fused,)
            return fused

        # 优先注册到 self_attn 子模块
        attn_module = self._find_attention_module(layer)
        if attn_module is not None:
            handle = attn_module.register_forward_hook(hook_fn)
            logger.debug(
                f"AGA hook 注册到 layer[{layer_idx}].{type(attn_module).__name__}"
            )
            return handle

        # 回退到整个层
        handle = layer.register_forward_hook(hook_fn)
        logger.debug(f"AGA hook 注册到 layer[{layer_idx}] (整层)")
        return handle

    @staticmethod
    def _find_attention_module(layer: nn.Module) -> Optional[nn.Module]:
        """
        在 Transformer 层中查找 attention 子模块

        vLLM 的 attention 模块命名可能不同于 HuggingFace:
        - self_attn (LLaMA, Qwen, Mistral, Gemma)
        - attention (GPT-NeoX, ChatGLM)
        - attn (Falcon, GPT-2)
        - self_attention (InternLM)
        """
        for attr_name in ["self_attn", "attention", "attn", "self_attention"]:
            module = getattr(layer, attr_name, None)
            if module is not None and isinstance(module, nn.Module):
                return module
        return None

    @staticmethod
    def check_compatibility(llm_engine) -> dict:
        """
        检查 vLLM 引擎与 AGA 的兼容性

        返回兼容性报告，包括:
        - vLLM 版本
        - 是否使用 CUDA Graph
        - 是否使用 Tensor Parallelism
        - 模型架构
        - 建议配置

        Args:
            llm_engine: vLLM 的 LLM 实例

        Returns:
            兼容性报告字典
        """
        report = {
            "compatible": True,
            "warnings": [],
            "suggestions": [],
            "vllm_version": None,
            "model_type": None,
            "cuda_graph": None,
            "tensor_parallel": None,
        }

        # 检查 vLLM 版本
        try:
            import vllm
            report["vllm_version"] = getattr(vllm, "__version__", "unknown")
        except ImportError:
            report["compatible"] = False
            report["warnings"].append("vLLM 未安装")
            return report

        # 检查模型类型
        try:
            model = VLLMAdapter.extract_model(llm_engine)
            report["model_type"] = type(model).__name__
        except AdapterError as e:
            report["compatible"] = False
            report["warnings"].append(f"无法提取模型: {e}")
            return report

        # 检查 CUDA Graph
        # 支持 LLM 对象 (llm.llm_engine) 和直接 LLMEngine 对象
        engine = getattr(llm_engine, "llm_engine", None) or llm_engine
        model_config = getattr(engine, "model_config", None)
        if model_config is not None:
            enforce_eager = getattr(model_config, "enforce_eager", None)
            if enforce_eager is False:
                report["cuda_graph"] = True
                report["warnings"].append(
                    "CUDA Graph 已启用。AGA 的条件门控可能导致 CUDA Graph "
                    "capture 失败。建议设置 enforce_eager=True。"
                )
                report["suggestions"].append(
                    'LLM(model="...", enforce_eager=True)'
                )
            else:
                report["cuda_graph"] = False

        # 检查 Tensor Parallelism
        parallel_config = getattr(engine, "parallel_config", None)
        if parallel_config is not None:
            tp_size = getattr(parallel_config, "tensor_parallel_size", 1)
            report["tensor_parallel"] = tp_size
            if tp_size > 1:
                report["warnings"].append(
                    f"Tensor Parallelism 已启用 (TP={tp_size})。"
                    f"AGA 的 KVStore 需要在所有 TP rank 上手动同步。"
                    f"当前版本不支持自动 TP 同步。"
                )

        # 检查 Continuous Batching（信息性，非阻塞）
        scheduler_config = getattr(engine, "scheduler_config", None)
        if scheduler_config is not None:
            max_num_seqs = getattr(scheduler_config, "max_num_seqs", 1)
            if max_num_seqs > 1:
                report["suggestions"].append(
                    f"连续批处理已启用 (max_num_seqs={max_num_seqs})。"
                    f"AGA 的全局知识库将对所有请求生效。"
                    f"如需 per-request 知识隔离，请使用 namespace 机制。"
                )

        return report

    def set_llm_engine_ref(self, llm_engine):
        """
        设置 vLLM 引擎引用（用于 CUDA Graph 检测等）

        在 extract_model 后、wrap_layer 前调用，可选。

        Args:
            llm_engine: vLLM 的 LLM 或 LLMEngine 实例
        """
        self._llm_engine_ref = llm_engine


class VLLMHookWorker:
    """
    IBM vLLM-Hook 兼容的 AGA Worker

    这是一个可选的集成路径，利用 IBM vLLM-Hook
    (https://github.com/IBM/vLLM-Hook) 的插件系统
    将 AGA 注入到 vLLM 中。

    不需要 fork vLLM — vLLM-Hook 本身就是一个插件库。

    使用方式:
        # 1. 安装 vLLM-Hook
        pip install -e vllm_hook_plugins

        # 2. 在 vLLM-Hook 的 registry 中注册 AGA worker
        from vllm_hook_plugins.registry import register_worker
        from aga.adapter.vllm import VLLMHookWorker

        register_worker("aga", VLLMHookWorker)

        # 3. 在 vLLM-Hook 配置中启用 AGA
        config = {
            "workers": ["aga"],
            "aga": {
                "hidden_dim": 4096,
                "bottleneck_dim": 64,
                "max_slots": 256,
                "knowledge_file": "knowledge.jsonl",
            }
        }

    vLLM-Hook 架构说明:
        vLLM-Hook 提供了 Worker/Analyzer 抽象:
        - Worker: 定义执行行为（如 AGA 的注入逻辑）
        - Analyzer: 可选的分析器（如 AGA 的诊断信息收集）
        AGA 作为一个 Worker 插件注册到 vLLM-Hook 中，
        由 vLLM-Hook 的 hook_llm 模块负责在正确的时机调用。
    """

    def __init__(self, config: dict = None):
        """
        初始化 vLLM-Hook AGA Worker

        Args:
            config: AGA 配置字典，包含:
                - hidden_dim: 隐藏维度
                - bottleneck_dim: 瓶颈维度
                - max_slots: 最大知识槽位
                - knowledge_file: 知识文件路径（可选）
                - device: 设备（默认 "cuda"）
                - layer_indices: 挂载层索引（默认 [-1, -2, -3]）
        """
        self._config = config or {}
        self._plugin = None
        self._adapter = None
        self._initialized = False

    def initialize(self, model: nn.Module, **kwargs):
        """
        由 vLLM-Hook 在模型加载后调用

        Args:
            model: vLLM 内部模型
            **kwargs: vLLM-Hook 传递的额外参数
        """
        # 延迟导入，避免循环依赖
        from ..plugin import AGAPlugin
        from ..config import AGAConfig

        aga_config = AGAConfig(
            hidden_dim=self._config.get("hidden_dim", 4096),
            bottleneck_dim=self._config.get("bottleneck_dim", 64),
            max_slots=self._config.get("max_slots", 256),
            device=self._config.get("device", "cuda"),
        )

        self._plugin = AGAPlugin(aga_config)
        self._adapter = VLLMAdapter(
            enforce_eager=self._config.get("enforce_eager", True),
        )

        # 加载知识
        knowledge_file = self._config.get("knowledge_file")
        if knowledge_file:
            count = self._plugin.load_from(knowledge_file)
            logger.info(f"vLLM-Hook AGA Worker: 加载了 {count} 条知识")

        # 挂载到模型
        layer_indices = self._config.get("layer_indices", [-1, -2, -3])
        self._plugin.attach(model, layer_indices=layer_indices, adapter=self._adapter)

        self._initialized = True
        logger.info(
            f"vLLM-Hook AGA Worker 初始化完成: "
            f"layers={layer_indices}, "
            f"knowledge={self._plugin.knowledge_count}"
        )

    def process(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        由 vLLM-Hook 在每个 forward step 调用（可选）

        注意: 如果使用 register_forward_hook 模式（推荐），
        此方法不会被调用。AGA 通过 hook 自动工作。

        此方法仅在 vLLM-Hook 的 "explicit call" 模式下使用。
        """
        if not self._initialized:
            return hidden_states
        # Hook 模式下，AGA 已经通过 forward hook 自动工作
        # 此方法仅作为 vLLM-Hook 接口的兼容实现
        return hidden_states

    def get_diagnostics(self) -> dict:
        """获取 AGA 诊断信息（供 vLLM-Hook Analyzer 使用）"""
        if self._plugin is None:
            return {"initialized": False}
        return self._plugin.get_diagnostics()

    def shutdown(self):
        """由 vLLM-Hook 在引擎关闭时调用"""
        if self._plugin is not None and self._plugin.is_attached:
            self._plugin.detach()
            logger.info("vLLM-Hook AGA Worker 已关闭")
