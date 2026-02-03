#!/bin/bash
# AGA - vLLM 模型启动脚本
# 
# 使用方法:
#   ./start_vllm.sh [model_name] [port] [tensor_parallel_size]
#
# 示例:
#   ./start_vllm.sh deepseek-ai/deepseek-coder-7b-instruct-v1.5 8001 1
#   ./start_vllm.sh Qwen/Qwen2.5-7B-Instruct 8001 2
#   ./start_vllm.sh meta-llama/Llama-3-8B-Instruct 8001 1
#   ./start_vllm.sh mistralai/Mistral-7B-Instruct-v0.2 8001 1

MODEL=${1:-deepseek-ai/deepseek-coder-7b-instruct-v1.5}
PORT=${2:-8001}
TP_SIZE=${3:-1}

echo "=========================================="
echo "AGA - vLLM Model Launcher"
echo "=========================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM is not installed."
    echo "Please install: pip install vllm"
    exit 1
fi

# Check GPU availability
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Warning: CUDA not available. vLLM requires GPU."
    echo "Attempting to start anyway..."
fi

echo "Starting vLLM server..."
echo ""

# Start vLLM with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --max-model-len 4096

# Note: Add these options as needed:
# --enable-lora              # Enable LoRA support
# --gpu-memory-utilization 0.9
# --dtype float16
# --quantization awq         # For quantized models

