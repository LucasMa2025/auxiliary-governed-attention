#!/bin/bash
# AGA - Ollama 模型启动脚本
# 
# 使用方法:
#   ./start_ollama.sh [model_name]
#
# 示例:
#   ./start_ollama.sh llama3.2:latest
#   ./start_ollama.sh qwen2.5:7b
#   ./start_ollama.sh deepseek-r1:7b

MODEL=${1:-llama3.2:latest}

echo "=========================================="
echo "AGA - Ollama Model Launcher"
echo "=========================================="
echo "Model: $MODEL"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed."
    echo "Please install from: https://ollama.ai/"
    exit 1
fi

# Check if Ollama service is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 3
fi

# Pull model if not exists
echo "Checking model availability..."
if ! ollama list | grep -q "$MODEL"; then
    echo "Pulling model $MODEL..."
    ollama pull $MODEL
fi

echo ""
echo "Model $MODEL is ready!"
echo ""
echo "Ollama API endpoint: http://localhost:11434"
echo ""
echo "To use with AGA Experiment Tool:"
echo "  1. Start the experiment tool: python aga_experiment_tool/app.py"
echo "  2. Use LLM adapter: OllamaAdapter(model='$MODEL')"
echo ""
echo "Available models:"
ollama list

