#!/bin/bash
# AGA Experiment Tool - Linux/macOS Launcher
# 
# Usage:
#   ./start_experiment_tool.sh [port]

PORT=${1:-8765}

echo "=========================================="
echo "AGA Experiment Tool Launcher"
echo "=========================================="
echo "Port: $PORT"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
pip3 show flask > /dev/null 2>&1 || pip3 install flask
pip3 show torch > /dev/null 2>&1 || pip3 install torch
pip3 show transformers > /dev/null 2>&1 || pip3 install transformers
pip3 show pyyaml > /dev/null 2>&1 || pip3 install pyyaml

echo ""
echo "Starting AGA Experiment Tool..."
echo "Access at: http://localhost:$PORT"
echo "Default password: aga_experiment_2026"
echo ""

cd "$(dirname "$0")/.."
python3 -m aga_experiment_tool.app --port $PORT

