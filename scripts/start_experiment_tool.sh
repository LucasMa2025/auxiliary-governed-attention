#!/bin/bash
# AGA Experiment Tool - Linux/macOS Launcher
# 
# Usage:
#   ./start_experiment_tool.sh [port] [--distributed]
#
# Options:
#   port          Server port (default: 8765)
#   --distributed Enable distributed mode

PORT=${1:-8765}
DISTRIBUTED=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
    esac
done

echo "=========================================="
echo "AGA Experiment Tool Launcher v3.1"
echo "=========================================="
echo "Port: $PORT"
echo "Distributed Mode: $DISTRIBUTED"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python Version: $PYTHON_VERSION"

# Check dependencies
echo ""
echo "Checking dependencies..."

# Core dependencies
pip3 show flask > /dev/null 2>&1 || pip3 install flask
pip3 show flask-cors > /dev/null 2>&1 || pip3 install flask-cors
pip3 show torch > /dev/null 2>&1 || pip3 install torch
pip3 show transformers > /dev/null 2>&1 || pip3 install transformers
pip3 show pyyaml > /dev/null 2>&1 || pip3 install pyyaml
pip3 show aiosqlite > /dev/null 2>&1 || pip3 install aiosqlite

# Distributed dependencies (optional)
if [ "$DISTRIBUTED" = true ]; then
    echo "Installing distributed dependencies..."
    pip3 show redis > /dev/null 2>&1 || pip3 install redis
fi

echo ""
echo "=========================================="
echo "Starting AGA Experiment Tool..."
echo "=========================================="
echo "Access at: http://localhost:$PORT"
echo "Default password: aga_experiment_2026"
echo ""
echo "Features:"
echo "  - Zero-training knowledge injection"
echo "  - Lifecycle management (PROBATIONARY → CONFIRMED → DEPRECATED → QUARANTINED)"
echo "  - Entropy gating (primary-first principle)"
echo "  - Multi-adapter persistence (SQLite/Redis/PostgreSQL)"
echo ""

cd "$(dirname "$0")/.."

if [ "$DISTRIBUTED" = true ]; then
    export AGA_DISTRIBUTED_ENABLED=true
    echo "Running in distributed mode..."
fi

python3 -m aga_experiment_tool.app --port $PORT
