#!/bin/bash
# ============================================================
# AGA REST API 启动脚本 (Linux/macOS)
# ============================================================
#
# 使用方式:
#   ./start_api.sh              # 默认配置启动
#   ./start_api.sh --dev        # 开发模式
#   ./start_api.sh --prod       # 生产模式
#   ./start_api.sh --help       # 显示帮助
#
# v3.5.0 新增:
#   --metrics-port PORT         # 独立指标端口
#   --log-format json|text      # 日志格式
#   --log-file PATH             # 日志文件路径
#
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认配置
HOST="0.0.0.0"
PORT=8081
HIDDEN_DIM=4096
BOTTLENECK_DIM=64
NUM_SLOTS=100
NUM_HEADS=32
PERSISTENCE_TYPE="sqlite"
DB_PATH="./aga_api_data.db"
WORKERS=1
RELOAD=""
EXTRA_ARGS=""

# v3.5.0 新增配置
METRICS_PORT=""
LOG_FORMAT="text"
LOG_FILE=""
LOG_LEVEL="INFO"
ENABLE_ANN="false"
ANN_CAPACITY=100000

# 显示帮助
show_help() {
    echo -e "${BLUE}AGA REST API 启动脚本 v3.5.0${NC}"
    echo ""
    echo "使用方式:"
    echo "  $0 [选项]"
    echo ""
    echo "基础选项:"
    echo "  --dev           开发模式 (自动重载, DEBUG 日志)"
    echo "  --prod          生产模式 (4 workers, WARNING 日志)"
    echo "  --port PORT     指定端口 (默认: 8081)"
    echo "  --host HOST     指定地址 (默认: 0.0.0.0)"
    echo "  --hidden-dim N  隐藏层维度 (默认: 4096)"
    echo "  --bottleneck N  瓶颈层维度 (默认: 64)"
    echo "  --slots N       槽位数 (默认: 100)"
    echo "  --postgres URL  使用 PostgreSQL"
    echo "  --no-persist    禁用持久化"
    echo ""
    echo -e "${CYAN}监控选项 (v3.5.0 新增):${NC}"
    echo "  --metrics-port PORT   独立 Prometheus 指标端口"
    echo "  --log-format FORMAT   日志格式: json, text (默认: text)"
    echo "  --log-file PATH       日志文件路径"
    echo "  --log-level LEVEL     日志级别: DEBUG, INFO, WARNING, ERROR"
    echo ""
    echo -e "${CYAN}大规模知识库选项 (v3.5.0 新增):${NC}"
    echo "  --enable-ann          启用 ANN 索引层"
    echo "  --ann-capacity N      ANN 索引容量 (默认: 100000)"
    echo ""
    echo "  --help          显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --dev --port 8082"
    echo "  $0 --prod --postgres postgresql://user:pass@localhost/aga"
    echo "  $0 --prod --log-format json --metrics-port 9090"
    echo "  $0 --prod --enable-ann --ann-capacity 1000000"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            RELOAD="--reload"
            WORKERS=1
            LOG_LEVEL="DEBUG"
            LOG_FORMAT="text"
            echo -e "${YELLOW}开发模式: 启用自动重载, DEBUG 日志${NC}"
            shift
            ;;
        --prod)
            WORKERS=4
            LOG_LEVEL="WARNING"
            LOG_FORMAT="json"
            echo -e "${GREEN}生产模式: ${WORKERS} workers, JSON 日志${NC}"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --bottleneck)
            BOTTLENECK_DIM="$2"
            shift 2
            ;;
        --slots)
            NUM_SLOTS="$2"
            shift 2
            ;;
        --postgres)
            PERSISTENCE_TYPE="postgres"
            DB_PATH="$2"
            shift 2
            ;;
        --no-persist)
            EXTRA_ARGS="$EXTRA_ARGS --no-persistence"
            shift
            ;;
        --metrics-port)
            METRICS_PORT="$2"
            shift 2
            ;;
        --log-format)
            LOG_FORMAT="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --enable-ann)
            ENABLE_ANN="true"
            shift
            ;;
        --ann-capacity)
            ANN_CAPACITY="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 切换到项目目录
cd "$PROJECT_ROOT"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3${NC}"
    exit 1
fi

# 检查依赖
echo -e "${BLUE}检查依赖...${NC}"
python3 -c "import fastapi" 2>/dev/null || {
    echo -e "${YELLOW}安装 FastAPI...${NC}"
    pip install fastapi uvicorn pydantic
}

# 检查 prometheus_client (可选)
if [ -n "$METRICS_PORT" ]; then
    python3 -c "import prometheus_client" 2>/dev/null || {
        echo -e "${YELLOW}安装 prometheus_client...${NC}"
        pip install prometheus_client
    }
fi

# 检查 ANN 依赖 (可选)
if [ "$ENABLE_ANN" = "true" ]; then
    python3 -c "import faiss" 2>/dev/null || {
        echo -e "${YELLOW}警告: faiss 未安装, ANN 功能可能受限${NC}"
        echo -e "${YELLOW}安装: pip install faiss-cpu 或 faiss-gpu${NC}"
    }
fi

# 设置环境变量
export AGA_LOG_FORMAT="$LOG_FORMAT"
export AGA_LOG_LEVEL="$LOG_LEVEL"
if [ -n "$LOG_FILE" ]; then
    export AGA_LOG_FILE="$LOG_FILE"
fi
if [ -n "$METRICS_PORT" ]; then
    export AGA_METRICS_PORT="$METRICS_PORT"
fi
if [ "$ENABLE_ANN" = "true" ]; then
    export AGA_ENABLE_ANN="true"
    export AGA_ANN_CAPACITY="$ANN_CAPACITY"
fi

# 显示配置
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}           AGA REST API v3.5.0${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "  ${CYAN}服务配置:${NC}"
echo -e "    主机:           ${GREEN}${HOST}:${PORT}${NC}"
echo -e "    Workers:        ${GREEN}${WORKERS}${NC}"
if [ -n "$RELOAD" ]; then
    echo -e "    自动重载:       ${GREEN}启用${NC}"
fi
echo ""
echo -e "  ${CYAN}AGA 配置:${NC}"
echo -e "    隐藏层维度:     ${GREEN}${HIDDEN_DIM}${NC}"
echo -e "    瓶颈层维度:     ${GREEN}${BOTTLENECK_DIM}${NC}"
echo -e "    槽位数:         ${GREEN}${NUM_SLOTS}${NC}"
echo -e "    持久化:         ${GREEN}${PERSISTENCE_TYPE}${NC}"
echo ""
echo -e "  ${CYAN}监控配置:${NC}"
echo -e "    日志格式:       ${GREEN}${LOG_FORMAT}${NC}"
echo -e "    日志级别:       ${GREEN}${LOG_LEVEL}${NC}"
if [ -n "$LOG_FILE" ]; then
    echo -e "    日志文件:       ${GREEN}${LOG_FILE}${NC}"
fi
if [ -n "$METRICS_PORT" ]; then
    echo -e "    指标端口:       ${GREEN}${METRICS_PORT}${NC}"
fi
echo ""
if [ "$ENABLE_ANN" = "true" ]; then
    echo -e "  ${CYAN}大规模知识库:${NC}"
    echo -e "    ANN 索引:       ${GREEN}启用${NC}"
    echo -e "    索引容量:       ${GREEN}${ANN_CAPACITY}${NC}"
    echo ""
fi
echo -e "  ${CYAN}访问地址:${NC}"
echo -e "    Swagger UI:     ${GREEN}http://${HOST}:${PORT}/docs${NC}"
echo -e "    ReDoc:          ${GREEN}http://${HOST}:${PORT}/redoc${NC}"
if [ -n "$METRICS_PORT" ]; then
    echo -e "    Metrics:        ${GREEN}http://${HOST}:${METRICS_PORT}/metrics${NC}"
fi
echo ""
echo -e "${BLUE}============================================================${NC}"
echo ""

# 启动服务
echo -e "${GREEN}启动 AGA API 服务...${NC}"
echo ""

python3 -m aga.api \
    --host "$HOST" \
    --port "$PORT" \
    --hidden-dim "$HIDDEN_DIM" \
    --bottleneck-dim "$BOTTLENECK_DIM" \
    --num-slots "$NUM_SLOTS" \
    --num-heads "$NUM_HEADS" \
    --persistence-type "$PERSISTENCE_TYPE" \
    --db-path "$DB_PATH" \
    --workers "$WORKERS" \
    $RELOAD \
    $EXTRA_ARGS
