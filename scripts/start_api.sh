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
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 显示帮助
show_help() {
    echo -e "${BLUE}AGA REST API 启动脚本${NC}"
    echo ""
    echo "使用方式:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --dev           开发模式 (自动重载)"
    echo "  --prod          生产模式 (4 workers)"
    echo "  --port PORT     指定端口 (默认: 8081)"
    echo "  --host HOST     指定地址 (默认: 0.0.0.0)"
    echo "  --hidden-dim N  隐藏层维度 (默认: 4096)"
    echo "  --bottleneck N  瓶颈层维度 (默认: 64)"
    echo "  --slots N       槽位数 (默认: 100)"
    echo "  --postgres URL  使用 PostgreSQL"
    echo "  --no-persist    禁用持久化"
    echo "  --help          显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --dev --port 8082"
    echo "  $0 --prod --postgres postgresql://user:pass@localhost/aga"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            RELOAD="--reload"
            WORKERS=1
            echo -e "${YELLOW}开发模式: 启用自动重载${NC}"
            shift
            ;;
        --prod)
            WORKERS=4
            echo -e "${GREEN}生产模式: ${WORKERS} workers${NC}"
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

# 显示配置
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}           AGA REST API (Demo Mode)${NC}"
echo -e "${BLUE}           Version: 3.4.0${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "  主机:           ${GREEN}${HOST}:${PORT}${NC}"
echo -e "  隐藏层维度:     ${GREEN}${HIDDEN_DIM}${NC}"
echo -e "  瓶颈层维度:     ${GREEN}${BOTTLENECK_DIM}${NC}"
echo -e "  槽位数:         ${GREEN}${NUM_SLOTS}${NC}"
echo -e "  持久化:         ${GREEN}${PERSISTENCE_TYPE}${NC}"
echo -e "  Workers:        ${GREEN}${WORKERS}${NC}"
if [ -n "$RELOAD" ]; then
    echo -e "  自动重载:       ${GREEN}启用${NC}"
fi
echo ""
echo -e "  Swagger UI:     ${GREEN}http://${HOST}:${PORT}/docs${NC}"
echo -e "  ReDoc:          ${GREEN}http://${HOST}:${PORT}/redoc${NC}"
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
