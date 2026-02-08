#!/bin/bash
# ============================================================
# AGA Portal 启动脚本 (Linux/macOS)
# ============================================================
#
# 使用方式:
#   ./scripts/start_portal.sh                    # 开发模式
#   ./scripts/start_portal.sh --prod             # 生产模式
#   ./scripts/start_portal.sh --config config.yaml
#
# v3.5.0 新增:
#   --metrics-port PORT         # 独立指标端口
#   --log-format json|text      # 日志格式
#   --log-file PATH             # 日志文件路径
#   --enable-ann                # 启用 ANN 索引
#
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认配置
HOST="0.0.0.0"
PORT="8081"
WORKERS="1"
RELOAD=""
LOG_LEVEL="info"
CONFIG=""
PERSISTENCE="sqlite"
MESSAGING="memory"
REDIS_HOST="localhost"
POSTGRES_URL=""

# v3.5.0 新增配置
METRICS_PORT=""
LOG_FORMAT="text"
LOG_FILE=""
ENABLE_ANN="false"
ANN_CAPACITY="100000"
HOT_POOL_SIZE="256"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod|--production)
            WORKERS="4"
            LOG_LEVEL="warning"
            LOG_FORMAT="json"
            PERSISTENCE="postgres"
            MESSAGING="redis"
            echo -e "${GREEN}生产模式: ${WORKERS} workers, JSON 日志${NC}"
            shift
            ;;
        --dev|--development)
            WORKERS="1"
            RELOAD="--reload"
            LOG_LEVEL="debug"
            LOG_FORMAT="text"
            echo -e "${YELLOW}开发模式: 自动重载, DEBUG 日志${NC}"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --redis)
            REDIS_HOST="$2"
            MESSAGING="redis"
            shift 2
            ;;
        --postgres)
            POSTGRES_URL="$2"
            PERSISTENCE="postgres"
            shift 2
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
        --hot-pool-size)
            HOT_POOL_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo -e "${BLUE}AGA Portal 启动脚本 v3.5.0${NC}"
            echo ""
            echo "使用方式:"
            echo "  $0 [选项]"
            echo ""
            echo "基础选项:"
            echo "  --dev, --development   开发模式 (自动重载)"
            echo "  --prod, --production   生产模式 (多 worker)"
            echo "  --config FILE          配置文件路径"
            echo "  --host HOST            监听地址 (默认: 0.0.0.0)"
            echo "  --port PORT            监听端口 (默认: 8081)"
            echo "  --workers N            工作进程数"
            echo "  --redis HOST           启用 Redis (消息同步)"
            echo "  --postgres URL         使用 PostgreSQL"
            echo ""
            echo -e "${CYAN}监控选项 (v3.5.0 新增):${NC}"
            echo "  --metrics-port PORT    独立 Prometheus 指标端口"
            echo "  --log-format FORMAT    日志格式: json, text"
            echo "  --log-file PATH        日志文件路径"
            echo "  --log-level LEVEL      日志级别"
            echo ""
            echo -e "${CYAN}大规模知识库选项 (v3.5.0 新增):${NC}"
            echo "  --enable-ann           启用 ANN 索引层"
            echo "  --ann-capacity N       ANN 索引容量 (默认: 100000)"
            echo "  --hot-pool-size N      Hot Pool 大小 (默认: 256)"
            echo ""
            echo "  --help, -h             显示帮助"
            echo ""
            echo "示例:"
            echo "  $0 --dev"
            echo "  $0 --prod --redis localhost --postgres postgresql://..."
            echo "  $0 --prod --log-format json --metrics-port 9090"
            echo "  $0 --prod --enable-ann --ann-capacity 1000000"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            exit 1
            ;;
    esac
done

# 检查依赖
echo -e "${BLUE}检查依赖...${NC}"
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo -e "${RED}错误: 缺少依赖。请运行: pip install fastapi uvicorn${NC}"
    exit 1
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
    export AGA_HOT_POOL_SIZE="$HOT_POOL_SIZE"
fi

# 显示配置
echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  AGA Portal v3.5.0${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "  ${CYAN}服务配置:${NC}"
echo -e "    地址:       ${GREEN}$HOST:$PORT${NC}"
echo -e "    进程数:     ${GREEN}$WORKERS${NC}"
echo -e "    日志级别:   ${GREEN}$LOG_LEVEL${NC}"
echo -e "    日志格式:   ${GREEN}$LOG_FORMAT${NC}"
echo ""
echo -e "  ${CYAN}存储配置:${NC}"
echo -e "    持久化:     ${GREEN}$PERSISTENCE${NC}"
echo -e "    消息队列:   ${GREEN}$MESSAGING${NC}"
if [ -n "$REDIS_HOST" ] && [ "$MESSAGING" = "redis" ]; then
    echo -e "    Redis:      ${GREEN}$REDIS_HOST${NC}"
fi
if [ -n "$CONFIG" ]; then
    echo -e "    配置文件:   ${GREEN}$CONFIG${NC}"
fi
echo ""
if [ -n "$METRICS_PORT" ]; then
    echo -e "  ${CYAN}监控配置:${NC}"
    echo -e "    指标端口:   ${GREEN}$METRICS_PORT${NC}"
    if [ -n "$LOG_FILE" ]; then
        echo -e "    日志文件:   ${GREEN}$LOG_FILE${NC}"
    fi
    echo ""
fi
if [ "$ENABLE_ANN" = "true" ]; then
    echo -e "  ${CYAN}大规模知识库:${NC}"
    echo -e "    ANN 索引:   ${GREEN}启用${NC}"
    echo -e "    索引容量:   ${GREEN}$ANN_CAPACITY${NC}"
    echo -e "    Hot Pool:   ${GREEN}$HOT_POOL_SIZE${NC}"
    echo ""
fi
echo -e "  ${CYAN}访问地址:${NC}"
echo -e "    API 文档:   ${GREEN}http://$HOST:$PORT/docs${NC}"
if [ -n "$METRICS_PORT" ]; then
    echo -e "    Metrics:    ${GREEN}http://$HOST:$METRICS_PORT/metrics${NC}"
fi
echo ""
echo -e "${BLUE}==========================================${NC}"
echo ""

# 构建命令
CMD="python3 -m aga.portal.app"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --persistence $PERSISTENCE"
CMD="$CMD --messaging $MESSAGING"

if [ -n "$REDIS_HOST" ] && [ "$MESSAGING" = "redis" ]; then
    CMD="$CMD --redis-host $REDIS_HOST"
fi

if [ -n "$POSTGRES_URL" ]; then
    CMD="$CMD --postgres-url $POSTGRES_URL"
fi

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi

if [ -n "$RELOAD" ]; then
    CMD="$CMD $RELOAD"
fi

# 启动
echo -e "${GREEN}启动 Portal...${NC}"
exec $CMD
