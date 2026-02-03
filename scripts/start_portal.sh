#!/bin/bash
#
# AGA Portal 启动脚本 (Linux/macOS)
#
# 使用方式:
#   ./scripts/start_portal.sh                    # 开发模式
#   ./scripts/start_portal.sh --prod             # 生产模式
#   ./scripts/start_portal.sh --config config.yaml
#

set -e

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

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod|--production)
            WORKERS="4"
            LOG_LEVEL="warning"
            PERSISTENCE="postgres"
            MESSAGING="redis"
            shift
            ;;
        --dev|--development)
            WORKERS="1"
            RELOAD="--reload"
            LOG_LEVEL="debug"
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
        --help|-h)
            echo "AGA Portal 启动脚本"
            echo ""
            echo "使用方式:"
            echo "  $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --dev, --development   开发模式 (自动重载)"
            echo "  --prod, --production   生产模式 (多 worker)"
            echo "  --config FILE          配置文件路径"
            echo "  --host HOST            监听地址 (默认: 0.0.0.0)"
            echo "  --port PORT            监听端口 (默认: 8081)"
            echo "  --workers N            工作进程数"
            echo "  --redis HOST           启用 Redis (消息同步)"
            echo "  --postgres URL         使用 PostgreSQL"
            echo "  --help, -h             显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查依赖
echo "检查依赖..."
python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "错误: 缺少依赖。请运行: pip install fastapi uvicorn"
    exit 1
}

# 显示配置
echo ""
echo "=========================================="
echo "  AGA Portal 启动配置"
echo "=========================================="
echo "  地址:     $HOST:$PORT"
echo "  进程数:   $WORKERS"
echo "  日志级别: $LOG_LEVEL"
echo "  持久化:   $PERSISTENCE"
echo "  消息队列: $MESSAGING"
if [ -n "$REDIS_HOST" ] && [ "$MESSAGING" = "redis" ]; then
    echo "  Redis:    $REDIS_HOST"
fi
if [ -n "$CONFIG" ]; then
    echo "  配置文件: $CONFIG"
fi
echo "=========================================="
echo ""

# 构建命令
CMD="python -m aga.portal.app"
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
echo "启动 Portal..."
exec $CMD
