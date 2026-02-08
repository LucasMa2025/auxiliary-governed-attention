@echo off
REM ============================================================
REM AGA Portal 启动脚本 (Windows)
REM ============================================================
REM
REM 使用方式:
REM   scripts\start_portal.bat                 开发模式
REM   scripts\start_portal.bat --prod          生产模式
REM
REM v3.5.0 新增:
REM   --metrics-port PORT        独立指标端口
REM   --log-format json|text     日志格式
REM   --enable-ann               启用 ANN 索引
REM
REM ============================================================

setlocal enabledelayedexpansion

REM 默认配置
set HOST=0.0.0.0
set PORT=8081
set WORKERS=1
set RELOAD=
set LOG_LEVEL=info
set PERSISTENCE=sqlite
set MESSAGING=memory
set REDIS_HOST=localhost
set POSTGRES_URL=

REM v3.5.0 新增配置
set METRICS_PORT=
set LOG_FORMAT=text
set LOG_FILE=
set ENABLE_ANN=false
set ANN_CAPACITY=100000
set HOT_POOL_SIZE=256

REM 解析参数
:parse_args
if "%~1"=="" goto :start
if "%~1"=="--prod" (
    set WORKERS=4
    set LOG_LEVEL=warning
    set LOG_FORMAT=json
    set MESSAGING=redis
    echo [生产模式] 4 workers, JSON 日志
    shift
    goto :parse_args
)
if "%~1"=="--dev" (
    set WORKERS=1
    set RELOAD=--reload
    set LOG_LEVEL=debug
    set LOG_FORMAT=text
    echo [开发模式] 自动重载, DEBUG 日志
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--redis" (
    set REDIS_HOST=%~2
    set MESSAGING=redis
    shift
    shift
    goto :parse_args
)
if "%~1"=="--postgres" (
    set POSTGRES_URL=%~2
    set PERSISTENCE=postgres
    shift
    shift
    goto :parse_args
)
if "%~1"=="--metrics-port" (
    set METRICS_PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--log-format" (
    set LOG_FORMAT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--log-file" (
    set LOG_FILE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--log-level" (
    set LOG_LEVEL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--enable-ann" (
    set ENABLE_ANN=true
    shift
    goto :parse_args
)
if "%~1"=="--ann-capacity" (
    set ANN_CAPACITY=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--hot-pool-size" (
    set HOT_POOL_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help

shift
goto :parse_args

:help
echo AGA Portal 启动脚本 v3.5.0
echo.
echo 使用方式:
echo   %~nx0 [选项]
echo.
echo 基础选项:
echo   --dev              开发模式 (自动重载)
echo   --prod             生产模式 (多 worker)
echo   --host HOST        监听地址 (默认: 0.0.0.0)
echo   --port PORT        监听端口 (默认: 8081)
echo   --redis HOST       启用 Redis (消息同步)
echo   --postgres URL     使用 PostgreSQL
echo.
echo 监控选项 (v3.5.0 新增):
echo   --metrics-port PORT   独立 Prometheus 指标端口
echo   --log-format FORMAT   日志格式: json, text
echo   --log-file PATH       日志文件路径
echo   --log-level LEVEL     日志级别
echo.
echo 大规模知识库选项 (v3.5.0 新增):
echo   --enable-ann          启用 ANN 索引层
echo   --ann-capacity N      ANN 索引容量 (默认: 100000)
echo   --hot-pool-size N     Hot Pool 大小 (默认: 256)
echo.
echo   --help, -h         显示帮助
echo.
echo 示例:
echo   %~nx0 --dev
echo   %~nx0 --prod --redis localhost
echo   %~nx0 --prod --log-format json --metrics-port 9090
echo   %~nx0 --prod --enable-ann --ann-capacity 1000000
goto :eof

:start
REM 检查依赖
echo 检查依赖...
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo 错误: 缺少依赖。请运行: pip install fastapi uvicorn
    exit /b 1
)

REM 设置环境变量
set AGA_LOG_FORMAT=%LOG_FORMAT%
set AGA_LOG_LEVEL=%LOG_LEVEL%
if not "%LOG_FILE%"=="" set AGA_LOG_FILE=%LOG_FILE%
if not "%METRICS_PORT%"=="" set AGA_METRICS_PORT=%METRICS_PORT%
if "%ENABLE_ANN%"=="true" (
    set AGA_ENABLE_ANN=true
    set AGA_ANN_CAPACITY=%ANN_CAPACITY%
    set AGA_HOT_POOL_SIZE=%HOT_POOL_SIZE%
)

REM 显示配置
echo.
echo ==========================================
echo   AGA Portal v3.5.0
echo ==========================================
echo.
echo   服务配置:
echo     地址:       %HOST%:%PORT%
echo     进程数:     %WORKERS%
echo     日志级别:   %LOG_LEVEL%
echo     日志格式:   %LOG_FORMAT%
echo.
echo   存储配置:
echo     持久化:     %PERSISTENCE%
echo     消息队列:   %MESSAGING%
if "%MESSAGING%"=="redis" (
    echo     Redis:      %REDIS_HOST%
)
echo.
if not "%METRICS_PORT%"=="" (
    echo   监控配置:
    echo     指标端口:   %METRICS_PORT%
    if not "%LOG_FILE%"=="" (
        echo     日志文件:   %LOG_FILE%
    )
    echo.
)
if "%ENABLE_ANN%"=="true" (
    echo   大规模知识库:
    echo     ANN 索引:   启用
    echo     索引容量:   %ANN_CAPACITY%
    echo     Hot Pool:   %HOT_POOL_SIZE%
    echo.
)
echo   访问地址:
echo     API 文档:   http://%HOST%:%PORT%/docs
if not "%METRICS_PORT%"=="" (
    echo     Metrics:    http://%HOST%:%METRICS_PORT%/metrics
)
echo.
echo ==========================================
echo.

REM 构建命令
set CMD=python -m aga.portal.app
set CMD=%CMD% --host %HOST%
set CMD=%CMD% --port %PORT%
set CMD=%CMD% --workers %WORKERS%
set CMD=%CMD% --persistence %PERSISTENCE%
set CMD=%CMD% --messaging %MESSAGING%

if "%MESSAGING%"=="redis" (
    set CMD=%CMD% --redis-host %REDIS_HOST%
)

if not "%POSTGRES_URL%"=="" (
    set CMD=%CMD% --postgres-url %POSTGRES_URL%
)

if not "%RELOAD%"=="" (
    set CMD=%CMD% %RELOAD%
)

REM 启动
echo 启动 Portal...
%CMD%
