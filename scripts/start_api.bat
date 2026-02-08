@echo off
REM ============================================================
REM AGA REST API 启动脚本 (Windows)
REM ============================================================
REM
REM 使用方式:
REM   start_api.bat              默认配置启动
REM   start_api.bat --dev        开发模式
REM   start_api.bat --prod       生产模式
REM   start_api.bat --help       显示帮助
REM
REM v3.5.0 新增:
REM   --metrics-port PORT        独立指标端口
REM   --log-format json|text     日志格式
REM   --log-file PATH            日志文件路径
REM
REM ============================================================

setlocal EnableDelayedExpansion

REM 默认配置
set HOST=0.0.0.0
set PORT=8081
set HIDDEN_DIM=4096
set BOTTLENECK_DIM=64
set NUM_SLOTS=100
set NUM_HEADS=32
set PERSISTENCE_TYPE=sqlite
set DB_PATH=./aga_api_data.db
set WORKERS=1
set RELOAD=
set EXTRA_ARGS=

REM v3.5.0 新增配置
set METRICS_PORT=
set LOG_FORMAT=text
set LOG_FILE=
set LOG_LEVEL=INFO
set ENABLE_ANN=false
set ANN_CAPACITY=100000

REM 获取脚本目录
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM 解析参数
:parse_args
if "%~1"=="" goto :after_args
if "%~1"=="--help" goto :show_help
if "%~1"=="--dev" (
    set RELOAD=--reload
    set WORKERS=1
    set LOG_LEVEL=DEBUG
    set LOG_FORMAT=text
    echo [开发模式] 启用自动重载, DEBUG 日志
    shift
    goto :parse_args
)
if "%~1"=="--prod" (
    set WORKERS=4
    set LOG_LEVEL=WARNING
    set LOG_FORMAT=json
    echo [生产模式] 4 workers, JSON 日志
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--hidden-dim" (
    set HIDDEN_DIM=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--bottleneck" (
    set BOTTLENECK_DIM=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--slots" (
    set NUM_SLOTS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--postgres" (
    set PERSISTENCE_TYPE=postgres
    set DB_PATH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-persist" (
    set EXTRA_ARGS=%EXTRA_ARGS% --no-persistence
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
echo 未知参数: %~1
goto :show_help

:after_args

REM 切换到项目目录
cd /d "%PROJECT_ROOT%"

REM 检查 Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [错误] 未找到 Python
    exit /b 1
)

REM 检查依赖
echo 检查依赖...
python -c "import fastapi" 2>nul
if %ERRORLEVEL% neq 0 (
    echo 安装 FastAPI...
    pip install fastapi uvicorn pydantic
)

REM 设置环境变量
set AGA_LOG_FORMAT=%LOG_FORMAT%
set AGA_LOG_LEVEL=%LOG_LEVEL%
if not "%LOG_FILE%"=="" set AGA_LOG_FILE=%LOG_FILE%
if not "%METRICS_PORT%"=="" set AGA_METRICS_PORT=%METRICS_PORT%
if "%ENABLE_ANN%"=="true" (
    set AGA_ENABLE_ANN=true
    set AGA_ANN_CAPACITY=%ANN_CAPACITY%
)

REM 显示配置
echo.
echo ============================================================
echo            AGA REST API v3.5.0
echo ============================================================
echo.
echo   服务配置:
echo     主机:           %HOST%:%PORT%
echo     Workers:        %WORKERS%
if defined RELOAD (
    echo     自动重载:       启用
)
echo.
echo   AGA 配置:
echo     隐藏层维度:     %HIDDEN_DIM%
echo     瓶颈层维度:     %BOTTLENECK_DIM%
echo     槽位数:         %NUM_SLOTS%
echo     持久化:         %PERSISTENCE_TYPE%
echo.
echo   监控配置:
echo     日志格式:       %LOG_FORMAT%
echo     日志级别:       %LOG_LEVEL%
if not "%LOG_FILE%"=="" (
    echo     日志文件:       %LOG_FILE%
)
if not "%METRICS_PORT%"=="" (
    echo     指标端口:       %METRICS_PORT%
)
echo.
if "%ENABLE_ANN%"=="true" (
    echo   大规模知识库:
    echo     ANN 索引:       启用
    echo     索引容量:       %ANN_CAPACITY%
    echo.
)
echo   访问地址:
echo     Swagger UI:     http://%HOST%:%PORT%/docs
echo     ReDoc:          http://%HOST%:%PORT%/redoc
if not "%METRICS_PORT%"=="" (
    echo     Metrics:        http://%HOST%:%METRICS_PORT%/metrics
)
echo.
echo ============================================================
echo.

REM 启动服务
echo 启动 AGA API 服务...
echo.

python -m aga.api ^
    --host %HOST% ^
    --port %PORT% ^
    --hidden-dim %HIDDEN_DIM% ^
    --bottleneck-dim %BOTTLENECK_DIM% ^
    --num-slots %NUM_SLOTS% ^
    --num-heads %NUM_HEADS% ^
    --persistence-type %PERSISTENCE_TYPE% ^
    --db-path %DB_PATH% ^
    --workers %WORKERS% ^
    %RELOAD% ^
    %EXTRA_ARGS%

goto :eof

:show_help
echo AGA REST API 启动脚本 v3.5.0
echo.
echo 使用方式:
echo   %~nx0 [选项]
echo.
echo 基础选项:
echo   --dev           开发模式 (自动重载, DEBUG 日志)
echo   --prod          生产模式 (4 workers, WARNING 日志)
echo   --port PORT     指定端口 (默认: 8081)
echo   --host HOST     指定地址 (默认: 0.0.0.0)
echo   --hidden-dim N  隐藏层维度 (默认: 4096)
echo   --bottleneck N  瓶颈层维度 (默认: 64)
echo   --slots N       槽位数 (默认: 100)
echo   --postgres URL  使用 PostgreSQL
echo   --no-persist    禁用持久化
echo.
echo 监控选项 (v3.5.0 新增):
echo   --metrics-port PORT   独立 Prometheus 指标端口
echo   --log-format FORMAT   日志格式: json, text (默认: text)
echo   --log-file PATH       日志文件路径
echo   --log-level LEVEL     日志级别: DEBUG, INFO, WARNING, ERROR
echo.
echo 大规模知识库选项 (v3.5.0 新增):
echo   --enable-ann          启用 ANN 索引层
echo   --ann-capacity N      ANN 索引容量 (默认: 100000)
echo.
echo   --help          显示帮助
echo.
echo 示例:
echo   %~nx0 --dev --port 8082
echo   %~nx0 --prod --postgres postgresql://user:pass@localhost/aga
echo   %~nx0 --prod --log-format json --metrics-port 9090
echo   %~nx0 --prod --enable-ann --ann-capacity 1000000
echo.
goto :eof
