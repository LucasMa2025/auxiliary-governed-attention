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
    echo [开发模式] 启用自动重载
    shift
    goto :parse_args
)
if "%~1"=="--prod" (
    set WORKERS=4
    echo [生产模式] 4 workers
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

REM 显示配置
echo.
echo ============================================================
echo            AGA REST API v3.1
echo ============================================================
echo.
echo   主机:           %HOST%:%PORT%
echo   隐藏层维度:     %HIDDEN_DIM%
echo   瓶颈层维度:     %BOTTLENECK_DIM%
echo   槽位数:         %NUM_SLOTS%
echo   持久化:         %PERSISTENCE_TYPE%
echo   Workers:        %WORKERS%
if defined RELOAD (
    echo   自动重载:       启用
)
echo.
echo   Swagger UI:     http://%HOST%:%PORT%/docs
echo   ReDoc:          http://%HOST%:%PORT%/redoc
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
echo AGA REST API 启动脚本
echo.
echo 使用方式:
echo   %~nx0 [选项]
echo.
echo 选项:
echo   --dev           开发模式 (自动重载)
echo   --prod          生产模式 (4 workers)
echo   --port PORT     指定端口 (默认: 8081)
echo   --host HOST     指定地址 (默认: 0.0.0.0)
echo   --hidden-dim N  隐藏层维度 (默认: 4096)
echo   --bottleneck N  瓶颈层维度 (默认: 64)
echo   --slots N       槽位数 (默认: 100)
echo   --postgres URL  使用 PostgreSQL
echo   --no-persist    禁用持久化
echo   --help          显示帮助
echo.
echo 示例:
echo   %~nx0 --dev --port 8082
echo   %~nx0 --prod --postgres postgresql://user:pass@localhost/aga
echo.
goto :eof
