"""
AGA API 命令行入口

使用方式:
    python -m aga.api --port 8081
"""
from .app import main

if __name__ == "__main__":
    main()
