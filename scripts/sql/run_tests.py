#!/usr/bin/env python
"""
AGA 测试运行脚本

用法:
    python run_tests.py                    # 运行所有测试
    python run_tests.py unit               # 只运行单元测试
    python run_tests.py component          # 只运行组件测试
    python run_tests.py integration        # 只运行集成测试
    python run_tests.py fault              # 只运行故障测试
    python run_tests.py performance        # 只运行性能测试
    python run_tests.py --fast             # 跳过慢速测试
    python run_tests.py --coverage         # 生成覆盖率报告
"""
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AGA 测试运行器")
    parser.add_argument(
        "suite",
        nargs="?",
        choices=["unit", "component", "integration", "fault", "performance", "all"],
        default="all",
        help="要运行的测试套件",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="跳过慢速测试",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="生成覆盖率报告",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "--failfast", "-x",
        action="store_true",
        help="遇到第一个失败就停止",
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        default=0,
        help="并行运行测试的进程数（需要 pytest-xdist）",
    )
    
    args = parser.parse_args()
    
    # 构建 pytest 命令
    cmd = ["python", "-m", "pytest"]
    
    # 测试路径
    test_path = Path(__file__).parent / "tests"
    
    if args.suite == "unit":
        cmd.append(str(test_path / "unit"))
        cmd.extend(["-m", "unit"])
    elif args.suite == "component":
        cmd.append(str(test_path / "component"))
        cmd.extend(["-m", "component"])
    elif args.suite == "integration":
        cmd.append(str(test_path / "integration"))
        cmd.extend(["-m", "integration"])
    elif args.suite == "fault":
        cmd.append(str(test_path / "fault"))
        cmd.extend(["-m", "fault"])
    elif args.suite == "performance":
        cmd.append(str(test_path / "performance"))
        cmd.extend(["-m", "performance"])
    else:
        cmd.append(str(test_path))
    
    # 选项
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    if args.coverage:
        cmd.extend(["--cov=aga", "--cov-report=html", "--cov-report=term-missing"])
    
    if args.verbose:
        cmd.append("-v")
    
    if args.failfast:
        cmd.append("-x")
    
    if args.parallel > 0:
        cmd.extend(["-n", str(args.parallel)])
    
    # 运行
    print(f"运行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
