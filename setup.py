"""
AGA - Auxiliary Governed Attention

A hot-pluggable knowledge injection system for frozen Transformers.
"""
from setuptools import setup, find_packages

# 从包中读取版本号，确保单一版本源
def get_version():
    """从 aga/__init__.py 读取版本号"""
    import re
    with open("aga/__init__.py", "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aga",
    version=get_version(),
    author="Lucas Ma",
    author_email="lucas_ma2025@126.com",
    description="Auxiliary Governed Attention - Zero-training knowledge injection for frozen Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucasMa2025/auxiliary-governed-attention",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "flask>=2.3.0",
        "pyyaml>=6.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.0",
        "httpx>=0.25.0",
        "aiosqlite>=0.19.0",
    ],
    extras_require={
        "llm": [
            "openai>=1.0.0",
        ],
        "compression": [
            "lz4>=4.3.0",
            "zstd>=1.5.0",
        ],
        "redis": [
            "redis>=5.0.0",
        ],
        "postgres": [
            "asyncpg>=0.29.0",
        ],
        "kafka": [
            "aiokafka>=0.10.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
            "opentelemetry-exporter-otlp>=1.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "openai>=1.0.0",
            "lz4>=4.3.0",
            "redis>=5.0.0",
            "asyncpg>=0.29.0",
            "prometheus-client>=0.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aga-experiment=aga_experiment_tool.app:main",
            "aga-portal=aga.portal.app:main",
        ],
    },
    project_urls={
        "Documentation": "https://github.com/LucasMa-Research/auxiliary-governed-attention/docs",
        "Bug Reports": "https://github.com/LucasMa-Research/auxiliary-governed-attention/issues",
        "Source": "https://github.com/LucasMa-Research/auxiliary-governed-attention",
    },
)
