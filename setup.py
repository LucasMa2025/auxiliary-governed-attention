"""
AGA - Auxiliary Governed Attention

A hot-pluggable knowledge injection system for frozen Transformers.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aga",
    version="0.1.0",
    author="Lucas Ma",
    author_email="lucas_ma2025@126.com",
    description="Auxiliary Governed Attention - Zero-training knowledge injection for frozen Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucasMa-Research/auxiliary-governed-attention",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "flask>=2.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "llm": [
            "openai>=1.0.0",
            "httpx>=0.25.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aga-experiment=aga_experiment_tool.app:main",
        ],
    },
)

