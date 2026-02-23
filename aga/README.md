# AGA — Auxiliary Governed Attention

<p align="center">
  <strong>极简注意力治理插件 · Minimalist Attention Governance Plugin</strong><br/>
  为冻结 LLM 提供推理时动态知识注入能力<br/>
  <em>Runtime dynamic knowledge injection for frozen LLMs</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-4.2.0-blue" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license"/>
  <img src="https://img.shields.io/badge/torch-2.0+-red" alt="torch"/>
</p>

---

## 📖 文档 / Documentation

| 语言 / Language | README                       | 用户手册 / User Manual                      | 产品文档 / Product Doc                      |
| --------------- | ---------------------------- | ------------------------------------------- | ------------------------------------------- |
| 🇨🇳 中文         | [README_zh.md](README_zh.md) | [user_manual_zh.md](docs/user_manual_zh.md) | [product_doc_zh.md](docs/product_doc_zh.md) |
| 🇬🇧 English      | [README_en.md](README_en.md) | [user_manual_en.md](docs/user_manual_en.md) | [product_doc_en.md](docs/product_doc_en.md) |

---

## ⚡ Quick Start / 快速开始

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)
output = model.generate(input_ids)  # AGA 自动工作 / AGA works automatically
```

## 🌊 Streaming / 流式生成

```python
session = plugin.create_streaming_session()
for token in model_generate_stream(input_ids):
    diag = session.get_step_diagnostics()
    if diag["aga_applied"]:
        print(f"AGA injected at step {diag['step']}")
summary = session.get_session_summary()
```

---

MIT License · Copyright (c) 2024-2026 AGA Team
