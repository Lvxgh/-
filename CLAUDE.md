# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

「本地优先的个人 AI 记忆系统」长期项目的**阶段 0** 产物：一个本地 Markdown 笔记 RAG 问答 CLI。四阶段路线图：阶段 0 RAG CLI → 阶段 1 本地 RAG 引擎（混合检索、增量索引）→ 阶段 2 记忆系统核心 + 评估体系 → 阶段 3 MCP 服务化与开源发布。

用户是学生/初学者，本项目同时是学习载体——代码改动应保持可读性和教学性（中文注释），不要过度抽象。

## 常用命令

```powershell
.venv\Scripts\Activate.ps1                  # 依赖：anthropic, sentence-transformers, numpy
python rag_cli.py index <笔记文件夹>         # 建索引，写入 .rag_index/
python rag_cli.py search "<问题>" -k 5       # 纯检索，无需 API key
python rag_cli.py ask "<问题>"               # 检索+生成，需要 $env:ANTHROPIC_API_KEY
```

没有测试和 lint 配置（阶段 0 刻意从简）。

## 环境注意事项

- 台式机上裸 `python` 指向 MSYS2 的 Python（无 pip）；建 venv 要用 `D:\python1\python.exe -m venv .venv`。
- `.venv/` 和 `.rag_index/` 是机器本地产物，不应进版本控制或跨机同步；换机器后重建即可。
- PowerShell 默认 GBK 编码，运行前先：`$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding = [Text.Encoding]::UTF8`。

## 架构

单文件 `rag_cli.py`（~190 行）实现完整 RAG 管线：

`chunk_text`（按段落切 ~500 字块）→ `cmd_index`（bge 向量化，存 `.rag_index/chunks.json` + `embeddings.npy`）→ `retrieve`（暴力余弦相似度 top-k）→ `cmd_ask`（片段注入 prompt，流式调用 Claude，默认 `claude-opus-4-8`）。

关键约定：

- embedding 模型是 `BAAI/bge-small-zh-v1.5`；**查询必须加 `QUERY_PREFIX` 前缀，文档不加**——这是 bge 模型的硬性要求，删掉会显著降低检索质量。
- 所有向量在 encode 时归一化（`normalize_embeddings=True`），因此点积即余弦相似度；改动 encode 逻辑时必须保持这一不变量。
- 刻意保留的简化（暴力检索、无 BM25/rerank、全量重建索引、不感知 Markdown 标题）是阶段 1 的计划升级项（见 README.md 末尾清单），**不要当作缺陷顺手"修复"**，升级时按路线图来。
