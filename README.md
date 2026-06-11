# local-memory · 阶段 0：本地笔记 RAG 问答工具

「本地优先的个人 AI 记忆系统」长期项目的第一步——一个约 200 行的命令行工具，
对本地 Markdown 笔记做检索和问答。目标是动手理解 RAG 的三个核心环节：
**切块（chunking）→ 向量化（embedding）→ 检索（retrieval）**。

## 安装

```powershell
# 用项目自带的虚拟环境（注意：这台机器上裸 `python` 指向 MSYS2 的 Python，没有 pip）
D:\python1\python.exe -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> 首次运行 `index` 时会自动从 HuggingFace 下载 embedding 模型 `BAAI/bge-small-zh-v1.5`（约 100MB）。

## 使用

```powershell
# 1. 对笔记文件夹建索引（生成 .rag_index/ 目录）
python rag_cli.py index sample_notes

# 2. 纯检索：看看哪些片段和问题最相关（不需要 API key）
python rag_cli.py search "怎么创建虚拟环境"

# 3. 检索 + 让 Claude 基于笔记回答（需要先设置 API key）
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python rag_cli.py ask "这个项目分几个阶段？"
```

把 `sample_notes` 换成你自己的笔记文件夹（支持 `.md` 和 `.txt`，递归扫描）。

## 代码结构（读懂这 5 个函数就理解了 RAG）

| 函数 | 作用 |
|---|---|
| `chunk_text` | 把长文档按段落切成 ~500 字的块 |
| `cmd_index` | 所有块向量化后存盘（numpy 数组 + JSON） |
| `retrieve` | 问题向量化，和所有块算余弦相似度，取 top-k |
| `cmd_search` | 打印检索结果，用于直观感受检索质量 |
| `cmd_ask` | 把检索到的片段塞进 prompt，流式调用 Claude 回答 |

## 刻意保持简单的地方（也是阶段 1 的升级方向）

- **暴力检索**：直接 numpy 点积，没有向量数据库 → 阶段 1 换 sqlite-vec / LanceDB
- **只有向量检索**：没有 BM25 关键词检索和 rerank → 阶段 1 做混合检索
- **全量重建索引**：每次 index 都重算所有 embedding → 阶段 1 做增量索引
- **固定切块**：不感知 Markdown 标题结构 → 阶段 1 按标题层级切块
