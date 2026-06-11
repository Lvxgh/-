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
#    默认增量：只重算新增/修改过的文件，没变的直接复用旧向量
#    加 --rebuild 可强制全量重建
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

## 升级进度（阶段 1 清单）

- [x] **增量索引**：按文件内容 SHA-256 判断变化，未变文件整体复用旧向量（`.rag_index/files.json` 记录指纹）
- [x] **结构感知切块**：`.md` 文件按标题层级切小节，块文本带「标题路径」上下文（如 `【Python 学习笔记 > 虚拟环境】`）。注意：改了切块逻辑后文件指纹不变，要手动 `--rebuild`
- [ ] **混合检索**：补上 BM25 关键词检索和 rerank，目前只有向量检索
- [ ] **向量数据库**：目前是 numpy 暴力点积，规模大了换 sqlite-vec / LanceDB
- [ ] **本地模型**：接入 Ollama，做到完全离线可用
