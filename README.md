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

# 2. 纯检索：混合检索（向量+BM25），不需要 API key
python rag_cli.py search "怎么创建虚拟环境"

# 3a. 检索 + 让 Claude 基于笔记回答（需要先设置 API key）
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python rag_cli.py ask "这个项目分几个阶段？"

# 3b. 或者走本地 Ollama，完全离线（需要先安装 https://ollama.com 并 pull 模型）
python rag_cli.py ask "这个项目分几个阶段？" --backend ollama

# 4. 检索质量不够时，加 --rerank 用 cross-encoder 重排（更准但更慢，首次下载约 1.1GB 模型）
python rag_cli.py search "怎么创建虚拟环境" --rerank

# 5. 跑问答测试集，量化检索命中率（hit@k / MRR）；加 --rerank 可对比重排前后的指标
python rag_cli.py eval eval_questions.jsonl

# 6. 个人记忆（阶段 2）：记住、召回、遗忘——和笔记不同，记忆是一条条带类型和重要性的独立事实
python rag_cli.py memory add "用户喜欢先结论、短答。" --type preference --importance 5
python rag_cli.py memory add "2026-06 完成阶段1评测。" --type episodic --importance 4
python rag_cli.py memory recall "用户喜欢怎样的回答风格？" -k 3   # 混合检索 + 重要性微调
python rag_cli.py memory list
python rag_cli.py memory forget m2
python rag_cli.py memory eval        # 跑记忆召回评测集（eval/memory_eval.json）
# 记忆存在 .memory_store/（个人数据，不进 Git）。评测前先按 eval/memory_eval.json
# 的预期内容添加 demo 记忆（见仓库提交历史或 CLAUDE.md）

# 7. 管理笔记：增 / 改 / 删之后都会自动增量更新索引，改完立刻能搜到
python rag_cli.py note add "读书笔记-原子习惯" "核心观点：习惯是复利……"
python rag_cli.py note append "读书笔记-原子习惯" "第二章：身份认同比目标更重要。"
python rag_cli.py note open "读书笔记-原子习惯"     # 用默认编辑器打开做大段修改（改完手动 index）
python rag_cli.py note list
python rag_cli.py note delete "读书笔记-原子习惯"
# 默认操作 sample_notes 文件夹，加 --folder <路径> 可指向自己的笔记库
```

把 `sample_notes` 换成你自己的笔记文件夹（支持 `.md` / `.txt` / `.pdf`，递归扫描）。

## 代码结构（读懂这 5 个函数就理解了 RAG）

| 函数 | 作用 |
|---|---|
| `chunk_markdown` / `chunk_text` | Markdown 按标题层级切小节（带标题路径）；纯文本按段落切 ~500 字块 |
| `cmd_index` | 增量索引：新增/修改的块向量化后存盘，未变文件复用旧向量 |
| `tokenize` / `BM25` | 中英文轻量分词 + 手写教科书版 BM25 |
| `retrieve` | 混合检索：向量与 BM25 两路排名，RRF 融合取 top-k；可选 cross-encoder 重排 |
| `cmd_ask` | 片段注入 prompt，流式调用 Claude（云端）或 Ollama（本地）回答 |
| `cmd_eval` | 跑 JSONL 问答测试集，输出 hit@1 / hit@k / MRR 检索指标 |
| `cmd_note_*` | 笔记增/改/删/列表/打开，改动后调用 `do_index` 自动增量更新索引 |
| `cmd_memory_*` | 个人记忆增/列/召回/遗忘/评测；召回 = 向量+BM25 混合 RRF + 重要性微调 |

## 升级进度（阶段 1 清单）

- [x] **增量索引**：按文件内容 SHA-256 判断变化，未变文件整体复用旧向量（`.rag_index/files.json` 记录指纹）
- [x] **结构感知切块**：`.md` 文件按标题层级切小节，块文本带「标题路径」上下文（如 `【Python 学习笔记 > 虚拟环境】`）。注意：改了切块逻辑后文件指纹不变，要手动 `--rebuild`
- [x] **混合检索**：向量（语义）+ 手写 BM25（关键词，中文用单字+双字分词）两路，RRF 倒数排名融合；`search` 会显示每路的分数
- [x] **PDF 接入**：`index` 支持 `.pdf`（pypdf 逐页抽取文本）
- [x] **本地模型**：`ask --backend ollama` 走本地 Ollama（默认 `qwen3:4b`），完全离线；`--backend claude`（默认）走云端
- [x] **rerank**：`--rerank` 用 cross-encoder（`bge-reranker-base`）对召回的前 20 个候选重排
- [x] **评测工具**：`eval` 命令跑 JSONL 测试集，输出 hit@1 / hit@k / MRR
- [x] **问答测试集**：50 条问答（`eval_questions.jsonl`，基于 11 篇模拟笔记），混合检索基线 hit@1 94% / MRR 0.970（阶段 1 收尾条件达成；换成真实笔记后照同样格式重建即可）
- [x] **笔记管理**：`note add / append / delete / list / open`，增删改后自动增量更新索引
- [x] **记忆系统最小闭环（阶段 2 第一步）**：`memory add / list / recall / forget`，三类记忆（preference/semantic/episodic）
- [x] **记忆评测 + 混合召回（阶段 2 第二步）**：`memory eval`（12 题，hit@1/3/5 + MRR）；召回升级为向量+BM25 RRF 混合后 hit@1 66.7%→75%、MRR 0.736→0.847
- [x] **记忆去重（阶段 2 第三步）**：`memory add` 自动查重，与已有记忆相似度 ≥0.92 时拒绝并提示，`--force` 可跳过
- [ ] **向量数据库**：目前是 numpy 暴力点积，规模大了换 sqlite-vec / LanceDB
