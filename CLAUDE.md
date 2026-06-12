# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目规划（中长期路线图）

**项目**：本地优先（local-first）的个人 AI 记忆系统——跨应用、跨会话的私人知识/记忆层，通过 MCP 协议可被任意 AI Agent 接入。2026-06 启动，预计 8-12 个月。

**差异化定位**："你的 AI 记忆属于你自己"——所有数据在本地，任何 Agent（Claude Code、Open WebUI、自建 Agent）通过 MCP 接入同一个记忆层。锚点是**本地优先 + 个人数据主权 + 中文场景**，避开 Mem0/Zep/Letta 等云优先竞品的主战场。

**目标**：兼顾求职竞争力（RAG 架构、向量检索、eval 能力）、商业化可能（开源核心 + 托管同步收费的 Obsidian 模式）、开源影响力。

**用户背景**：学生/初学者，每周可投入 25+ 小时。本项目同时是学习载体——代码改动应保持可读性和教学性（中文注释），不要过度抽象；重要改动后向用户解释设计思路。

### 阶段 0：打地基（2-3 周）✅ 已完成

- 200 行以内的本地笔记 RAG CLI（切块 → bge 向量化 → 余弦检索 → Claude 流式问答）
- 技术选型：Python；embedding 用 bge 系列（中英双语，本地跑）；向量先用 numpy 暴力检索

### 阶段 1：本地 RAG 引擎（1-2 个月）✅ 收尾条件已达成

目标：把玩具变成自己每天真实使用的工具。

- [x] 增量索引（文件变更检测，不全量重建）
- [x] 结构感知切块（Markdown 标题层级 + 标题路径上下文）
- [x] 混合检索：手写 BM25 + 向量，RRF 融合
- [x] PDF 接入（pypdf）
- [x] Ollama 本地模型支持（`ask --backend ollama`，完全离线）
- [x] rerank：`--rerank` 用 cross-encoder（bge-reranker-base）对前 20 个候选重排
- [x] 评测工具：`eval` 命令（hit@1 / hit@k / MRR）
- [x] 问答测试集：50 条（`eval_questions.jsonl`，基于 11 篇模拟笔记 sample_notes/）——阶段 1 收尾条件达成，基线 hit@1 94% / MRR 0.970
- [x] 笔记管理：`note add/append/delete/list/open`，增删改后自动增量重建索引
- [ ] 向量存储换 sqlite-vec / LanceDB（嵌入式、零运维）
- [ ] 网页剪藏接入
- [ ] 轻量 Web UI（CLI 优先）
- **里程碑**：日常用它查自己的笔记，比直接搜文件好用
- **检验标准**：用自己笔记构建 50-100 条问答测试集，检索命中率有量化数字

### 阶段 2：记忆系统核心（2-3 个月）⭐ 项目的灵魂 ◀ 当前阶段

目标：从"检索文档"升级为"拥有记忆"，与所有 RAG 工具拉开差距。

- [x] 最小闭环：`memory add/list/recall/forget`，三类记忆 + 重要性加权召回（`.memory_store/`，行号对齐同 RAG 索引）
- [ ] 记忆提取：从对话/文档自动提取结构化记忆，分 episodic（事件）/ semantic（事实）/ preference（偏好）三类
- [x] 去重（最小版）：add 时与全库算余弦相似度，≥0.92（`DUP_THRESHOLD`）拒绝入库并提示已有记忆，`--force` 可跳过
- [ ] 记忆生命周期（其余）：相似记忆合并、冲突解决（新旧信息矛盾时如何更新）、时间衰减与遗忘策略
- [x] 记忆评测：`memory eval`（hit@1/3/5 + MRR），改打分策略前后必跑、不许回退；失败样本自动打印期望关键词和 top5
- [x] 记忆召回升级为混合检索：向量 + BM25 RRF 融合 + 0.0001×重要性微调（12 题旧集上 hit@1 66.7%→75%）
- [x] 评测集扩充到 37 题（偏好 12 / 事实 13 / 事件 12，含 2 题仅按类型判定）+ 29 条 demo 记忆；问法刻意同义/间接化防刷分。失败样本全是"间接问法与记忆原文无词面交集"型，是后续 rerank/查询改写的素材
- [x] 召回超参网格实验：`MEMORY_RRF_K` 60→20。混合检索基线：hit@1 59.5% / hit@3 81.1% / hit@5 89.2% / MRR 0.699
- [x] cross-encoder rerank 真正实现（`memory recall/eval --rerank`，复用 bge-reranker-base）：**hit@1 59.5%→75.7%，hit@3 91.9%，hit@5 94.6%，MRR 0.830**——5 个零词面交集失败题翻回 4 个
- [x] LLM 自动记忆提取：`memory extract <文件>|--text`（`--dry-run` 预览、`--max-n` 限量），source 标记 extracted，逐条过查重；解析器容忍围栏/废话/坏行。**端到端待 API key 或 Ollama 就绪后验证**
- [x] DeepSeek 后端（`--backend deepseek`，ask 和 extract 通用）：OpenAI 兼容接口、urllib 裸调零新依赖，作为 Claude API 的低成本替代；协议解析已用本地 mock 服务器验证，真实 key 端到端验证待用户
- [x] recall 接入 ask：ask 自动注入 top3 相关记忆（`<关于用户的记忆>` 块，`--no-memory` 关闭）；生成质量观察同样待 LLM 后端
- [ ] 时间衰减：暂缓——当前 29 条记忆 created_at 全是同一天，衰减实验零信号；等记忆跨越足够时间再做
- [ ] 记忆生命周期（其余）：相似记忆合并、冲突解决
- [ ] **评估体系**（求职含金量最高）：跑 LoCoMo、LongMemEval 等公开 benchmark；自建回归测试，改记忆策略指标不许倒退；LLM-as-judge 评估记忆质量
- **里程碑**：至少一个公开 benchmark 上有可对比成绩，写技术博客分析结果

### 阶段 3：MCP 服务化与开源发布（2-3 个月）

- MCP server：暴露 `remember` / `recall` / `forget` 等工具，让任意支持 MCP 的客户端共享同一记忆层
- 开源工程化：英文为主的 README、快速上手文档、Docker 一键部署、CI 测试
- 发布运营：GitHub + Hacker News / r/LocalLLaMA / V2EX / 即刻；持续响应 issue
- **里程碑**：有自己以外的真实用户在用并提 issue

### 阶段 4：演进方向（按反馈选择，不预先承诺）

- 多设备同步（端到端加密）——本地优先软件的经典难题，技术含金量极高
- 浏览器/输入法/聊天软件的记忆采集插件
- 商业化探索：开源核心 + 托管同步服务收费

### 原则

- **不烂尾的唯一保证**：每个阶段结束时，产出物必须是用户自己每天真实在用的东西
- 每阶段结束写技术博客 + 尽早开源，用外部反馈校准方向
- 各阶段可切出独立中小项目（检索 CLI、中文记忆评测数据集、MCP 插件）单独发布

## 常用命令

```powershell
.venv\Scripts\Activate.ps1                  # 依赖：anthropic, sentence-transformers, numpy, pypdf
python rag_cli.py index <笔记文件夹>         # 增量建索引（.md/.txt/.pdf），写入 .rag_index/
python rag_cli.py search "<问题>" -k 5       # 混合检索，无需 API key；--rerank 开重排
python rag_cli.py ask "<问题>"               # 检索+生成，默认走 DeepSeek，需 $env:DEEPSEEK_API_KEY
python rag_cli.py ask "<问题>" --backend claude  # Claude API，需 $env:ANTHROPIC_API_KEY
python rag_cli.py ask "<问题>" --backend ollama  # 本地模型，完全离线（需安装 Ollama）
python rag_cli.py eval eval_questions.jsonl  # 跑问答测试集，输出 hit@k / MRR
python rag_cli.py note add "<标题>" "<内容>"  # 增/改/删笔记后自动增量更新索引
python rag_cli.py note append "<标题>" "<内容>"  # 另有 delete / list / open
python rag_cli.py memory add "<内容>" --type preference --importance 5  # 记住
python rag_cli.py memory recall "<问题>" -k 5  # 召回；另有 list / forget <id>
```

没有测试和 lint 配置（阶段 0 刻意从简）。

## 环境注意事项

- 台式机上裸 `python` 指向 MSYS2 的 Python（无 pip）；建 venv 要用 `D:\python1\python.exe -m venv .venv`。
- `.venv/` 和 `.rag_index/` 是机器本地产物，不应进版本控制或跨机同步；换机器后重建即可。
- PowerShell 默认 GBK 编码，运行前先：`$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding = [Text.Encoding]::UTF8`。
- API 密钥用 `setx DEEPSEEK_API_KEY "sk-..."` 永久写入 Windows 用户环境变量（新开的终端才生效）；**密钥只放环境变量，绝不写进任何会提交到 Git 的文件**。

## 架构

单文件 `rag_cli.py`（~190 行）实现完整 RAG 管线：

`read_document`（`.pdf` 用 pypdf 抽文本，其余按 UTF-8 读）→ `chunk_markdown`（`.md` 按标题层级切小节，块前缀「【标题路径】」；其余走 `chunk_text` 固定切块；超长小节滑窗细分）→ `cmd_index`（bge 向量化，存 `.rag_index/chunks.json` + `embeddings.npy` + `files.json` 文件指纹）→ `retrieve`（混合检索：向量余弦 + 手写 BM25 两路，RRF 融合，每路取前 50；`--rerank` 时对前 20 个候选用 cross-encoder `bge-reranker-base` 重排）→ `cmd_ask`（片段注入 prompt，三个后端：`--backend deepseek`（默认）走 DeepSeek 的 OpenAI 兼容接口（标准库 urllib 裸调 + SSE 流式解析，默认模型 `deepseek-v4-pro`，密钥读环境变量 `DEEPSEEK_API_KEY`，零新增依赖）、`--backend claude` 流式调 Claude API（默认 `claude-opus-4-8`）、`--backend ollama` 走本地 http://localhost:11434。后端→默认模型的映射在 `default_model_for()`）。

混合检索的约定：BM25 的中文分词是单字+双字滑窗（`tokenize`），BM25 索引在查询时现建（个人笔记量级下足够快）；RRF 只融合排名不融合分数，BM25 零分的块不参与融合。

评测：`cmd_eval` 读 JSONL 测试集（字段 `question` / `expect_source` / 可选 `expect_text`，命中=来源路径含 expect_source 且块文本含 expect_text），输出 hit@1 / hit@k / MRR。embedding 和 rerank 模型缓存在模块级变量（`_EMBEDDER` / `_RERANKER`），eval 连跑多题只加载一次。

记忆系统（阶段 2，`cmd_memory_*`）：记忆存 `.memory_store/memories.json` + `embeddings.npy`，行号一一对应（同 RAG 索引约定，forget 时用 `np.delete` 同步删行）。每条记忆含 id（m1、m2…取最大号+1）/ content / type（preference/semantic/episodic）/ importance（1-5）/ created_at / updated_at / source（manual，将来有 extracted）/ tags。`.memory_store/` 是个人数据，已加入 .gitignore。

记忆召回（`recall_memories`，recall 命令与 memory eval 共用）：混合检索——向量余弦 + BM25（复用 RAG 的 BM25 类和 tokenize）RRF 融合（`MEMORY_RRF_K=20`，独立于 RAG 的 K=60），再加 `IMPORTANCE_COEF=0.0001 × importance` 微调。两个超参都经过 37 题评测网格实验（K∈{20,60,100}×系数∈{0,0.0001,0.0003}）选定。`--rerank` 时取约 3 倍候选池（`pool = max(top_k*3, 15)`）过 `rerank_with_cross_encoder()`（真实现，bge-reranker-base 逐对打分、按分数降序）再裁回 top_k；不开启则直接取 top_k。超参实验已穷尽：K∈{10,20,40,60,100}、系数∈{0,0.0001,0.0003,0.03} 共 9 组，hit@1 始终 59.5%（系数 0.03 会崩到 29.7%——重要性碾压相关性的量化反例）；rerank 一举把 hit@1 提到 75.7%，验证了"零词面交集失败只能靠 cross-encoder"的判断。

记忆提取（`cmd_memory_extract`）：`llm_complete()` 非流式调 LLM（claude/deepseek/ollama），`EXTRACT_PROMPT` 要求每行一个 JSON（注意拼提示词用 `.replace` 不能 `.format`，JSON 示例的大括号会撞占位符），`parse_extracted_memories()` 宽容解析（跳坏行、type 不合法归 semantic、importance 夹到 1-5），逐条走 `add_memory(source="extracted")` 过查重。`add_memory()` 是 add/extract 共用底层，成功返回 (True, 新记忆)，重复返回 (False, (相似度, 已有记忆))。

ask 记忆注入：`cmd_ask` 自动召回 top3 记忆拼进 `<关于用户的记忆>` 块（system 提示"是背景不是笔记内容"），`--no-memory` 关闭；记忆库为空时静默跳过。记忆内容是文档侧不加 bge 前缀，查询加。**改打分逻辑前后必须跑 `memory eval` 对比，指标不许回退**。

记忆评测（`cmd_memory_eval`）：读 `eval/memory_eval.json`（JSON 数组，字段 query / expected_contains / expected_type），命中规则优先按 expected_contains 任一关键词匹配内容，没给关键词才退回比对 type；失败样本打印期望关键词 + top5 便于诊断。当前 37 题、29 条 demo 记忆下基线：hit@1 59.5% / hit@3 78.4% / hit@5 86.5% / MRR 0.690（比旧 12 题集低是因为题更难、库更大，**不要为拉高指标把 query 改写得和记忆原文一样**）。

索引是**增量**的：`files.json` 存每个文件内容的 SHA-256，未变文件整体复用旧 embeddings 行（全复用时不加载模型）；`--rebuild` 强制全量。chunks 和 embeddings 按行号一一对应，任何改动必须保持这个对齐。

**改动切块逻辑或 embedding 模型后必须 `--rebuild`**——文件指纹不变，增量模式会继续复用按旧逻辑算出的块。

关键约定：

- embedding 模型是 `BAAI/bge-small-zh-v1.5`；**查询必须加 `QUERY_PREFIX` 前缀，文档不加**——这是 bge 模型的硬性要求，删掉会显著降低检索质量。
- 所有向量在 encode 时归一化（`normalize_embeddings=True`），因此点积即余弦相似度；改动 encode 逻辑时必须保持这一不变量。
- 刻意保留的简化（暴力检索、无 BM25/rerank、全量重建索引、不感知 Markdown 标题）是阶段 1 的计划升级项（见 README.md 末尾清单），**不要当作缺陷顺手"修复"**，升级时按路线图来。
