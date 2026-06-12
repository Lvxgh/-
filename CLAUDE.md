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
- [x] 记忆提取：从对话/文档自动提取结构化记忆，分 episodic（事件）/ semantic（事实）/ preference（偏好）三类（详见下方 LLM 自动记忆提取条目）
- [x] 去重（最小版）：add 时与全库算余弦相似度，≥0.92（`DUP_THRESHOLD`）拒绝入库并提示已有记忆，`--force` 可跳过
- [x] 记忆生命周期最小闭环（2026-06-12）：`memory update <id>`（部分字段更新，content 变了重算该行向量）+ `memory merge <id1> <id2> --content`（手动合并：保留 id1、tags 取并集、importance 取高、删 id2 行）+ `judge_memory_action()` LLM 守门员（判断新记忆该 add/duplicate/update/conflict/ignore，只建议不改库）+ `extract --review` 安全模式（候选→召回 top5 相似→守门判断→存待审清单，`memory pending list/apply/reject` 人工处理）。5 场景实测全对：换说法的重复→duplicate、偏好反转→conflict、同向细化→update+合并稿、琐事→提取层就丢弃（judge 直测也给 ignore）、独立新事实→add。提示词两条要点：偏好变化宁报 conflict 不悄悄 update；话题相关但讲不同方面的独立事实选 add 不选 update（第一版没这条时把"项目用什么后端"并进了"项目是什么"）。eval 基线与 rerank 指标逐位不变（召回路径未动）
- [x] 记忆评测：`memory eval`（hit@1/3/5 + MRR），改打分策略前后必跑、不许回退；失败样本自动打印期望关键词和 top5
- [x] 记忆召回升级为混合检索：向量 + BM25 RRF 融合 + 0.0001×重要性微调（12 题旧集上 hit@1 66.7%→75%）
- [x] 评测集扩充到 37 题（偏好 12 / 事实 13 / 事件 12，含 2 题仅按类型判定）+ 29 条 demo 记忆；问法刻意同义/间接化防刷分。失败样本全是"间接问法与记忆原文无词面交集"型，是后续 rerank/查询改写的素材
- [x] 召回超参网格实验：`MEMORY_RRF_K` 60→20。混合检索基线：hit@1 59.5% / hit@3 81.1% / hit@5 89.2% / MRR 0.699
- [x] cross-encoder rerank 真正实现（`memory recall/eval --rerank`，复用 bge-reranker-base）：**hit@1 59.5%→75.7%，hit@3 91.9%，hit@5 94.6%，MRR 0.830**——5 个零词面交集失败题翻回 4 个
- [x] LLM 自动记忆提取：`memory extract <文件>|--text`（`--dry-run` 预览、`--max-n` 限量），source 标记 extracted，逐条过查重；解析器容忍围栏/废话/坏行。已用 DeepSeek 后端端到端验证：三类记忆分类正确、琐事正确丢弃
- [x] recall 接入 ask：ask 自动注入 top3 相关记忆（`<关于用户的记忆>` 块，`--no-memory` 关闭）；DeepSeek 实测回答风格符合注入的"先结论、短答"偏好
- [x] DeepSeek 后端：`--backend deepseek`（ask 和 memory extract 均可用），OpenAI 兼容接口用 urllib 直连零新依赖，默认 `deepseek-v4-pro`，密钥读环境变量 `DEEPSEEK_API_KEY`；`--backend` 不填时 `pick_backend()` 按密钥可用性自动选（claude→deepseek→ollama）
- [x] v4-pro 质量评估（2026-06-12 实测）：三个 LLM 场景当前阶段全部够用——提取格式纪律强（每行一个 JSON 全照办，宽容解析器基本没派上用场，琐事正确丢弃）；ask 防编造行为正确（没有就直说）、风格跟随注入的偏好记忆；改写对提示词敏感但终版稳定。未测边界：整本 PDF 级长文提取、将来冲突解决/LLM-as-judge 需要的裁判型判断力。待办：用现有 memory eval 对比 `deepseek-v4-flash` 做改写够不够用（更快更省）
- [x] LLM 查询改写（HyDE，`memory recall/eval --rewrite`）：`rewrite_query()` 让 LLM 把问题改写成"假想记忆"（猜答案的陈述句），召回时多融合两路（假想记忆的向量+BM25）。四配置对比（37 题）：基线 59.5%/0.699 → +rewrite **67.6%/0.783** → +rerank 75.7%/0.830 → 两者叠加 75.7%/0.829（无增益，修的是同一批间接问法题；rerank 用原问题打分，会把 rewrite 捞回的 #34 再次挤出）。结论：有 API 无 GPU/不想加载 1.1GB 重排模型时用 rewrite，否则 rerank 仍是最优单项。提示词调教记录：第一版"多带术语"→编造 MoSCoW/RICE；第二版"别编专有名词"→退化成复述问题；第三版"猜答案+示例"才对——HyDE 的灵魂是猜答案不是改问法
- [ ] 时间衰减：暂缓——当前 29 条记忆 created_at 全是同一天，衰减实验零信号；等记忆跨越足够时间再做
- [x] 全库记忆体检 `memory consolidate`（2026-06-12）：`find_similar_pairs()` 用现成 embeddings 矩阵乘两两算相似度（阈值默认 0.82、同 type 对优先、上限 `--max-pairs 20` 控制 LLM 调用数），`judge_memory_pair()` 逐对判断 keep/duplicate/merge/conflict（解析失败保守 keep——和 extract 守门员的保守 conflict 相反，因为体检对象是已入库旧记忆，错动比漏检代价大）。默认 dry-run；`--save-pending` 把 keep 之外的建议存待审清单（`kind: "consolidate"`、`target_ids` 两条），pending apply：merge→`merge_memories()`（从 cmd_memory_merge 抽出的共用底层）、duplicate→保留前者遗忘后者、conflict→拒绝执行强制人工。4 类场景全判对；提示词教训同 extract：要明说"同主题下不同事实选 keep 不要硬合并"（没这句时把"项目是什么"和"未来接 MCP"合成了一坨）。真实库 0.82 下 0 对（干净），0.7 下 LLM 对"方法+指标"类相关对仍偏爱 merge——所以默认阈值保守、改库必须人工确认
- [x] 生命周期判断评测 `memory lifecycle-eval`（2026-06-12）：30 题静态评测集 `eval/memory_lifecycle_eval.json`（action 任务 15 题：add/duplicate/update/conflict/ignore 各 3；pair 任务 15 题：keep 4/duplicate 3/merge 4/conflict 4），直接复用 `judge_memory_action`/`judge_memory_pair`，完全不碰 .memory_store。守门函数解析失败返回的保守动作在评测里一律记失败（碰巧对也不算），原始输出打印前 300 字。结果：**v4-pro 28/30=93.3%，v4-flash 28/30=93.3%**，错的是同两道题且都在 merge/keep 边界（"方法 vs 指标"被判 merge、"代码风格两侧面"被判 keep）——这条边界本身模糊，靠人工确认兜底，不再调提示词追分。结论：守门场景 flash 完全够用，可作降本默认
- [x] 守门员模型路由（2026-06-12）：常量集中在顶部——`DEFAULT_ASK_MODEL=v4-pro`（生成类：ask、extract 提取）、`DEFAULT_JUDGE_MODEL=v4-flash`（守门判断，`judge_model()` 选择）、`DEFAULT_FALLBACK_JUDGE_MODEL=v4-pro`（复核）。judge 输出加 confidence 1-5（`parse_confidence()`，缺省按 3）；`judge_with_fallback()` 路由：低置信度（<=3）/解析失败/该给 target 没给（uncertain 标记）时换 pro 复核一次，结果带 fallback_from/first_action 轨迹；只在 deepseek 后端且首判不是复核模型时生效。review/consolidate 默认开（--no-fallback 关），lifecycle-eval 默认关（--fallback-pro 开，评测单模型不混复核）。**实测重要发现**：lifecycle-eval 上两模型平均 conf 5.0、fallback 触发 0 次——模型在边界题上"自信地错"，置信度路由抓不到这类错，只能兜住解析失败/缺 target；所以人工 apply 这道闸不能撤。另外边界题在 temperature 0 下仍会随轮次翻面（flash 两轮错的题不同：19/23 → 21/26），单轮 ±2 题的波动别过度解读
- [ ] 记忆生命周期（其余）：merge 合并文案自动化（LLM 起草、人确认）、conflict 的交互式确认体验
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
python rag_cli.py ask "<问题>"               # 检索+生成，需要 $env:ANTHROPIC_API_KEY
python rag_cli.py ask "<问题>" --backend deepseek  # DeepSeek 云端，需要 $env:DEEPSEEK_API_KEY
python rag_cli.py ask "<问题>" --backend ollama  # 本地模型，完全离线（需安装 Ollama）
python rag_cli.py eval eval_questions.jsonl  # 跑问答测试集，输出 hit@k / MRR
python rag_cli.py note add "<标题>" "<内容>"  # 增/改/删笔记后自动增量更新索引
python rag_cli.py note append "<标题>" "<内容>"  # 另有 delete / list / open
python rag_cli.py memory add "<内容>" --type preference --importance 5  # 记住
python rag_cli.py memory recall "<问题>" -k 5  # 召回；另有 list / forget <id>；--rerank 重排 / --rewrite LLM 改写
python rag_cli.py memory update m3 --content "<新内容>"   # 部分更新；merge <id1> <id2> --content 合并两条
python rag_cli.py memory extract <文件> --review  # LLM 守门建议（不入库），memory pending list/apply/reject 处理
python rag_cli.py memory consolidate              # 全库体检（dry-run）；--threshold 0.82 / --save-pending 存待审
python rag_cli.py memory lifecycle-eval           # 守门判断准确率评测（静态集，不碰库）；--model 对比 pro/flash
```

没有测试和 lint 配置（阶段 0 刻意从简）。

## 环境注意事项

- 台式机上裸 `python` 指向 MSYS2 的 Python（无 pip）；建 venv 要用 `D:\python1\python.exe -m venv .venv`。
- `.venv/` 和 `.rag_index/` 是机器本地产物，不应进版本控制或跨机同步；换机器后重建即可。
- PowerShell 默认 GBK 编码，运行前先：`$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding = [Text.Encoding]::UTF8`。
- 用户有 DeepSeek API key（密钥本身绝不写进仓库任何文件）；无 ANTHROPIC_API_KEY、未装 Ollama，所以 LLM 相关测试用 `--backend deepseek`。可用模型：`deepseek-v4-pro`（默认）、`deepseek-v4-flash`。

## 架构

单文件 `rag_cli.py`（~190 行）实现完整 RAG 管线：

`read_document`（`.pdf` 用 pypdf 抽文本，其余按 UTF-8 读）→ `chunk_markdown`（`.md` 按标题层级切小节，块前缀「【标题路径】」；其余走 `chunk_text` 固定切块；超长小节滑窗细分）→ `cmd_index`（bge 向量化，存 `.rag_index/chunks.json` + `embeddings.npy` + `files.json` 文件指纹）→ `retrieve`（混合检索：向量余弦 + 手写 BM25 两路，RRF 融合，每路取前 50；`--rerank` 时对前 20 个候选用 cross-encoder `bge-reranker-base` 重排）→ `cmd_ask`（片段注入 prompt，`--backend claude` 流式调 Claude API（默认 `claude-opus-4-8`）、`--backend deepseek` 走 DeepSeek 云端（OpenAI 兼容 SSE 流，urllib 直连）或 `--backend ollama` 走本地 http://localhost:11434）。

混合检索的约定：BM25 的中文分词是单字+双字滑窗（`tokenize`），BM25 索引在查询时现建（个人笔记量级下足够快）；RRF 只融合排名不融合分数，BM25 零分的块不参与融合。

评测：`cmd_eval` 读 JSONL 测试集（字段 `question` / `expect_source` / 可选 `expect_text`，命中=来源路径含 expect_source 且块文本含 expect_text），输出 hit@1 / hit@k / MRR。embedding 和 rerank 模型缓存在模块级变量（`_EMBEDDER` / `_RERANKER`），eval 连跑多题只加载一次。

记忆系统（阶段 2，`cmd_memory_*`）：记忆存 `.memory_store/memories.json` + `embeddings.npy`，行号一一对应（同 RAG 索引约定，forget 时用 `np.delete` 同步删行）。每条记忆含 id（m1、m2…取最大号+1）/ content / type（preference/semantic/episodic）/ importance（1-5）/ created_at / updated_at / source（manual，将来有 extracted）/ tags。`.memory_store/` 是个人数据，已加入 .gitignore。

记忆召回（`recall_memories`，recall 命令与 memory eval 共用）：混合检索——向量余弦 + BM25（复用 RAG 的 BM25 类和 tokenize）RRF 融合（`MEMORY_RRF_K=20`，独立于 RAG 的 K=60），再加 `IMPORTANCE_COEF=0.0001 × importance` 微调。两个超参都经过 37 题评测网格实验（K∈{20,60,100}×系数∈{0,0.0001,0.0003}）选定。`--rerank` 时取约 3 倍候选池（`pool = max(top_k*3, 15)`）过 `rerank_with_cross_encoder()`（真实现，bge-reranker-base 逐对打分、按分数降序）再裁回 top_k；不开启则直接取 top_k。超参实验已穷尽：K∈{10,20,40,60,100}、系数∈{0,0.0001,0.0003,0.03} 共 9 组，hit@1 始终 59.5%（系数 0.03 会崩到 29.7%——重要性碾压相关性的量化反例）；rerank 一举把 hit@1 提到 75.7%，验证了"零词面交集失败只能靠 cross-encoder"的判断。`--rewrite`（HyDE）时 `recall_memories` 收 `hyde` 参数，原问题+假想记忆各贡献向量/BM25 两路、共 4 路 RRF；展示用的 sim/bm25 分数始终取原问题的；rerank 的 cross-encoder 也始终用原问题打分。

记忆提取（`cmd_memory_extract`）：`llm_complete()` 非流式调 LLM（claude/ollama），`EXTRACT_PROMPT` 要求每行一个 JSON（注意拼提示词用 `.replace` 不能 `.format`，JSON 示例的大括号会撞占位符），`parse_extracted_memories()` 宽容解析（跳坏行、type 不合法归 semantic、importance 夹到 1-5），逐条走 `add_memory(source="extracted")` 过查重。`add_memory()` 是 add/extract 共用底层，成功返回 (True, 新记忆)，重复返回 (False, (相似度, 已有记忆))。

记忆生命周期（守门员模式）：`judge_memory_action()` 把新记忆和召回的 top5 相似记忆给 LLM，返回 `{"action", "target_id", "merged_content", "reason"}`（`JUDGE_PROMPT` 五动作 add/duplicate/update/conflict/ignore；LLM 编的 target_id 一律作废；解析失败保守返回 conflict 拦下来要人看）。`extract --review` 不写记忆库，建议存 `.memory_store/pending_memories.json`（id 取 p1、p2…），`memory pending apply` 才真正执行：add 走 `add_memory()`（仍过查重）、update/conflict 走 `update_memory()`（与 memory update 命令共用底层，content 变了只重算该行 embedding，行号对齐和所有 id 都不变）、duplicate/ignore 直接丢弃。核心原则：**LLM 只建议，任何改库动作必须人工 apply**。`memory merge` 是手动版（--content 必填），合并文案自动化留给后续。

守门员模型路由：模型常量集中在文件顶部（`DEFAULT_ASK_MODEL`/`DEFAULT_JUDGE_MODEL`/`DEFAULT_FALLBACK_JUDGE_MODEL`），不要把模型名散落进函数。生成类走 `default_model()`（deepseek 默认 pro），守门判断走 `judge_model()`（deepseek 默认 flash），用户显式 `--model` 永远优先（extract --review 里它同时覆盖提取和守门两步）。`judge_with_fallback(judge_call, backend, model, fallback)` 是统一路由层：judge_call 是绑好参数的单参 lambda；触发复核 = uncertain 或 confidence<=3；复核仍只是建议，改库必须人工 pending apply。已知局限：模型会"自信地错"（错题 conf 也是 5），置信度路由只能兜解析失败/缺 target 这类硬伤。

守门判断评测（`cmd_memory_lifecycle_eval`）：读 `eval/memory_lifecycle_eval.json`（JSON 数组，`task` 字段区分 action/pair 两类样本，静态给出 new_memory+existing_candidates 或 memory_a/b 和 expected_action），逐题调对应 judge 函数比对动作，输出总准确率、分任务、分动作和失败样本（含 LLM 理由）。注意：judge 函数解析失败时返回保守动作（action→conflict、pair→keep），评测靠 reason 里的"无法解析"识别并一律记失败。改守门提示词或换模型前后必跑。

全库体检（`cmd_memory_consolidate`）：`find_similar_pairs()` 拿现成 embeddings 矩阵乘出全对相似度，≥阈值（默认 0.82）的对按"同 type 优先、组内相似度降序"排序、截到 `--max-pairs`；`judge_memory_pair()` 逐对给 LLM 判 keep/duplicate/merge/conflict（解析失败保守 keep 并打印原始输出）。默认 dry-run；`--save-pending` 存的待审条目带 `kind: "consolidate"` 和 `target_ids`（pending 的 list/apply/reject 都按 kind 分支——extract 条目没有 kind 字段）。apply 语义：merge→`merge_memories()`、duplicate→保留 target_ids[0] 删 [1]、conflict→sys.exit 拒绝（人工 update/forget 后 reject 清单）。两个守门提示词的共同教训：必须明说"一条记忆只说一件事，同主题不同事实选 keep/add"，否则 LLM 偏爱把相关事实合成一坨。

ask 记忆注入：`cmd_ask` 自动召回 top3 记忆拼进 `<关于用户的记忆>` 块（system 提示"是背景不是笔记内容"），`--no-memory` 关闭；记忆库为空时静默跳过。记忆内容是文档侧不加 bge 前缀，查询加。**改打分逻辑前后必须跑 `memory eval` 对比，指标不许回退**。

记忆评测（`cmd_memory_eval`）：读 `eval/memory_eval.json`（JSON 数组，字段 query / expected_contains / expected_type），命中规则优先按 expected_contains 任一关键词匹配内容，没给关键词才退回比对 type；失败样本打印期望关键词 + top5 便于诊断。当前 37 题、29 条 demo 记忆下基线：hit@1 59.5% / hit@3 78.4% / hit@5 86.5% / MRR 0.690（比旧 12 题集低是因为题更难、库更大，**不要为拉高指标把 query 改写得和记忆原文一样**）。

索引是**增量**的：`files.json` 存每个文件内容的 SHA-256，未变文件整体复用旧 embeddings 行（全复用时不加载模型）；`--rebuild` 强制全量。chunks 和 embeddings 按行号一一对应，任何改动必须保持这个对齐。

**改动切块逻辑或 embedding 模型后必须 `--rebuild`**——文件指纹不变，增量模式会继续复用按旧逻辑算出的块。

关键约定：

- embedding 模型是 `BAAI/bge-small-zh-v1.5`；**查询必须加 `QUERY_PREFIX` 前缀，文档不加**——这是 bge 模型的硬性要求，删掉会显著降低检索质量。
- 所有向量在 encode 时归一化（`normalize_embeddings=True`），因此点积即余弦相似度；改动 encode 逻辑时必须保持这一不变量。
- 刻意保留的简化（暴力检索、无 BM25/rerank、全量重建索引、不感知 Markdown 标题）是阶段 1 的计划升级项（见 README.md 末尾清单），**不要当作缺陷顺手"修复"**，升级时按路线图来。
