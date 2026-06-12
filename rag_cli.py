# -*- coding: utf-8 -*-
"""
rag_cli.py —— 阶段 0：最朴素的本地笔记 RAG 问答工具

命令一览：
    python rag_cli.py index  <笔记文件夹>     # 建索引（默认增量；支持 .md/.txt/.pdf）
    python rag_cli.py search <问题>           # 混合检索：向量 + BM25，RRF 融合（--rerank 重排）
    python rag_cli.py ask    <问题>           # 检索 + 生成回答
                                              #   --backend claude（默认，需 ANTHROPIC_API_KEY）
                                              #   --backend deepseek（云端，需 DEEPSEEK_API_KEY）
                                              #   --backend ollama（本地模型，完全离线）
    python rag_cli.py note   add/append/...   # 笔记增删改，改完自动更新索引
    python rag_cli.py memory add/recall/...   # 个人记忆：记住 / 召回 / 遗忘（阶段 2）
    python rag_cli.py eval   <测试集.jsonl>   # 检索质量评测：hit@k / MRR

存储刻意保持朴素：embedding 直接存 numpy 数组，向量检索是暴力点积。
个人笔记的量级（几千个片段）下这完全够用，也最利于理解 RAG 的每一步。
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

INDEX_DIR = Path(".rag_index")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文优先的小型 embedding 模型（~100MB，本地运行）
# bge 系列模型要求：查询（短问题）加这个前缀，文档不加，检索效果才好
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
RERANK_MODEL = "BAAI/bge-reranker-base"  # cross-encoder 重排模型（~1.1GB，仅 --rerank 时加载）
DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"  # DeepSeek 云端 API（OpenAI 兼容格式）
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def default_model(backend: str) -> str:
    """--model 未指定时，按后端选默认模型。"""
    return {"claude": DEFAULT_MODEL, "ollama": DEFAULT_OLLAMA_MODEL,
            "deepseek": DEFAULT_DEEPSEEK_MODEL}[backend]


def pick_backend(backend: str | None) -> str:
    """--backend 未指定时自动挑一个可用的后端：看哪家的密钥已配置，都没有就回退本地 Ollama。"""
    if backend:
        return backend
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    return "ollama"

# 模型加载很慢（数秒），缓存到模块级变量——eval 连续跑几十个问题时只加载一次
_EMBEDDER = None
_RERANKER = None


def load_embedder():
    # 延迟导入：index/search 之外的命令（如 --help）不用等模型加载
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        print(f"加载 embedding 模型 {EMBED_MODEL}（首次运行会自动下载）...", file=sys.stderr)
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def load_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        print(f"加载 rerank 模型 {RERANK_MODEL}（首次运行会自动下载）...", file=sys.stderr)
        _RERANKER = CrossEncoder(RERANK_MODEL)
    return _RERANKER


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """按标题把 Markdown 切成小节，返回 (标题路径, 正文) 列表。

    标题路径形如 "Python 学习笔记 > 虚拟环境"，用标题栈维护层级：
    遇到 N 级标题就弹掉栈里所有 >= N 级的标题再入栈。
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []  # (标题级别, 标题文字)
    cur_lines: list[str] = []

    def flush():
        body = "\n".join(cur_lines).strip()
        if body:
            path = " > ".join(title for _, title in stack)
            sections.append((path, body))

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            cur_lines = []
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            cur_lines.append(line)
    flush()
    return sections


def chunk_markdown(text: str) -> list[str]:
    """结构感知切块：每个小节独立成块，块文本带上标题路径作为上下文。

    标题路径既帮 embedding 理解"这段在讲什么主题下的内容"，
    也让检索结果和送给 Claude 的片段自带出处层级。超长小节再滑窗细分。
    """
    chunks: list[str] = []
    for path, body in split_markdown_sections(text):
        for piece in chunk_text(body):
            chunks.append(f"【{path}】\n{piece}" if path else piece)
    return chunks


def chunk_text(text: str, max_chars: int = 500, overlap: int = 100) -> list[str]:
    """按空行切成段落，再合并成不超过 max_chars 的块；超长段落滑窗切分。"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = f"{cur}\n\n{p}".strip()
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        while len(p) > max_chars:
            chunks.append(p[:max_chars])
            p = p[max_chars - overlap:]
        cur = p
    if cur:
        chunks.append(cur)
    return chunks


def file_hash(path: Path) -> str:
    """文件内容的 SHA-256 指纹——增量索引判断'文件是否变过'的依据。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_document(f: Path) -> str:
    """读取一个文档的纯文本。PDF 逐页抽取，其余按 UTF-8 文本读。"""
    if f.suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(f).pages)
    return f.read_text(encoding="utf-8", errors="ignore")


def cmd_index(args):
    do_index(Path(args.folder), rebuild=args.rebuild)


def do_index(folder: Path, rebuild: bool = False):
    """对文件夹建索引（默认增量）。note 命令改动笔记后也直接调用它。"""
    if not folder.is_dir():
        sys.exit(f"错误：{folder} 不是文件夹")

    files = sorted(
        list(folder.rglob("*.md")) + list(folder.rglob("*.txt")) + list(folder.rglob("*.pdf"))
    )
    if not files:
        sys.exit(f"错误：{folder} 下没有找到 .md / .txt / .pdf 文件")

    # 增量索引：读旧索引作为复用基础；--rebuild 强制从零重建
    old_chunks: list[dict] = []
    old_embeddings = None
    old_hashes: dict[str, str] = {}
    files_json = INDEX_DIR / "files.json"
    if not rebuild and files_json.exists():
        try:
            old_chunks, old_embeddings = load_index()
            old_hashes = json.loads(files_json.read_text(encoding="utf-8"))
        except Exception:
            print("旧索引损坏或不完整，改为全量重建", file=sys.stderr)
            old_chunks, old_embeddings, old_hashes = [], None, {}

    # 旧块按来源文件分组，未变化的文件可以整体复用
    old_by_source: dict[str, list[int]] = {}
    for i, c in enumerate(old_chunks):
        old_by_source.setdefault(c["source"], []).append(i)

    new_hashes: dict[str, str] = {}
    kept_chunks: list[dict] = []   # 复用的旧块
    kept_rows: list[int] = []      # 它们在旧 embeddings 里的行号
    fresh_chunks: list[dict] = []  # 需要重新向量化的新块
    n_unchanged = n_changed = 0

    for f in files:
        key = str(f)
        h = file_hash(f)
        new_hashes[key] = h
        if old_hashes.get(key) == h and key in old_by_source:
            n_unchanged += 1
            for i in old_by_source[key]:
                kept_chunks.append(old_chunks[i])
                kept_rows.append(i)
        else:
            n_changed += 1
            text = read_document(f)
            chunker = chunk_markdown if f.suffix == ".md" else chunk_text
            for ci, c in enumerate(chunker(text)):
                fresh_chunks.append({"source": key, "chunk_id": ci, "text": c})

    n_deleted = len(set(old_hashes) - set(new_hashes))
    print(f"文件：{n_unchanged} 个未变（复用向量），{n_changed} 个新增/修改（重新计算），{n_deleted} 个已删除")

    parts = []
    if kept_rows:
        parts.append(old_embeddings[kept_rows])
    if fresh_chunks:
        embedder = load_embedder()  # 全部复用时连模型都不用加载
        parts.append(
            embedder.encode(
                [c["text"] for c in fresh_chunks],
                normalize_embeddings=True,  # 归一化后，余弦相似度 = 点积
                show_progress_bar=True,
            ).astype(np.float32)
        )
    if not parts:
        sys.exit("没有可索引的内容")

    chunks = kept_chunks + fresh_chunks
    embeddings = np.vstack(parts)

    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    files_json.write_text(
        json.dumps(new_hashes, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"索引共 {len(chunks)} 块，已保存到 {INDEX_DIR.resolve()}")


def load_index():
    chunks_file = INDEX_DIR / "chunks.json"
    emb_file = INDEX_DIR / "embeddings.npy"
    if not chunks_file.exists() or not emb_file.exists():
        sys.exit("错误：还没有索引，先运行 python rag_cli.py index <笔记文件夹>")
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    embeddings = np.load(emb_file)
    return chunks, embeddings


TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """轻量中英文分词：英文按单词，中文按 单字 + 相邻双字（bigram）。

    不用 jieba 等分词库——双字滑窗对 BM25 来说已经够用，且零依赖。
    例："创建虚拟环境" -> [创, 建, 虚, 拟, 环, 境, 创建, 建虚, 虚拟, 拟环, 环境]
    """
    tokens: list[str] = []
    for w in TOKEN_RE.findall(text.lower()):
        if re.match(r"[一-鿿]", w):
            tokens.extend(w)
            tokens.extend(w[i : i + 2] for i in range(len(w) - 1))
        else:
            tokens.append(w)
    return tokens


class BM25:
    """教科书版 BM25（Okapi）。打分 = Σ idf(词) * 饱和化的词频。

    直觉：罕见词权重高（idf），词频带来的收益递减（k1 饱和），
    长文档做长度惩罚（b）。个人笔记量级下每次查询现建索引即可，无需持久化。
    """

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tf = [Counter(d) for d in docs_tokens]
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(docs_tokens) else 1.0
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.doc_tf), dtype=np.float32)
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf_counter in enumerate(self.doc_tf):
                tf = tf_counter.get(t, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


def retrieve(query: str, top_k: int, rerank: bool = False) -> list[dict]:
    """混合检索：向量（语义）+ BM25（关键词）两路，RRF 融合排名。

    RRF（Reciprocal Rank Fusion）：每路只贡献 1/(60+名次)，
    不直接加分数——这样就不用纠结余弦相似度和 BM25 分数量纲不同的问题。

    rerank=True 时再加一步 cross-encoder 重排：把查询和候选块**拼在一起**
    送进模型逐对打分。比"查询、文档各自独立编码再比距离"（bi-encoder）精
    准得多，但每对都要过一遍模型，慢——所以只对召回的前 20 个候选做。
    """
    chunks, embeddings = load_index()

    # 通道 1：向量语义检索
    embedder = load_embedder()
    q_emb = embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    vec_scores = embeddings @ q_emb  # 归一化向量的点积 = 余弦相似度
    vec_rank = np.argsort(vec_scores)[::-1]

    # 通道 2：BM25 关键词检索
    bm25 = BM25([tokenize(c["text"]) for c in chunks])
    bm_scores = bm25.scores(tokenize(query))
    bm_rank = np.argsort(bm_scores)[::-1]

    # RRF 融合（每路取前 50）
    K = 60
    rrf: dict[int, float] = defaultdict(float)
    for r, i in enumerate(vec_rank[:50]):
        rrf[int(i)] += 1 / (K + r + 1)
    for r, i in enumerate(bm_rank[:50]):
        if bm_scores[i] <= 0:  # 关键词完全不匹配的不参与
            break
        rrf[int(i)] += 1 / (K + r + 1)

    # rerank 时多召回一些候选（前 20），给重排留出"翻盘"空间
    pool = max(top_k * 4, 20) if rerank else top_k
    top = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:pool]
    hits = [
        {
            **chunks[i],
            "score": rrf[i],
            "vec": float(vec_scores[i]),
            "bm25": float(bm_scores[i]),
        }
        for i in top
    ]

    if rerank and hits:
        reranker = load_reranker()
        rr_scores = reranker.predict([(query, h["text"]) for h in hits])
        for h, s in zip(hits, rr_scores):
            h["rerank"] = float(s)
        hits.sort(key=lambda h: h["rerank"], reverse=True)
        hits = hits[:top_k]
    return hits


def cmd_search(args):
    hits = retrieve(args.query, args.top_k, rerank=args.rerank)
    for rank, h in enumerate(hits, 1):
        rr = f"rerank {h['rerank']:.3f} | " if "rerank" in h else ""
        print(
            f"\n[{rank}] {rr}RRF {h['score']:.4f}（向量 {h['vec']:.3f} | BM25 {h['bm25']:.2f}）"
            f"  来源: {h['source']} #{h['chunk_id']}"
        )
        print("-" * 60)
        print(h["text"])


def ask_claude(model: str, system: str, user_msg: str):
    import anthropic

    client = anthropic.Anthropic()  # 从环境变量 ANTHROPIC_API_KEY 读取密钥
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)


def ask_ollama(model: str, system: str, user_msg: str):
    """走本地 Ollama 的流式聊天接口（http://localhost:11434），完全离线。"""
    import urllib.error
    import urllib.request

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:  # Ollama 流式返回：每行一个 JSON
                data = json.loads(line)
                print(data.get("message", {}).get("content", ""), end="", flush=True)
                if data.get("done"):
                    break
    except urllib.error.URLError:
        sys.exit(
            "错误：连不上 Ollama（http://localhost:11434）。\n"
            "请先安装并启动 Ollama（https://ollama.com），"
            f"然后执行 ollama pull {model} 下载模型。"
        )


def deepseek_request(model: str, system: str, user_msg: str, stream: bool):
    """构造 DeepSeek 的 HTTP 请求。它是 OpenAI 兼容接口，所以和 Ollama 一样用 urllib 直连，
    不需要再装一个 SDK。密钥从环境变量 DEEPSEEK_API_KEY 读取（不要写进代码里）。"""
    import urllib.request

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit(
            "错误：未设置 DEEPSEEK_API_KEY 环境变量。\n"
            '  PowerShell 里执行：$env:DEEPSEEK_API_KEY = "sk-..."\n'
            "  密钥在 https://platform.deepseek.com 申请和管理。"
        )
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_msg}],
        "stream": stream,
    }
    if not stream:
        # 非流式都是内部结构化用途（记忆提取、查询改写），温度 0 求稳定可复现；
        # ask 的流式生成保持默认温度
        body["temperature"] = 0
    return urllib.request.Request(
        DEEPSEEK_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )


def ask_deepseek(model: str, system: str, user_msg: str):
    """走 DeepSeek 云端 API，流式输出。

    OpenAI 兼容的流式格式是 SSE：每条事件一行 "data: {...}"，以 "data: [DONE]" 结束，
    增量文本在 choices[0].delta.content 里（和 Ollama 每行一个完整 JSON 不同）。
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(deepseek_request(model, system, user_msg, stream=True)) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices")
                if not choices:  # 流末尾可能有只带用量统计、没有 choices 的块
                    continue
                print(choices[0].get("delta", {}).get("content") or "", end="", flush=True)
    except urllib.error.HTTPError as e:
        sys.exit(f"错误：DeepSeek API 返回 {e.code}：{e.read().decode('utf-8', 'replace')[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"错误：连不上 DeepSeek API（{e.reason}），请检查网络。")


def cmd_ask(args):
    hits = retrieve(args.question, args.top_k, rerank=args.rerank)
    context = "\n\n".join(
        f"<片段 来源=\"{h['source']}\">\n{h['text']}\n</片段>" for h in hits
    )

    # 注入个人记忆：回答不仅基于笔记，还贴合用户的偏好和背景（阶段 2 的记忆闭环）
    mems: list[dict] = []
    if not args.no_memory and (MEMORY_DIR / "memories.json").exists():
        memories, _ = load_memories()
        if memories:
            mems = recall_memories(args.question, 3)
    memory_block = (
        "\n\n<关于用户的记忆>\n"
        + "\n".join(f"- [{m['type']}] {m['content']}" for m in mems)
        + "\n</关于用户的记忆>"
    ) if mems else ""

    system = (
        "你是用户的个人笔记问答助手。请只根据 <笔记内容> 中提供的片段回答问题，"
        "并指出答案来自哪个文件。如果笔记中没有相关信息，就直说没有找到，不要编造。"
        + (
            "<关于用户的记忆> 是用户本人的背景和偏好，用来调整回答的角度和风格，"
            "不要把它当作笔记内容来引用。" if mems else ""
        )
    )
    user_msg = f"<笔记内容>\n{context}\n</笔记内容>{memory_block}\n\n问题：{args.question}"

    backend = pick_backend(args.backend)
    model = args.model or default_model(backend)
    mem_note = f"，注入 {len(mems)} 条记忆" if mems else ""
    print(f"\n[检索到 {len(hits)} 个相关片段{mem_note}，{backend}/{model} 生成回答...]\n", file=sys.stderr)
    if backend == "claude":
        ask_claude(model, system, user_msg)
    elif backend == "deepseek":
        ask_deepseek(model, system, user_msg)
    else:
        ask_ollama(model, system, user_msg)
    print()


# ---------- 笔记管理：增 / 改 / 删，每次操作后自动增量重建索引 ----------

def note_file(folder: Path, title: str) -> Path:
    """标题即文件名：<folder>/<标题>.md。标题里不能带 Windows 文件名禁用字符。"""
    if re.search(r'[\\/:*?"<>|]', title):
        sys.exit('错误：标题不能包含 \\ / : * ? " < > | 这些字符')
    return folder / f"{title}.md"


def cmd_note_add(args):
    folder = Path(args.folder)
    folder.mkdir(exist_ok=True)
    f = note_file(folder, args.title)
    if f.exists():
        sys.exit(f"错误：{f} 已存在。用 note append 追加内容，或先 note delete 删除")
    body = f"\n{args.content}\n" if args.content else "\n"
    f.write_text(f"# {args.title}\n{body}", encoding="utf-8")
    print(f"已创建 {f}")
    do_index(folder)


def cmd_note_append(args):
    f = note_file(Path(args.folder), args.title)
    if not f.exists():
        sys.exit(f"错误：找不到 {f}（用 note list 查看现有笔记）")
    with f.open("a", encoding="utf-8") as fp:
        fp.write(f"\n{args.content}\n")
    print(f"已追加到 {f}")
    do_index(Path(args.folder))


def cmd_note_delete(args):
    f = note_file(Path(args.folder), args.title)
    if not f.exists():
        sys.exit(f"错误：找不到 {f}（用 note list 查看现有笔记）")
    f.unlink()
    print(f"已删除 {f}")
    do_index(Path(args.folder))


def cmd_note_list(args):
    folder = Path(args.folder)
    files = sorted(folder.glob("*.md")) if folder.is_dir() else []
    if not files:
        sys.exit(f"{folder} 下还没有笔记（用 note add <标题> <内容> 创建）")
    for f in files:
        first = f.read_text(encoding="utf-8", errors="ignore").lstrip().splitlines()
        print(f"{f.stem:<20} {f.stat().st_size:>6} 字节  {first[0] if first else '(空)'}")


def cmd_note_open(args):
    """用系统默认编辑器打开笔记，自由修改大段内容。改完记得 index 一下。"""
    import os

    f = note_file(Path(args.folder), args.title)
    if not f.exists():
        sys.exit(f"错误：找不到 {f}（用 note list 查看现有笔记）")
    os.startfile(f)  # Windows：交给默认关联程序打开
    print(f"已打开 {f}。编辑保存后运行 python rag_cli.py index {args.folder} 更新索引")


# ---------- 记忆系统（阶段 2 最小闭环）：手动记忆的 增 / 列 / 召回 / 遗忘 ----------
#
# 和"笔记检索"的本质区别：笔记是文档的切块，记忆是一条条独立的结构化事实，
# 有类型（preference 偏好 / semantic 长期事实 / episodic 事件经历）和重要性。
# 存储沿用 RAG 索引的约定：memories.json 与 embeddings.npy 按行号一一对应。

MEMORY_DIR = Path(".memory_store")
MEMORY_TYPES = ["preference", "semantic", "episodic"]
# add 时的查重阈值：新记忆与已有记忆余弦相似度达到该值即视为重复。
# bge-small 上近似改写一般 0.9+，主题相关但内容不同一般 0.5-0.8
DUP_THRESHOLD = 0.92


def load_memories() -> tuple[list[dict], "np.ndarray | None"]:
    mem_file = MEMORY_DIR / "memories.json"
    memories = json.loads(mem_file.read_text(encoding="utf-8")) if mem_file.exists() else []
    emb_file = MEMORY_DIR / "embeddings.npy"
    embeddings = np.load(emb_file) if emb_file.exists() else None
    if memories and (embeddings is None or len(memories) != len(embeddings)):
        sys.exit("错误：.memory_store 里 memories.json 和 embeddings.npy 行数不一致，记忆库已损坏")
    return memories, embeddings


def save_memories(memories: list[dict], embeddings):
    MEMORY_DIR.mkdir(exist_ok=True)
    (MEMORY_DIR / "memories.json").write_text(
        json.dumps(memories, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if embeddings is not None and len(embeddings):
        np.save(MEMORY_DIR / "embeddings.npy", embeddings)
    else:
        (MEMORY_DIR / "embeddings.npy").unlink(missing_ok=True)


def add_memory(content: str, mtype: str = "semantic", importance: int = 3,
               tags: list[str] | None = None, source: str = "manual",
               force: bool = False):
    """添加一条记忆的底层实现（memory add 和 memory extract 共用）。

    成功返回 (True, 新记忆)；被查重拦截返回 (False, (相似度, 已有记忆))。
    每次调用整库读写——个人记忆量级（几百条）下简单优先，不做批量优化。
    """
    memories, embeddings = load_memories()
    # id 取现存最大编号 +1。注意：删掉最大号后再 add 会复用该编号——
    # 旧记忆已不存在所以无碍；将来若要 id 永不复用，需把计数器持久化
    next_n = max((int(m["id"][1:]) for m in memories), default=0) + 1
    now = datetime.now().isoformat(timespec="seconds")
    mem = {
        "id": f"m{next_n}",
        "content": content,
        "type": mtype,
        "importance": importance,
        "created_at": now,
        "updated_at": now,
        "source": source,  # manual=手动添加，extracted=LLM 自动提取
        "tags": tags or [],
    }
    embedder = load_embedder()
    # 记忆内容是"文档"一侧，按 bge 约定不加查询前缀
    emb = embedder.encode([content], normalize_embeddings=True).astype(np.float32)

    # 查重：和已有记忆几乎一样的内容不再入库，避免记忆库被重复污染
    if embeddings is not None and not force:
        sims = embeddings @ emb[0]
        nearest = int(np.argmax(sims))
        if sims[nearest] >= DUP_THRESHOLD:
            return False, (float(sims[nearest]), memories[nearest])

    memories.append(mem)
    embeddings = emb if embeddings is None else np.vstack([embeddings, emb])
    save_memories(memories, embeddings)
    return True, mem


def cmd_memory_add(args):
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    ok, info = add_memory(args.content, args.type, args.importance, tags=tags, force=args.force)
    if not ok:
        sim, old = info
        sys.exit(
            f"未添加：与已有记忆 [{old['id']}] 相似度 {sim:.3f}（阈值 {DUP_THRESHOLD}）\n"
            f"  已有：{old['content']}\n"
            f"  新增：{args.content}\n"
            f"确认不是重复就加 --force 强制添加；想更新旧记忆请先 memory forget {old['id']}"
        )
    print(f"已记住 [{info['id']}]（{info['type']}，重要性 {info['importance']}）：{info['content']}")


def cmd_memory_list(args):
    memories, _ = load_memories()
    if not memories:
        sys.exit('记忆库是空的。用 memory add "内容" --type semantic 添加一条')
    for m in memories:
        tags = f"  #{','.join(m['tags'])}" if m["tags"] else ""
        src = "  ←extracted" if m.get("source") == "extracted" else ""
        print(
            f"[{m['id']:>4}] {m['type']:<10} 重要性 {m['importance']}"
            f"  {m['created_at'][:10]}  {m['content']}{tags}{src}"
        )
    print(f"共 {len(memories)} 条记忆")


# ---------- LLM 自动记忆提取：从对话/文档抽取结构化记忆（source=extracted） ----------

EXTRACT_PROMPT = (
    "你是一个记忆提取器。从用户提供的文本中提取值得长期记住的信息，"
    "分三类：preference（用户偏好）、semantic（长期事实）、episodic（带时间的事件经历）。\n"
    "要求：\n"
    "- 每条记忆独立完整，单独拿出来也能看懂；以第三人称陈述（如「用户……」）\n"
    "- 只提取有长期价值的信息，寒暄和过程性细节不要\n"
    '- importance 取 1-5：5=核心身份或原则，3=一般信息，1=琐事\n'
    '- 每行输出一个 JSON 对象：{"content": "...", "type": "semantic", "importance": 3}\n'
    "- 除 JSON 行外不要输出任何其他文字；最多提取 {max_n} 条"
)


def llm_complete(backend: str, model: str, system: str, user_msg: str) -> str:
    """非流式调一次 LLM，拿完整回复（提取等内部用途；ask 的流式输出走 ask_claude/ask_ollama）。"""
    if backend == "claude":
        import anthropic

        client = anthropic.Anthropic()  # 从环境变量 ANTHROPIC_API_KEY 读取密钥
        resp = client.messages.create(
            model=model, max_tokens=4000, system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    import urllib.error
    import urllib.request

    if backend == "deepseek":
        try:
            with urllib.request.urlopen(deepseek_request(model, system, user_msg, stream=False)) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            sys.exit(f"错误：DeepSeek API 返回 {e.code}：{e.read().decode('utf-8', 'replace')[:500]}")
        except urllib.error.URLError as e:
            sys.exit(f"错误：连不上 DeepSeek API（{e.reason}），请检查网络。")

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_msg}],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["message"]["content"]
    except urllib.error.URLError:
        sys.exit(
            "错误：连不上 Ollama（http://localhost:11434）。\n"
            "请先安装并启动 Ollama（https://ollama.com），"
            f"然后执行 ollama pull {model} 下载模型。"
        )


REWRITE_PROMPT = (
    "你是检索查询改写器。用户的问题会拿去检索一个「个人记忆库」，"
    "里面是一条条第三人称陈述的事实（如「用户偏好……」「项目阶段 1 做……」）。\n"
    "请猜一条「能回答这个问题的记忆」长什么样（不知道真实答案没关系，给最常见的合理猜测）：\n"
    "- 陈述句、第三人称（「用户……」「项目……」）；写答案本身，不是把问题换个说法\n"
    "- 用朴素的日常说法，多铺几个近义表述；不要编造问题里没有的专有名词和数字\n"
    "- 只输出这一句话，不要解释、不要引号\n"
    "示例：问「写代码有什么风格要求？」→ 用户要求代码简洁清晰、注重可读性，不要过度封装抽象"
)


def rewrite_query(query: str, backend: str, model: str) -> str:
    """把间接问法改写成「假想记忆」再去检索（HyDE 思路）。

    动机：评测里的顽固失败题全是"问题与记忆原文零词面交集"型，向量和 BM25 都够不着。
    让 LLM 先猜一条"答案大概长什么样"的陈述句，拿它的向量和关键词去搜，
    词面和语义就都对上了。改写失败（输出为空）时退回原问题，不让检索挂掉。
    """
    text = llm_complete(backend, model, REWRITE_PROMPT, query).strip()
    return (text.splitlines()[0].strip() or query) if text else query


def parse_extracted_memories(text: str) -> list[dict]:
    """解析 LLM 输出的记忆列表：每行一个 JSON 对象。

    对模型的小毛病保持宽容：跳过 ```json 围栏和夹杂的说明文字、容忍行尾逗号；
    type 不合法归为 semantic，importance 越界夹到 1-5。解析失败的行直接丢弃。
    """
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip().strip("`").strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = str(obj.get("content", "")).strip()
        if not content:
            continue
        mtype = obj.get("type") if obj.get("type") in MEMORY_TYPES else "semantic"
        try:
            importance = max(1, min(5, int(obj.get("importance", 3))))
        except (TypeError, ValueError):
            importance = 3
        out.append({"content": content, "type": mtype, "importance": importance})
    return out


# ---------- 记忆生命周期：新记忆进库前先判断 新增/重复/更新/冲突/忽略 ----------
#
# 动机：extract 能自动提取记忆之后，如果全部直接写库，库会被慢慢污染——
# 同一件事换个说法就重复一条、偏好变了却新旧并存互相打架。
# 所以让 LLM 当"守门员"只给建议，真正改库的动作一律由人确认（pending apply）。

JUDGE_PROMPT = (
    "你是个人记忆库的守门员。给你一条「新记忆」，以及从库里按相似度召回的几条「已有记忆」，"
    "判断新记忆该怎么处理，从五个动作里选一个：\n"
    "- add：和已有记忆都不重叠，是独立的新信息，应该新增\n"
    "- duplicate：和某条已有记忆说的是同一件事（换了说法、中英文混用也算），不必再写入\n"
    "- update：是某条已有记忆的补充、细化或更准确的版本（方向一致，只是信息更多），建议更新那条旧记忆\n"
    "- conflict：和某条已有记忆相互矛盾（比如偏好反转、事实对不上），不能自动处理，要人工确认\n"
    "- ignore：琐事或临时状态（吃了什么、今天天气），没有长期记住的价值\n"
    "注意：偏好发生变化时宁可报 conflict 也不要悄悄 update——改写用户偏好必须经人确认。\n"
    "注意：只是话题相关、但讲的是不同方面的独立事实（如旧记忆讲项目是什么，新记忆讲项目用了什么工具），"
    "选 add 而不是 update——一条记忆只说一件事，检索才精准。\n"
    "只输出一个 JSON 对象，不要任何其他文字：\n"
    '{"action": "add/duplicate/update/conflict/ignore 五选一", '
    '"target_id": "涉及的那条已有记忆的 id（duplicate/update/conflict 必填，add/ignore 填 null）", '
    '"merged_content": "update/conflict 时给出建议的更新后内容（一句完整陈述），其余填 null", '
    '"reason": "一句话原因"}'
)


def judge_memory_action(new_memory: dict, candidates: list[dict],
                        backend: str, model: str) -> dict:
    """让 LLM 判断一条新记忆该怎么进库，返回 {"action", "target_id", "merged_content", "reason"}。

    本函数只产出建议、绝不改库。为什么：改错一条偏好的代价远大于多确认一次，
    所以 update/conflict 必须经 memory pending apply 由人执行。
    解析失败或动作不认识时，宁可报 conflict 拦下来要人看，也不默认放行入库。
    """
    cand_text = "\n".join(
        f"- id={m['id']} [{m['type']}，重要性 {m['importance']}] {m['content']}"
        for m in candidates
    ) or "（记忆库是空的，没有相似记忆）"
    user_msg = (
        f"新记忆：[{new_memory['type']}，重要性 {new_memory['importance']}] {new_memory['content']}\n\n"
        f"已有记忆（按相似度召回）：\n{cand_text}"
    )
    reply = llm_complete(backend, model, JUDGE_PROMPT, user_msg)
    # 宽容解析：取回复里第一个 { 到最后一个 } 之间的部分（容忍 ```json 围栏等装饰）
    start, end = reply.find("{"), reply.rfind("}")
    try:
        obj = json.loads(reply[start:end + 1]) if start != -1 else {}
    except json.JSONDecodeError:
        obj = {}
    action = obj.get("action")
    if action not in ("add", "duplicate", "update", "conflict", "ignore"):
        print(f"[警告：LLM 输出无法解析。原始输出：{reply[:300]}]", file=sys.stderr)
        return {"action": "conflict", "target_id": None, "merged_content": None,
                "reason": f"LLM 输出无法解析（动作={action!r}），需要人工判断"}
    target = obj.get("target_id")
    if target not in {m["id"] for m in candidates}:
        target = None  # LLM 偶尔会编一个不存在的 id，编的一律作废
    reason = str(obj.get("reason") or "").strip() or "（LLM 没给原因）"
    if action in ("duplicate", "update", "conflict") and target is None:
        reason += "（注意：LLM 未给出有效 target_id）"
    merged = obj.get("merged_content")
    merged = str(merged).strip() if merged and str(merged).strip().lower() != "null" else None
    return {"action": action, "target_id": target, "merged_content": merged, "reason": reason}


# ---------- 全库记忆体检（consolidate）：扫描存量记忆里的重复/可合并/冲突 ----------
#
# extract --review 守的是"新记忆进库"这道门，但门建好之前入库的存量记忆没人管。
# consolidate 就是定期体检：先用现成的 embeddings 两两算相似度挑出可疑对（便宜），
# 再让 LLM 只对这些可疑对逐一判断（贵的步骤只花在刀刃上）。默认 dry-run 不动库。

JUDGE_PAIR_PROMPT = (
    "你是个人记忆库的体检员。给你同一个库里的两条记忆 A 和 B（它们的向量相似度较高），"
    "判断这一对该怎么处理，从四个动作里选一个：\n"
    "- keep：只是话题相关，各自讲的是独立的事实/偏好，两条都不该动\n"
    "- duplicate：说的是同一件事（换说法、中英文混用也算），保留一条即可\n"
    "- merge：讲的是同一件事的不同侧面，合并成一条更完整的记忆更好\n"
    "- conflict：互相矛盾（偏好相反、事实对不上），不能自动处理，要人工确认\n"
    "注意：一条记忆只说一件事，检索才精准——话题相关但各说各事的选 keep，不要硬合成一坨。\n"
    "注意：merge 只用于两条说的是**同一件事**、只是各漏了点细节的情况；"
    "同一个项目/主题下的不同事实（一条讲它是什么、另一条讲未来计划或用了什么工具）选 keep。\n"
    "注意：merged_content 只能用 A、B 原文里已有的信息，不要编造。\n"
    "只输出一个 JSON 对象，不要任何其他文字：\n"
    '{"action": "keep/duplicate/merge/conflict 四选一", '
    '"merged_content": "merge 时给出合并后的内容（一句完整陈述），其余填 null", '
    '"reason": "一句话原因"}'
)


def judge_memory_pair(mem_a: dict, mem_b: dict, backend: str, model: str) -> dict:
    """让 LLM 判断一对已入库的相似记忆该怎么处理，返回 {"action", "merged_content", "reason"}。

    解析失败的保守值和 extract 守门员相反：那边拿不准报 conflict 拦下来，
    这边拿不准报 keep 不动——体检对象是已经在库里的旧记忆，错动比漏检代价大。
    """
    user_msg = (
        f"记忆 A：id={mem_a['id']} [{mem_a['type']}，重要性 {mem_a['importance']}] {mem_a['content']}\n"
        f"记忆 B：id={mem_b['id']} [{mem_b['type']}，重要性 {mem_b['importance']}] {mem_b['content']}"
    )
    reply = llm_complete(backend, model, JUDGE_PAIR_PROMPT, user_msg)
    start, end = reply.find("{"), reply.rfind("}")
    try:
        obj = json.loads(reply[start:end + 1]) if start != -1 else {}
    except json.JSONDecodeError:
        obj = {}
    action = obj.get("action")
    if action not in ("keep", "duplicate", "merge", "conflict"):
        print(f"[警告：LLM 输出无法解析，按 keep 处理。原始输出：{reply[:300]}]", file=sys.stderr)
        return {"action": "keep", "merged_content": None, "reason": "LLM 输出无法解析，保守不动"}
    merged = obj.get("merged_content")
    merged = str(merged).strip() if merged and str(merged).strip().lower() != "null" else None
    reason = str(obj.get("reason") or "").strip() or "（LLM 没给原因）"
    return {"action": action, "merged_content": merged, "reason": reason}


def find_similar_pairs(memories: list[dict], embeddings,
                       threshold: float, max_pairs: int) -> list[tuple[float, int, int]]:
    """全库两两余弦相似度（向量已归一化，矩阵乘一把出），挑出 >= threshold 的对。

    同 type 的对排前面（同是偏好/事实才最可能重复或合并），组内按相似度降序；
    最多返回 max_pairs 对，控制 LLM 调用次数。返回 (相似度, 下标i, 下标j) 列表。
    """
    sims = embeddings @ embeddings.T
    pairs = [
        (float(sims[i, j]), i, j)
        for i in range(len(memories))
        for j in range(i + 1, len(memories))
        if sims[i, j] >= threshold
    ]
    pairs.sort(key=lambda p: (memories[p[1]]["type"] != memories[p[2]]["type"], -p[0]))
    return pairs[:max_pairs]


def cmd_memory_consolidate(args):
    """全库记忆体检：相似对扫描 → LLM 逐对判断 → 打印报告。

    默认 dry-run 只看不动；--save-pending 把可操作的建议（duplicate/merge/conflict）
    存入待审清单，库本身依然不动，由 memory pending apply 人工执行。
    """
    memories, embeddings = load_memories()
    if len(memories) < 2:
        sys.exit("记忆不足 2 条，没有可体检的")
    pairs = find_similar_pairs(memories, embeddings, args.threshold, args.max_pairs)
    print(f"记忆体检报告：共 {len(memories)} 条记忆 | 相似度阈值 {args.threshold} | 候选 {len(pairs)} 对"
          f"（上限 {args.max_pairs}）")
    if not pairs:
        print("没有相似度超过阈值的记忆对，库很干净")
        return

    backend = pick_backend(args.backend)
    model = args.model or default_model(backend)
    print(f"[{backend}/{model} 逐对判断中...]", file=sys.stderr)
    actionable = []  # keep 之外的建议，--save-pending 时存入待审清单
    for n, (sim, i, j) in enumerate(pairs, 1):
        a, b = memories[i], memories[j]
        verdict = judge_memory_pair(a, b, backend, model)
        print(f"\n[{n}] 相似度 {sim:.3f}")
        print(f"{a['id']}: [{a['type']}] {a['content']}")
        print(f"{b['id']}: [{b['type']}] {b['content']}")
        print(f"建议：{verdict['action']}")
        if verdict["merged_content"]:
            print(f"合并草稿：{verdict['merged_content']}")
        print(f"原因：{verdict['reason']}")
        if verdict["action"] != "keep":
            actionable.append((a, b, verdict))

    n_keep = len(pairs) - len(actionable)
    print(f"\n体检完成：keep {n_keep} 对，可操作建议 {len(actionable)} 条。", end="")
    if not args.save_pending:
        print("（dry-run，库没有任何改动"
              + ("；加 --save-pending 可存入待审清单）" if actionable else "）"))
        return
    if not actionable:
        print("（没有要存入待审清单的）")
        return
    pending = load_pending()
    next_p = max((int(p["id"][1:]) for p in pending), default=0) + 1
    for a, b, verdict in actionable:
        pending.append({
            "id": f"p{next_p}", "kind": "consolidate",
            "action": verdict["action"], "target_ids": [a["id"], b["id"]],
            "contents": [a["content"], b["content"]],  # 快照，pending list 展示用
            "merged_content": verdict["merged_content"], "reason": verdict["reason"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
        next_p += 1
    save_pending(pending)
    print(f"已存入待审清单（memory pending list 查看，apply 执行，reject 丢弃）。")


def cmd_memory_lifecycle_eval(args):
    """评测两个 LLM 守门判断的准确率：judge_memory_action（新记忆进库的五动作）
    和 judge_memory_pair（全库体检的四动作）。

    和 memory eval 同一套方法论：改守门提示词、换模型、降成本之前后各跑一遍，
    量化对比而不是凭手感。评测集全是静态数据，完全不读不写 .memory_store/。
    """
    path = Path(args.dataset)
    if not path.exists():
        sys.exit(f"错误：找不到评测集 {path}")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not cases:
        sys.exit(f"错误：{path} 是空的")

    backend = pick_backend(args.backend)
    model = args.model or default_model(backend)
    print(f"记忆生命周期判断评测：{path} | {backend}/{model} | 共 {len(cases)} 题")

    task_stats = defaultdict(lambda: [0, 0])    # task -> [对, 总]
    action_stats = defaultdict(lambda: [0, 0])  # 期望动作 -> [对, 总]
    failures = []
    for n, case in enumerate(cases, 1):
        if case["task"] == "action":
            verdict = judge_memory_action(case["new_memory"], case["existing_candidates"],
                                          backend, model)
        else:
            verdict = judge_memory_pair(case["memory_a"], case["memory_b"], backend, model)
        expected, got = case["expected_action"], verdict["action"]
        # 解析失败时守门函数会返回保守动作（conflict/keep），就算碰巧等于期望也算失败——
        # 那是兜底救的，不是模型判断对的
        parse_failed = "无法解析" in verdict["reason"]
        ok = got == expected and not parse_failed
        task_stats[case["task"]][0] += ok
        task_stats[case["task"]][1] += 1
        action_stats[expected][0] += ok
        action_stats[expected][1] += 1
        mark = "✓" if ok else "✗"
        print(f"[{n:>2}] {mark} {case['task']:<6} 期望 {expected:<9} 得到 {got:<9} {case['name']}")
        if not ok:
            failures.append((n, case, got, verdict["reason"]))

    total = len(cases)
    n_ok = sum(s[0] for s in task_stats.values())
    print("-" * 60)
    print(f"总准确率：{n_ok}/{total} = {n_ok / total:.1%}")
    print("分任务：" + "  ".join(
        f"{t} {s[0]}/{s[1]} = {s[0] / s[1]:.1%}" for t, s in sorted(task_stats.items())))
    print("分动作：" + "  ".join(
        f"{a} {s[0]}/{s[1]}" for a, s in sorted(action_stats.items())))
    if failures:
        print("\n失败样本：")
        for n, case, got, reason in failures:
            print(f"[{n}] {case['task']} / {case['name']}")
            print(f"    期望 {case['expected_action']}，得到 {got}；LLM 理由：{reason}")


def cmd_memory_extract(args):
    if args.text:
        text = args.text
    elif args.source_file:
        f = Path(args.source_file)
        if not f.exists():
            sys.exit(f"错误：找不到文件 {f}")
        text = read_document(f)
    else:
        sys.exit("错误：请给出要提取的文件路径，或用 --text 直接给一段文本")

    backend = pick_backend(args.backend)
    model = args.model or default_model(backend)
    print(f"[{backend}/{model} 提取记忆中...]", file=sys.stderr)
    # 注意不能用 .format()——提示词里的 JSON 示例带大括号，会被当成占位符
    reply = llm_complete(backend, model, EXTRACT_PROMPT.replace("{max_n}", str(args.max_n)), text)
    extracted = parse_extracted_memories(reply)[: args.max_n]
    if not extracted:
        sys.exit("没有提取到任何记忆（模型输出格式不对，或文本里没有可提取的内容）")

    if args.review:
        review_candidates(extracted, backend, model)
        return

    n_added = n_dup = 0
    for em in extracted:
        if args.dry_run:
            print(f"[预览]（{em['type']}，重要性 {em['importance']}）{em['content']}")
            continue
        ok, info = add_memory(em["content"], em["type"], em["importance"], source="extracted")
        if ok:
            n_added += 1
            print(f"已记住 [{info['id']}]（{em['type']}，重要性 {em['importance']}，extracted）：{em['content']}")
        else:
            n_dup += 1
            sim, old = info
            print(f"跳过重复（与 [{old['id']}] 相似度 {sim:.3f}）：{em['content']}")
    if not args.dry_run:
        print(f"提取完成：新增 {n_added} 条，跳过重复 {n_dup} 条")


# ---------- 待审清单（pending）：extract --review 的候选存在这里，由人 apply / reject ----------
#
# 文件和记忆库放在一起（.memory_store/pending_memories.json，同样不进 Git），
# 每条记录候选内容 + LLM 的建议动作，apply 才真正改库。

PENDING_FILE = MEMORY_DIR / "pending_memories.json"


def load_pending() -> list[dict]:
    return json.loads(PENDING_FILE.read_text(encoding="utf-8")) if PENDING_FILE.exists() else []


def save_pending(items: list[dict]):
    MEMORY_DIR.mkdir(exist_ok=True)
    PENDING_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def review_candidates(extracted: list[dict], backend: str, model: str):
    """--review 模式：每条候选先召回 top5 相似旧记忆，让 LLM 判断动作。

    只打印建议并存入待审清单，不写记忆库——这是 extract 的安全模式，
    防止自动提取的内容（重复、偏好反转）不经确认就污染长期记忆。
    """
    memories, _ = load_memories()
    pending = load_pending()
    next_p = max((int(p["id"][1:]) for p in pending), default=0) + 1
    for n, em in enumerate(extracted, 1):
        candidates = recall_memories(em["content"], 5) if memories else []
        verdict = judge_memory_action(em, candidates, backend, model)
        item = {"id": f"p{next_p}", "candidate": em, **verdict,
                "created_at": datetime.now().isoformat(timespec="seconds")}
        pending.append(item)
        next_p += 1
        print(f"\n[候选 {n}]（待审 {item['id']}）")
        print(f"content: {em['content']}")
        print(f"type: {em['type']}    importance: {em['importance']}")
        target = f"  target: {verdict['target_id']}" if verdict["target_id"] else ""
        print(f"LLM 建议：{verdict['action']}{target}")
        if verdict["merged_content"]:
            print(f"merged_content: {verdict['merged_content']}")
        print(f"reason: {verdict['reason']}")
    save_pending(pending)
    print(f"\n以上 {len(extracted)} 条候选都没有入库，已存入待审清单。"
          f"用 memory pending list 查看，apply <id> 执行建议，reject <id> 丢弃。")


def find_pending(pending_id: str) -> tuple[list[dict], int]:
    pending = load_pending()
    idx = next((i for i, p in enumerate(pending) if p["id"] == pending_id), None)
    if idx is None:
        sys.exit(f"错误：没有 id 为 {pending_id} 的待审项（用 memory pending list 查看）")
    return pending, idx


def cmd_memory_pending_list(args):
    pending = load_pending()
    if not pending:
        sys.exit("待审清单是空的（memory extract --review 和 memory consolidate --save-pending 会往这里存候选）")
    for p in pending:
        # 两种来源：extract --review 的单条候选（默认），consolidate 的记忆对（kind=consolidate）
        if p.get("kind") == "consolidate":
            ids = p["target_ids"]
            print(f"[{p['id']:>4}] 体检建议 {p['action']}  {' + '.join(ids)}")
            for mid, content in zip(ids, p.get("contents", [])):
                print(f"       {mid}：{content}")
        else:
            c = p["candidate"]
            target = f" → {p['target_id']}" if p.get("target_id") else ""
            print(f"[{p['id']:>4}] 建议 {p['action']}{target}  （{c['type']}，重要性 {c['importance']}）{c['content']}")
        print(f"       理由：{p['reason']}")
        if p.get("merged_content"):
            print(f"       合并稿：{p['merged_content']}")
    print(f"共 {len(pending)} 条待审")


def apply_consolidate_item(item: dict):
    """执行一条体检建议（两条都是已入库的旧记忆，和 extract 候选的处理完全不同）：
    duplicate → 保留第一条，遗忘第二条；merge → 用合并稿合并；
    conflict → 拒绝自动执行，矛盾的旧记忆必须人工 update/forget 后再 reject 本条。
    """
    a, b = item["target_ids"]
    action = item["action"]
    if action == "conflict":
        sys.exit(
            f"错误：conflict 不能自动执行——{a} 和 {b} 互相矛盾，得由你决定留哪边：\n"
            f"  改：memory update <id> --content \"...\"   删：memory forget <id>\n"
            f"  处理完后 memory pending reject {item['id']} 清掉本条"
        )
    if action == "merge":
        if not item.get("merged_content"):
            sys.exit("错误：该建议没有合并稿，无法执行。可手动 memory merge 后 reject 本条")
        keep, drop = merge_memories(a, b, item["merged_content"])
        print(f"已合并 [{drop['id']}] → [{keep['id']}]：{keep['content']}")
        return
    # duplicate：保留第一条，遗忘第二条（逻辑同 memory forget）
    memories, embeddings = load_memories()
    idx = next((i for i, m in enumerate(memories) if m["id"] == b), None)
    if idx is None:
        sys.exit(f"错误：记忆 {b} 已不存在（可能已被处理过），直接 reject 本条即可")
    gone = memories.pop(idx)
    embeddings = np.delete(embeddings, idx, axis=0)
    save_memories(memories, embeddings)
    print(f"已保留 [{a}]，遗忘重复的 [{gone['id']}]：{gone['content']}")


def cmd_memory_pending_apply(args):
    """执行 LLM 的建议动作（人工确认这一步就是现在）：
    add → 入库（仍过 add_memory 的查重）；update/conflict → 把目标记忆更新成合并稿
    （conflict 走到这里说明用户确认了偏好/事实确实变了）；duplicate/ignore → 按建议丢弃。
    consolidate 来源的条目走 apply_consolidate_item（动作含义不同）。
    """
    pending, idx = find_pending(args.pending_id)
    item = pending[idx]
    if item.get("kind") == "consolidate":
        apply_consolidate_item(item)  # conflict 等会 sys.exit，条目留在清单里
        pending.pop(idx)
        save_pending(pending)
        return
    cand, action = item["candidate"], item["action"]
    if action == "add":
        ok, info = add_memory(cand["content"], cand["type"], cand["importance"], source="extracted")
        if ok:
            print(f"已新增 [{info['id']}]（{info['type']}，重要性 {info['importance']}，extracted）：{info['content']}")
        else:
            sim, old = info
            print(f"查重拦截：与 [{old['id']}] 相似度 {sim:.3f}，未入库（库里已有等价记忆）")
    elif action in ("update", "conflict"):
        if not item.get("target_id"):
            sys.exit("错误：该建议没有有效的 target_id，无法执行。可 reject 后手动 memory add / update")
        new_content = item.get("merged_content") or cand["content"]
        m = update_memory(item["target_id"], content=new_content)
        print(f"已把 [{m['id']}] 更新为：{m['content']}")
    else:  # duplicate / ignore：建议本身就是"不入库"
        print(f"建议是 {action}（不入库），已按建议丢弃该候选")
    pending.pop(idx)
    save_pending(pending)


def cmd_memory_pending_reject(args):
    pending, idx = find_pending(args.pending_id)
    gone = pending.pop(idx)
    save_pending(pending)
    # consolidate 条目没有 candidate 字段，描述用涉及的记忆 id
    desc = (" + ".join(gone["target_ids"]) if gone.get("kind") == "consolidate"
            else gone["candidate"]["content"])
    print(f"已丢弃待审 [{gone['id']}]（建议 {gone['action']}）：{desc}")


# 重要性加分系数。RRF 里相邻名次的分差约 0.00026，取 0.0001 让重要性只在
# 两条记忆排名接近时起"加时赛"作用，不至于让高重要性碾压真正相关的记忆。
# v1 曾用 final = 相似度 + 0.03×importance，被 memory eval 量出两处反噬后改为混合检索。
IMPORTANCE_COEF = 0.0001
# 记忆召回的 RRF 平滑常数：越小头部名次权重越大（更尖），越大两路融合越平。
# 独立于 retrieve() 里 RAG 检索的 K=60，方便单独做实验。
# 37 题评测网格实验（K∈{20,60,100} × 系数∈{0,0.0001,0.0003}）：K=20 时
# hit@3 78.4%→81.1%、hit@5 86.5%→89.2%、MRR 0.690→0.699，全指标不回退，故选 20。
MEMORY_RRF_K = 20


def rerank_with_cross_encoder(query: str, candidates: list[dict]) -> list[dict]:
    """cross-encoder 重排：查询和每条候选记忆拼成一对，逐对过模型打分后重新排序。

    向量检索是 bi-encoder——查询、记忆各自独立编码，模型看不到两者的交互；
    cross-encoder 能逐词比对两边，对"查询与记忆零词面交集"的间接问法最有效
    （37 题评测里 5 个失败全是这种类型，调参实验已证明超参救不了它们）。
    复用 RAG 检索同一个 bge-reranker-base，调用方传入约 3 倍候选池、重排后裁回。
    """
    reranker = load_reranker()
    scores = reranker.predict([(query, m["content"]) for m in candidates])
    for m, s in zip(candidates, scores):
        m["rerank"] = float(s)
    return sorted(candidates, key=lambda m: m["rerank"], reverse=True)


def recall_memories(query: str, top_k: int, rerank: bool = False,
                    hyde: str | None = None) -> list[dict]:
    """记忆召回核心：混合检索（向量 + BM25，RRF 融合）+ 重要性微调。
    recall 命令和 memory eval 共用，保证评测的就是线上真实在跑的那套逻辑。
    rerank=True 时再过一遍 cross-encoder（更准但更慢，需加载约 1.1GB 模型）。
    hyde 是 LLM 把问题改写成的「假想记忆」（rewrite_query 生成）：
    给了就多融合两路（假想记忆的向量 + BM25），不给则和原来完全一样。

    和笔记检索的 retrieve() 同一套思路，直接复用 BM25 类和 tokenize。
    改这里的任何打分逻辑，前后都要跑 memory eval 对比，指标不许回退。
    """
    memories, embeddings = load_memories()
    if not memories:
        sys.exit("记忆库是空的，没有可召回的内容")
    embedder = load_embedder()
    queries = [query] + ([hyde] if hyde else [])
    qvecs = embedder.encode([QUERY_PREFIX + q for q in queries], normalize_embeddings=True)
    bm25 = BM25([tokenize(m["content"]) for m in memories])
    # sim/bm 始终取原问题的分数：展示给用户看的应该是"问题与记忆"的关系，不是假想记忆的
    sims = embeddings @ qvecs[0]
    bm = bm25.scores(tokenize(query))

    # 每个查询各贡献两路排名（向量 + BM25），RRF 把所有路融合在一起
    rrf = np.zeros(len(memories), dtype=np.float32)
    for qv in qvecs:
        s = embeddings @ qv
        for r, i in enumerate(np.argsort(s)[::-1]):
            rrf[i] += 1 / (MEMORY_RRF_K + r + 1)
    for q in queries:
        b = bm25.scores(tokenize(q))
        for r, i in enumerate(np.argsort(b)[::-1]):
            if b[i] <= 0:  # 关键词完全不匹配的不参与
                break
            rrf[i] += 1 / (MEMORY_RRF_K + r + 1)

    importance = np.array([m["importance"] for m in memories], dtype=np.float32)
    finals = rrf + IMPORTANCE_COEF * importance
    # rerank 时多取约 3 倍候选，给 cross-encoder 留"翻盘空间"，重排后再裁回 top_k
    pool = max(top_k * 3, 15) if rerank else top_k
    hits = [
        {
            **memories[int(i)],
            "final": float(finals[i]),
            "sim": float(sims[i]),
            "bm25": float(bm[i]),
        }
        for i in np.argsort(finals)[::-1][:pool]
    ]
    if rerank:
        hits = rerank_with_cross_encoder(query, hits)[:top_k]
    return hits


def cmd_memory_recall(args):
    hyde = None
    if args.rewrite:
        backend = pick_backend(args.backend)
        hyde = rewrite_query(args.query, backend, args.model or default_model(backend))
        print(f"[改写为假想记忆：{hyde}]", file=sys.stderr)
    for rank, m in enumerate(recall_memories(args.query, args.top_k, rerank=args.rerank, hyde=hyde), 1):
        rr = f"rerank {m['rerank']:.3f} | " if "rerank" in m else ""
        print(
            f"[{rank}] {rr}RRF {m['final']:.4f}（向量 {m['sim']:.3f} | BM25 {m['bm25']:.2f}"
            f" | 重要性 {m['importance']}）  {m['type']}  {m['id']}"
        )
        print(f"    {m['content']}")


def memory_hit(mem: dict, case: dict) -> bool:
    """一条召回结果是否命中评测样本。

    优先按内容关键词判断（expected_contains 中任一出现即命中）——内容比 id 稳定；
    样本没给关键词时才退回按类型判断（只看 type 容易过宽，所以是兜底而非首选）。
    """
    contains = case.get("expected_contains")
    if contains:
        return any(kw in mem["content"] for kw in contains)
    return mem["type"] == case.get("expected_type")


def cmd_memory_eval(args):
    """跑记忆召回评测集，输出 hit@1/3/5 和 MRR。

    和 RAG 的 cmd_eval 同一套方法论：改打分策略前后各跑一遍，指标不许倒退。
    """
    path = Path(args.dataset)
    if not path.exists():
        sys.exit(f"错误：找不到评测集 {path}")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not cases:
        sys.exit(f"错误：{path} 是空的")

    if args.rewrite:
        backend = pick_backend(args.backend)
        model = args.model or default_model(backend)
        print(f"[查询改写：{backend}/{model}，每题先改写再召回]", file=sys.stderr)

    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    for n, case in enumerate(cases, 1):
        hyde = rewrite_query(case["query"], backend, model) if args.rewrite else None
        hits = recall_memories(case["query"], 5, rerank=args.rerank, hyde=hyde)
        rank = next((r for r, m in enumerate(hits, 1) if memory_hit(m, case)), 0)
        if rank:
            mrr += 1 / rank
            hit1 += rank == 1
            hit3 += rank <= 3
            hit5 += 1
        mark = f"命中 rank={rank}" if rank else "未命中 ✗　"
        print(f"[{n:>2}] {mark}  {case['query']}")
        if not rank:  # 失败样本输出详情，方便诊断是"没召回"还是"排太靠后"
            print(f"     期望: {case.get('expected_contains') or '类型=' + case['expected_type']}")
            for r, m in enumerate(hits, 1):
                print(f"     top{r} [{m['type']:<10}] {m['content'][:36]}")

    n = len(cases)
    print("-" * 60)
    print(f"记忆评测：{path}，共 {n} 题")
    extras = [name for name, on in [("rewrite", args.rewrite), ("rerank", args.rerank)] if on]
    print(
        f"hit@1 {hit1 / n:.1%} | hit@3 {hit3 / n:.1%}"
        f" | hit@5 {hit5 / n:.1%} | MRR {mrr / n:.3f}"
        + (f"（已开启 {' + '.join(extras)}）" if extras else "")
    )


def cmd_memory_forget(args):
    memories, embeddings = load_memories()
    idx = next((i for i, m in enumerate(memories) if m["id"] == args.memory_id), None)
    if idx is None:
        sys.exit(f"错误：没有 id 为 {args.memory_id} 的记忆（用 memory list 查看）")
    gone = memories.pop(idx)
    embeddings = np.delete(embeddings, idx, axis=0)  # 同步删掉对应行，保持行号对齐
    save_memories(memories, embeddings)
    print(f"已遗忘 [{gone['id']}]：{gone['content']}")


def update_memory(memory_id: str, content: str | None = None, mtype: str | None = None,
                  importance: int | None = None, tags: list[str] | None = None) -> dict:
    """部分更新一条记忆（memory update 和 pending apply 共用）。

    只动这一条：content 变了就重算它那一行的 embedding，其余行原封不动，
    行号对齐不变，所有记忆的 id 也不变。
    """
    memories, embeddings = load_memories()
    idx = next((i for i, m in enumerate(memories) if m["id"] == memory_id), None)
    if idx is None:
        sys.exit(f"错误：没有 id 为 {memory_id} 的记忆（用 memory list 查看）")
    m = memories[idx]
    if content is not None and content != m["content"]:
        m["content"] = content
        embedder = load_embedder()
        # 和 add_memory 一致：记忆内容是"文档"一侧，不加查询前缀
        embeddings[idx] = embedder.encode([content], normalize_embeddings=True).astype(np.float32)[0]
    if mtype is not None:
        m["type"] = mtype
    if importance is not None:
        m["importance"] = importance
    if tags is not None:
        m["tags"] = tags
    m["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_memories(memories, embeddings)
    return m


def cmd_memory_update(args):
    tags = ([t.strip() for t in args.tags.split(",") if t.strip()]
            if args.tags is not None else None)
    if args.content is None and args.type is None and args.importance is None and tags is None:
        sys.exit("错误：至少给一个要更新的字段（--content / --type / --importance / --tags）")
    m = update_memory(args.memory_id, content=args.content, mtype=args.type,
                      importance=args.importance, tags=tags)
    tag_str = f"  #{','.join(m['tags'])}" if m["tags"] else ""
    print(f"已更新 [{m['id']}]（{m['type']}，重要性 {m['importance']}）：{m['content']}{tag_str}")


def merge_memories(id1: str, id2: str, content: str, mtype: str | None = None) -> tuple[dict, dict]:
    """合并两条记忆的底层实现（memory merge 和 pending apply 共用）：
    id1 保留并改成合并后的内容，id2 删除；tags 取并集、importance 取较高值。
    返回 (保留的记忆, 被删的记忆)。
    """
    memories, embeddings = load_memories()
    pos = {m["id"]: i for i, m in enumerate(memories)}
    if id1 == id2:
        sys.exit("错误：两个 id 相同，没有可合并的")
    for mid in (id1, id2):
        if mid not in pos:
            sys.exit(f"错误：没有 id 为 {mid} 的记忆（用 memory list 查看）")
    keep, drop = memories[pos[id1]], memories[pos[id2]]
    keep["content"] = content
    keep["type"] = mtype or keep["type"]
    keep["importance"] = max(keep["importance"], drop["importance"])
    keep["tags"] = keep["tags"] + [t for t in drop["tags"] if t not in keep["tags"]]
    keep["updated_at"] = datetime.now().isoformat(timespec="seconds")
    embedder = load_embedder()
    # 先改写保留行的向量，再删另一行——np.delete 之后行号会前移，顺序不能反
    embeddings[pos[id1]] = embedder.encode([content], normalize_embeddings=True).astype(np.float32)[0]
    drop_idx = pos[id2]
    memories.pop(drop_idx)
    embeddings = np.delete(embeddings, drop_idx, axis=0)
    save_memories(memories, embeddings)
    return keep, drop


def cmd_memory_merge(args):
    # 合并文案必须人工用 --content 给出——consolidate 的 LLM 合并草稿可以拿来当参考
    keep, drop = merge_memories(args.id1, args.id2, args.content, args.type)
    print(f"已合并 [{drop['id']}] → [{keep['id']}]（{keep['type']}，重要性 {keep['importance']}）：{keep['content']}")


def cmd_eval(args):
    """跑问答测试集，量化检索质量。

    测试集是 JSONL，每行一个用例：
        {"question": "怎么创建虚拟环境？", "expect_source": "python学习笔记", "expect_text": "venv"}
    expect_source：命中块的来源文件路径须包含这个子串；
    expect_text（可选）：命中块的文本还须包含这个子串（防止"碰巧召回同文件无关段落"算命中）。

    指标：
        hit@k —— 前 k 个结果里有命中的问题占比（k=1 和 k=top_k）
        MRR  —— 平均倒数排名：命中排第 1 得 1 分、第 2 得 1/2、没命中 0 分，再求平均。
                比 hit@k 更细腻：能反映"命中了，但排得靠不靠前"。
    """
    path = Path(args.testset)
    if not path.exists():
        sys.exit(f"错误：找不到测试集 {path}")
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        sys.exit(f"错误：{path} 是空的")

    hit1 = hitk = 0
    mrr = 0.0
    for n, case in enumerate(cases, 1):
        hits = retrieve(case["question"], args.top_k, rerank=args.rerank)
        rank = 0  # 0 表示没命中
        for r, h in enumerate(hits, 1):
            if case["expect_source"] in h["source"] and case.get("expect_text", "") in h["text"]:
                rank = r
                break
        if rank == 1:
            hit1 += 1
        if rank:
            hitk += 1
            mrr += 1 / rank
        mark = f"命中@{rank}" if rank else "未命中 ✗"
        print(f"[{n:>2}] {mark:　<6} {case['question']}")

    n = len(cases)
    print("-" * 60)
    print(
        f"共 {n} 题 | hit@1 {hit1 / n:.0%} | hit@{args.top_k} {hitk / n:.0%}"
        f" | MRR {mrr / n:.3f}" + ("（已开启 rerank）" if args.rerank else "")
    )


def main():
    parser = argparse.ArgumentParser(description="本地笔记 RAG 问答工具（阶段 0）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="对笔记文件夹建立向量索引（默认增量）")
    p_index.add_argument("folder", help="包含 .md / .txt 笔记的文件夹")
    p_index.add_argument("--rebuild", action="store_true", help="忽略旧索引，全量重建")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="检索最相关的笔记片段")
    p_search.add_argument("query", help="查询内容")
    p_search.add_argument("-k", "--top-k", type=int, default=5)
    p_search.add_argument("--rerank", action="store_true", help="用 cross-encoder 对候选重排（更准但更慢）")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="检索并基于笔记生成回答")
    p_ask.add_argument("question", help="要问的问题")
    p_ask.add_argument("-k", "--top-k", type=int, default=5)
    p_ask.add_argument("--rerank", action="store_true", help="用 cross-encoder 对候选重排（更准但更慢）")
    p_ask.add_argument(
        "--backend", choices=["claude", "deepseek", "ollama"], default=None,
        help="不填则自动检测：有 ANTHROPIC_API_KEY 用 claude，有 DEEPSEEK_API_KEY 用 deepseek，否则 ollama",
    )
    p_ask.add_argument("--model", default=None, help="模型名，不填则按后端用默认值")
    p_ask.add_argument("--no-memory", action="store_true", help="不注入个人记忆，只用笔记回答")
    p_ask.set_defaults(func=cmd_ask)

    # note 命令下再分一层子命令（add/append/delete/list/open），结构同 git remote add
    p_note = sub.add_parser("note", help="管理笔记：增/改/删后自动更新索引")
    note_sub = p_note.add_subparsers(dest="action", required=True)
    for name, func, help_text, with_content in [
        ("add", cmd_note_add, "新建一篇笔记", True),
        ("append", cmd_note_append, "往已有笔记追加一段内容", True),
        ("delete", cmd_note_delete, "删除一篇笔记", False),
        ("open", cmd_note_open, "用默认编辑器打开笔记做大段修改", False),
    ]:
        p = note_sub.add_parser(name, help=help_text)
        p.add_argument("title", help="笔记标题（即文件名，不含 .md）")
        if with_content:
            p.add_argument("content", nargs="?", default="", help="笔记内容")
        p.add_argument("--folder", default="sample_notes", help="笔记文件夹（默认 sample_notes）")
        p.set_defaults(func=func)
    p_list = note_sub.add_parser("list", help="列出所有笔记")
    p_list.add_argument("--folder", default="sample_notes")
    p_list.set_defaults(func=cmd_note_list)

    # memory 命令：阶段 2 记忆系统（结构同 note，二级子命令）
    p_mem = sub.add_parser("memory", help="个人记忆：add / list / recall / forget")
    mem_sub = p_mem.add_subparsers(dest="action", required=True)
    p_madd = mem_sub.add_parser("add", help="记住一条记忆")
    p_madd.add_argument("content", help="记忆内容（一条独立的事实/偏好/事件）")
    p_madd.add_argument(
        "--type", choices=MEMORY_TYPES, default="semantic",
        help="preference=偏好 / semantic=长期事实（默认）/ episodic=事件经历",
    )
    p_madd.add_argument("--importance", type=int, choices=range(1, 6), default=3,
                        help="重要性 1-5（默认 3），召回时加权")
    p_madd.add_argument("--tags", default="", help="逗号分隔的标签，如 学习,项目")
    p_madd.add_argument("--force", action="store_true", help="跳过查重，强制添加")
    p_madd.set_defaults(func=cmd_memory_add)
    p_mlist = mem_sub.add_parser("list", help="列出所有记忆")
    p_mlist.set_defaults(func=cmd_memory_list)
    p_mrecall = mem_sub.add_parser("recall", help="按问题召回最相关的记忆")
    p_mrecall.add_argument("query", help="要回忆什么")
    p_mrecall.add_argument("-k", "--top-k", type=int, default=5)
    p_mrecall.add_argument("--rerank", action="store_true", help="cross-encoder 重排（更准但更慢）")
    p_mrecall.add_argument("--rewrite", action="store_true",
                           help="先让 LLM 把问题改写成「假想记忆」再检索（HyDE，对间接问法最有效，需 LLM 后端）")
    p_mrecall.add_argument("--backend", choices=["claude", "deepseek", "ollama"], default=None,
                           help="--rewrite 用哪个 LLM，不填自动检测（同 ask）")
    p_mrecall.add_argument("--model", default=None, help="--rewrite 用的模型名，不填按后端用默认值")
    p_mrecall.set_defaults(func=cmd_memory_recall)
    p_mforget = mem_sub.add_parser("forget", help="按 id 删除一条记忆")
    p_mforget.add_argument("memory_id", help="记忆 id，如 m3（memory list 可查）")
    p_mforget.set_defaults(func=cmd_memory_forget)
    p_mupd = mem_sub.add_parser("update", help="按 id 更新一条记忆的部分字段（content 变了会重算向量）")
    p_mupd.add_argument("memory_id", help="记忆 id，如 m3（memory list 可查）")
    p_mupd.add_argument("--content", default=None, help="新的记忆内容")
    p_mupd.add_argument("--type", choices=MEMORY_TYPES, default=None, help="新类型，不填不改")
    p_mupd.add_argument("--importance", type=int, choices=range(1, 6), default=None, help="新重要性 1-5，不填不改")
    p_mupd.add_argument("--tags", default=None, help='逗号分隔的新标签（整体替换；--tags "" 可清空）')
    p_mupd.set_defaults(func=cmd_memory_update)
    p_mmerge = mem_sub.add_parser("merge", help="合并两条记忆：保留第一个 id，删除第二个")
    p_mmerge.add_argument("id1", help="保留的记忆 id")
    p_mmerge.add_argument("id2", help="被合并删除的记忆 id")
    p_mmerge.add_argument("--content", required=True, help="合并后的记忆内容（手动给出）")
    p_mmerge.add_argument("--type", choices=MEMORY_TYPES, default=None, help="不填保留第一条的类型")
    p_mmerge.set_defaults(func=cmd_memory_merge)
    p_mcons = mem_sub.add_parser("consolidate",
                                 help="全库记忆体检：扫相似对，LLM 逐对判断 keep/duplicate/merge/conflict（默认 dry-run 不动库）")
    p_mcons.add_argument("--threshold", type=float, default=0.82,
                         help="相似对的余弦阈值（默认 0.82；add 查重的 0.92 只抓近乎复读，体检要松一点）")
    p_mcons.add_argument("--max-pairs", type=int, default=20, dest="max_pairs",
                         help="最多体检几对（默认 20，控制 LLM 调用次数；同 type 的对优先）")
    p_mcons.add_argument("--save-pending", action="store_true", dest="save_pending",
                         help="把 keep 之外的建议存入待审清单（库本身仍不动，之后 memory pending apply 执行）")
    p_mcons.add_argument("--backend", choices=["claude", "deepseek", "ollama"], default=None,
                         help="不填则自动检测（同 ask）")
    p_mcons.add_argument("--model", default=None, help="模型名，不填按后端用默认值")
    p_mcons.set_defaults(func=cmd_memory_consolidate)
    p_mlce = mem_sub.add_parser("lifecycle-eval",
                                help="评测 LLM 守门判断准确率（extract 的五动作 + consolidate 的四动作），静态评测集不碰记忆库")
    p_mlce.add_argument("--dataset", default="eval/memory_lifecycle_eval.json",
                        help="评测集路径（默认 eval/memory_lifecycle_eval.json）")
    p_mlce.add_argument("--backend", choices=["claude", "deepseek", "ollama"], default=None,
                        help="不填则自动检测（同 ask）")
    p_mlce.add_argument("--model", default=None,
                        help="模型名，不填按后端用默认值（可用来对比 deepseek-v4-pro 和 deepseek-v4-flash）")
    p_mlce.set_defaults(func=cmd_memory_lifecycle_eval)
    p_mpend = mem_sub.add_parser("pending", help="待审清单：处理 extract --review / consolidate --save-pending 存下的候选")
    pend_sub = p_mpend.add_subparsers(dest="pending_action", required=True)
    pend_sub.add_parser("list", help="列出所有待审候选").set_defaults(func=cmd_memory_pending_list)
    p_papply = pend_sub.add_parser("apply", help="执行 LLM 建议（add 入库 / update·conflict 更新目标 / duplicate·ignore 丢弃）")
    p_papply.add_argument("pending_id", help="待审 id，如 p1")
    p_papply.set_defaults(func=cmd_memory_pending_apply)
    p_prej = pend_sub.add_parser("reject", help="丢弃一条待审候选，不改库")
    p_prej.add_argument("pending_id", help="待审 id，如 p1")
    p_prej.set_defaults(func=cmd_memory_pending_reject)
    p_meval = mem_sub.add_parser("eval", help="跑记忆召回评测集，输出 hit@1/3/5 / MRR")
    p_meval.add_argument("--dataset", default="eval/memory_eval.json",
                         help="评测集路径（默认 eval/memory_eval.json）")
    p_meval.add_argument("--rerank", action="store_true", help="开启重排后评测，便于对比")
    p_meval.add_argument("--rewrite", action="store_true",
                         help="开启 LLM 查询改写后评测（每题一次 LLM 调用，慢且花钱，但可与 --rerank 叠加）")
    p_meval.add_argument("--backend", choices=["claude", "deepseek", "ollama"], default=None,
                         help="--rewrite 用哪个 LLM，不填自动检测（同 ask）")
    p_meval.add_argument("--model", default=None, help="--rewrite 用的模型名，不填按后端用默认值")
    p_meval.set_defaults(func=cmd_memory_eval)
    p_mext = mem_sub.add_parser("extract", help="用 LLM 从文件/文本自动提取记忆（source=extracted）")
    p_mext.add_argument("source_file", nargs="?", default=None, help="要提取的文件（.md/.txt/.pdf）")
    p_mext.add_argument("--text", default=None, help="直接给一段文本（与文件二选一）")
    p_mext.add_argument("--backend", choices=["claude", "deepseek", "ollama"], default=None,
                        help="不填则自动检测（同 ask）")
    p_mext.add_argument("--model", default=None, help="模型名，不填按后端用默认值")
    p_mext.add_argument("--max-n", type=int, default=10, dest="max_n", help="最多提取几条（默认 10）")
    p_mext.add_argument("--dry-run", action="store_true", help="只预览提取结果，不入库")
    p_mext.add_argument("--review", action="store_true",
                        help="审核模式：对每条候选召回相似旧记忆，让 LLM 判断 add/duplicate/update/conflict/ignore，"
                             "只给建议并存入待审清单，不直接入库（用 memory pending apply/reject 处理）")
    p_mext.set_defaults(func=cmd_memory_extract)

    p_eval = sub.add_parser("eval", help="跑问答测试集，输出 hit@k / MRR 检索指标")
    p_eval.add_argument("testset", help="JSONL 测试集，每行含 question / expect_source / 可选 expect_text")
    p_eval.add_argument("-k", "--top-k", type=int, default=5)
    p_eval.add_argument("--rerank", action="store_true", help="开启重排后再评测，便于对比效果")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
