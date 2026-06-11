# -*- coding: utf-8 -*-
"""
rag_cli.py —— 阶段 0：最朴素的本地笔记 RAG 问答工具

命令一览：
    python rag_cli.py index  <笔记文件夹>     # 建索引（默认增量；支持 .md/.txt/.pdf）
    python rag_cli.py search <问题>           # 混合检索：向量 + BM25，RRF 融合（--rerank 重排）
    python rag_cli.py ask    <问题>           # 检索 + 生成回答
                                              #   --backend claude（默认，需 ANTHROPIC_API_KEY）
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


def cmd_ask(args):
    hits = retrieve(args.question, args.top_k, rerank=args.rerank)
    context = "\n\n".join(
        f"<片段 来源=\"{h['source']}\">\n{h['text']}\n</片段>" for h in hits
    )

    system = (
        "你是用户的个人笔记问答助手。请只根据 <笔记内容> 中提供的片段回答问题，"
        "并指出答案来自哪个文件。如果笔记中没有相关信息，就直说没有找到，不要编造。"
    )
    user_msg = f"<笔记内容>\n{context}\n</笔记内容>\n\n问题：{args.question}"

    # --model 未指定时按后端选默认值
    model = args.model or (DEFAULT_MODEL if args.backend == "claude" else DEFAULT_OLLAMA_MODEL)
    print(f"\n[检索到 {len(hits)} 个相关片段，{args.backend}/{model} 生成回答...]\n", file=sys.stderr)
    if args.backend == "claude":
        ask_claude(model, system, user_msg)
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


def cmd_memory_add(args):
    memories, embeddings = load_memories()
    # id 取现存最大编号 +1。注意：删掉最大号后再 add 会复用该编号——
    # 旧记忆已不存在所以无碍；将来若要 id 永不复用，需把计数器持久化
    next_n = max((int(m["id"][1:]) for m in memories), default=0) + 1
    now = datetime.now().isoformat(timespec="seconds")
    mem = {
        "id": f"m{next_n}",
        "content": args.content,
        "type": args.type,
        "importance": args.importance,
        "created_at": now,
        "updated_at": now,
        "source": "manual",  # 阶段 2 后期会有 "extracted"（从对话自动提取）
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else [],
    }
    embedder = load_embedder()
    # 记忆内容是"文档"一侧，按 bge 约定不加查询前缀
    emb = embedder.encode([args.content], normalize_embeddings=True).astype(np.float32)

    # 查重：和已有记忆几乎一样的内容不再入库，避免记忆库被重复污染
    if embeddings is not None and not args.force:
        sims = embeddings @ emb[0]
        nearest = int(np.argmax(sims))
        if sims[nearest] >= DUP_THRESHOLD:
            old = memories[nearest]
            sys.exit(
                f"未添加：与已有记忆 [{old['id']}] 相似度 {sims[nearest]:.3f}（阈值 {DUP_THRESHOLD}）\n"
                f"  已有：{old['content']}\n"
                f"  新增：{args.content}\n"
                f"确认不是重复就加 --force 强制添加；想更新旧记忆请先 memory forget {old['id']}"
            )

    memories.append(mem)
    embeddings = emb if embeddings is None else np.vstack([embeddings, emb])
    save_memories(memories, embeddings)
    print(f"已记住 [{mem['id']}]（{mem['type']}，重要性 {mem['importance']}）：{mem['content']}")


def cmd_memory_list(args):
    memories, _ = load_memories()
    if not memories:
        sys.exit('记忆库是空的。用 memory add "内容" --type semantic 添加一条')
    for m in memories:
        tags = f"  #{','.join(m['tags'])}" if m["tags"] else ""
        print(
            f"[{m['id']:>4}] {m['type']:<10} 重要性 {m['importance']}"
            f"  {m['created_at'][:10]}  {m['content']}{tags}"
        )
    print(f"共 {len(memories)} 条记忆")


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
    """记忆重排的占位接口：将来接 cross-encoder（如 bge-reranker-base，模型已在本地）。

    37 题评测里 5 个失败全是"查询与记忆零词面交集"型，正是 cross-encoder 的强项。
    当前版本不加载模型、原序返回，先把调用点固定在召回管线里。
    """
    # TODO: 复用 load_reranker() 对 (query, content) 逐对打分后重排
    return candidates


def recall_memories(query: str, top_k: int) -> list[dict]:
    """记忆召回核心：混合检索（向量 + BM25，RRF 融合）+ 重要性微调。
    recall 命令和 memory eval 共用，保证评测的就是线上真实在跑的那套逻辑。

    和笔记检索的 retrieve() 同一套思路，直接复用 BM25 类和 tokenize。
    改这里的任何打分逻辑，前后都要跑 memory eval 对比，指标不许回退。
    """
    memories, embeddings = load_memories()
    if not memories:
        sys.exit("记忆库是空的，没有可召回的内容")
    embedder = load_embedder()
    q = embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    sims = embeddings @ q
    bm = BM25([tokenize(m["content"]) for m in memories]).scores(tokenize(query))

    rrf = np.zeros(len(memories), dtype=np.float32)
    for r, i in enumerate(np.argsort(sims)[::-1]):
        rrf[i] += 1 / (MEMORY_RRF_K + r + 1)
    for r, i in enumerate(np.argsort(bm)[::-1]):
        if bm[i] <= 0:  # 关键词完全不匹配的不参与
            break
        rrf[i] += 1 / (MEMORY_RRF_K + r + 1)

    importance = np.array([m["importance"] for m in memories], dtype=np.float32)
    finals = rrf + IMPORTANCE_COEF * importance
    hits = [
        {
            **memories[int(i)],
            "final": float(finals[i]),
            "sim": float(sims[i]),
            "bm25": float(bm[i]),
        }
        for i in np.argsort(finals)[::-1][:top_k]
    ]
    return rerank_with_cross_encoder(query, hits)


def cmd_memory_recall(args):
    for rank, m in enumerate(recall_memories(args.query, args.top_k), 1):
        print(
            f"[{rank}] 总分 {m['final']:.4f}（向量 {m['sim']:.3f} | BM25 {m['bm25']:.2f}"
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

    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    for n, case in enumerate(cases, 1):
        hits = recall_memories(case["query"], 5)
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
    print(
        f"hit@1 {hit1 / n:.1%} | hit@3 {hit3 / n:.1%}"
        f" | hit@5 {hit5 / n:.1%} | MRR {mrr / n:.3f}"
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
        "--backend", choices=["claude", "ollama"], default="claude",
        help="claude=云端 API（默认）；ollama=本地模型，完全离线",
    )
    p_ask.add_argument("--model", default=None, help="模型名，不填则按后端用默认值")
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
    p_mrecall.set_defaults(func=cmd_memory_recall)
    p_mforget = mem_sub.add_parser("forget", help="按 id 删除一条记忆")
    p_mforget.add_argument("memory_id", help="记忆 id，如 m3（memory list 可查）")
    p_mforget.set_defaults(func=cmd_memory_forget)
    p_meval = mem_sub.add_parser("eval", help="跑记忆召回评测集，输出 hit@1/3/5 / MRR")
    p_meval.add_argument("--dataset", default="eval/memory_eval.json",
                         help="评测集路径（默认 eval/memory_eval.json）")
    p_meval.set_defaults(func=cmd_memory_eval)

    p_eval = sub.add_parser("eval", help="跑问答测试集，输出 hit@k / MRR 检索指标")
    p_eval.add_argument("testset", help="JSONL 测试集，每行含 question / expect_source / 可选 expect_text")
    p_eval.add_argument("-k", "--top-k", type=int, default=5)
    p_eval.add_argument("--rerank", action="store_true", help="开启重排后再评测，便于对比效果")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
