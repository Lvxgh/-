# -*- coding: utf-8 -*-
"""
rag_cli.py —— 阶段 0：最朴素的本地笔记 RAG 问答工具

三个命令：
    python rag_cli.py index  <笔记文件夹>     # 建立索引（切块 + 向量化，存到 .rag_index/）
    python rag_cli.py search <问题>           # 只检索：打印最相关的笔记片段
    python rag_cli.py ask    <问题>           # 检索 + 调用 Claude 生成回答（需要 ANTHROPIC_API_KEY）

设计刻意保持朴素：embedding 直接存 numpy 数组，检索用暴力余弦相似度。
个人笔记的量级（几千个片段）下这完全够用，也最利于理解 RAG 的每一步。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

INDEX_DIR = Path(".rag_index")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文优先的小型 embedding 模型（~100MB，本地运行）
# bge 系列模型要求：查询（短问题）加这个前缀，文档不加，检索效果才好
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
DEFAULT_MODEL = "claude-opus-4-8"


def load_embedder():
    # 延迟导入：index/search 之外的命令（如 --help）不用等模型加载
    from sentence_transformers import SentenceTransformer

    print(f"加载 embedding 模型 {EMBED_MODEL}（首次运行会自动下载）...", file=sys.stderr)
    return SentenceTransformer(EMBED_MODEL)


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


def cmd_index(args):
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"错误：{folder} 不是文件夹")

    files = sorted(list(folder.rglob("*.md")) + list(folder.rglob("*.txt")))
    if not files:
        sys.exit(f"错误：{folder} 下没有找到 .md / .txt 文件")

    # 增量索引：读旧索引作为复用基础；--rebuild 强制从零重建
    old_chunks: list[dict] = []
    old_embeddings = None
    old_hashes: dict[str, str] = {}
    files_json = INDEX_DIR / "files.json"
    if not args.rebuild and files_json.exists():
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
            text = f.read_text(encoding="utf-8", errors="ignore")
            for ci, c in enumerate(chunk_text(text)):
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


def retrieve(query: str, top_k: int) -> list[dict]:
    chunks, embeddings = load_index()
    embedder = load_embedder()
    q_emb = embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = embeddings @ q_emb  # 归一化向量的点积 = 余弦相似度
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_idx]


def cmd_search(args):
    hits = retrieve(args.query, args.top_k)
    for rank, h in enumerate(hits, 1):
        print(f"\n[{rank}] 相似度 {h['score']:.3f}  来源: {h['source']} #{h['chunk_id']}")
        print("-" * 60)
        print(h["text"])


def cmd_ask(args):
    import anthropic

    hits = retrieve(args.question, args.top_k)
    context = "\n\n".join(
        f"<片段 来源=\"{h['source']}\">\n{h['text']}\n</片段>" for h in hits
    )

    system = (
        "你是用户的个人笔记问答助手。请只根据 <笔记内容> 中提供的片段回答问题，"
        "并指出答案来自哪个文件。如果笔记中没有相关信息，就直说没有找到，不要编造。"
    )
    user_msg = f"<笔记内容>\n{context}\n</笔记内容>\n\n问题：{args.question}"

    client = anthropic.Anthropic()  # 从环境变量 ANTHROPIC_API_KEY 读取密钥
    print(f"\n[检索到 {len(hits)} 个相关片段，正在生成回答...]\n", file=sys.stderr)
    with client.messages.stream(
        model=args.model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


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
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="检索并让 Claude 基于笔记回答")
    p_ask.add_argument("question", help="要问的问题")
    p_ask.add_argument("-k", "--top-k", type=int, default=5)
    p_ask.add_argument("--model", default=DEFAULT_MODEL)
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
