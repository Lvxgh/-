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


def cmd_index(args):
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"错误：{folder} 不是文件夹")

    files = sorted(list(folder.rglob("*.md")) + list(folder.rglob("*.txt")))
    if not files:
        sys.exit(f"错误：{folder} 下没有找到 .md / .txt 文件")

    chunks: list[dict] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i, c in enumerate(chunk_text(text)):
            chunks.append({"source": str(f), "chunk_id": i, "text": c})
    print(f"共 {len(files)} 个文件，切分出 {len(chunks)} 个文本块")

    embedder = load_embedder()
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,  # 归一化后，余弦相似度 = 点积
        show_progress_bar=True,
    )

    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    np.save(INDEX_DIR / "embeddings.npy", embeddings.astype(np.float32))
    print(f"索引已保存到 {INDEX_DIR.resolve()}")


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

    p_index = sub.add_parser("index", help="对笔记文件夹建立向量索引")
    p_index.add_argument("folder", help="包含 .md / .txt 笔记的文件夹")
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
