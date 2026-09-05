#!/usr/bin/env python3
"""
rag.py - local per-user knowledge base for AURA, built on ChromaDB.

Each user's docs live in their own physically separate Chroma collection
(kb_<user_id>), so one user's files can never leak into another user's
search results (see the README's "Per-user knowledge base" section). Source
docs come from the knowledge/ folder next to this file -- plain text and
markdown files are chunked, embedded with Chroma's built-in ONNX MiniLM
embedding function (ships with chromadb -- no separate model download or
extra package beyond what's already in requirements.txt), and stored.

CLI:
    python rag.py build [user_id] [--force]   # (re)index knowledge/ for user_id
    python rag.py query [user_id] "..."       # search and print top matches

    user_id defaults to "local" (AURA is single-user; this matches
    jarvis.py's CLI_USER_ID and jarvis_server.py's LOCAL_USER).

Library:
    import rag
    rag.query_as_context(user_id, query, n_results=4) -> str
        Returns the top matching chunks as one formatted string, ready to
        drop into the model's context. Raises RuntimeError if that user
        has no collection yet (i.e. `build` hasn't been run) -- both
        jarvis.py and jarvis_server.py catch this and surface a friendly
        "knowledge search unavailable" message instead of crashing.

IMPORTANT: this file must stay a self-contained CLI script. A previous
version of this repo accidentally had rag.py's content overwritten with a
copy of run_dev.py -- since Render's build command is
`pip install -r requirements.txt && python rag.py build`, that made the
build step launch a Flask dev server and hang forever instead of indexing
anything and exiting. If you ever touch this file, make sure `python
rag.py build` still exits on its own.
"""
import argparse
import os
import sys

import chromadb

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(HERE, "knowledge")
CHROMA_DIR = os.path.join(HERE, "chroma_db")

# Simple fixed-size chunking with overlap. Good enough for notes/resume-style
# text files: long enough to keep a paragraph together, short enough that no
# single chunk dominates the whole context window.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _collection_name(user_id):
    # Chroma only allows alnum/underscore/hyphen names, 3-63 chars.
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(user_id))
    return f"kb_{safe}"[:63]


def _client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _chunk_text(text):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _iter_knowledge_files():
    if not os.path.isdir(KNOWLEDGE_DIR):
        return
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for fname in files:
            if fname.lower().endswith((".txt", ".md")):
                yield os.path.join(root, fname)


def build(user_id, force=False):
    """(Re)index every .txt/.md file under knowledge/ into this user's
    Chroma collection. With force=True, drops the collection first and
    rebuilds from scratch; otherwise upsert-by-id keeps re-running build
    cheap (unchanged chunks are just overwritten with the same content,
    new files/chunks get added)."""
    client = _client()
    name = _collection_name(user_id)

    if force:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name)

    ids, docs, metadatas = [], [], []
    file_count = 0
    for path in _iter_knowledge_files():
        file_count += 1
        rel = os.path.relpath(path, KNOWLEDGE_DIR)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for i, chunk in enumerate(_chunk_text(text)):
            ids.append(f"{rel}::{i}")
            docs.append(chunk)
            metadatas.append({"source": rel, "chunk": i})

    if not docs:
        print(f"No .txt/.md files found under {KNOWLEDGE_DIR} -- nothing indexed.")
        return 0

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    print(f"Indexed {len(docs)} chunk(s) from {file_count} file(s) into collection '{name}'.")
    return len(docs)


def query(user_id, text, n_results=4):
    """Raw search against a user's collection -- returns Chroma's result dict."""
    client = _client()
    name = _collection_name(user_id)
    try:
        collection = client.get_collection(name)
    except Exception:
        raise RuntimeError(
            f"No knowledge base found for user '{user_id}' -- run "
            f"`python rag.py build {user_id}` first."
        )
    return collection.query(query_texts=[text], n_results=n_results)


def query_as_context(user_id, text, n_results=4):
    """Used by jarvis.py / jarvis_server.py's search_knowledge tool. Returns
    a plain-text block of the top matching chunks (source + content), or a
    friendly message if nothing matches. Raises RuntimeError (caught by
    both callers) if the user's collection doesn't exist yet."""
    if not text.strip():
        return "No search query provided."

    results = query(user_id, text, n_results=n_results)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "No matching results found in the knowledge base."

    parts = []
    for doc, meta in zip(docs, metas):
        source = (meta or {}).get("source", "unknown")
        parts.append(f"[{source}]\n{doc.strip()}")
    return "\n\n---\n\n".join(parts)


def _main():
    parser = argparse.ArgumentParser(description="AURA local knowledge base (Chroma-backed RAG).")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="(Re)index knowledge/ for a user.")
    build_p.add_argument("user_id", nargs="?", default="local")
    build_p.add_argument("--force", action="store_true", help="Wipe and rebuild from scratch.")

    query_p = sub.add_parser("query", help="Search a user's knowledge base.")
    query_p.add_argument("user_id", nargs="?", default="local")
    query_p.add_argument("text")
    query_p.add_argument("-n", "--n-results", type=int, default=4)

    args = parser.parse_args()

    if args.command == "build":
        build(args.user_id, force=args.force)
    elif args.command == "query":
        try:
            print(query_as_context(args.user_id, args.text, n_results=args.n_results))
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    _main()
