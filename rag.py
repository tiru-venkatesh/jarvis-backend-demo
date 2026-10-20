#!/usr/bin/env python3
"""
rag.py - local per-user knowledge base for AURA, built on ChromaDB.

Each user's docs live in their own physically separate Chroma collection
(kb_<user_id>), so one user's files can never leak into another user's
search results (see the README's "Per-user knowledge base" section). Source
docs come from the knowledge/ folder next to this file -- plain text and
markdown files are chunked, embedded with Chroma's built-in ONNX MiniLM
embedding function (ships with chromadb), and stored.

Retrieval quality features (v2):
  * Structure-aware chunking: split on markdown headings, then pack whole
    paragraphs/bullets; every chunk carries its heading path ("Technical
    Skills > AI/ML") so it is self-describing both to the embedder and the LLM.
  * Hybrid search: dense vectors (semantic) + BM25 (exact keywords/names/
    numbers), merged with Reciprocal Rank Fusion.
  * Query expansion (acronyms/synonyms), near-duplicate removal, and a
    relevance floor so unrelated questions return "no match" instead of junk.
  * Re-index removes stale chunks of edited/deleted files.
  * Reads .txt .md and .pdf (pdf needs pypdf; skipped if missing).
  * If the embedding model can't load, search degrades to BM25 only.

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
import hashlib
import math
import os
import re
import sys
from collections import Counter

import chromadb

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(HERE, "knowledge")
CHROMA_DIR = os.path.join(HERE, "chroma_db")

# Target chunk size (chars). Chunks are built from whole paragraphs/bullets,
# so they land near this size but rarely cut a thought in half.
CHUNK_TARGET = 900
CHUNK_MAX = 1400
OVERLAP_CHARS = 150          # tail of previous chunk repeated (sentence-aligned)
MIN_CHUNK = 60               # drop crumbs

CANDIDATES = 24              # per retriever before fusion
RRF_K = 60
DENSE_MAX_DISTANCE = 0.88    # cosine distance (MiniLM); above this a dense hit is noise
MIN_BM25_SCORE = 1.0         # keyword-only hits below this are too weak to trust
DEDUPE_JACCARD = 0.72

TEXT_EXT = (".txt", ".md")
PDF_EXT = (".pdf",)

_STOP = set("""a an and are as at be but by can did do does for from had has have
how i if in into is it its me my of on or our so that the their them then there
these they this to was we were what when where which who whom why will with you
your tell about please give show""".split())

# Small domain-agnostic expansion map: query word -> extra terms to also match.
_SYNONYMS = {
    "college": ["university", "jntu", "education", "btech"],
    "university": ["college", "education"],
    "study": ["education", "degree", "btech"],
    "degree": ["btech", "education"],
    "skills": ["skill", "technical", "stack", "programming"],
    "stack": ["skills", "technologies", "tools"],
    "project": ["projects", "built", "deployed", "app"],
    "projects": ["project", "built", "deployed", "app"],
    "built": ["project", "made", "created"],
    "contact": ["email", "linkedin", "github"],
    "email": ["mail", "contact"],
    "job": ["career", "role", "internship"],
    "internship": ["intern", "career"],
    "ml": ["machine", "learning"],
    "ai": ["artificial", "intelligence", "llm"],
    "llm": ["llms", "language", "model"],
    "rag": ["retrieval", "augmented", "generation"],
    "cv": ["vision", "computer"],
    "hackathon": ["hackathons", "competition"],
    "deploy": ["deployed", "deployment", "hosting", "render", "vercel", "netlify"],
    "deployed": ["deployment", "hosting", "render", "vercel", "netlify"],
    "website": ["site", "web", "portfolio"],
    "resume": ["cv", "profile"],
    "goal": ["goals", "ambition", "direction"],
    "ambition": ["goal", "persona"],
}


def _collection_name(user_id):
    # Chroma only allows alnum/underscore/hyphen names, 3-63 chars.
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(user_id))
    return f"kb_{safe}"[:63]


def _client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


# --------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------
def _iter_knowledge_files():
    if not os.path.isdir(KNOWLEDGE_DIR):
        return
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for fname in sorted(files):
            if fname.lower().endswith(TEXT_EXT + PDF_EXT):
                yield os.path.join(root, fname)


def _read_file(path):
    if path.lower().endswith(PDF_EXT):
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"  (skipping {os.path.basename(path)}: pip install pypdf to index PDFs)")
            return ""
        try:
            reader = PdfReader(path)
            return "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            print(f"  (could not read {os.path.basename(path)}: {e})")
            return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[*])")


def _clean_heading(h):
    return re.sub(r"[*_`]", "", h).strip()


def _split_sections(text):
    """Yield (heading_path, body) for each markdown section. Text with no
    headings (plain .txt, PDFs) is a single section with an empty path."""
    stack = []          # [(level, title)]
    buf = []
    sections = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            # Drop the level-1 document title from the path (the file name is
            # added separately) unless it is the only heading we have.
            parts = [t for lv, t in stack if lv > 1] or [t for _, t in stack]
            sections.append((" > ".join(parts), body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), _clean_heading(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        elif re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", line):
            continue            # horizontal rules are noise
        else:
            buf.append(line)
    flush()
    return sections


def _units(body):
    """Break a section body into atomic units: paragraphs, with oversized
    ones split on sentence boundaries (or hard-wrapped as a last resort)."""
    out = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_MAX:
            out.append(para)
            continue
        # long paragraph / bullet list: split by line first, then sentence
        pieces = []
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) <= CHUNK_MAX:
                pieces.append(line)
            else:
                pieces.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
        for p in pieces:
            while len(p) > CHUNK_MAX:               # hard wrap, last resort
                cut = p.rfind(" ", 0, CHUNK_MAX)
                cut = cut if cut > CHUNK_MAX // 2 else CHUNK_MAX
                out.append(p[:cut].strip())
                p = p[cut:].strip()
            if p:
                out.append(p)
    return out


def _tail_overlap(text):
    """Last ~OVERLAP_CHARS of text, starting at a sentence/line boundary."""
    if len(text) <= OVERLAP_CHARS:
        return text
    tail = text[-OVERLAP_CHARS:]
    for sep in ("\n", ". ", "; "):
        i = tail.find(sep)
        if 0 <= i < len(tail) - 20:
            return tail[i + len(sep):].strip()
    i = tail.find(" ")
    return tail[i + 1:] if i >= 0 else tail


def _chunk_document(text):
    """Return list of (heading_path, chunk_text)."""
    chunks = []
    for path, body in _split_sections(text):
        cur, cur_len = [], 0
        prev_tail = ""
        for u in _units(body):
            if cur and cur_len + len(u) + 1 > CHUNK_TARGET:
                joined = "\n".join(cur)
                chunks.append((path, joined))
                prev_tail = _tail_overlap(joined)
                cur, cur_len = ([prev_tail] if prev_tail else []), len(prev_tail)
            cur.append(u)
            cur_len += len(u) + 1
        if cur:
            joined = "\n".join(cur)
            # don't emit a chunk that is only the overlap tail
            if joined.strip() != prev_tail.strip():
                chunks.append((path, joined))
    # merge crumbs into the previous chunk of the same section
    merged = []
    for path, c in chunks:
        if merged and len(c) < MIN_CHUNK and merged[-1][0] == path:
            merged[-1] = (path, merged[-1][1] + "\n" + c)
        elif len(c.strip()) >= 15:
            merged.append((path, c))
    return merged


def _contextualize(source, path, chunk):
    """Text that is actually embedded/searched: chunk prefixed with where it
    came from, so 'Skills' chunks know they are about Technical Skills of
    this profile even if the chunk text itself is just a bullet list."""
    title = os.path.splitext(os.path.basename(source))[0].replace("_", " ")
    ctx = f"{title} > {path}" if path else title
    return f"[{ctx}]\n{chunk}"


# --------------------------------------------------------------------------
# Tokenisation / BM25
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-_/@]*[a-z0-9+#]|[a-z0-9]")


def _stem(w):
    for suf in ("ingly", "edly", "ings", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
            break
    return w


def _tokens(text):
    toks = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP:
            continue
        toks.append(_stem(t))
        # also index the pieces of compound tokens: "gpt-oss-120b" -> gpt, oss, 120b
        if re.search(r"[-_./@]", t):
            toks.extend(_stem(p) for p in re.split(r"[-_./@]+", t) if p and p not in _STOP)
    return toks


class _BM25:
    def __init__(self, docs_tokens, k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = [len(d) for d in docs_tokens]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q_tokens):
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q_tokens:
                f = tf.get(t)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


_BM25_CACHE = {}   # collection name -> (signature, ids, docs, metas, bm25)


def _load_index(collection):
    name = collection.name
    count = collection.count()
    cached = _BM25_CACHE.get(name)
    if cached and cached[0] == count:
        return cached[1:]
    data = collection.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    bm = _BM25([_tokens(d) for d in docs])
    _BM25_CACHE[name] = (count, ids, docs, metas, bm)
    return ids, docs, metas, bm


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build(user_id, force=False):
    """(Re)index knowledge/ into this user's Chroma collection. Stale chunks
    from edited/removed files are deleted, so re-running build always leaves
    the index matching the files on disk."""
    client = _client()
    name = _collection_name(user_id)

    if force:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    ids, docs, metadatas = [], [], []
    file_count = 0
    for path in _iter_knowledge_files():
        file_count += 1
        rel = os.path.relpath(path, KNOWLEDGE_DIR).replace("\\", "/")
        text = _read_file(path)
        for i, (hpath, chunk) in enumerate(_chunk_document(text)):
            digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:10]
            ids.append(f"{rel}::{i}::{digest}")
            docs.append(_contextualize(rel, hpath, chunk))
            metadatas.append({"source": rel, "chunk": i, "section": hpath})

    if not docs:
        print(f"No .txt/.md/.pdf files found under {KNOWLEDGE_DIR} -- nothing indexed.")
        return 0

    # Remove anything in the collection that's no longer produced by the files.
    try:
        existing = set(collection.get(include=[])["ids"])
    except Exception:
        existing = set()
    stale = list(existing - set(ids))
    if stale:
        for i in range(0, len(stale), 500):
            collection.delete(ids=stale[i:i + 500])

    for i in range(0, len(ids), 100):
        collection.upsert(ids=ids[i:i + 100], documents=docs[i:i + 100],
                          metadatas=metadatas[i:i + 100])
    _BM25_CACHE.pop(name, None)
    print(f"Indexed {len(docs)} chunk(s) from {file_count} file(s) into collection "
          f"'{name}' ({len(stale)} stale removed).")
    return len(docs)


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------
def _expand_query(text):
    base = _tokens(text)
    extra = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        for s in _SYNONYMS.get(w, []):
            extra.append(_stem(s))
    return base, extra


def _get_collection(user_id):
    client = _client()
    name = _collection_name(user_id)
    try:
        return client.get_collection(name)
    except Exception:
        raise RuntimeError(
            f"No knowledge base found for user '{user_id}' -- run "
            f"`python rag.py build {user_id}` first."
        )


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def search(user_id, text, n_results=5):
    """Hybrid search. Returns a list of dicts: {text, source, section, score,
    dense_rank, bm25_rank} best-first."""
    collection = _get_collection(user_id)
    if collection.count() == 0:
        return []
    ids, docs, metas, bm = _load_index(collection)
    pos = {i: k for k, i in enumerate(ids)}

    # --- sparse (BM25) with light query expansion (expansion terms weigh less)
    base, extra = _expand_query(text)
    sparse_scores = bm.scores(base)
    if extra:
        ex_scores = bm.scores(extra)
        sparse_scores = [s + 0.35 * e for s, e in zip(sparse_scores, ex_scores)]
    sparse_order = sorted(range(len(ids)), key=lambda i: sparse_scores[i], reverse=True)
    sparse_order = [i for i in sparse_order if sparse_scores[i] > 0][:CANDIDATES]

    # --- dense (semantic); degrade gracefully if the embedder is unavailable
    dense_order, dense_dist = [], {}
    try:
        res = collection.query(
            query_texts=[text + (" " + " ".join(extra[:6]) if extra else "")],
            n_results=min(CANDIDATES, len(ids)),
            include=["distances"],
        )
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            if cid in pos:
                dense_order.append(pos[cid])
                dense_dist[pos[cid]] = dist
    except Exception as e:
        print(f"[rag] dense search unavailable, using keyword search only: {e}", file=sys.stderr)

    # --- reciprocal rank fusion
    fused = Counter()
    for r, i in enumerate(dense_order):
        fused[i] += 1.0 / (RRF_K + r + 1)
    for r, i in enumerate(sparse_order):
        fused[i] += 1.0 / (RRF_K + r + 1) * 1.15      # exact matches get a small edge
    sparse_rank = {i: r for r, i in enumerate(sparse_order)}
    dense_rank = {i: r for r, i in enumerate(dense_order)}

    # --- relevance floor: keep a hit if it's a decent dense match OR a keyword hit
    def relevant(i):
        d = dense_dist.get(i)
        return (d is not None and d <= DENSE_MAX_DISTANCE) or (i in sparse_rank and sparse_scores[i] >= MIN_BM25_SCORE)

    ranked = [i for i, _ in fused.most_common() if relevant(i)]
    if not dense_order and not sparse_order:
        return []

    # --- drop near-duplicates (overlap regions), keep order
    picked, picked_tok = [], []
    for i in ranked:
        tok = _tokens(docs[i])
        if any(_jaccard(tok, pt) >= DEDUPE_JACCARD for pt in picked_tok):
            continue
        picked.append(i)
        picked_tok.append(tok)
        if len(picked) >= n_results:
            break

    return [{
        "text": docs[i],
        "source": (metas[i] or {}).get("source", "unknown"),
        "section": (metas[i] or {}).get("section", ""),
        "score": round(fused[i], 5),
        "dense_rank": dense_rank.get(i),
        "bm25_rank": sparse_rank.get(i),
    } for i in picked]


def query(user_id, text, n_results=5):
    """Back-compat: Chroma-style dict of the top hits."""
    hits = search(user_id, text, n_results=n_results)
    return {
        "documents": [[h["text"] for h in hits]],
        "metadatas": [[{"source": h["source"], "section": h["section"]} for h in hits]],
    }


def query_as_context(user_id, text, n_results=5):
    """Used by jarvis.py / jarvis_server.py's search_knowledge tool. Returns
    a plain-text block of the top matching chunks (source > section +
    content), or a friendly message if nothing is relevant. Raises
    RuntimeError (caught by both callers) if the collection doesn't exist."""
    if not text or not text.strip():
        return "No search query provided."

    hits = search(user_id, text, n_results=n_results)
    if not hits:
        return "No matching results found in the knowledge base."

    parts = []
    for n, h in enumerate(hits, 1):
        # h["text"] already starts with "[title > section]" context line
        parts.append(f"({n}) {h['text'].strip()}")
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
    query_p.add_argument("-n", "--n-results", type=int, default=5)
    query_p.add_argument("--debug", action="store_true", help="Show scores and ranks.")

    args = parser.parse_args()

    if args.command == "build":
        build(args.user_id, force=args.force)
    elif args.command == "query":
        try:
            if args.debug:
                for h in search(args.user_id, args.text, args.n_results):
                    print(f"score={h['score']} dense#={h['dense_rank']} bm25#={h['bm25_rank']} "
                          f"{h['source']} :: {h['section']}")
            print(query_as_context(args.user_id, args.text, n_results=args.n_results))
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    _main()
