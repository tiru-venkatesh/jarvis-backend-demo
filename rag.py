"""
rag.py - lightweight, per-user local knowledge base for AURA.

Uses ChromaDB's built-in embedding function so the Render deployment
does NOT need sentence-transformers, PyTorch, or CUDA packages.

Per-user isolation
------------------
Conversations and long-term facts are already scoped by user_id (see
memory_store.py). Knowledge used to be the exception: one global
collection that every signed-in user shared, so the moment a second user
signed up, user A's search_knowledge could surface user B's private
documents (and vice versa). That's a data-leak, not a feature gap.

The fix here is deliberately *physical*, not a filter: each user gets
their own Chroma collection and their own knowledge/ subfolder. Isolation
therefore can't be defeated by a forgotten `where={"user_id": ...}` clause
on a query -- a query against user A's collection simply has no way to
return user B's vectors. For personal docs (about_tiru.txt is Tiru's own
bio) that stronger guarantee is worth the small extra bookkeeping.

Layout on disk:
    knowledge/                      root-level files belong to the OWNER
        about_tiru.txt              (KNOWLEDGE_OWNER_ID, default "local")
        <user_id>/                  each other user's private docs
            notes.txt
    chroma_db/
        manifest_<safe_user_id>.json  per-user change-tracking manifest

Backward compatibility: existing installs keep about_tiru.txt at the
knowledge/ root, which maps to the owner id automatically, and the owner's
collection auto-builds on first query -- no manual migration step.
"""

import hashlib
import json
import os
import re
import sys

KNOWLEDGE_DIR = "knowledge"
CHROMA_DIR = "chroma_db"

# Root-level files in knowledge/ (not inside a per-user subfolder) belong to
# this user id. In CLI mode that's "local" (see jarvis.CLI_USER_ID), which is
# also Tiru's own bio in the current single-user setup. On the multi-user
# server every real user has a Supabase uuid and their own knowledge/<uuid>/
# folder instead.
KNOWLEDGE_OWNER_ID = os.environ.get("KNOWLEDGE_OWNER_ID", "local")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _get_chromadb():
    try:
        import chromadb
    except ImportError:
        raise RuntimeError(
            "ChromaDB is not installed. Add 'chromadb' to requirements.txt."
        )
    return chromadb


# One PersistentClient for the process; collections are cached per user so a
# hot path (repeated search_knowledge for the same user) doesn't re-open the
# collection every call.
_chroma_client = None
_collections = {}


def _safe_id(user_id: str) -> str:
    """Turn an arbitrary user_id into a filesystem/collection-safe token.

    Supabase ids are uuids (already safe), but user_id is untrusted input in
    principle, so sanitize defensively and append a short hash so two
    different ids can never collapse to the same sanitized string (which
    would silently merge their knowledge -- the exact leak we're preventing)."""
    raw = str(user_id or "").strip() or "anon"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower() or "u"
    digest = hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{slug[:40]}_{digest}"


def _collection_name(user_id: str) -> str:
    # Chroma requires 3-512 chars, [a-zA-Z0-9._-], starting/ending
    # alphanumeric. "kb_" + _safe_id() satisfies all of that.
    return f"kb_{_safe_id(user_id)}"


def _manifest_file(user_id: str) -> str:
    return os.path.join(CHROMA_DIR, f"manifest_{_safe_id(user_id)}.json")


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        chromadb = _get_chromadb()
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _get_collection(user_id: str):
    key = _collection_name(user_id)
    if key not in _collections:
        client = _get_client()
        _collections[key] = client.get_or_create_collection(
            key, metadata={"hnsw:space": "cosine"}
        )
    return _collections[key]


def _user_knowledge_dir(user_id: str) -> str:
    """Where this user's source documents live.

    The owner reads from the knowledge/ root (backward compat with the old
    single-user layout); everyone else reads from knowledge/<user_id>/."""
    if str(user_id) == KNOWLEDGE_OWNER_ID:
        return KNOWLEDGE_DIR
    return os.path.join(KNOWLEDGE_DIR, str(user_id))


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0:
            break
    return [c.strip() for c in chunks if c.strip()]


def _file_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_files(user_id: str) -> dict:
    """Load this user's .txt/.md files.

    For the owner (knowledge/ root) we deliberately DON'T descend into
    per-user subfolders -- os.walk would otherwise sweep every other user's
    knowledge/<uuid>/ into the owner's index, re-creating the global-leak
    this module exists to prevent. Non-owner users read their own subfolder
    recursively, which is fine because that whole tree is theirs."""
    folder = _user_knowledge_dir(user_id)
    if not os.path.isdir(folder):
        return {}

    docs = {}
    is_owner = str(user_id) == KNOWLEDGE_OWNER_ID

    if is_owner:
        # Root-level files only; skip subdirectories (they belong to others).
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if os.path.isfile(path) and fname.lower().endswith((".txt", ".md")):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    docs[path] = f.read()
    else:
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith((".txt", ".md")):
                    path = os.path.join(root, fname)
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        docs[path] = f.read()
    return docs


def _load_manifest(user_id: str) -> dict:
    mf = _manifest_file(user_id)
    if os.path.exists(mf):
        with open(mf, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(user_id: str, manifest: dict) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(_manifest_file(user_id), "w") as f:
        json.dump(manifest, f)


def build_index(user_id: str, force: bool = False) -> None:
    """(Re)build one user's index from their knowledge folder. Incremental by
    default -- only files whose contents changed (by hash) are re-embedded."""
    docs = _load_files(user_id)
    if not docs:
        folder = _user_knowledge_dir(user_id)
        print(f"[rag] No .txt/.md files found for user '{user_id}' in {folder}/.")
        return

    client = _get_client()
    name = _collection_name(user_id)

    if force:
        existing = [c.name for c in client.list_collections()]
        if name in existing:
            client.delete_collection(name)
        _collections.pop(name, None)
        manifest = {}
    else:
        manifest = _load_manifest(user_id)

    collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    _collections[name] = collection

    to_process = {p: t for p, t in docs.items() if manifest.get(p) != _file_hash(t)}
    removed = [p for p in manifest if p not in docs]

    if not to_process and not removed:
        print(f"[rag] User '{user_id}' already up to date ({len(docs)} file(s)).")
        return

    for path in list(to_process.keys()) + removed:
        try:
            collection.delete(where={"source": path})
        except Exception:
            pass
        manifest.pop(path, None)

    for path, text in to_process.items():
        chunks = _chunk_text(text)
        if not chunks:
            continue
        ids = [f"{path}::{i}" for i in range(len(chunks))]
        metadata = [{"source": path} for _ in chunks]
        collection.add(ids=ids, documents=chunks, metadatas=metadata)
        manifest[path] = _file_hash(text)

    _save_manifest(user_id, manifest)
    print(
        f"[rag] User '{user_id}' index updated: "
        f"{collection.count()} chunks, {len(docs)} files."
    )


def _ensure_index(user_id: str) -> None:
    """Lazy self-heal: if this user has documents on disk but an empty
    collection (fresh deploy, first run, or the one-time switch from the old
    global collection name), build it on demand so there's no manual
    `python rag.py build` step and no cold-start empty results."""
    collection = _get_collection(user_id)
    if collection.count() > 0:
        return
    if _load_files(user_id):
        build_index(user_id)


def query(user_id: str, question: str, k: int = 5) -> list:
    _ensure_index(user_id)
    collection = _get_collection(user_id)
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(k, collection.count()),
    )

    hits = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(documents, metadatas):
        hits.append({"text": doc, "source": meta.get("source", "unknown")})
    return hits


def query_as_context(user_id: str, question: str, k: int = 5) -> str:
    hits = query(user_id, question, k=k)
    if not hits:
        return "No relevant information found in the local knowledge base."

    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(parts)


def _print_cli_usage():
    print(
        "Usage:\n"
        "  python rag.py build [user_id] [--force]\n"
        '  python rag.py query [user_id] "your question"\n\n'
        f"user_id defaults to the owner ('{KNOWLEDGE_OWNER_ID}') when omitted."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _print_cli_usage()
        sys.exit(1)

    command = sys.argv[1]
    rest = [a for a in sys.argv[2:] if a != "--force"]
    force = "--force" in sys.argv

    if command == "build":
        # Optional positional user_id; defaults to the owner.
        user_id = rest[0] if rest else KNOWLEDGE_OWNER_ID
        build_index(user_id, force=force)

    elif command == "query":
        if not rest:
            print('Usage: python rag.py query [user_id] "your question"')
            sys.exit(1)
        # If the first token looks like a bare user id (no spaces) AND there's
        # more after it, treat it as the user id; otherwise it's the question
        # and we default to the owner.
        if len(rest) > 1:
            user_id, question = rest[0], " ".join(rest[1:])
        else:
            user_id, question = KNOWLEDGE_OWNER_ID, rest[0]
        print(query_as_context(user_id, question))

    else:
        print(f"Unknown command '{command}'. Use 'build' or 'query'.")
        _print_cli_usage()
