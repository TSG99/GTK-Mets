"""
Mets Articles Retrieval (embedding + search)
-----------------------------------------------
Embeds the child chunks from chunk_articles.py using a local Ollama
embedding model (nomic-embed-text) -- no API key, no network call, no
per-user cost. Stores the vectors locally and provides a retrieve()
function for semantic search over the article corpus.

Requires:
    ollama pull nomic-embed-text
    ollama serve                      (if not already running)
    pip install ollama

nomic-embed-text was trained with asymmetric task prefixes -- documents
get "search_document: " prepended, queries get "search_query: " prepended.
Skipping this measurably hurts retrieval quality with this model, so both
build_index() and retrieve() add it automatically; you don't need to
think about it when calling either function.

Usage:
    python retrieval.py --build              # embeds all children, builds the index
    python retrieval.py --query "how are the Mets doing at the trade deadline"
"""

import argparse
import json
import os

import numpy as np
import ollama

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "Database")
ARTICLES_DIR = os.path.join(DB_DIR, "articles")
CHUNKED_CORPUS_FILE = os.path.join(ARTICLES_DIR, "chunked_corpus.json")
INDEX_DIR = os.path.join(ARTICLES_DIR, "index")
VECTORS_FILE = os.path.join(INDEX_DIR, "vectors.npy")
INDEX_META_FILE = os.path.join(INDEX_DIR, "index_meta.json")

EMBED_MODEL = "nomic-embed-text"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def ensure_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)


def build_index(batch_size: int = 32):
    """Embed every child chunk and save vectors + metadata locally."""
    if not os.path.exists(CHUNKED_CORPUS_FILE):
        raise FileNotFoundError(
            f"{CHUNKED_CORPUS_FILE} not found. Run build_articles.py then chunk_articles.py first."
        )

    with open(CHUNKED_CORPUS_FILE) as f:
        chunked = json.load(f)

    children = chunked["children"]
    parents = {p["parent_id"]: p for p in chunked["parents"]}

    client = ollama.Client()  # talks to localhost:11434 by default

    all_vectors = []
    metadata = []

    print(f"Embedding {len(children)} chunks with {EMBED_MODEL} (local, via Ollama)...")
    for i in range(0, len(children), batch_size):
        batch = children[i:i + batch_size]
        texts = [DOCUMENT_PREFIX + c["text"] for c in batch]

        result = client.embed(model=EMBED_MODEL, input=texts)
        all_vectors.extend(result.embeddings)

        for c in batch:
            parent = parents.get(c["parent_id"], {})
            metadata.append({
                "chunk_id": c["chunk_id"],
                "parent_id": c["parent_id"],
                "article_id": c["article_id"],
                "source": c["source"],
                "text": c["text"],
                "parent_text": parent.get("parent_text"),
                "url": parent.get("url"),
                "published": parent.get("published"),
                "angle": parent.get("angle"),
            })

        print(f"  embedded {min(i + batch_size, len(children))}/{len(children)}")

    ensure_dir()
    vectors_array = np.array(all_vectors, dtype=np.float32)
    np.save(VECTORS_FILE, vectors_array)
    with open(INDEX_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {vectors_array.shape[0]} vectors (dim={vectors_array.shape[1]}) -> {VECTORS_FILE}")
    print(f"Saved metadata -> {INDEX_META_FILE}")


def load_index():
    if not (os.path.exists(VECTORS_FILE) and os.path.exists(INDEX_META_FILE)):
        raise FileNotFoundError("Index not built yet. Run with --build first.")

    vectors = np.load(VECTORS_FILE)
    with open(INDEX_META_FILE) as f:
        metadata = json.load(f)
    return vectors, metadata


def retrieve(query: str, k: int = 3):
    """Embed the query, return the top-k most similar chunks (with parent context)."""
    vectors, metadata = load_index()

    client = ollama.Client()
    result = client.embed(model=EMBED_MODEL, input=[QUERY_PREFIX + query])
    query_vec = np.array(result.embeddings[0], dtype=np.float32)

    # Ollama's embed() output isn't guaranteed unit-normalized like Voyage's
    # was, so normalize both sides explicitly before scoring -- otherwise
    # this quietly turns into un-normalized dot product instead of cosine
    # similarity and rankings get skewed by vector magnitude.
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query_vec_norm = query_vec / np.linalg.norm(query_vec)
    scores = vectors_norm @ query_vec_norm
    top_k_idx = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_k_idx:
        entry = metadata[idx].copy()
        entry["score"] = float(scores[idx])
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description="Build or query the Mets articles retrieval index.")
    parser.add_argument("--build", action="store_true", help="Embed all chunks and build the index.")
    parser.add_argument("--query", type=str, default=None, help="Run a test query against the index.")
    parser.add_argument("--k", type=int, default=3, help="Number of results to return.")
    args = parser.parse_args()

    if args.build:
        build_index()

    if args.query:
        results = retrieve(args.query, k=args.k)
        print(f"\nTop {len(results)} results for: \"{args.query}\"\n")
        for r in results:
            print(f"[{r['score']:.3f}] {r['source']} ({r['published']}) — {r['angle']}")
            print(f"  chunk: {r['text'][:150]}...")
            print()


if __name__ == "__main__":
    main()
