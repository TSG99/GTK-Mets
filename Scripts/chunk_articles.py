"""
Mets Articles Chunker (parent-child, no network)
--------------------------------------------------
Reads the raw scraped article text (from build_articles.py) and produces
chunks for retrieval. No network calls — safe to re-run repeatedly while
you experiment with chunk size/strategy.

Parent-child model:
    - Each PARENT is the full article (or a section of it, see --parent-mode).
    - Each CHILD is a smaller unit (paragraph, or fixed-size word window)
      that gets embedded/searched. At generation time, look up the child's
      parent_id to pull the fuller context instead of the bare chunk.

Usage:
    python chunk_articles.py                                  # defaults: paragraph children, whole-article parents
    python chunk_articles.py --child-mode fixed --window-words 60 --overlap-words 15
    python chunk_articles.py --parent-mode section --section-size 3
    python chunk_articles.py --min-chars 100
"""

import argparse
import json
import os

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "Database")
ARTICLES_DIR = os.path.join(DB_DIR, "articles")
RAW_CORPUS_FILE = os.path.join(ARTICLES_DIR, "raw_corpus.json")
CHUNKED_CORPUS_FILE = os.path.join(ARTICLES_DIR, "chunked_corpus.json")


# =======================================================================================
# CHILD CHUNKING STRATEGIES
# =======================================================================================
def chunk_paragraphs(text: str, min_chars: int):
    """Split on blank lines / newlines, drop anything too short to be real content."""
    raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return [p for p in raw_paragraphs if len(p) >= min_chars]


def chunk_fixed_window(text: str, window_words: int, overlap_words: int, min_chars: int):
    """Sliding word-window chunking with overlap, for when paragraph boundaries
    are too uneven (some outlets write very long or very short paragraphs)."""
    words = text.split()
    if not words:
        return []

    step = max(window_words - overlap_words, 1)
    chunks = []
    i = 0
    while i < len(words):
        window = words[i:i + window_words]
        chunk_text = " ".join(window)
        if len(chunk_text) >= min_chars:
            chunks.append(chunk_text)
        if i + window_words >= len(words):
            break
        i += step
    return chunks


# =======================================================================================
# PARENT GROUPING STRATEGIES
# =======================================================================================
def build_parents_whole_article(article: dict):
    """One parent per article: the entire article text."""
    return [{
        "parent_id": article["article_id"],
        "parent_text": article["raw_text"],
    }]


def build_parents_by_section(article: dict, section_size: int):
    """Group every N paragraphs into a parent 'section', so a child's
    context window is a chunk of the article rather than the whole thing."""
    paragraphs = [p.strip() for p in article["raw_text"].split("\n") if p.strip()]
    parents = []
    for idx, start in enumerate(range(0, len(paragraphs), section_size)):
        section_paragraphs = paragraphs[start:start + section_size]
        parents.append({
            "parent_id": f"{article['article_id']}_sec{idx}",
            "parent_text": "\n\n".join(section_paragraphs),
        })
    return parents


def assign_child_to_parent(child_text: str, parents: list):
    """Find which parent's text contains this child chunk (substring match).
    Falls back to the first parent if no exact match is found (can happen
    with fixed-window chunking crossing paragraph/section boundaries)."""
    for parent in parents:
        if child_text[:60] in parent["parent_text"]:
            return parent["parent_id"]
    return parents[0]["parent_id"] if parents else None


# =======================================================================================
# MAIN BUILD
# =======================================================================================
def build_chunked_corpus(args):
    if not os.path.exists(RAW_CORPUS_FILE):
        raise FileNotFoundError(
            f"{RAW_CORPUS_FILE} not found. Run build_articles.py first to scrape articles."
        )

    with open(RAW_CORPUS_FILE) as f:
        raw_articles = json.load(f)

    all_parents = []
    all_children = []

    for article in raw_articles:
        # Build parents for this article
        if args.parent_mode == "article":
            parents = build_parents_whole_article(article)
        else:  # "section"
            parents = build_parents_by_section(article, args.section_size)

        for p in parents:
            p.update({
                "article_id": article["article_id"],
                "source": article["source"],
                "published": article["published"],
                "angle": article["angle"],
                "url": article["url"],
            })
        all_parents.extend(parents)

        # Build children for this article
        if args.child_mode == "paragraph":
            child_texts = chunk_paragraphs(article["raw_text"], args.min_chars)
        else:  # "fixed"
            child_texts = chunk_fixed_window(
                article["raw_text"], args.window_words, args.overlap_words, args.min_chars
            )

        for i, child_text in enumerate(child_texts):
            parent_id = assign_child_to_parent(child_text, parents)
            all_children.append({
                "chunk_id": f"{article['article_id']}_c{i}",
                "parent_id": parent_id,
                "article_id": article["article_id"],
                "source": article["source"],
                "text": child_text,
            })

    output = {
        "config": {
            "child_mode": args.child_mode,
            "min_chars": args.min_chars,
            "window_words": args.window_words if args.child_mode == "fixed" else None,
            "overlap_words": args.overlap_words if args.child_mode == "fixed" else None,
            "parent_mode": args.parent_mode,
            "section_size": args.section_size if args.parent_mode == "section" else None,
        },
        "parents": all_parents,
        "children": all_children,
    }

    with open(CHUNKED_CORPUS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Built {len(all_parents)} parents, {len(all_children)} children -> {CHUNKED_CORPUS_FILE}")
    print(f"Config: {output['config']}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Chunk the scraped Mets articles (parent-child).")
    parser.add_argument("--child-mode", choices=["paragraph", "fixed"], default="paragraph",
                         help="How to split children: paragraph boundaries or fixed word windows.")
    parser.add_argument("--min-chars", type=int, default=80,
                         help="Minimum character length for a child chunk to be kept.")
    parser.add_argument("--window-words", type=int, default=60,
                         help="(fixed mode) Words per child chunk.")
    parser.add_argument("--overlap-words", type=int, default=15,
                         help="(fixed mode) Overlapping words between consecutive chunks.")
    parser.add_argument("--parent-mode", choices=["article", "section"], default="article",
                         help="Parent granularity: whole article, or grouped sections.")
    parser.add_argument("--section-size", type=int, default=3,
                         help="(section mode) Number of paragraphs per parent section.")
    args = parser.parse_args()

    build_chunked_corpus(args)


if __name__ == "__main__":
    main()
