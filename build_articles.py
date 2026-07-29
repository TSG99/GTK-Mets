"""
Mets Articles Scraper (raw text only, no chunking)

Grabs a small set of predetermined articles and extracts
article text (using trafilatura to clean) 
Does NOT chunk: that's handled separately by chunk_articles.py 
This maintains modularity

Saves raw extracted text per article into yourpath/Data/Database/articles/raw_corpus.json.

Usage:
    python build_articles.py                # builds cache if not present
    python build_articles.py --update        # forces a re-scrape of all articles
"""

import argparse
import hashlib
import json
import os
from datetime import datetime

import requests
import trafilatura

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "Database")
ARTICLES_DIR = os.path.join(DB_DIR, "articles")
RAW_CORPUS_FILE = os.path.join(ARTICLES_DIR, "raw_corpus.json")

# Hand-picked, reputable sources — swap/add here
ARTICLE_SOURCES = [
    (
        "https://www.mlb.com/news/mets-2026-season-preview-predictions",
        "MLB.com",
        "2026-03-23",
        "preseason predictions",
    ),
    (
        "https://www.qchron.com/editions/queenswide/mets-2026-season-preview/article_bc1b0222-20e3-5413-8ab9-e3f248da7e36.html",
        "Queens Chronicle",
        "2026-03-25",
        "season preview / roster overhaul recap",
    ),
    (
        "https://www.mlb.com/news/mets-approach-to-2026-trade-deadline",
        "MLB.com",
        "2026-05-14",
        "mid-season trade deadline outlook",
    ),
    (
        "https://www.cbssports.com/mlb/news/mets-trade-deadline-david-peterson-freddy-peralta-luke-weaver-brooks-raley-clay-holmes/",
        "CBS Sports",
        "2026-06-25",
        "trade deadline sell-off analysis",
    ),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def ensure_dir():
    os.makedirs(ARTICLES_DIR, exist_ok=True)

def make_article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def fetch_raw_text(url: str, source: str, published: str, angle: str):
    print(f"Fetching: {source} — {url}")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    extracted = trafilatura.extract(resp.text, url=url, favor_recall=True)
    if not extracted:
        print(f"  WARNING: trafilatura could not extract article text from {url}")
        return None

    article_id = make_article_id(url)
    print(f"  -> extracted {len(extracted)} characters of raw text")

    return {
        "article_id": article_id,
        "url": url,
        "source": source,
        "published": published,
        "angle": angle,
        "raw_text": extracted,
        "fetched_at": datetime.now().isoformat(),
    }

def build_raw_corpus():
    ensure_dir()
    corpus = []
    for url, source, published, angle in ARTICLE_SOURCES:
        try:
            record = fetch_raw_text(url, source, published, angle)
            if record:
                corpus.append(record)
        except requests.exceptions.RequestException as e:
            print(f"  ERROR fetching {url}: {e}")

    with open(RAW_CORPUS_FILE, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"\nSaved {len(corpus)} raw articles -> {RAW_CORPUS_FILE}")
    print("Run chunk_articles.py next to generate chunks (no network needed).")
    return corpus


def load_raw_corpus(force_update: bool = False):
    if force_update or not os.path.exists(RAW_CORPUS_FILE):
        reason = "forced update" if force_update else "no cache found"
        print(f"Building raw article corpus ({reason})...")
        return build_raw_corpus()
    else:
        print(f"Loading cached raw corpus from {RAW_CORPUS_FILE} (no network call)...")
        with open(RAW_CORPUS_FILE) as f:
            return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Scrape the Mets articles (raw text only).")
    parser.add_argument("--update", action="store_true", help="Force a full re-scrape.")
    args = parser.parse_args()

    corpus = load_raw_corpus(force_update=args.update)

    print(f"\n{len(corpus)} raw articles available:")
    for a in corpus:
        print(f"  [{a['source']}] {a['published']} — {a['angle']} ({len(a['raw_text'])} chars)")

if __name__ == "__main__":
    main()
