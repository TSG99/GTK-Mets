"""
Mets RAG Pipeline (query routing + hybrid retrieval + generation)
-------------------------------------------------------------------
This is the piece that ties the structured stats DB (build_db.py) and the
unstructured article index (chunk_articles.py + retrieval.py) together into
an actual retrieval-augmented generation pipeline.

Flow for a single question:
    1. ROUTE    -- ask the local model whether the question needs stats,
                   articles, or both, and pull out any player names it
                   should filter on.
    2. RETRIEVE -- pull the relevant stats rows (structured) and/or the
                   top-k article chunks with parent context (unstructured).
    3. GENERATE -- hand both context blocks to the local model and have it
                   write a grounded answer that cites its sources (stat row
                   or article + date), and says so plainly when the corpus
                   doesn't cover something.

Runs entirely against a local Ollama server -- no API key required, which
matters for sharing this with other people. No network calls happen at
import time. Building the article index (via retrieval.py --build) and
having mets_batting.json / mets_pitching.json present are both
prerequisites -- this module doesn't build either, it just consumes them.

Requires:
    ollama pull qwen2.5:14b
    ollama serve                      (if not already running)
    pip install ollama

Note on hardware: qwen2.5:14b wants roughly 10-12GB of RAM/VRAM free to run
at a reasonable speed. If someone running this app hits an out-of-memory
error or the app feels unusably slow, swap DEFAULT_MODEL below for a
smaller one (e.g. "llama3.1:8b" or "mistral:7b") -- everything else in this
file stays the same.

Usage (as a library):
    from rag_pipeline import answer_question
    result = answer_question("How has Soto looked at the plate this year?")
    print(result["answer"])
    print(result["sources"])
"""

# =======================================================================================
# STEP 0: IMPORTS AND GLOBAL CONFIG
# =======================================================================================
import json
import os
import re

import ollama
import pandas as pd

from retrieval import retrieve  # reuses the existing embed/search index

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "Database")
BATTING_FILE = os.path.join(DB_DIR, "mets_batting.json")
PITCHING_FILE = os.path.join(DB_DIR, "mets_pitching.json")

DEFAULT_MODEL = "qwen2.5:14b"
GENERATION_MODEL = DEFAULT_MODEL
ROUTER_MODEL = DEFAULT_MODEL
ARTICLE_TOP_K = 4

_client = None


def get_client() -> ollama.Client:
    """Lazy singleton so importing this module never requires the Ollama
    server to already be up. Talks to localhost:11434 by default."""
    global _client
    if _client is None:
        _client = ollama.Client()
    return _client


# =======================================================================================
# STEP 1: QUERY ROUTING
# =======================================================================================
ROUTER_SYSTEM_PROMPT = """You are a query router for a New York Mets question-answering \
system with two data sources:

- STATS: a small structured table of current-season Mets batting and pitching stats \
(AVG/OBP/SLG/HR/RBI/SB for hitters, ERA/WHIP/W/L/SV/SO for pitchers).
- ARTICLES: a handful of recent news articles covering season predictions, roster \
moves, and trade deadline analysis.

Given a user question, decide which source(s) are needed to answer it well.

Respond with ONLY a JSON object, no other text, no markdown fences:
{"need_stats": true/false, "need_articles": true/false, "player_names": ["..."]}

player_names should list any specific players named or clearly implied in the question \
(empty list if the question is about the team/roster in general, e.g. "who's hitting \
the best right now"). Use the player's most likely full name as it would appear on a \
roster.

If the question is ambiguous or could benefit from both sources, set both flags true \
rather than guessing wrong -- being unhelpful is worse than pulling one extra source."""


def route_query(question: str) -> dict:
    """Ask the local model to classify what this question needs. Uses
    Ollama's format="json" mode so the model is constrained to valid JSON
    at decode time (not just prompted to produce it). Falls back to 'pull
    everything' if parsing still fails somehow, since that's the safe
    direction -- an extra retrieval call is cheap, a wrongly-skipped one
    isn't."""
    response = get_client().chat(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        format="json",
    )
    raw = response.message.content.strip()

    try:
        parsed = json.loads(raw)
        return {
            "need_stats": bool(parsed.get("need_stats", True)),
            "need_articles": bool(parsed.get("need_articles", True)),
            "player_names": list(parsed.get("player_names", [])),
        }
    except (json.JSONDecodeError, AttributeError):
        print(f"  WARNING: router returned unparseable output ({raw!r}), pulling both sources.")
        return {"need_stats": True, "need_articles": True, "player_names": []}


# =======================================================================================
# STEP 2A-i: DETERMINISTIC STAT SUPERLATIVES (best/worst/highest/lowest at X)
# =======================================================================================
# Small local models are unreliable at scanning an unsorted table and picking
# the true max/min -- in practice they lean toward whichever row appears
# first (or most recently) in context, not the row that's actually highest.
# Rather than trust generation with that arithmetic, we detect "who has the
# best/worst <stat>" style questions here and compute the answer directly
# with pandas. The model gets handed a fact, not a table to do math over.

# Minimum sample size before a rate stat counts -- otherwise a player with a
# handful of at-bats/innings can top the leaderboard on noise (e.g. a
# September call-up hitting .500 in 6 at-bats).
MIN_QUALIFYING_AT_BATS = 50
MIN_QUALIFYING_INNINGS = 10.0

METRIC_CONFIG = {
    "AVG":  {"table": "batting",  "keywords": ["batting average", "avg", "hitting average"], "better": "high"},
    "OBP":  {"table": "batting",  "keywords": ["on-base", "on base percentage", "obp"], "better": "high"},
    "SLG":  {"table": "batting",  "keywords": ["slugging"], "better": "high"},
    "HR":   {"table": "batting",  "keywords": ["home run", "homers", "hr"], "better": "high"},
    "RBI":  {"table": "batting",  "keywords": ["rbi", "runs batted in"], "better": "high"},
    "SB":   {"table": "batting",  "keywords": ["stolen base", "steals", "sb"], "better": "high"},
    "ERA":  {"table": "pitching", "keywords": ["era", "earned run average"], "better": "low"},
    "WHIP": {"table": "pitching", "keywords": ["whip"], "better": "low"},
    "W":    {"table": "pitching", "keywords": ["wins", "win"], "better": "high"},
    "SV":   {"table": "pitching", "keywords": ["save"], "better": "high"},
    "SO":   {"table": "pitching", "keywords": ["strikeout", "punchout"], "better": "high"},
}

# Literal direction words: these name a numeric direction directly, independent
# of which direction happens to be "good" for a given metric. "lowest WHIP"
# means ascending order full stop -- it should NOT get treated the same as
# "worst WHIP" just because both contain a low/bad-sounding word.
LITERAL_DIRECTION_WORDS = {
    "highest": "desc", "most": "desc",
    "lowest": "asc", "least": "asc", "fewest": "asc",
}
# Semantic best/worst words: these refer to performance quality, so they DO
# depend on the metric's "better" direction (worst ERA = highest number).
SEMANTIC_WORDS = {"worst": "worse"}


def _keyword_present(keyword: str, text: str) -> bool:
    """Whole-word match (plus optional trailing 's' for plurals like
    'strikeouts'/'home runs') so short keywords like 'hr' or 'era' don't
    false-positive inside unrelated words (e.g. 'era' inside 'overall')."""
    return re.search(r"\b" + re.escape(keyword) + r"s?\b", text) is not None


def detect_stat_leader(question: str, batting_df: pd.DataFrame, pitching_df: pd.DataFrame,
                        top_n: int = 3):
    """If the question is asking for a best/worst-at-a-stat leaderboard,
    compute it directly rather than leaving it to the model. Returns a
    formatted string, or None if this doesn't look like that kind of
    question (or the relevant table/column isn't available)."""
    q = question.lower()

    metric, config = None, None
    for m, cfg in METRIC_CONFIG.items():
        if any(_keyword_present(kw, q) for kw in cfg["keywords"]):
            metric, config = m, cfg
            break

    if metric is None:
        return None

    df = batting_df if config["table"] == "batting" else pitching_df
    if df.empty or metric not in df.columns:
        return None

    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if config["table"] == "batting" and "AtBats" in df.columns:
        df = df[df["AtBats"] >= MIN_QUALIFYING_AT_BATS]
        qualifier = f"min {MIN_QUALIFYING_AT_BATS} at-bats"
    elif config["table"] == "pitching" and "InningsPitched" in df.columns:
        innings = pd.to_numeric(df["InningsPitched"], errors="coerce")
        df = df[innings >= MIN_QUALIFYING_INNINGS]
        qualifier = f"min {MIN_QUALIFYING_INNINGS} innings pitched"
    else:
        qualifier = "no minimum sample filter available"

    if df.empty:
        return None

    # Direction resolution, in priority order:
    #   1. A literal direction word ("highest"/"lowest"/"most"/etc.) wins
    #      outright -- it's unambiguous regardless of the metric.
    #   2. Otherwise fall back to the metric's own "better" direction,
    #      flipped if "worst" is present.
    literal_hit = next((LITERAL_DIRECTION_WORDS[w] for w in LITERAL_DIRECTION_WORDS
                         if _keyword_present(w, q)), None)

    if literal_hit is not None:
        ascending = literal_hit == "asc"
    else:
        ascending = config["better"] == "low"
        if _keyword_present("worst", q):
            ascending = not ascending

    direction_label = "lowest" if ascending else "highest"
    ranked = df.sort_values(metric, ascending=ascending).head(top_n)

    lines = [f"Computed leaderboard -- {metric} ({direction_label} first, {qualifier}):"]
    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append(f"{i}. {row['Name']} -- {metric}: {row[metric]}")

    return "\n".join(lines)


# =======================================================================================
# STEP 2A-ii: STRUCTURED STATS RETRIEVAL
# =======================================================================================
def load_stats():
    """Load the batting/pitching tables. Returns empty DataFrames (not an
    error) if the DB hasn't been built yet -- generation should degrade
    gracefully, not crash."""
    batting_df = pd.read_json(BATTING_FILE) if os.path.exists(BATTING_FILE) else pd.DataFrame()
    pitching_df = pd.read_json(PITCHING_FILE) if os.path.exists(PITCHING_FILE) else pd.DataFrame()
    return batting_df, pitching_df


def get_stats_context(question: str, player_names: list) -> str:
    """Filter to named players if given, otherwise return full tables --
    both tables together are ~32 rows, small enough to pass in full. If the
    question looks like a best/worst-at-a-stat question, a deterministically
    computed leaderboard is prepended so the model doesn't have to (and
    can't get wrong by) scanning the raw table itself."""
    batting_df, pitching_df = load_stats()

    leader_note = detect_stat_leader(question, batting_df, pitching_df)

    if player_names:
        name_mask_b = batting_df["Name"].str.contains(
            "|".join(player_names), case=False, na=False
        ) if not batting_df.empty else pd.Series(dtype=bool)
        name_mask_p = pitching_df["Name"].str.contains(
            "|".join(player_names), case=False, na=False
        ) if not pitching_df.empty else pd.Series(dtype=bool)

        filtered_batting = batting_df[name_mask_b] if not batting_df.empty else batting_df
        filtered_pitching = pitching_df[name_mask_p] if not pitching_df.empty else pitching_df

        # If the named players didn't match anyone (nickname, misspelling, etc.),
        # fall back to the full tables rather than silently returning nothing.
        if filtered_batting.empty and filtered_pitching.empty:
            filtered_batting, filtered_pitching = batting_df, pitching_df
    else:
        filtered_batting, filtered_pitching = batting_df, pitching_df

    parts = []
    if leader_note:
        parts.append(leader_note)
    if not filtered_batting.empty:
        parts.append("BATTING STATS:\n" + filtered_batting.to_markdown(index=False))
    if not filtered_pitching.empty:
        parts.append("PITCHING STATS:\n" + filtered_pitching.to_markdown(index=False))

    return "\n\n".join(parts) if parts else "(no stats available)"


# =======================================================================================
# STEP 2B: UNSTRUCTURED ARTICLE RETRIEVAL
# =======================================================================================
def get_article_context(question: str, k: int = ARTICLE_TOP_K) -> tuple[str, list]:
    """Runs the existing retrieval.py search and formats results (with
    parent-context) into a single context block, plus a list of source
    dicts for citation display."""
    try:
        results = retrieve(question, k=k)
    except FileNotFoundError:
        return "(article index not built yet -- run retrieval.py --build)", []

    context_blocks = []
    sources = []
    for r in results:
        context_blocks.append(
            f"[{r['source']} — {r['published']} — {r['angle']}]\n{r['parent_text']}"
        )
        sources.append({
            "type": "article",
            "source": r["source"],
            "published": r["published"],
            "angle": r["angle"],
            "url": r["url"],
            "score": round(r["score"], 3),
        })

    return "\n\n---\n\n".join(context_blocks), sources


# =======================================================================================
# STEP 3: ANSWER GENERATION
# =======================================================================================
GENERATION_SYSTEM_PROMPT = """You are a Mets analyst answering questions using ONLY the \
context provided below. Follow these rules:

- Ground every specific claim (a stat, a trade, a roster move, a prediction) in the \
provided context. Don't use outside knowledge of players or trades beyond what's given.
- When you cite a stat, name the player. When you cite an article claim, name the \
source and date (e.g. "per CBS Sports, June 25").
- If the stats context includes a line starting with "Computed leaderboard", that \
ranking was computed directly from the data and is authoritative -- use it as-is for \
best/worst/highest/lowest questions rather than re-deriving your own ranking by eye \
from the raw table.
- If the context doesn't cover something the question asks about, say so plainly \
instead of guessing or filling the gap with general baseball knowledge.
- Be direct and concise -- this is a stats/news assistant, not a storyteller."""


def generate_answer(question: str, stats_context: str, article_context: str) -> str:
    user_content = (
        f"QUESTION: {question}\n\n"
        f"=== STATS CONTEXT ===\n{stats_context}\n\n"
        f"=== ARTICLE CONTEXT ===\n{article_context}"
    )
    response = get_client().chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.message.content


# =======================================================================================
# STEP 4: PUBLIC ENTRY POINT
# =======================================================================================
def answer_question(question: str, verbose: bool = False) -> dict:
    """Full pipeline: route -> retrieve -> generate. Returns a dict with the
    answer text plus a structured list of sources actually used, so a UI
    layer can render citations separately from the prose answer."""
    route = route_query(question)
    if verbose:
        print(f"  route: {route}")

    stats_context = get_stats_context(question, route["player_names"]) if route["need_stats"] else "(not needed for this question)"
    article_context, article_sources = (
        get_article_context(question) if route["need_articles"] else ("(not needed for this question)", [])
    )

    answer = generate_answer(question, stats_context, article_context)

    return {
        "answer": answer,
        "route": route,
        "sources": article_sources,
    }


# =======================================================================================
# STEP 5: CLI FOR QUICK TESTING (no GUI)
# =======================================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ask the Mets RAG pipeline a question.")
    parser.add_argument("question", type=str, help="Question to ask.")
    parser.add_argument("--verbose", action="store_true", help="Print routing decision.")
    args = parser.parse_args()

    result = answer_question(args.question, verbose=args.verbose)
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - [{s['source']}, {s['published']}] {s['angle']} ({s['url']})")


if __name__ == "__main__":
    main()
