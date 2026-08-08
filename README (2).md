# Mets RAG System

A hybrid retrieval-augmented system for answering natural language questions about the New York Mets, combining unstructured article retrieval with structured statistical data from the MLB Stats API. Runs fully locally.

## Overview

Most baseball Q&A falls into two buckets: **qualitative** ("what did the manager say about the bullpen") and **quantitative** ("what's Pete Alonso's OPS this season"). A single RAG pipeline handles the first well and the second poorly — local LLMs are prone to positional/recency bias when asked to reason over numeric leaderboards from retrieved text. This system routes each query to the retrieval path suited to it, and combines both when a question needs both.

## Architecture

```
Query
  │
  ▼
[1] Query Classifier (LLM)
  │   → qual | quan | both
  ▼
[2] Entity/Database Check
  │   → does the query map to a known player/team + supported StatsAPI endpoint?
  ▼
  ├── quan or both, DB match found ──► [3] Structured Retrieval
  │                                        - LLM generates structured query params
  │                                        - Params validated against endpoint schema
  │                                        - pandas layer pulls + formats stat table
  │
  ├── qual or both, (DB check no, or qual requested) ──► [4] Unstructured Retrieval
  │                                        - paragraph-level semantic search
  │                                        - falls back to section-level if below
  │                                          similarity threshold
  │                                        - relevancy-check agent (LLM) validates
  │                                          the retrieved chunk actually answers
  │                                          the query; if not → "No data on this
  │                                          question"
  │
  ▼
[5] Output Assembly
  │
  ├── 3a. RAG only        → cited paragraph/section text, no LLM-generated claims
  ├── 3b. Stats only      → structured stat table, no LLM-generated claims
  └── 3c. RAG + Stats     → synthesis step (LLM) combines both into a single
                             answer, with the stat table and source paragraph
                             shown alongside as citations (not just concatenated)
```

### Degrade paths

- **quan-only, no DB match:** falls through to unstructured retrieval rather than dead-ending, since the article corpus may still cover it in prose (e.g., a stat not exposed by StatsAPI).
- **"both" requested, one side fails relevancy/validation:** falls back to the single successful branch's output mode (3a or 3b) rather than presenting a half-empty combined answer, with a note on which half was unavailable.
- **both branches fail:** returns "Currently no data on this question."

### Design rationale

- **Router before retrieval, not after:** avoids running both retrieval paths on every query. Cuts unnecessary API/LLM calls and keeps latency down.
- **Deterministic stat layer:** stats are pulled via pandas against validated query params, not generated freeform by the LLM — this is the fix for positional bias observed when local models (qwen2.5:14b) were asked to reason directly over leaderboard-style data.
- **Relevancy-check agent on both branches:** a generated stats query can be structurally valid but wrong in context (wrong season, wrong split), same failure mode as a semantically-near-but-irrelevant RAG chunk. Both get validated before being surfaced.
- **3a/3c distinction:** RAG-only output is deliberately citation-only with no LLM claims layered on top — this is a design choice (grounding over generation), not a missing feature.

### Known tradeoffs

This pipeline costs a minimum of 3 LLM calls per query (classify → generate/retrieve → relevancy check), more for the "both" path with synthesis. Acceptable for a portfolio-scale demo; a production version would cache classification results for repeated query patterns and consider a smaller/distilled model for the classifier and relevancy-check steps specifically, reserving the larger model for synthesis.

## Stack

- **LLM inference:** Ollama, `qwen2.5:14b`
- **Embeddings:** `nomic-embed-text`
- **Structured data:** MLB StatsAPI + pandas
- **UI:** Gradio
- **Vector store:** *(update this line to match what you're actually running — Chroma is assumed in requirements.txt below; swap if you're on FAISS or something else)*

## Project Structure

```
mets_rag/
├── router/              # query classification + entity/DB check
├── retrieval/
│   ├── unstructured/    # paragraph/section RAG + relevancy check
│   └── structured/      # StatsAPI query generation, validation, pandas layer
├── synthesis/           # combines RAG + stats output for 3c
├── ui/                  # Gradio app
├── data/                # article corpus, embeddings cache
└── README.md
```

## Setup

```bash
# 1. Install Ollama and pull models
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the app
python ui/app.py
```

## Status

Active build. Router logic (classification, DB check, relevancy validation, degrade paths) is designed and documented here; see `DESIGN_DECISIONS.md` for the full rationale writeup used for interview prep.
