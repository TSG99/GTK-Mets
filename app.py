"""
Mets RAG App (Gradio UI)
---------------------------
Chat-style front end over rag_pipeline.py. Ask a question, get a grounded
answer, see exactly which stats rows and/or articles were used to write it.

Runs on a local Ollama server -- no API keys
Ollama is needed though

Usage:
    ollama pull qwen2.5:14b
    ollama pull nomic-embed-text
    ollama serve                      (if not already running)
    python retrieval.py --build       (only needed once, or after re-scraping)
    python app.py
"""

# =======================================================================================
# STEP 0: IMPORTS
# =======================================================================================
import gradio as gr

from rag_pipeline import answer_question

EXAMPLE_QUESTIONS = [
    "How is Juan Soto hitting this year?",
    "What's the Mets' outlook heading into the trade deadline?",
    "Who's been the most reliable arm out of the bullpen?",
    "How does the rotation look compared to preseason expectations?",
]


# =======================================================================================
# STEP 1: FORMAT A SOURCES BLOCK FOR DISPLAY UNDERNEATH THE ANSWER
# =======================================================================================
def format_sources(route: dict, sources: list) -> str:
    lines = []

    if route.get("need_stats"):
        players = ", ".join(route.get("player_names", [])) or "full roster"
        lines.append(f"**Stats used:** {players}")

    if sources:
        lines.append("**Articles used:**")
        for s in sources:
            lines.append(f"- [{s['source']}, {s['published']}] {s['angle']} (relevance: {s['score']})")
    elif route.get("need_articles"):
        lines.append("**Articles used:** none matched (index may not be built)")

    return "\n".join(lines) if lines else "_No sources needed for this question._"

# =======================================================================================
# STEP 2: CHAT CALLBACK
# =======================================================================================
def respond(message: str, history: list):
    result = answer_question(message)
    sources_md = format_sources(result["route"], result["sources"])
    full_reply = f"{result['answer']}\n\n---\n{sources_md}"
    return full_reply

# =======================================================================================
# STEP 3: BUILD THE INTERFACE
# =======================================================================================
def build_app():
    demo = gr.ChatInterface(
        fn=respond,
        title="🧢 Mets RAG Assistant",
        description=(
            "Ask about current Mets stats, roster moves, or trade deadline outlook. "
            "Answers are grounded in a stats DB (MLB StatsAPI) and a small hand-picked "
            "article corpus -- not general knowledge."
        ),
        examples=EXAMPLE_QUESTIONS,
    )
    return demo

# =======================================================================================
# STEP 4: MAIN
# =======================================================================================
if __name__ == "__main__":
    app = build_app()
    app.launch()
