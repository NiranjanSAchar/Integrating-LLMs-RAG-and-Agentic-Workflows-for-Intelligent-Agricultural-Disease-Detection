"""
AgriSense — Intelligent Plant Disease Detection with Agentic RAG
================================================================
Course: UE23CS342BA9 — Generative AI and its Applications

System described in:
  "Integrating LLMs, RAG, and Agentic Workflows for
   Intelligent Agricultural Disease Detection"

Implements:
  - TF-IDF retrieval over an agronomic knowledge base (RAG pipeline, §III-C)
  - Plan → Draft → Reflect agentic workflow (§III-D)
  - Three experiment modes: Baseline / RAG-only / Agentic RAG (§IV)
"""

from __future__ import annotations

import time
from pathlib import Path
import re

import streamlit as st
from dotenv import load_dotenv

from src.agent import AgenticRAG
from src.config import load_settings
from src.llm import build_llm
from src.rag import SimpleRAG


# ─── UI Helpers ───────────────────────────────────────────────────────────────

def normalize_chunk_text(text: str, max_chars: int = 700) -> str:
    """Strip markdown headings and collapse whitespace for chunk preview."""
    cleaned = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip() + "…"
    return cleaned


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-a: #071c0f;
            --bg-b: #0d2016;
            --card: rgba(15, 35, 22, 0.80);
            --line: rgba(100, 200, 130, 0.22);
            --brand: #22c55e;
            --brand-2: #34d399;
            --text: #eafaf1;
            --muted: #a3c9b0;
        }

        .stApp {
            background: radial-gradient(1200px 500px at 10% 0%, #0d3320 0%, rgba(13, 50, 32, 0.05) 55%),
                        linear-gradient(145deg, var(--bg-a), var(--bg-b));
            color: var(--text);
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .main .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            letter-spacing: -0.02em;
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.10), rgba(52, 211, 153, 0.07));
            border-radius: 18px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 1rem;
        }

        .card {
            border: 1px solid var(--line);
            background: var(--card);
            border-radius: 16px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
        }

        .section-title {
            font-size: 0.88rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--brand);
            margin-bottom: 0.35rem;
        }

        .metric-chip {
            display: inline-block;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.28rem 0.70rem;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
            color: var(--muted);
            font-size: 0.76rem;
            font-family: 'IBM Plex Mono', monospace;
            background: rgba(10, 28, 18, 0.65);
        }

        .experiment-badge {
            display: inline-block;
            border-radius: 8px;
            padding: 0.22rem 0.60rem;
            font-size: 0.78rem;
            font-weight: 700;
            font-family: 'IBM Plex Mono', monospace;
            margin-bottom: 0.5rem;
        }
        .badge-baseline  { background: rgba(234,179,8,0.18);  color: #fde047; border: 1px solid rgba(234,179,8,0.3); }
        .badge-rag       { background: rgba(59,130,246,0.18); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
        .badge-agentic   { background: rgba(34,197,94,0.18);  color: #86efac; border: 1px solid rgba(34,197,94,0.3); }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 28, 18, 0.97), rgba(7, 18, 11, 0.97));
            border-right: 1px solid var(--line);
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid transparent;
            background: linear-gradient(135deg, var(--brand), var(--brand-2));
            color: #021a08;
            font-weight: 800;
            padding: 0.62rem 1rem;
        }

        .stButton > button:hover {
            filter: brightness(1.08);
            border: 1px solid rgba(255,255,255,0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_filename(name: str) -> str:
    """Sanitize an uploaded filename for safe local writes."""
    base = Path(name).name
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "uploaded_doc.md"


def latency_badge(seconds: float) -> str:
    color = "#86efac" if seconds < 3 else "#fde047" if seconds < 6 else "#fca5a5"
    return (
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;'
        f'color:{color};">⏱ {seconds:.2f}s</span>'
    )


# ─── Experiment Modes ─────────────────────────────────────────────────────────

EXPERIMENT_MODES = {
    "Agentic RAG  (Plan → Draft → Reflect)": "agentic",
    "RAG-Only  (Retrieve + Single-pass Generate)": "rag_only",
    "Baseline  (Vanilla LLM, no retrieval)": "baseline",
}

EXPERIMENT_DESCRIPTIONS = {
    "agentic": (
        "Full system from §III of the paper. The agent plans a diagnostic strategy, "
        "retrieves top-k agronomic chunks, drafts a grounded answer with citations, "
        "then reflects and refines for factual accuracy."
    ),
    "rag_only": (
        "Retrieval-Augmented Generation only (§III-C). Fetches top-k chunks and "
        "generates a single-pass answer. No planning or reflection stage."
    ),
    "baseline": (
        "Vanilla LLM prompting (§IV-B control). No retrieval context is injected. "
        "Use this to measure the hallucination rate baseline reported in Table II."
    ),
}


# ─── Main App ─────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    settings = load_settings()

    st.set_page_config(
        page_title="AgriSense — Agentic RAG",
        page_icon="🌿",
        layout="wide",
    )
    inject_custom_css()

    st.markdown(
        """
        <div class="hero">
          <h1 style="margin:0;">🌿 AgriSense — Plant Disease Detection</h1>
          <p style="margin:0.4rem 0 0.1rem 0; color:#a3c9b0; font-size:0.95rem;">
            Intelligent advisory via <strong>Prompt Engineering</strong> ·
            <strong>RAG</strong> · <strong>Agentic Workflow</strong>
          </p>
          <p style="margin:0; color:#6b9f7e; font-size:0.78rem;">
            UE23CS342BA9 — Generative AI and its Applications
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("🔑 Runtime")

        runtime_api_key = st.text_input(
            "OPENAI_API_KEY (optional)",
            type="password",
            placeholder="Paste key — or leave blank for MockLLM",
            help="Without a key the app runs in demo/mock mode.",
        ).strip()
        effective_api_key = runtime_api_key or settings.openai_api_key

        active_model = settings.openai_model if effective_api_key else "MockLLM"
        st.write(f"**Model:** `{active_model}`")
        st.write(f"**Top-k retrieval:** `{settings.top_k}`")
        st.write(f"**Chunk size:** `{settings.chunk_size}` tokens")

        st.divider()
        st.markdown("### 📂 Knowledge Base")
        st.caption(
            "Upload agronomic `.md` / `.txt` files — disease guides, pesticide "
            "protocols, ICAR advisories — to populate the RAG retrieval index."
        )

        uploaded_files = st.file_uploader(
            "Upload .md or .txt files",
            type=["md", "txt"],
            accept_multiple_files=True,
            help="Files are saved into the local data/ folder and indexed on next run.",
        )

        data_path = Path(__file__).parent / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        if st.button("💾 Save Files to data/", use_container_width=True):
            if not uploaded_files:
                st.warning("Select at least one file first.")
            else:
                saved = 0
                for up in uploaded_files:
                    fname = safe_filename(up.name)
                    (data_path / fname).write_bytes(up.getbuffer())
                    saved += 1
                st.success(f"Saved {saved} file(s) — rerun to reindex.")

        existing_files = sorted(
            [p.name for p in data_path.glob("*.md")]
            + [p.name for p in data_path.glob("*.txt")]
        )
        if existing_files:
            st.caption(f"Indexed files ({len(existing_files)})")
            for name in existing_files[:10]:
                st.write(f"  `{name}`")
            if len(existing_files) > 10:
                st.write(f"  … and {len(existing_files) - 10} more")
        else:
            st.info("No files in `data/` yet. Starter `knowledge_base.md` will be used.")

        st.divider()
        st.markdown("### 🧪 Experiment Mode")
        st.caption(
            "Switch modes to reproduce Table II from the paper. "
            "Record latency and quality scores in `reports/`."
        )
        mode_label = st.selectbox(
            "Select configuration",
            options=list(EXPERIMENT_MODES.keys()),
            index=0,
        )
        experiment_mode = EXPERIMENT_MODES[mode_label]

        badge_cls = {"agentic": "badge-agentic", "rag_only": "badge-rag", "baseline": "badge-baseline"}[experiment_mode]
        badge_label = {"agentic": "Agentic RAG", "rag_only": "RAG-Only", "baseline": "Baseline LLM"}[experiment_mode]
        st.markdown(
            f'<div class="experiment-badge {badge_cls}">{badge_label}</div>',
            unsafe_allow_html=True,
        )
        st.caption(EXPERIMENT_DESCRIPTIONS[experiment_mode])

    # ── Index and build agent ─────────────────────────────────────────────────
    rag = SimpleRAG(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunk_count = rag.index_folder(str(data_path))
    llm = build_llm(effective_api_key, settings.openai_model)
    agent = AgenticRAG(llm=llm, rag=rag, top_k=settings.top_k)

    if not effective_api_key:
        st.warning(
            "🔶 **MockLLM mode** — no `OPENAI_API_KEY` detected. "
            "Outputs are demo placeholders. Add your key in the sidebar or `.env` for real generations."
        )

    # ── Status chips ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <span class="metric-chip">📄 {chunk_count} chunks indexed</span>
        <span class="metric-chip">🔍 Top-{settings.top_k} retrieval</span>
        <span class="metric-chip">🤖 {active_model}</span>
        <span class="metric-chip">Mode: {badge_label}</span>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Query input ───────────────────────────────────────────────────────────
    st.markdown("#### 🌾 Ask your Agricultural Disease Advisor")
    st.caption(
        "Enter a query about a crop disease, symptoms, or treatment. "
        "The system will retrieve relevant agronomic knowledge and generate a grounded answer."
    )

    EXAMPLE_QUERIES = [
        "My tomato leaves show dark spots with yellow halos. What disease is this and how do I treat it?",
        "What are the best organic treatments for powdery mildew on cucumber plants?",
        "How should I evaluate an agentic RAG system for my course project?",
        "What is the difference between early blight and late blight in tomatoes?",
        "Which fungicides are recommended for rice blast disease management?",
    ]

    selected_example = st.selectbox(
        "Or pick an example query",
        options=["— type your own below —"] + EXAMPLE_QUERIES,
        index=0,
    )

    default_text = (
        selected_example
        if selected_example != "— type your own below —"
        else "My tomato leaves show dark brown spots with yellow halos. What disease could this be and how do I treat it organically?"
    )

    question = st.text_area(
        "Your query",
        value=default_text,
        height=110,
        label_visibility="collapsed",
    )

    run_btn = st.button("🚀 Run Agentic Workflow", type="primary", use_container_width=False)

    if run_btn:
        if not question.strip():
            st.error("Please enter a question before running.")
            st.stop()

        t_start = time.perf_counter()
        with st.spinner("🌿 Planning → Retrieving → Drafting → Reflecting…"):
            result = agent.ask(question, mode=experiment_mode)
        elapsed = time.perf_counter() - t_start

        st.divider()
        col_answer, col_context = st.columns([2, 1])

        # ── Answer column ──────────────────────────────────────────────────
        with col_answer:
            st.markdown(
                f'<div class="card">'
                f'<div class="section-title">✅ Final Answer</div>',
                unsafe_allow_html=True,
            )
            st.write(result.final)
            st.markdown(
                f'{latency_badge(elapsed)}&nbsp;&nbsp;'
                f'<span style="font-size:0.75rem;color:#6b9f7e;">experiment: <code>{experiment_mode}</code></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Only show plan/draft for agentic mode
            if experiment_mode == "agentic":
                with st.expander("🗺 Show Plan (Stage 1)", expanded=False):
                    st.markdown(
                        "_The agent decomposed the query into a structured diagnostic strategy "
                        "before retrieval — see §III-D of the paper._"
                    )
                    st.write(result.plan)

                with st.expander("📝 Show First Draft (Stage 2)", expanded=False):
                    st.markdown(
                        "_Raw generation after retrieval, before the reflection pass._"
                    )
                    st.write(result.draft)

            elif experiment_mode == "rag_only":
                with st.expander("📝 Show Draft (single-pass)", expanded=False):
                    st.write(result.draft)

            # Experiment logger
            with st.expander("📊 Log this result (for Table II)", expanded=False):
                st.markdown(
                    "Rate this response and record in `reports/` for your paper:\n\n"
                    "| Metric | Value |\n|---|---|\n"
                    f"| Experiment mode | `{experiment_mode}` |\n"
                    f"| Latency (s) | `{elapsed:.2f}` |\n"
                    "| Hallucination? (Y/N) | — |\n"
                    "| Factual correctness (1–5) | — |\n"
                    "| Answer relevance (1–5) | — |\n"
                    "| Citation correctness (Y/N) | — |\n"
                )

        # ── Context column ─────────────────────────────────────────────────
        with col_context:
            st.markdown(
                '<div class="card">'
                '<div class="section-title">📚 Retrieved Chunks (RAG Context)</div>',
                unsafe_allow_html=True,
            )

            if experiment_mode == "baseline":
                st.info("No context retrieved in Baseline mode. This is the control condition for Table II.")
            elif not result.context:
                st.warning(
                    "No chunks retrieved — add more agronomic documents to `data/` "
                    "and rerun to populate the knowledge base."
                )
            else:
                for i, chunk in enumerate(result.context, start=1):
                    with st.expander(f"[{i}] {chunk.source}", expanded=(i == 1)):
                        st.text(normalize_chunk_text(chunk.text))

            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
