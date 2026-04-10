from __future__ import annotations

from pathlib import Path
import re

import streamlit as st
from dotenv import load_dotenv

from src.agent import AgenticRAG
from src.config import load_settings
from src.llm import build_llm
from src.rag import SimpleRAG


def normalize_chunk_text(text: str, max_chars: int = 700) -> str:
    # Convert markdown-ish content to plain preview text for stable UI rendering.
    cleaned = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip() + "..."
    return cleaned


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-a: #071325;
            --bg-b: #0e1d2f;
            --card: rgba(19, 33, 56, 0.72);
            --line: rgba(150, 190, 230, 0.24);
            --brand: #17c9b2;
            --brand-2: #35b6ff;
            --text: #eaf2ff;
            --muted: #b5c6de;
        }

        .stApp {
            background: radial-gradient(1200px 500px at 10% 0%, #183056 0%, rgba(24, 48, 86, 0.05) 55%),
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
            background: linear-gradient(135deg, rgba(23, 201, 178, 0.12), rgba(53, 182, 255, 0.08));
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .card {
            border: 1px solid var(--line);
            background: var(--card);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
        }

        .section-title {
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: var(--brand);
            margin-bottom: 0.3rem;
        }

        .metric-chip {
            display: inline-block;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.28rem 0.64rem;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
            color: var(--muted);
            font-size: 0.78rem;
            font-family: 'IBM Plex Mono', monospace;
            background: rgba(20, 34, 58, 0.65);
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(14, 28, 46, 0.95), rgba(10, 20, 34, 0.95));
            border-right: 1px solid var(--line);
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid transparent;
            background: linear-gradient(135deg, var(--brand), var(--brand-2));
            color: #03111b;
            font-weight: 800;
            padding: 0.62rem 1rem;
        }

        .stButton > button:hover {
            filter: brightness(1.06);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_filename(name: str) -> str:
    base = Path(name).name
    # Keep filenames simple/safe for local writes.
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "uploaded_note.md"


def main() -> None:
    load_dotenv()
    settings = load_settings()
    st.set_page_config(page_title="Agentic RAG Project", page_icon="AI", layout="wide")
    inject_custom_css()
    st.markdown(
        """
        <div class="hero">
          <h1 style="margin:0;">Agentic RAG Studio</h1>
          <p style="margin:0.35rem 0 0 0;color:#b5c6de;">Prompt engineering + retrieval + planning/reflection in one workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rag = SimpleRAG(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    data_path = Path(__file__).parent / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    data_dir = str(data_path)

    with st.sidebar:
        st.subheader("Runtime")

        runtime_api_key = st.text_input(
            "OPENAI_API_KEY (optional)",
            type="password",
            placeholder="Paste key to use real model",
            help="If empty, app runs in MockLLM mode.",
        ).strip()
        effective_api_key = runtime_api_key or settings.openai_api_key

        st.write(f"Model: {settings.openai_model if effective_api_key else 'MockLLM'}")
        st.write(f"Top-k retrieval: {settings.top_k}")

        st.markdown("### Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload .md or .txt files",
            type=["md", "txt"],
            accept_multiple_files=True,
            help="Files are saved into the local data/ folder.",
        )

        if st.button("Save Files to data/", use_container_width=True):
            if not uploaded_files:
                st.warning("Select at least one .md or .txt file first.")
            else:
                saved = 0
                for up in uploaded_files:
                    fname = safe_filename(up.name)
                    out_path = data_path / fname
                    out_path.write_bytes(up.getbuffer())
                    saved += 1
                st.success(f"Saved {saved} file(s).")

        existing_files = sorted(
            [p.name for p in data_path.glob("*.md")] + [p.name for p in data_path.glob("*.txt")]
        )
        if existing_files:
            st.caption("Indexed source files")
            for name in existing_files[:12]:
                st.write(f"- {name}")
            if len(existing_files) > 12:
                st.write(f"... and {len(existing_files) - 12} more")
        else:
            st.info("No files in data/ yet.")

        st.info("Upload, then click Save Files to data/. The app auto-reindexes on rerun.")

    chunk_count = rag.index_folder(data_dir)
    llm = build_llm(effective_api_key, settings.openai_model)
    agent = AgenticRAG(llm=llm, rag=rag, top_k=settings.top_k)

    if not effective_api_key:
        st.warning(
            "Running in MOCK MODE because OPENAI_API_KEY is not set. "
            "Outputs are demo-quality placeholders, not real LLM generations."
        )

    st.caption(f"Indexed chunks: {chunk_count}")

    st.markdown(
        """
        <span class="metric-chip">Agentic: Plan</span>
        <span class="metric-chip">Agentic: Draft</span>
        <span class="metric-chip">Agentic: Reflect</span>
        <span class="metric-chip">RAG: Top-k Context</span>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Ask your project assistant",
        value="How should I evaluate an agentic RAG system for my course project?",
        height=120,
    )

    if st.button("Run Agentic Workflow", type="primary"):
        with st.spinner("Planning, retrieving, drafting, and reflecting..."):
            result = agent.ask(question)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Final Answer</div>', unsafe_allow_html=True)
            st.write(result.final)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Show Plan", expanded=False):
                st.write(result.plan)

            with st.expander("Show First Draft", expanded=False):
                st.write(result.draft)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Retrieved Chunks</div>', unsafe_allow_html=True)
            if not result.context:
                st.warning("No context retrieved")
            else:
                for i, chunk in enumerate(result.context, start=1):
                    with st.expander(f"[{i}] {chunk.source}", expanded=(i == 1)):
                        st.text(normalize_chunk_text(chunk.text))
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
