"""
agent.py — Agentic workflow controller for AgriSense.
Implements the Plan → Draft → Reflect pipeline described in §III-D of the paper.

Design patterns used (Singh et al., 2024, DOI: 10.1109/AIIoT61789.2024.10578990):
  - Planning:    decomposes the query into a structured diagnostic strategy
  - Tool use:    calls the RAG retriever as an external tool
  - Reflection:  self-critique pass to improve factual accuracy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.llm import BaseLLM
from src.rag import Chunk, SimpleRAG


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Holds all intermediate and final outputs of one agentic run."""
    plan: str = ""
    context: List[Chunk] = field(default_factory=list)
    draft: str = ""
    reflection: str = ""
    final: str = ""


# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PLANNER = """\
You are an expert agricultural AI assistant helping diagnose and treat plant diseases.
Your task is to create a short, structured diagnostic plan (3–5 steps) for answering
the user's query. The plan will guide retrieval and answer generation.
Be specific about what information to look for in the knowledge base.
"""

SYSTEM_DRAFTER = """\
You are an expert agricultural AI assistant. Answer the user's query using ONLY
the provided context from the agronomic knowledge base. Cite context sources where
relevant using [Source: filename]. If the context does not contain enough information,
say so clearly rather than guessing. Be concise, factual, and actionable.
"""

SYSTEM_REFLECTOR = """\
You are a quality reviewer for agricultural AI advisory responses.
Review the draft answer for:
1. Factual accuracy — are all claims supported by the retrieved context?
2. Completeness — does it address all parts of the query?
3. Safety — are any chemical dosages or treatment advice potentially harmful if misapplied?
If the draft is satisfactory, output it unchanged.
If improvements are needed, output a corrected version with your changes.
"""

SYSTEM_DIRECT = """\
You are an expert agricultural AI assistant. Answer the user's query to the best
of your knowledge. Be factual and acknowledge uncertainty where appropriate.
"""

SYSTEM_RAG_ONLY = """\
You are an expert agricultural AI assistant. Answer the user's query using the
provided context from the agronomic knowledge base. Cite sources where relevant.
If context is insufficient, acknowledge it.
"""


# ─── Agent ────────────────────────────────────────────────────────────────────

class AgenticRAG:
    """
    Orchestrates the three-stage agentic workflow:
      Stage 1 — Plan:    LLM generates a diagnostic strategy
      Stage 2 — Draft:   RAG retrieval + grounded LLM generation
      Stage 3 — Reflect: LLM self-critique and refinement
    """

    def __init__(self, llm: BaseLLM, rag: SimpleRAG, top_k: int = 4) -> None:
        self._llm = llm
        self._rag = rag
        self._top_k = top_k

    def _format_context(self, chunks: List[Chunk]) -> str:
        if not chunks:
            return "No relevant context found in the knowledge base."
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(f"[{i}] Source: {chunk.source}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    # ── Stage 1: Plan ─────────────────────────────────────────────────────────

    def _plan(self, query: str) -> str:
        return self._llm.complete(
            system=SYSTEM_PLANNER,
            user=f"User query: {query}\n\nCreate a diagnostic plan.",
        )

    # ── Stage 2: Draft ────────────────────────────────────────────────────────

    def _draft(self, query: str, plan: str, context: List[Chunk]) -> str:
        ctx_text = self._format_context(context)
        user_prompt = (
            f"Diagnostic plan:\n{plan}\n\n"
            f"Retrieved knowledge base context:\n{ctx_text}\n\n"
            f"User query: {query}\n\n"
            f"Write a complete, grounded answer citing the context."
        )
        return self._llm.complete(system=SYSTEM_DRAFTER, user=user_prompt)

    # ── Stage 3: Reflect ──────────────────────────────────────────────────────

    def _reflect(self, query: str, draft: str, context: List[Chunk]) -> str:
        ctx_text = self._format_context(context)
        user_prompt = (
            f"User query: {query}\n\n"
            f"Retrieved context:\n{ctx_text}\n\n"
            f"Draft answer to review:\n{draft}\n\n"
            f"Review and output the final corrected answer."
        )
        return self._llm.complete(system=SYSTEM_REFLECTOR, user=user_prompt)

    # ── Public entry point ────────────────────────────────────────────────────

    def ask(self, query: str, mode: str = "agentic") -> AgentResult:
        """
        Run the agentic workflow in one of three experiment modes:
          - 'agentic'  : full Plan → RAG → Draft → Reflect pipeline
          - 'rag_only' : RAG retrieval + single-pass generation (no plan/reflect)
          - 'baseline' : vanilla LLM with no retrieval (control condition)
        """
        result = AgentResult()

        if mode == "baseline":
            # Control condition — no retrieval, no planning
            result.final = self._llm.complete(
                system=SYSTEM_DIRECT,
                user=query,
            )
            return result

        if mode == "rag_only":
            # RAG pipeline without agentic orchestration
            result.context = self._rag.retrieve(query, top_k=self._top_k)
            ctx_text = self._format_context(result.context)
            result.draft = self._llm.complete(
                system=SYSTEM_RAG_ONLY,
                user=f"Context:\n{ctx_text}\n\nQuery: {query}",
            )
            result.final = result.draft
            return result

        # Full agentic mode: Plan → Retrieve → Draft → Reflect
        result.plan = self._plan(query)
        result.context = self._rag.retrieve(query, top_k=self._top_k)
        result.draft = self._draft(query, result.plan, result.context)
        result.final = self._reflect(query, result.draft, result.context)
        result.reflection = result.final  # reflection output IS the final answer

        return result
