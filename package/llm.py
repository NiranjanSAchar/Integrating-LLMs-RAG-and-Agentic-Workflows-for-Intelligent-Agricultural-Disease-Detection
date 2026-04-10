"""
llm.py — LLM client abstraction for AgriSense.
Provides a real OpenAI client and a MockLLM for offline/demo use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


# ─── Base interface ───────────────────────────────────────────────────────────

class BaseLLM(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a system + user prompt and return the completion string."""


# ─── OpenAI client ────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # lazy import so MockLLM works without openai
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


# ─── Mock LLM (offline / demo mode) ──────────────────────────────────────────

class MockLLM(BaseLLM):
    """
    Returns structured placeholder responses so the full agentic pipeline
    runs end-to-end without an API key. Useful for demos and offline testing.
    """

    def complete(self, system: str, user: str) -> str:
        prompt_lower = user.lower()

        # Plan stage
        if "plan" in system.lower() or "strategy" in system.lower():
            return (
                "**Diagnostic Plan (MockLLM)**\n\n"
                "1. Identify visible symptoms from the query (spots, lesions, colour changes).\n"
                "2. Match symptom pattern to known diseases in the knowledge base.\n"
                "3. Retrieve treatment protocols for the matched disease.\n"
                "4. Provide organic and chemical treatment options with dosage guidance.\n"
                "5. Recommend preventive measures for future seasons."
            )

        # Reflect stage
        if "reflect" in system.lower() or "critique" in system.lower():
            return (
                "**Reflection (MockLLM)**\n\n"
                "The draft answer correctly identifies the disease category and cites "
                "retrieved context. Dosage figures should be verified against ICAR guidelines "
                "before field application. The organic option is appropriate for early-stage "
                "infection. Overall factual accuracy: satisfactory."
            )

        # Draft / answer stage
        if any(w in prompt_lower for w in ["tomato", "blight", "spot", "mildew", "rust", "blast"]):
            return (
                "**Agricultural Advisory (MockLLM — demo mode)**\n\n"
                "Based on the described symptoms, this appears consistent with **Early Blight** "
                "(*Alternaria solani*), one of the most common tomato foliar diseases.\n\n"
                "**Symptoms:** Dark brown spots with concentric rings and yellow halos on older leaves, "
                "typically progressing upward from the base of the plant.\n\n"
                "**Organic treatment:**\n"
                "- Neem oil spray (2–3 ml/L water) every 7 days.\n"
                "- Copper-based fungicide (Bordeaux mixture 1%) as a preventive spray.\n\n"
                "**Chemical treatment:**\n"
                "- Mancozeb 75% WP at 2 g/L, or Chlorothalonil 75% WP at 2 g/L.\n"
                "- Apply every 10–14 days; rotate fungicides to prevent resistance.\n\n"
                "**Prevention:**\n"
                "- Remove and destroy infected leaves immediately.\n"
                "- Avoid overhead irrigation; water at the base.\n"
                "- Maintain plant spacing for good airflow.\n\n"
                "_Source: AgriSense knowledge base (MockLLM placeholder — connect OpenAI API for real generations)._"
            )

        # Generic fallback
        return (
            "**AgriSense Advisory (MockLLM — demo mode)**\n\n"
            "This is a placeholder response. To get real LLM-generated answers, "
            "add your `OPENAI_API_KEY` in the sidebar or your `.env` file.\n\n"
            "The agentic pipeline (Plan → Draft → Reflect) ran successfully in mock mode. "
            "All three stages are functional and ready for evaluation once an API key is provided."
        )


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_llm(api_key: str, model: str) -> BaseLLM:
    """Return an OpenAILLM if an API key is available, else MockLLM."""
    if api_key:
        return OpenAILLM(api_key=api_key, model=model)
    return MockLLM()
