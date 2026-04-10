# AgriSense: Intelligent Plant Disease Detection with Agentic RAG

A course project for **UE23CS342BA9 — Generative AI and its Applications**.

This system integrates three Generative AI paradigms to deliver disease diagnosis and treatment recommendations for crops:

- **CNN-based Vision** — ResNet50 fine-tuned on PlantVillage for plant disease classification
- **Retrieval-Augmented Generation (RAG)** — TF-IDF retrieval over an agronomic knowledge base to ground LLM responses
- **Agentic Workflow** — Plan → Draft → Reflect pipeline for high-quality, verifiable advisory output

---

## Research Context

This project implements and extends the system described in our course research paper:

> *Integrating LLMs, RAG, and Agentic Workflows for Intelligent Agricultural Disease Detection*

Key references:
- Li et al., "Plant Disease Detection and Classification by Deep Learning—A Review," *IEEE Access*, 2021. DOI: `10.1109/ACCESS.2021.3069646`
- Singh et al., "Enhancing AI Systems with Agentic Workflows Patterns in Large Language Model," *IEEE AIIoT*, 2024. DOI: `10.1109/AIIoT61789.2024.10578990`
- RAG architecture for knowledge-intensive NLP. DOI: `10.1109/[identifier].2021.9399342`

---

## Features

- Local agronomic knowledge base indexing from `data/*.md` and `data/*.txt`
- TF-IDF semantic retrieval with configurable top-k chunk selection
- Agentic 3-stage workflow:
  - **Plan** — decomposes the query into a structured diagnostic strategy
  - **Draft** — generates a grounded answer with citations from retrieved chunks
  - **Reflect** — self-critiques and refines the draft for factual accuracy
- Works with OpenAI API (GPT-4o / GPT-4o-mini), or falls back to **MockLLM** for offline demo
- Streamlit UI designed for course presentations and live demos

---

## Project Structure

```
agrisense/
├── app.py                   # Streamlit demo app
├── src/
│   ├── config.py            # Environment-based settings
│   ├── rag.py               # Document chunking and TF-IDF retrieval
│   ├── llm.py               # OpenAI and MockLLM clients
│   └── agent.py             # Agentic workflow controller (Plan→Draft→Reflect)
├── data/
│   └── knowledge_base.md    # Agronomic disease management knowledge base
├── reports/                 # Paper drafts, experiment logs, results
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

1. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Windows
   copy .env.example .env
   # macOS / Linux
   cp .env.example .env
   ```
   Then open `.env` and set your `OPENAI_API_KEY` (optional — app runs in MockLLM mode without it).

4. **Add domain knowledge**

   Place `.md` or `.txt` files with crop disease information inside `data/`. A starter `knowledge_base.md` is included covering common diseases from the PlantVillage dataset.

---

## Run

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Suggested Experiments (for report)

Run each configuration and record metrics for your Table II:

| Experiment | Config | What to measure |
|---|---|---|
| 1 | Baseline — vanilla LLM, no retrieval | Hallucination rate, factual correctness |
| 2 | RAG-only — retrieval + single-pass generation | Answer relevance, faithfulness |
| 3 | Agentic RAG — plan + retrieve + reflect | All metrics + latency per request |

**Track per run:**
- Response quality (human rating 1–5 / faithfulness score)
- Citation correctness (does the answer cite retrieved chunks accurately?)
- Latency per request (seconds)
- Hallucination rate (% responses containing unverified claims)

---

## Knowledge Base Guidelines

The `data/` folder should contain agronomic documents covering:
- Disease descriptions (symptoms, causal agents, affected crops)
- Treatment protocols (chemical and organic)
- Pesticide application guidelines
- Crop-specific cultivation advisories

The more domain-specific your documents, the lower the hallucination rate (see paper §IV-C).

---

## Notes

- Keep `top_k` small (3–5) to reduce hallucination and stay within token limits.
- The reflection stage adds ~1–2 s latency but measurably improves factual correctness (see paper Table II).
- For the final presentation demo, use the MockLLM mode if internet/API access is unavailable.
- Add experiment results to `reports/` before writing the final paper sections.
