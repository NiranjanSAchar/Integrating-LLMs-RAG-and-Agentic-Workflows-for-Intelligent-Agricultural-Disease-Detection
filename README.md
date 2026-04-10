# Agentic RAG Project (Course-Ready)

This project is a complete starter for a Generative AI course project using:
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)
- Agentic Workflow (plan -> draft -> reflect)

## Features
- Simple local document indexing from data/*.md and data/*.txt
- TF-IDF retrieval with top-k context selection
- Agentic 3-stage flow:
  - Planning
  - Grounded drafting with citations
  - Reflection/self-check pass
- Works with OpenAI API, or falls back to MockLLM for offline demo
- Streamlit UI for easy presentation/demo

## Project Structure
- app.py: Streamlit demo app
- src/config.py: environment-based settings
- src/rag.py: chunking and retrieval
- src/llm.py: OpenAI and mock LLM clients
- src/agent.py: agentic workflow controller
- data/knowledge_base.md: starter knowledge base
- reports/: place for paper drafts or experiment outputs

## Setup
1. Create and activate a virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Copy env template:
   copy .env.example .env
4. (Optional) set OPENAI_API_KEY in .env

## Run
streamlit run app.py

## Suggested Experiments (for report)
1. Baseline prompting (no retrieval)
2. RAG only (retrieval + single generation)
3. Agentic RAG (planning + retrieval + reflection)

Track:
- Response quality (human rating / faithfulness)
- Citation correctness
- Latency per request

## Citation Seed
You can cite this paper in your report for agentic-workflow motivation:
- Enhancing AI Systems with Agentic Workflows Patterns in Large Language Model
- DOI: 10.1109/AIIoT61789.2024.10578990

## Notes
- Add more domain documents to data/ before final experiments.
- Keep top-k small to reduce hallucination and token usage.
