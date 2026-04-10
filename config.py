"""
config.py — Environment-based settings for AgriSense.
Reads from .env (via python-dotenv) with safe fallback defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    openai_api_key: str
    openai_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    max_reflect_iterations: int
    data_dir: str
    reports_dir: str


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        top_k=int(os.getenv("TOP_K", "4")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        max_reflect_iterations=int(os.getenv("MAX_REFLECT_ITERATIONS", "1")),
        data_dir=os.getenv("DATA_DIR", "data"),
        reports_dir=os.getenv("REPORTS_DIR", "reports"),
    )
