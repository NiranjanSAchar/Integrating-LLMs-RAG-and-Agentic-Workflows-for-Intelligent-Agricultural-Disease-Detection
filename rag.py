"""
rag.py — Document chunking and TF-IDF retrieval for AgriSense.
Implements the RAG knowledge module described in §III-C of the paper.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    """A single retrieved document chunk with its source filename."""
    text: str
    source: str


@dataclass
class SimpleRAG:
    """
    Lightweight RAG retriever using TF-IDF + cosine similarity.
    No external vector database required — runs fully locally.
    """
    chunk_size: int = 500
    chunk_overlap: int = 100

    _chunks: List[Chunk] = field(default_factory=list, init=False, repr=False)
    _vectorizer: object = field(default=None, init=False, repr=False)
    _matrix: object = field(default=None, init=False, repr=False)

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _split_text(self, text: str, source: str) -> List[Chunk]:
        """Split text into overlapping word-count chunks."""
        words = text.split()
        chunks: List[Chunk] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, source=source))
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_folder(self, folder: str) -> int:
        """
        Read all .md and .txt files from `folder`, chunk them, and build
        a TF-IDF index. Returns the total number of indexed chunks.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._chunks = []
        folder_path = Path(folder)

        for ext in ("*.md", "*.txt"):
            for fpath in sorted(folder_path.glob(ext)):
                try:
                    raw = fpath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                # Strip markdown heading markers for cleaner TF-IDF tokens
                clean = re.sub(r"^#{1,6}\s*", "", raw, flags=re.MULTILINE)
                self._chunks.extend(self._split_text(clean, fpath.name))

        if not self._chunks:
            # Index an empty placeholder so the app doesn't crash
            self._chunks = [Chunk(
                text="No documents indexed. Add .md or .txt files to the data/ folder.",
                source="(empty)"
            )]

        self._vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(
            [c.text for c in self._chunks]
        )
        return len(self._chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        """
        Return the top-k most relevant chunks for the given query
        using cosine similarity over TF-IDF vectors.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        if self._vectorizer is None or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self._chunks[i] for i in top_indices if scores[i] > 0]
