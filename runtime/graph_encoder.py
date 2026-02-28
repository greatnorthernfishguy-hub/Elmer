"""
Graph Encoder — SubstrateSignal → NG Substrate Encoding

Encodes SubstrateSignals into a format compatible with the NG ecosystem's
record_outcome / get_recommendations API.  This is the bridge between
Elmer's signal bus and the learning substrate.

Phase 1: Hash-based embedding for testing.  Phase 2+: real embeddings
via sentence-transformers or Ollama.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 initial creation.
#   What: GraphEncoder with encode() method producing NG-compatible dicts.
#   Why:  Decouples signal → embedding logic from the engine and sockets.
# -------------------
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional

import numpy as np

from ng_ecosystem import SubstrateSignal

logger = logging.getLogger("elmer.graph_encoder")


class GraphEncoder:
    """Encodes SubstrateSignals for NG substrate learning.

    Produces a graph_encoding dict with:
      - embedding: np.ndarray (384-dim, L2-normalized)
      - target_id: str (derived from signal type + source)
      - metadata: dict (signal context for learning attribution)

    Phase 1: Hash-based deterministic embeddings (no ML model needed).
    Phase 2: Pluggable embedding backend (sentence-transformers, Ollama).
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self._embedding_dim = embedding_dim

    def encode(self, signal: SubstrateSignal) -> Dict[str, Any]:
        """Encode a signal for the NG substrate.

        Args:
            signal: SubstrateSignal to encode.

        Returns:
            Dict with 'embedding', 'target_id', and 'metadata' keys.
        """
        text = self._extract_text(signal)
        embedding = self._embed(text)
        target_id = f"{signal.signal_type.value}:{signal.source_socket}"

        return {
            "embedding": embedding.tolist(),
            "target_id": target_id,
            "metadata": {
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type.value,
                "source_socket": signal.source_socket,
                "confidence": signal.confidence,
                "priority": signal.priority,
            },
        }

    def _extract_text(self, signal: SubstrateSignal) -> str:
        """Extract representative text from a signal's payload."""
        payload = signal.payload
        # Try common text fields
        for key in ("text", "content", "message", "query"):
            if key in payload and isinstance(payload[key], str):
                return payload[key]
        # Fallback: stringify payload
        return str(payload)

    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding vector from text.

        Phase 1: Hash-based deterministic embedding (no ML deps).
        Phase 2: sentence-transformers or Ollama embedding.
        """
        h = hashlib.sha256(text.encode()).digest()
        # Repeat hash bytes to fill embedding dimension
        repeats = (self._embedding_dim * 4 // len(h)) + 1
        raw = np.frombuffer((h * repeats)[:self._embedding_dim * 4], dtype=np.float32)
        vec = raw[:self._embedding_dim]
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec
