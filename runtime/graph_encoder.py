"""
Graph Encoder — SubstrateSignal ↔ GraphSnapshot / NG Encoding  (PRD §9)

Bridges between Elmer's §6.1 SubstrateSignal format and the §5.2.2
GraphSnapshot format consumed by sockets.  Also produces NG-compatible
embedding dicts for ecosystem recording.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §5.2.2/§6.1/§9 compliant rewrite.
#   What: GraphEncoder with signal_to_snapshot(), encode(), and
#         hash-based embedding for testing.
#   Why:  PRD v0.2.0 §9 requires GraphSnapshot routing through sockets.
# -------------------
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from core.base_socket import GraphSnapshot
from core.substrate_signal import SubstrateSignal

logger = logging.getLogger("elmer.graph_encoder")


class GraphEncoder:
    """Encodes SubstrateSignals for socket processing and NG learning.

    Two roles:
      1. signal_to_snapshot() — build a GraphSnapshot for socket routing
      2. encode() — produce NG-compatible embedding dict for ecosystem

    Ref: PRD §9
    """

    # [2026-03-29] Fixed: 384→768 to match ecosystem standard (punchlist #101)
    def __init__(self, embedding_dim: int = 768) -> None:
        self._embedding_dim = embedding_dim

    def signal_to_snapshot(self, signal: SubstrateSignal) -> GraphSnapshot:
        """Convert a SubstrateSignal into a GraphSnapshot for socket processing.

        Builds a minimal graph with the signal as a node and its scored
        fields as edge weights.

        Ref: PRD §5.2.2
        """
        node = {
            "id": signal.signal_id,
            "type": signal.signal_type,
            "module_id": signal.module_id,
            "description": signal.description,
            "coherence_score": signal.coherence_score,
            "health_score": signal.health_score,
            "anomaly_level": signal.anomaly_level,
            "novelty": signal.novelty,
            "confidence": signal.confidence,
        }

        edges = []
        # If metadata contains a parent signal, create an edge
        parent_id = signal.metadata.get("parent_signal")
        if parent_id:
            edges.append({
                "source": parent_id,
                "target": signal.signal_id,
                "weight": signal.confidence,
                "type": "derived_from",
            })

        return GraphSnapshot(
            nodes=[node],
            edges=edges,
            metadata={
                "source_signal": signal.signal_id,
                "signal_type": signal.signal_type,
            },
            timestamp=signal.timestamp,
        )

    def encode(self, signal: SubstrateSignal) -> Dict[str, Any]:
        """Encode a signal for NG ecosystem recording.

        Args:
            signal: SubstrateSignal to encode.

        Returns:
            Dict with 'embedding', 'target_id', and 'metadata'.
        """
        text = signal.description
        embedding = self._embed(text)
        target_id = f"{signal.signal_type}:{signal.module_id}"

        return {
            "embedding": embedding,
            "target_id": target_id,
            "metadata": {
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type,
                "module_id": signal.module_id,
                "coherence_score": signal.coherence_score,
                "confidence": signal.confidence,
            },
        }

    def _embed(self, text: str) -> np.ndarray:
        """Embed via ng_embed (ecosystem standard).

        Snowflake/snowflake-arctic-embed-m-v1.5, 768-dim, ONNX.
        Falls back to zero vector if embedding fails.
        """
        try:
            from ng_embed import embed
            return embed(text, normalize=True)
        except Exception:
            return np.zeros(self._embedding_dim, dtype=np.float32)
