"""
Memory Pipeline — Cross-Session Persistence and Recall Chain  (PRD §8)

Manages substrate signal persistence, cross-session memory, and
associative recall through the NG ecosystem.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §6.1/§8 compliant rewrite.
#   What: MemoryPipeline with §6.1 observation signals for store/recall.
#   Why:  PRD v0.2.0 §6.1 mandates flat scored signal fields.
# -------------------
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.substrate_signal import SubstrateSignal

logger = logging.getLogger("elmer.pipeline.memory")


class MemoryPipeline:
    """Cross-session memory and recall pipeline.

    In-memory signal buffer with bounded capacity.  NG substrate-backed
    associative recall planned for future phases.

    Ref: PRD §8
    """

    def __init__(self, max_signals: int = 1000) -> None:
        self._buffer: List[SubstrateSignal] = []
        self._max_signals = max_signals
        self._store_count = 0
        self._recall_count = 0

    def store(self, signal: SubstrateSignal) -> SubstrateSignal:
        """Store a signal in the memory buffer.

        Args:
            signal: Signal to remember.

        Returns:
            §6.1 SubstrateSignal of type "observation" confirming storage.
        """
        self._buffer.append(signal)
        if len(self._buffer) > self._max_signals:
            self._buffer = self._buffer[-self._max_signals:]

        self._store_count += 1

        return SubstrateSignal.create(
            signal_type="observation",
            description=f"Memory stored signal {signal.signal_id[:8]}",
            coherence_score=1.0,
            health_score=1.0,
            anomaly_level=0.0,
            novelty=0.0,
            confidence=1.0,
            severity=0.0,
            temporal_window=0.0,
            metadata={
                "pipeline": "memory",
                "action": "stored",
                "stored_signal_id": signal.signal_id,
                "buffer_size": len(self._buffer),
            },
        )

    def recall(self, query: str, k: int = 5) -> SubstrateSignal:
        """Recall signals from memory (most recent).

        Args:
            query: Search query (semantic recall in future phases).
            k: Max signals to return.

        Returns:
            §6.1 SubstrateSignal with recalled signal IDs in metadata.
        """
        self._recall_count += 1
        recent = self._buffer[-k:] if self._buffer else []

        return SubstrateSignal.create(
            signal_type="observation",
            description=f"Memory recall: {len(recent)} signals for '{query[:50]}'",
            coherence_score=1.0,
            health_score=1.0,
            anomaly_level=0.0,
            novelty=0.0,
            confidence=0.5,
            severity=0.0,
            temporal_window=0.0,
            metadata={
                "pipeline": "memory",
                "action": "recalled",
                "query": query,
                "recalled_count": len(recent),
                "signal_ids": [s.signal_id for s in recent],
            },
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "pipeline": "memory",
            "buffer_size": len(self._buffer),
            "store_count": self._store_count,
            "recall_count": self._recall_count,
        }
