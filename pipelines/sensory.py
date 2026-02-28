"""
Sensory Pipeline — Input Processing Chain

Processes raw sensory input (text, embeddings, external signals) into
structured SubstrateSignals for downstream processing.

Phase 1: Stub that wraps raw text into a SENSORY SubstrateSignal.
Phase 2+: Multi-modal input parsing, tokenization, embedding generation.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 stub.
# -------------------
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ng_ecosystem import SubstrateSignal, SignalType

logger = logging.getLogger("elmer.pipeline.sensory")


class SensoryPipeline:
    """Processes raw input into structured sensory signals.

    Phase 1: Minimal text → SubstrateSignal wrapping.
    Phase 2+: Multi-modal parsing, tokenization, embedding.
    """

    def __init__(self) -> None:
        self._process_count = 0

    def process(self, text: str) -> SubstrateSignal:
        """Convert raw text input into a SENSORY SubstrateSignal.

        Args:
            text: Raw input text.

        Returns:
            SENSORY SubstrateSignal with text payload.
        """
        self._process_count += 1

        return SubstrateSignal.create(
            source_socket="pipeline:sensory",
            signal_type=SignalType.SENSORY,
            payload={"text": text, "input_length": len(text)},
            confidence=1.0,
            priority=5,
            metadata={"pipeline": "sensory", "version": "0.1.0"},
        )

    def stats(self) -> Dict[str, Any]:
        return {"pipeline": "sensory", "process_count": self._process_count}
