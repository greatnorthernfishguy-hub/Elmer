"""
Sensory Pipeline — Input Processing Chain  (PRD §8)

Processes raw sensory input (text, embeddings, external signals) into
SubstrateSignals with proper §6.1 scored fields.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §6.1/§8 compliant rewrite.
#   What: SensoryPipeline producing §6.1-compliant SubstrateSignals
#         with flat scored fields.
#   Why:  PRD v0.2.0 §6.1 mandates specific signal schema.
# -------------------
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ng_ecosystem import SubstrateSignal

logger = logging.getLogger("elmer.pipeline.sensory")


class SensoryPipeline:
    """Ingests raw input and produces observation SubstrateSignals.

    Ref: PRD §8
    """

    def __init__(self) -> None:
        self._process_count = 0

    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> SubstrateSignal:
        """Process raw text into an observation signal.

        Args:
            text: Raw input text.
            metadata: Optional extra context.

        Returns:
            §6.1 SubstrateSignal of type "observation".
        """
        self._process_count += 1
        text_len = len(text) if text else 0
        novelty = min(text_len / 1000.0, 1.0)

        return SubstrateSignal.create(
            signal_type="observation",
            description=f"Sensory input: {text_len} chars",
            coherence_score=1.0,
            health_score=1.0,
            anomaly_level=0.0,
            novelty=novelty,
            confidence=0.9,
            severity=0.0,
            temporal_window=0.0,
            metadata={
                "pipeline": "sensory",
                "text_length": text_len,
                "text_preview": (text[:200] if text else ""),
                **(metadata or {}),
            },
        )

    def stats(self) -> Dict[str, Any]:
        return {"pipeline": "sensory", "process_count": self._process_count}
