"""
Signal Decoder — SubstrateSignal → Output Dict  (PRD §9)

Decodes §6.1 SubstrateSignals into structured output dicts for the
OpenClaw hook, CLI, or API consumers.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §6.1/§9 compliant rewrite.
#   What: SignalDecoder extracting §6.1 flat scored fields.
#   Why:  PRD v0.2.0 §6.1 signal schema changed from generic payload.
# -------------------
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.substrate_signal import SubstrateSignal

logger = logging.getLogger("elmer.signal_decoder")


class SignalDecoder:
    """Decodes SubstrateSignals into structured output dicts.

    Ref: PRD §9
    """

    def decode(self, signal: SubstrateSignal) -> Dict[str, Any]:
        """Decode a processed signal into output format.

        Extracts all §6.1 scored fields plus Elmer-specific extensions.
        """
        return {
            "signal_id": signal.signal_id,
            "module_id": signal.module_id,
            "signal_type": signal.signal_type,
            "description": signal.description,
            "coherence_score": signal.coherence_score,
            "health_score": signal.health_score,
            "anomaly_level": signal.anomaly_level,
            "novelty": signal.novelty,
            "confidence": signal.confidence,
            "severity": signal.severity,
            "temporal_window": signal.temporal_window,
            "identity_coherence": signal.identity_coherence,
            "pruning_pressure": signal.pruning_pressure,
            "topology_health": signal.topology_health,
            "coherence_status": signal.coherence_status,
            "metadata": signal.metadata,
            "timestamp": signal.timestamp,
        }
