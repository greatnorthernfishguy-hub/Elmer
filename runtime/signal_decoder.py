"""
Signal Decoder — SubstrateSignal → Human-Readable Output

Decodes SubstrateSignals from the processing pipeline into structured
output suitable for consumption by the OpenClaw hook, CLI, or API.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 initial creation.
#   What: SignalDecoder with decode() method.
#   Why:  Clean separation between internal signal format and output.
# -------------------
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ng_ecosystem import SubstrateSignal

logger = logging.getLogger("elmer.signal_decoder")


class SignalDecoder:
    """Decodes SubstrateSignals into structured output dicts.

    Phase 1: Direct extraction of signal fields.
    Phase 2+: Rich formatting, template rendering, multi-modal output.
    """

    def decode(self, signal: SubstrateSignal) -> Dict[str, Any]:
        """Decode a processed signal into output format.

        Args:
            signal: Processed SubstrateSignal.

        Returns:
            Dict with decoded output fields.
        """
        return {
            "signal_id": signal.signal_id,
            "signal_type": signal.signal_type.value,
            "source_socket": signal.source_socket,
            "payload": signal.payload,
            "confidence": signal.confidence,
            "priority": signal.priority,
            "metadata": signal.metadata,
            "graph_encoding": signal.graph_encoding,
        }
