"""
Comprehension Socket — Language Understanding Processing Unit

Phase 1 stub.  The ComprehensionSocket will evolve to handle:
  - Semantic parsing of incoming text
  - Intent classification
  - Entity extraction
  - Context framing for downstream sockets

Currently: pass-through with signal wrapping and health reporting.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 stub.
#   What: ComprehensionSocket implementing ElmerSocket ABC.
#   Why:  Phase 1 requires socket registration and signal flow.
#         Full NLU pipeline deferred to Phase 2.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from core.base_socket import ElmerSocket
from ng_ecosystem import SubstrateSignal, SignalType

logger = logging.getLogger("elmer.comprehension")


class ComprehensionSocket(ElmerSocket):
    """Language comprehension processing socket.

    Phase 1: Pass-through that validates signals and reports health.
    Phase 2+: Full NLU pipeline with semantic parsing, intent
    classification, and entity extraction.
    """

    SOCKET_ID = "elmer:comprehension"
    SOCKET_TYPE = "sensory"

    @property
    def socket_id(self) -> str:
        return self.SOCKET_ID

    @property
    def socket_type(self) -> str:
        return self.SOCKET_TYPE

    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        self._connect_time = time.time()
        logger.info("ComprehensionSocket connected")

    def disconnect(self) -> None:
        self._connected = False
        logger.info("ComprehensionSocket disconnected")

    def process(self, signal: SubstrateSignal) -> SubstrateSignal:
        """Phase 1: Pass-through with metadata annotation."""
        if not self._connected:
            raise RuntimeError("ComprehensionSocket not connected")

        self._process_count += 1
        self._last_process_time = time.time()

        # Phase 1: annotate and pass through
        enriched_metadata = {
            **signal.metadata,
            "comprehension_processed": True,
            "comprehension_version": "0.1.0",
        }

        return signal.with_updates(
            source_socket=self.socket_id,
            metadata=enriched_metadata,
        )

    def health_check(self) -> Dict[str, Any]:
        base = self._base_health()
        base["status"] = "healthy" if self._connected else "offline"
        return base
