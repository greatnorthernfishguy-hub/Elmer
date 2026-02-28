"""
Monitoring Socket — System Health and Telemetry Processing Unit

Phase 1 stub.  The MonitoringSocket will evolve to handle:
  - Continuous health signal processing
  - Anomaly detection in substrate signals
  - Performance metric aggregation
  - Alert generation for degraded states

Currently: pass-through with health signal enrichment.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 stub.
#   What: MonitoringSocket implementing ElmerSocket ABC.
#   Why:  Phase 1 requires health endpoint and signal flow.
#         Full monitoring pipeline deferred to Phase 2.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from core.base_socket import ElmerSocket
from ng_ecosystem import SubstrateSignal, SignalType

logger = logging.getLogger("elmer.monitoring")


class MonitoringSocket(ElmerSocket):
    """System monitoring and health processing socket.

    Phase 1: Pass-through that tracks health signals and reports status.
    Phase 2+: Full monitoring pipeline with anomaly detection,
    metric aggregation, and alert generation.
    """

    SOCKET_ID = "elmer:monitoring"
    SOCKET_TYPE = "health"

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
        logger.info("MonitoringSocket connected")

    def disconnect(self) -> None:
        self._connected = False
        logger.info("MonitoringSocket disconnected")

    def process(self, signal: SubstrateSignal) -> SubstrateSignal:
        """Phase 1: Pass-through with health metadata annotation."""
        if not self._connected:
            raise RuntimeError("MonitoringSocket not connected")

        self._process_count += 1
        self._last_process_time = time.time()

        enriched_metadata = {
            **signal.metadata,
            "monitoring_processed": True,
            "monitoring_version": "0.1.0",
        }

        return signal.with_updates(
            source_socket=self.socket_id,
            metadata=enriched_metadata,
        )

    def health_check(self) -> Dict[str, Any]:
        base = self._base_health()
        base["status"] = "healthy" if self._connected else "offline"
        return base
