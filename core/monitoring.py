"""
Monitoring Socket — System Health and Telemetry Processing Unit  (PRD §5.2.1)

Processes graph snapshots for anomaly detection and health assessment.
Produces health signals with anomaly_level and severity scores.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §5.2.1 compliant rewrite.
#   What: MonitoringSocket implementing full §5.2.1 interface.
#   Why:  Align with PRD v0.2.0 §5.2 socket specification.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from core.base_socket import (
    ElmerSocket,
    GraphSnapshot,
    HardwareRequirements,
    SocketHealth,
    SocketOutput,
)
from ng_ecosystem import (
    COHERENCE_CRITICAL,
    COHERENCE_DEGRADED,
    COHERENCE_HEALTHY,
    SubstrateSignal,
)

logger = logging.getLogger("elmer.monitoring")


class MonitoringSocket(ElmerSocket):
    """System monitoring and health processing socket.

    Ref: PRD §5.2.1

    Produces health signals with anomaly detection and severity
    scoring against §14 threshold constants.
    """

    SOCKET_ID = "elmer:monitoring"
    SOCKET_TYPE = "monitoring"

    @property
    def socket_id(self) -> str:
        return self.SOCKET_ID

    @property
    def socket_type(self) -> str:
        return self.SOCKET_TYPE

    def declare_requirements(self) -> HardwareRequirements:
        return HardwareRequirements(
            min_memory_mb=128,
            gpu_required=False,
            cpu_cores=1,
            disk_mb=0,
        )

    def load(self, model_path: str) -> bool:
        if self._loaded:
            return True
        self._loaded = True
        self._load_time = time.time()
        logger.info("MonitoringSocket loaded (model_path=%s)", model_path)
        return True

    def unload(self) -> None:
        self._loaded = False
        logger.info("MonitoringSocket unloaded")

    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Assess system health from graph snapshot.

        Checks coherence against §14 thresholds and flags anomalies.
        """
        if not self._loaded:
            raise RuntimeError("MonitoringSocket not loaded")

        t0 = time.time()
        self._process_count += 1

        node_count = len(snapshot.nodes)
        edge_count = len(snapshot.edges)

        # Health heuristic: graph connectivity
        max_edges = max(node_count * (node_count - 1) / 2, 1)
        health_score = min(edge_count / max_edges, 1.0) if node_count > 1 else 1.0

        # Anomaly: disconnected nodes (nodes with no edges)
        connected_ids = set()
        for e in snapshot.edges:
            connected_ids.add(e.get("source"))
            connected_ids.add(e.get("target"))
        disconnected = sum(
            1 for n in snapshot.nodes if n.get("id") not in connected_ids
        )
        anomaly_level = disconnected / max(node_count, 1)

        # Severity from §14 thresholds
        if health_score >= COHERENCE_HEALTHY:
            severity = 0.0
        elif health_score >= COHERENCE_DEGRADED:
            severity = 0.3
        elif health_score >= COHERENCE_CRITICAL:
            severity = 0.7
        else:
            severity = 1.0

        signal_type = "health" if severity < 0.5 else "anomaly"

        elapsed = time.time() - t0
        self._total_latency += elapsed

        signal = SubstrateSignal.create(
            signal_type=signal_type,
            description=f"Monitoring: health={health_score:.2f} anomaly={anomaly_level:.2f}",
            coherence_score=health_score,
            health_score=health_score,
            anomaly_level=anomaly_level,
            novelty=0.0,
            confidence=0.9,
            severity=severity,
            temporal_window=elapsed,
            topology_health=health_score,
            metadata={
                "socket": self.socket_id,
                "node_count": node_count,
                "edge_count": edge_count,
                "disconnected_nodes": disconnected,
            },
        )

        return SocketOutput(
            signal=signal,
            graph_delta=None,
            confidence=0.9,
            processing_time=elapsed,
        )

    def health(self) -> SocketHealth:
        return self._make_health("healthy")
