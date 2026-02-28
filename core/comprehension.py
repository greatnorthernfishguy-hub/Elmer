"""
Comprehension Socket — Sensory Input Processing Unit  (PRD §5.2.1)

Processes incoming graph snapshots for semantic comprehension — pattern
recognition, novelty detection, and coherence assessment.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §5.2.1 compliant rewrite.
#   What: ComprehensionSocket implementing declare_requirements, load,
#         unload, process(GraphSnapshot, context), health.
#   Why:  Align with PRD v0.2.0 §5.2 socket interface.
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
from ng_ecosystem import SubstrateSignal

logger = logging.getLogger("elmer.comprehension")


class ComprehensionSocket(ElmerSocket):
    """Sensory comprehension and pattern-recognition socket.

    Ref: PRD §5.2.1

    Produces observation signals with coherence and novelty scores
    derived from graph topology.
    """

    SOCKET_ID = "elmer:comprehension"
    SOCKET_TYPE = "comprehension"

    @property
    def socket_id(self) -> str:
        return self.SOCKET_ID

    @property
    def socket_type(self) -> str:
        return self.SOCKET_TYPE

    def declare_requirements(self) -> HardwareRequirements:
        return HardwareRequirements(
            min_memory_mb=256,
            gpu_required=False,
            cpu_cores=1,
            disk_mb=0,
        )

    def load(self, model_path: str) -> bool:
        if self._loaded:
            return True
        self._loaded = True
        self._load_time = time.time()
        logger.info("ComprehensionSocket loaded (model_path=%s)", model_path)
        return True

    def unload(self) -> None:
        self._loaded = False
        logger.info("ComprehensionSocket unloaded")

    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Assess coherence and novelty from the graph snapshot.

        Ref: PRD §5.2.1
        """
        if not self._loaded:
            raise RuntimeError("ComprehensionSocket not loaded")

        t0 = time.time()
        self._process_count += 1

        node_count = len(snapshot.nodes)
        edge_count = len(snapshot.edges)

        # Coherence heuristic: ratio of edges to possible edges
        max_edges = max(node_count * (node_count - 1) / 2, 1)
        coherence = min(edge_count / max_edges, 1.0) if node_count > 1 else 1.0

        # Novelty heuristic: inverse of node count (more nodes → more known)
        novelty = 1.0 / (1.0 + node_count * 0.1)

        elapsed = time.time() - t0
        self._total_latency += elapsed

        signal = SubstrateSignal.create(
            signal_type="observation",
            description=f"Comprehension: {node_count} nodes, {edge_count} edges",
            coherence_score=coherence,
            health_score=1.0,
            anomaly_level=0.0,
            novelty=novelty,
            confidence=0.8,
            severity=0.0,
            temporal_window=elapsed,
            metadata={
                "socket": self.socket_id,
                "node_count": node_count,
                "edge_count": edge_count,
            },
        )

        return SocketOutput(
            signal=signal,
            graph_delta=None,
            confidence=0.8,
            processing_time=elapsed,
        )

    def health(self) -> SocketHealth:
        return self._make_health("healthy")
