"""
MyelinationSocket — Elmer's oligodendrocyte.

Extracts myelination-relevant patterns from the substrate and produces
recommendations for which tracts should be upgraded (file→mmap) or
downgraded (mmap→file).

Biological analog: oligodendrocytes observe axon activity through the
chemical environment and respond by wrapping active nerves in myelin.
The axon doesn't know it's being myelinated.  The glial cell decides.

This socket follows the same pattern as ComprehensionSocket and
MonitoringSocket.  It receives a GraphSnapshot, extracts patterns,
and returns a SubstrateSignal with myelination recommendations in
the metadata.

The tract itself remains dumb — it conducts, it does not observe.
Myelination state is runtime-only (not persisted).  If the process
restarts, all tracts start unmyelinated and re-earn myelination
through the substrate.

Apprentice-tier bootstrap: until the substrate has enough data to
make substrate-learned decisions, the socket uses a simple heuristic
based on recent Commons deposit frequency per source module.  This
scaffolding lives here in Elmer's code, not in the tract infrastructure.

Source attribution uses the target_id namespace convention:
  - 3-segment IDs (metrics:neurograph:kind, health:elmer:kind, error:module:type)
    → segment[1] is the depositing module name.
  - 2-segment or fixed-source namespaces (topology, experience, autonomic, etc.)
    → mapped via _NAMESPACE_TO_MODULE.

# ---- Changelog ----
# [2026-03-23] Claude (Opus 4.6) — Initial creation (punchlist #53 v0.4)
#   What: MyelinationSocket implementing ElmerSocket interface.
#   Why:  Elmer is the oligodendrocyte — decides which tracts get
#         myelinated based on substrate-learned patterns.  The tract
#         bridge gained mmap transport in the same v0.4 pass.
#   How:  Extracts pathway activity patterns from GraphSnapshot.
#         Apprentice heuristic counts peer events.  Produces
#         SubstrateSignal with myelination_recommendations metadata.
# [2026-06-30] Claude Code (Sonnet 4.6) — #326 re-source from Commons
#   What: _apprentice_score() now buckets recent Commons deposits and
#         counts by source module instead of reading the deleted
#         NGPeerBridge._peer_events field (NGPeerBridge removed 2026-06-03).
#   Why:  getattr(bridge_ref, '_peer_events', []) always returned []
#         after ng_peer_bridge.py was deleted → myelination socket
#         permanently dark (0 recommendations ever fired post Phase 3).
#   How:  _get_commons() tries self._commons_ref (test injection) then
#         falls back to get_commons() singleton.  bucket_recent(200)
#         gives a ~6-minute window of activity; target_id namespace
#         parsing extracts the depositing module. set_bridge_ref()
#         retained for engine backwards-compat but unused in heuristic.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.base_socket import (
    ElmerSocket,
    GraphSnapshot,
    HardwareRequirements,
    SocketHealth,
    SocketOutput,
)
from core.substrate_signal import SubstrateSignal

logger = logging.getLogger("elmer.myelination")

# Apprentice-tier thresholds (bootstrap scaffolding)
# These live in Elmer's code, not in the tract infrastructure.
# They graduate through the competence model (#79).
_MIN_EVENTS_TO_RECOMMEND = 10   # Don't recommend until we've seen this many deposits
_MYELINATE_PERCENTILE = 0.70    # Top 30% of peers by activity → myelinate
_DEMYELINATE_PERCENTILE = 0.15  # Bottom 15% → demyelinate candidate

# Fixed-source namespaces: target_ids in these namespaces always come from the same module.
# For 3-segment IDs (metrics:neurograph:kind, health:elmer:kind, error:module:type)
# segment[1] is the source module name and takes precedence over this table.
_NAMESPACE_TO_MODULE: Dict[str, str] = {
    "topology": "neurograph",
    "experience": "neurograph",
    "autonomic": "immunis",
    "threat": "immunis",
    "response": "immunis",
    "perimeter": "trollguard",
    "violation": "elmer",
    "repair": "thc",
}


class MyelinationSocket(ElmerSocket):
    """Elmer's myelination extraction bucket.

    Observes the substrate for tract activity patterns and produces
    recommendations for which pathways should be myelinated (file→mmap)
    or demyelinated (mmap→file).

    The decision is substrate-learned at Journeyman/Master tiers.
    At Apprentice tier, a simple heuristic based on peer event frequency
    serves as bootstrap scaffolding.
    """

    SOCKET_ID = "elmer:myelination"
    SOCKET_TYPE = "myelination"

    def __init__(self) -> None:
        super().__init__()
        self._peer_bridge_ref = None  # Retained for engine backwards-compat; not used in heuristic
        self._commons_ref = None      # Injected in tests; production uses get_commons() singleton

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
        logger.info("MyelinationSocket loaded")
        return True

    def unload(self) -> None:
        self._loaded = False
        logger.info("MyelinationSocket unloaded")

    def set_bridge_ref(self, bridge) -> None:
        """Engine provides a reference to the peer bridge.

        Retained for engine backwards-compat. The Apprentice heuristic no
        longer reads _peer_events (NGPeerBridge removed 2026-06-03); it
        buckets the Commons instead. This method is a safe no-op.
        """
        self._peer_bridge_ref = bridge

    def set_commons_ref(self, commons) -> None:
        """Inject a Commons instance for test isolation.

        Production code leaves this None and _get_commons() falls back to
        the get_commons() process singleton (in-process, always available
        when running inside neurograph_rpc.py).
        """
        self._commons_ref = commons

    def _get_commons(self) -> Optional[Any]:
        """Return the Commons singleton (or test-injected mock)."""
        if self._commons_ref is not None:
            return self._commons_ref
        try:
            from commons import get_commons  # noqa: PLC0415 — in-process import
            return get_commons()
        except Exception:
            return None

    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Extract myelination recommendations from the substrate.

        At Apprentice tier: counts peer events from the bridge's cache
        and recommends myelination for high-activity peers.

        At Journeyman/Master tier (future): extracts pathway activity
        patterns from the GraphSnapshot's learned topology — nodes
        with cross-module metadata, edge weights reflecting downstream
        impact, activation patterns corresponding to tract traffic.
        """
        if not self._loaded:
            raise RuntimeError("MyelinationSocket not loaded")

        t0 = time.time()
        self._process_count += 1

        # --- Apprentice tier: heuristic based on peer event counts ---
        pathway_scores = self._apprentice_score(snapshot, context)
        myelinate_list, demyelinate_list = self._apprentice_recommend(pathway_scores)

        elapsed = time.time() - t0
        self._total_latency += elapsed

        signal = SubstrateSignal.create(
            signal_type="health",
            description=(
                f"Myelination assessment: {len(myelinate_list)} upgrade, "
                f"{len(demyelinate_list)} downgrade candidates"
            ),
            coherence_score=1.0,
            health_score=1.0,
            anomaly_level=0.0,
            novelty=0.0,
            confidence=0.5,  # Apprentice — low confidence
            severity=0.0,
            temporal_window=elapsed,
            metadata={
                "socket": self.SOCKET_ID,
                "myelination_recommendations": {
                    "myelinate": myelinate_list,
                    "demyelinate": demyelinate_list,
                },
                "pathway_scores": pathway_scores,
                "tier": "apprentice",
            },
        )

        return SocketOutput(
            signal=signal,
            graph_delta=None,
            confidence=0.5,
            processing_time=elapsed,
        )

    def health(self) -> SocketHealth:
        return self._make_health("healthy")

    # -------------------------------------------------------------------
    # Apprentice-tier heuristics (bootstrap scaffolding)
    # -------------------------------------------------------------------

    def _apprentice_score(
        self, snapshot: GraphSnapshot, context: dict,
    ) -> Dict[str, float]:
        """Score each peer pathway by recent Commons deposit frequency.

        Buckets the last 200 Commons entries and counts deposits per source
        module, inferred from the target_id namespace convention:
          - 3-segment IDs (metrics:neurograph:kind, health:elmer:kind) →
            segment[1] is the module name.
          - Fixed-source namespaces (topology, experience, autonomic, etc.) →
            looked up in _NAMESPACE_TO_MODULE.
        Normalizes counts to [0, 1].  Apprentice scaffolding — at Journeyman
        tier this would read learned topology patterns from the snapshot.
        """
        commons = self._get_commons()
        if commons is None:
            return {}

        try:
            recs = commons.bucket_recent(limit=200)
        except Exception:
            return {}

        if len(recs) < _MIN_EVENTS_TO_RECOMMEND:
            return {}

        counts: Dict[str, int] = {}
        for entry in recs:
            target_id = entry[0] if isinstance(entry, (tuple, list)) else str(entry)
            parts = target_id.split(":", 2)
            namespace = parts[0]
            if namespace in _NAMESPACE_TO_MODULE:
                module_id = _NAMESPACE_TO_MODULE[namespace]
            elif len(parts) >= 3:
                # e.g. "metrics:neurograph:anomaly" → "neurograph"
                module_id = parts[1]
            else:
                continue
            if module_id:
                counts[module_id] = counts.get(module_id, 0) + 1

        if not counts:
            return {}

        max_count = max(counts.values())
        if max_count == 0:
            return {}

        return {module: count / max_count for module, count in counts.items()}

    def _apprentice_recommend(
        self, scores: Dict[str, float],
    ) -> tuple:
        """Produce myelination/demyelination recommendations from scores.

        Top peers (above _MYELINATE_PERCENTILE) → myelinate.
        Bottom peers (below _DEMYELINATE_PERCENTILE) → demyelinate.
        """
        if not scores:
            return [], []

        myelinate = [
            peer for peer, score in scores.items()
            if score >= _MYELINATE_PERCENTILE
        ]
        demyelinate = [
            peer for peer, score in scores.items()
            if score <= _DEMYELINATE_PERCENTILE
        ]

        return myelinate, demyelinate
