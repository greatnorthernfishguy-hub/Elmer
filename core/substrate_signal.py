"""
SubstrateSignal — Elmer's canonical inter-socket data format (PRD §6.1)

Frozen dataclass with flat scored fields (not a generic payload dict).
JSON-serializable. Compatible with NGEcosystem.record_outcome().
Includes Elmer-specific extensions: identity_coherence, pruning_pressure,
topology_health.

NOTE: This is Elmer's extraction vocabulary — the shape of Elmer's bucket
when it dips into the River. It is NOT an inter-module protocol. No module
serializes a SubstrateSignal and sends it to another module. See
ARCHITECTURE.md §6-7.

Previously lived inside the vendored ng_ecosystem.py (Law 2 violation).
Extracted to this Elmer-local file 2026-03-18.

# ---- Changelog ----
# [2026-03-18] Claude (CC) — Extracted from vendored ng_ecosystem.py
# What: Moved SubstrateSignal, COHERENCE_* thresholds, SIGNAL_TYPES
#   from ng_ecosystem.py to this Elmer-local file.
# Why: Law 2 — vendored files are sacred. Module-specific code cannot
#   live inside vendored files. Elmer's ng_ecosystem.py had ~150 lines
#   of Elmer-specific code injected, preventing vendored file sync.
# How: Extracted verbatim. Updated all Elmer imports to source from
#   core.substrate_signal instead of ng_ecosystem.
# -------------------
# [2026-02-28] Claude (Opus 4.6) — §6.1 compliant rewrite.
#   What: Flat scored fields, Elmer-specific extensions, signal_type as
#         string enum matching et_module.json signal_types array.
#   Why:  PRD v0.2.0 §6.1 mandates specific schema for cross-module
#         interoperability with NG ecosystem.
# -------------------
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


# Threshold constants (PRD §14) — bootstrap defaults.
# SVG Phase 3: config-backed via CoherenceConfig. These module-level
# constants are retained as fallbacks and for direct import compatibility.
COHERENCE_HEALTHY: float = 0.70
COHERENCE_DEGRADED: float = 0.40
COHERENCE_CRITICAL: float = 0.15

# Valid signal_type values (PRD §6.1, Appendix B)
SIGNAL_TYPES = ("observation", "anomaly", "coherence", "health")


@dataclass(frozen=True)
class SubstrateSignal:
    """Immutable signal passed between Elmer sockets through the substrate bus.

    Every piece of data flowing through Elmer's processing pipeline is
    wrapped in a SubstrateSignal.  Flat scored fields ensure uniform
    logging, routing, threshold checks, and NG substrate compatibility.

    Ref: PRD §6.1

    Attributes:
        signal_id:          UUID4 unique identifier.
        module_id:          Always "elmer" for signals originating here.
        signal_type:        One of: observation, anomaly, coherence, health.
        coherence_score:    Graph-topology coherence [0.0, 1.0].
        health_score:       Overall substrate health [0.0, 1.0].
        anomaly_level:      Detected anomaly magnitude [0.0, 1.0].
        novelty:            Novelty relative to known patterns [0.0, 1.0].
        confidence:         Producer's confidence in this signal [0.0, 1.0].
        severity:           Severity for alerting [0.0, 1.0].
        temporal_window:    Observation window in seconds.
        description:        Human-readable description.
        metadata:           Additional context (source, lineage, debug).
        timestamp:          Unix timestamp of signal creation.
        identity_coherence: Elmer-specific: self-model consistency [0.0, 1.0].
        pruning_pressure:   Elmer-specific: need for graph pruning [0.0, 1.0].
        topology_health:    Elmer-specific: graph topology health [0.0, 1.0].
    """
    signal_id: str
    module_id: str
    signal_type: str
    coherence_score: float
    health_score: float
    anomaly_level: float
    novelty: float
    confidence: float
    severity: float
    temporal_window: float
    description: str
    metadata: Dict[str, Any]
    timestamp: float
    # Elmer-specific extensions (PRD §6.1)
    identity_coherence: float = 1.0
    pruning_pressure: float = 0.0
    topology_health: float = 1.0

    def __post_init__(self) -> None:
        if self.signal_type not in SIGNAL_TYPES:
            raise ValueError(
                f"signal_type must be one of {SIGNAL_TYPES}, got {self.signal_type!r}"
            )

    @classmethod
    def create(
        cls,
        signal_type: str,
        description: str,
        *,
        coherence_score: float = 1.0,
        health_score: float = 1.0,
        anomaly_level: float = 0.0,
        novelty: float = 0.0,
        confidence: float = 1.0,
        severity: float = 0.0,
        temporal_window: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        identity_coherence: float = 1.0,
        pruning_pressure: float = 0.0,
        topology_health: float = 1.0,
    ) -> "SubstrateSignal":
        """Factory with auto-generated signal_id and timestamp."""
        return cls(
            signal_id=str(uuid.uuid4()),
            module_id="elmer",
            signal_type=signal_type,
            coherence_score=coherence_score,
            health_score=health_score,
            anomaly_level=anomaly_level,
            novelty=novelty,
            confidence=confidence,
            severity=severity,
            temporal_window=temporal_window,
            description=description,
            metadata=metadata or {},
            timestamp=time.time(),
            identity_coherence=identity_coherence,
            pruning_pressure=pruning_pressure,
            topology_health=topology_health,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubstrateSignal":
        """Deserialize from a dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def with_updates(self, **kwargs: Any) -> "SubstrateSignal":
        """Return a new signal with specified fields replaced (frozen-safe)."""
        d = asdict(self)
        d.update(kwargs)
        return SubstrateSignal(**d)

    @property
    def coherence_status(self) -> str:
        """Classify coherence_score against §14 thresholds."""
        if self.coherence_score >= COHERENCE_HEALTHY:
            return "healthy"
        if self.coherence_score >= COHERENCE_DEGRADED:
            return "degraded"
        if self.coherence_score >= COHERENCE_CRITICAL:
            return "warning"
        return "critical"
