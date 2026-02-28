"""
NG Autonomic — Autonomic Nervous System State for E-T Modules

Vendored file providing read-only access to the shared autonomic state
that governs ecosystem-wide processing intensity.

Metaphor: biological autonomic nervous system.
  - PARASYMPATHETIC: calm / consolidation mode.  Low priority, gentle
    pruning, memory consolidation.
  - SYMPATHETIC: alert / danger mode.  High priority, aggressive
    processing, anomaly response.

The state is written by the ecosystem coordinator (NeuroGraph / Immunis)
and read by consumer modules like Elmer.

  >>> Elmer reads but NEVER writes autonomic state.  (PRD §7)

State file: ~/.et_modules/autonomic_state.json

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Initial creation per Elmer PRD §7.
#   What: AutonomicState enum, AutonomicMonitor reader class.
#   Why:  Elmer must modulate processing based on ecosystem arousal
#         level — gentle in PARASYMPATHETIC, aggressive in SYMPATHETIC.
# -------------------
"""

from __future__ import annotations

import enum
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ng_autonomic")

__version__ = "1.0.0"

# Default state file location
_DEFAULT_STATE_PATH = os.path.expanduser("~/.et_modules/autonomic_state.json")


class AutonomicState(enum.Enum):
    """Ecosystem arousal level.  (PRD §7)"""
    PARASYMPATHETIC = "parasympathetic"  # calm / consolidation
    SYMPATHETIC = "sympathetic"          # alert / danger


@dataclass(frozen=True)
class AutonomicSnapshot:
    """Point-in-time read of the autonomic state file.

    Attributes:
        state:        Current arousal level.
        intensity:    How strongly the state is expressed [0.0, 1.0].
        changed_at:   Unix timestamp when state last changed.
        source:       Module that wrote the state.
        metadata:     Additional context from the state writer.
        read_at:      Unix timestamp when this snapshot was taken.
    """
    state: AutonomicState = AutonomicState.PARASYMPATHETIC
    intensity: float = 0.0
    changed_at: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    read_at: float = 0.0


class AutonomicMonitor:
    """Read-only monitor for the shared autonomic state.

    Caches reads to avoid excessive disk I/O.  Default cache TTL
    is 5 seconds — autonomic transitions are rare events.

    Usage:
        monitor = AutonomicMonitor()
        snap = monitor.read()
        if snap.state == AutonomicState.SYMPATHETIC:
            # increase processing priority
            ...
    """

    def __init__(
        self,
        state_path: Optional[str] = None,
        cache_ttl: float = 5.0,
    ):
        self._path = Path(state_path or _DEFAULT_STATE_PATH)
        self._cache_ttl = cache_ttl
        self._cached: Optional[AutonomicSnapshot] = None
        self._cache_time: float = 0.0

    def read(self) -> AutonomicSnapshot:
        """Read the current autonomic state.

        Returns a cached snapshot if within TTL.  Falls back to
        PARASYMPATHETIC if the state file is missing or corrupt.
        """
        now = time.time()
        if self._cached and (now - self._cache_time) < self._cache_ttl:
            return self._cached

        snap = self._read_file(now)
        self._cached = snap
        self._cache_time = now
        return snap

    @property
    def state(self) -> AutonomicState:
        """Convenience: current state without full snapshot."""
        return self.read().state

    @property
    def is_sympathetic(self) -> bool:
        return self.state == AutonomicState.SYMPATHETIC

    @property
    def is_parasympathetic(self) -> bool:
        return self.state == AutonomicState.PARASYMPATHETIC

    def _read_file(self, now: float) -> AutonomicSnapshot:
        """Parse the state file, with graceful fallback."""
        if not self._path.exists():
            return AutonomicSnapshot(read_at=now)

        try:
            data = json.loads(self._path.read_text())
            state_str = data.get("state", "parasympathetic")
            try:
                state = AutonomicState(state_str)
            except ValueError:
                state = AutonomicState.PARASYMPATHETIC

            return AutonomicSnapshot(
                state=state,
                intensity=float(data.get("intensity", 0.0)),
                changed_at=float(data.get("changed_at", 0.0)),
                source=str(data.get("source", "")),
                metadata=data.get("metadata", {}),
                read_at=now,
            )
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Failed to read autonomic state: %s", exc)
            return AutonomicSnapshot(read_at=now)
