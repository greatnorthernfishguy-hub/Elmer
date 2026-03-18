"""
ElmerSocket — Abstract Base Class for Elmer Processing Sockets  (PRD §5.2.1)

Every processing unit in Elmer implements this interface.  Sockets receive
a GraphSnapshot, process it, and return a SocketOutput containing a
SubstrateSignal and optional graph delta.

Supporting data structures (PRD §5.2.2):
  GraphSnapshot   — point-in-time view of the NG-Lite graph
  SocketOutput    — result of socket processing
  HardwareRequirements — what a socket needs to run
  SocketHealth    — runtime health telemetry

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §5.2.1 / §5.2.2 compliant rewrite.
#   What: ElmerSocket ABC with declare_requirements, load, unload,
#         process(GraphSnapshot, context) -> SocketOutput, health.
#         Plus four supporting dataclasses.
#   Why:  PRD v0.2.0 §5.2 mandates this interface for uniform socket
#         management and hardware-aware orchestration.
# -------------------
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.substrate_signal import SubstrateSignal


# --------------------------------------------------------------------------
# Data structures  (PRD §5.2.2)
# --------------------------------------------------------------------------

@dataclass
class GraphSnapshot:
    """Point-in-time view of the NG-Lite graph fed to sockets.

    Ref: PRD §5.2.2
    """
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    @classmethod
    def empty(cls) -> "GraphSnapshot":
        return cls(timestamp=time.time())


@dataclass
class SocketOutput:
    """Result of socket processing.

    Ref: PRD §5.2.2
    """
    signal: SubstrateSignal
    graph_delta: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class HardwareRequirements:
    """What a socket needs to run.

    Ref: PRD §5.2.2
    """
    min_memory_mb: int = 128
    gpu_required: bool = False
    cpu_cores: int = 1
    disk_mb: int = 0


@dataclass
class SocketHealth:
    """Runtime health telemetry for a socket.

    Ref: PRD §5.2.2
    """
    status: str = "offline"        # "healthy" | "degraded" | "offline"
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    error_count: int = 0
    last_check: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# ElmerSocket ABC  (PRD §5.2.1)
# --------------------------------------------------------------------------

class ElmerSocket(ABC):
    """Abstract base class for all Elmer processing sockets.

    Lifecycle: construct → load(model_path) → process()* → unload()

    Ref: PRD §5.2.1
    """

    def __init__(self) -> None:
        self._loaded = False
        self._process_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        self._load_time = 0.0

    # -----------------------------------------------------------------
    # Abstract interface  (PRD §5.2.1)
    # -----------------------------------------------------------------

    @property
    @abstractmethod
    def socket_id(self) -> str:
        """Unique identifier for this socket instance."""
        ...

    @property
    @abstractmethod
    def socket_type(self) -> str:
        """Socket type classification (e.g., 'comprehension', 'monitoring')."""
        ...

    @abstractmethod
    def declare_requirements(self) -> HardwareRequirements:
        """Declare hardware requirements for this socket.

        Called by SocketManager before load() to verify resource
        availability and allocate hardware.
        """
        ...

    @abstractmethod
    def load(self, model_path: str) -> bool:
        """Load models/resources and prepare for processing.

        Args:
            model_path: Path to model directory or file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release models/resources.  Should be idempotent."""
        ...

    @abstractmethod
    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Process a graph snapshot and return socket output.

        Args:
            snapshot: Current graph state.
            context: Processing context (autonomic state, config, etc.).

        Returns:
            SocketOutput with SubstrateSignal and optional graph delta.
        """
        ...

    @abstractmethod
    def health(self) -> SocketHealth:
        """Return current health telemetry."""
        ...

    # -----------------------------------------------------------------
    # Concrete helpers
    # -----------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _make_health(self, status: str = "healthy") -> SocketHealth:
        """Build a SocketHealth from internal counters."""
        avg_latency = (
            (self._total_latency / self._process_count * 1000)
            if self._process_count > 0
            else 0.0
        )
        return SocketHealth(
            status=status if self._loaded else "offline",
            latency_ms=avg_latency,
            memory_mb=0.0,  # placeholder — real monitoring in Phase 3
            error_count=self._error_count,
            last_check=time.time(),
        )
