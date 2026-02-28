"""
ElmerSocket — Abstract Base Class for Elmer Processing Sockets

Every processing unit in Elmer implements this interface.  Sockets are
the atomic processing elements: they receive a SubstrateSignal, transform
it, and emit a new SubstrateSignal.  The SocketManager handles lifecycle,
hardware allocation, and signal routing.

Design: Inspired by the neuroscience concept of "processing sockets" —
specific brain regions that receive, transform, and relay signals
through the substrate.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 initial creation.
#   What: ElmerSocket ABC defining the socket contract.
#   Why:  All Elmer processing units share this interface so the
#         SocketManager can manage them uniformly.
# -------------------
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ng_ecosystem import SubstrateSignal

logger = logging.getLogger("elmer.socket")


class ElmerSocket(ABC):
    """Abstract base class for all Elmer processing sockets.

    A socket is a self-contained processing unit that:
      1. Connects to hardware resources (GPU, CPU, NPU)
      2. Receives SubstrateSignals from the bus
      3. Processes them (transform, enrich, filter)
      4. Emits new SubstrateSignals

    Lifecycle: construct → connect() → process()* → disconnect()

    Subclasses MUST implement all abstract methods and properties.
    """

    def __init__(self) -> None:
        self._connected = False
        self._process_count = 0
        self._error_count = 0
        self._last_process_time = 0.0
        self._connect_time = 0.0

    # -----------------------------------------------------------------
    # Abstract interface
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
    def connect(self) -> None:
        """Initialize hardware resources and prepare for processing.

        Called by SocketManager during startup.  Should be idempotent.
        Raises RuntimeError if resources are unavailable.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Release hardware resources and clean up.

        Called by SocketManager during shutdown.  Should be idempotent.
        """
        ...

    @abstractmethod
    def process(self, signal: SubstrateSignal) -> SubstrateSignal:
        """Process an incoming signal and return the transformed result.

        This is the core processing method.  Implementations should:
          - Validate the incoming signal
          - Apply domain-specific transformation
          - Return a new SubstrateSignal (signals are immutable)

        Args:
            signal: Incoming substrate signal.

        Returns:
            Transformed substrate signal.

        Raises:
            ValueError: If the signal is incompatible with this socket.
        """
        ...

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return current health status of this socket.

        Returns:
            Dict with at minimum:
              - status: "healthy" | "degraded" | "offline"
              - socket_id: str
              - socket_type: str
              - uptime: float (seconds since connect)
        """
        ...

    # -----------------------------------------------------------------
    # Concrete properties and methods
    # -----------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Whether this socket is currently connected and ready."""
        return self._connected

    @property
    def hardware_affinity(self) -> str:
        """Preferred hardware type: 'gpu', 'cpu', or 'npu'.

        Override in subclasses that have specific hardware preferences.
        Default is 'cpu'.
        """
        return "cpu"

    def _base_health(self) -> Dict[str, Any]:
        """Common health fields for all sockets."""
        now = time.time()
        return {
            "socket_id": self.socket_id,
            "socket_type": self.socket_type,
            "connected": self._connected,
            "uptime": now - self._connect_time if self._connected else 0.0,
            "process_count": self._process_count,
            "error_count": self._error_count,
            "last_process_time": self._last_process_time,
        }
