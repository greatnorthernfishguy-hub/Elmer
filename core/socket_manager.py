"""
Socket Manager — Lifecycle, Hardware Allocation, and Signal Routing

Manages all ElmerSocket instances: registration, connection lifecycle,
hardware detection/allocation, signal routing, and health monitoring.

Hardware detection reuses the pattern from TID's hardware.py:
  - GPU: check for CUDA/ROCm via torch or environment
  - NPU: check for vendor-specific libs
  - CPU: always available as fallback

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 initial creation.
#   What: SocketManager with register, connect_all, disconnect_all,
#         route_signal, health_report, and hardware detection.
#   Why:  Central coordination for all Elmer processing sockets.
# -------------------
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from core.base_socket import ElmerSocket
from ng_ecosystem import SubstrateSignal, SignalType

logger = logging.getLogger("elmer.socket_manager")


class SocketManager:
    """Manages Elmer socket lifecycle and signal routing.

    Usage:
        manager = SocketManager()
        manager.register(ComprehensionSocket())
        manager.register(MonitoringSocket())
        manager.connect_all()

        result = manager.route_signal(signal)
        health = manager.health_report()

        manager.disconnect_all()
    """

    def __init__(self, max_sockets: int = 16) -> None:
        self._sockets: Dict[str, ElmerSocket] = {}
        self._type_index: Dict[str, List[str]] = {}
        self._max_sockets = max_sockets
        self._hardware: Optional[Dict[str, Any]] = None
        self._started = False

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def register(self, socket: ElmerSocket) -> None:
        """Register a socket with the manager.

        Args:
            socket: Socket instance to register.

        Raises:
            ValueError: If socket_id is already registered or at capacity.
        """
        if socket.socket_id in self._sockets:
            raise ValueError(f"Socket already registered: {socket.socket_id}")
        if len(self._sockets) >= self._max_sockets:
            raise ValueError(
                f"Socket limit reached ({self._max_sockets}). "
                f"Cannot register {socket.socket_id}"
            )

        self._sockets[socket.socket_id] = socket

        # Index by type for routing
        stype = socket.socket_type
        if stype not in self._type_index:
            self._type_index[stype] = []
        self._type_index[stype].append(socket.socket_id)

        logger.info(
            "Registered socket: %s (type=%s, hw=%s)",
            socket.socket_id, socket.socket_type, socket.hardware_affinity,
        )

    def unregister(self, socket_id: str) -> None:
        """Remove a socket from the manager."""
        socket = self._sockets.pop(socket_id, None)
        if socket is None:
            return

        stype = socket.socket_type
        if stype in self._type_index:
            self._type_index[stype] = [
                sid for sid in self._type_index[stype] if sid != socket_id
            ]

        if socket.is_connected:
            try:
                socket.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting %s: %s", socket_id, exc)

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def connect_all(self) -> Dict[str, bool]:
        """Connect all registered sockets.

        Returns:
            Dict of socket_id -> success (True/False).
        """
        if self._hardware is None:
            self._hardware = self.detect_hardware()

        results: Dict[str, bool] = {}
        for sid, socket in self._sockets.items():
            try:
                socket.connect()
                results[sid] = True
                logger.info("Connected socket: %s", sid)
            except Exception as exc:
                results[sid] = False
                logger.error("Failed to connect %s: %s", sid, exc)

        self._started = True
        return results

    def disconnect_all(self) -> None:
        """Disconnect all sockets gracefully."""
        for sid, socket in self._sockets.items():
            if socket.is_connected:
                try:
                    socket.disconnect()
                    logger.info("Disconnected socket: %s", sid)
                except Exception as exc:
                    logger.warning("Error disconnecting %s: %s", sid, exc)

        self._started = False

    # -----------------------------------------------------------------
    # Signal routing
    # -----------------------------------------------------------------

    def route_signal(self, signal: SubstrateSignal) -> SubstrateSignal:
        """Route a signal to the appropriate socket(s) for processing.

        Routing strategy (Phase 1):
          - Match signal_type to socket_type
          - If multiple sockets match, use the first connected one
          - If no match, return the signal unchanged (pass-through)

        Args:
            signal: Incoming substrate signal.

        Returns:
            Processed substrate signal.
        """
        target_type = signal.signal_type.value

        socket_ids = self._type_index.get(target_type, [])
        for sid in socket_ids:
            socket = self._sockets.get(sid)
            if socket and socket.is_connected:
                try:
                    result = socket.process(signal)
                    return result
                except Exception as exc:
                    logger.error(
                        "Socket %s failed to process signal %s: %s",
                        sid, signal.signal_id, exc,
                    )

        # No matching socket — pass through
        return signal

    def get_socket(self, socket_id: str) -> Optional[ElmerSocket]:
        """Retrieve a socket by ID."""
        return self._sockets.get(socket_id)

    def list_sockets(self) -> List[Dict[str, Any]]:
        """List all registered sockets with basic info."""
        return [
            {
                "socket_id": s.socket_id,
                "socket_type": s.socket_type,
                "connected": s.is_connected,
                "hardware_affinity": s.hardware_affinity,
            }
            for s in self._sockets.values()
        ]

    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    def health_report(self) -> Dict[str, Any]:
        """Aggregate health report for all sockets.

        Returns:
            Dict with overall status and per-socket health.
        """
        socket_health: Dict[str, Any] = {}
        statuses: List[str] = []

        for sid, socket in self._sockets.items():
            try:
                h = socket.health_check()
                socket_health[sid] = h
                statuses.append(h.get("status", "unknown"))
            except Exception as exc:
                socket_health[sid] = {"status": "error", "error": str(exc)}
                statuses.append("error")

        # Overall status: healthy if all healthy, degraded if any degraded, offline if all offline
        if not statuses:
            overall = "no_sockets"
        elif all(s == "healthy" for s in statuses):
            overall = "healthy"
        elif all(s in ("offline", "error") for s in statuses):
            overall = "offline"
        else:
            overall = "degraded"

        return {
            "status": overall,
            "socket_count": len(self._sockets),
            "connected_count": sum(1 for s in self._sockets.values() if s.is_connected),
            "hardware": self._hardware,
            "sockets": socket_health,
        }

    # -----------------------------------------------------------------
    # Hardware detection (pattern: TID hardware.py)
    # -----------------------------------------------------------------

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Detect available hardware accelerators.

        Returns:
            Dict with gpu, npu, and cpu capabilities.
        """
        hw: Dict[str, Any] = {
            "cpu": {
                "available": True,
                "cores": os.cpu_count() or 1,
            },
            "gpu": {
                "available": False,
                "devices": [],
                "cuda_available": False,
            },
            "npu": {
                "available": False,
            },
        }

        # GPU detection via PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                hw["gpu"]["available"] = True
                hw["gpu"]["cuda_available"] = True
                hw["gpu"]["devices"] = [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_mem,
                    }
                    for i in range(torch.cuda.device_count())
                ]
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                hw["gpu"]["available"] = True
                hw["gpu"]["devices"] = [{"index": 0, "name": "Apple MPS"}]
        except ImportError:
            pass

        # ROCm detection
        if not hw["gpu"]["available"]:
            try:
                rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
                if os.path.isdir(rocm_path):
                    hw["gpu"]["available"] = True
                    hw["gpu"]["devices"] = [{"index": 0, "name": "AMD ROCm"}]
            except Exception:
                pass

        logger.info(
            "Hardware detected: CPU=%d cores, GPU=%s, NPU=%s",
            hw["cpu"]["cores"],
            "yes" if hw["gpu"]["available"] else "no",
            "yes" if hw["npu"]["available"] else "no",
        )

        return hw
