"""
Socket Manager — Lifecycle, Hardware Allocation, and Signal Routing  (PRD §5.2)

Manages all ElmerSocket instances: registration, load/unload lifecycle,
hardware requirements verification, GraphSnapshot routing, and health
aggregation.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §5.2.1/§5.2.2 compliant rewrite.
#   What: SocketManager using declare_requirements, load/unload,
#         process(GraphSnapshot, context), aggregated health.
#   Why:  Align with PRD v0.2.0 §5.2 socket management contract.
# -------------------
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from core.base_socket import (
    ElmerSocket,
    GraphSnapshot,
    SocketHealth,
    SocketOutput,
)

logger = logging.getLogger("elmer.socket_manager")


class SocketManager:
    """Manages ElmerSocket lifecycle and routing.

    Ref: PRD §5.2

    Responsibilities:
      - Register / unregister sockets
      - Verify hardware requirements via declare_requirements()
      - Coordinate load / unload lifecycle
      - Route GraphSnapshots to appropriate sockets
      - Aggregate health reports
      - Hardware detection (GPU / CPU / NPU)
    """

    def __init__(self, max_sockets: int = 16, model_dir: str = "models") -> None:
        self._sockets: Dict[str, ElmerSocket] = {}
        self._max_sockets = max_sockets
        self._model_dir = model_dir

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def register(self, socket: ElmerSocket) -> None:
        """Register a socket.  Raises if duplicate or at capacity."""
        sid = socket.socket_id
        if sid in self._sockets:
            raise ValueError(f"Socket already registered: {sid}")
        if len(self._sockets) >= self._max_sockets:
            raise RuntimeError(f"Max sockets ({self._max_sockets}) reached")
        self._sockets[sid] = socket
        logger.info("Registered socket: %s (%s)", sid, socket.socket_type)

    def unregister(self, socket_id: str) -> None:
        """Unregister a socket.  Unloads first if loaded."""
        sock = self._sockets.pop(socket_id, None)
        if sock and sock.is_loaded:
            sock.unload()
        logger.info("Unregistered socket: %s", socket_id)

    def get_socket(self, socket_id: str) -> Optional[ElmerSocket]:
        return self._sockets.get(socket_id)

    def list_sockets(self) -> List[Dict[str, Any]]:
        return [
            {
                "socket_id": s.socket_id,
                "socket_type": s.socket_type,
                "loaded": s.is_loaded,
                "requirements": s.declare_requirements().__dict__,
            }
            for s in self._sockets.values()
        ]

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def load_all(self) -> Dict[str, bool]:
        """Load all registered sockets.  Returns socket_id -> success."""
        results = {}
        for sid, sock in self._sockets.items():
            model_path = os.path.join(self._model_dir, sock.socket_type)
            results[sid] = sock.load(model_path)
        return results

    def unload_all(self) -> None:
        """Unload all sockets."""
        for sock in self._sockets.values():
            if sock.is_loaded:
                sock.unload()

    # -----------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------

    def route(
        self,
        snapshot: GraphSnapshot,
        context: dict,
        socket_type: Optional[str] = None,
    ) -> List[SocketOutput]:
        """Route a GraphSnapshot to matching loaded sockets.

        If socket_type is given, only route to sockets of that type.
        Otherwise route to all loaded sockets.

        Returns list of SocketOutputs (one per matching socket).
        """
        outputs: List[SocketOutput] = []
        for sock in self._sockets.values():
            if not sock.is_loaded:
                continue
            if socket_type and sock.socket_type != socket_type:
                continue
            try:
                out = sock.process(snapshot, context)
                outputs.append(out)
            except Exception as exc:
                logger.error("Socket %s process error: %s", sock.socket_id, exc)
        return outputs

    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    def health_report(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate health from all sockets."""
        return {
            sid: sock.health().to_dict()
            for sid, sock in self._sockets.items()
        }

    # -----------------------------------------------------------------
    # Hardware detection  (PRD §5.2)
    # -----------------------------------------------------------------

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Detect available hardware (GPU/CPU/NPU)."""
        hw: Dict[str, Any] = {
            "cpu_cores": os.cpu_count() or 1,
            "gpu_available": False,
            "gpu_name": None,
            "npu_available": False,
        }

        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                hw["gpu_available"] = True
                hw["gpu_name"] = result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            hw["disk_free_mb"] = free // (1024 * 1024)
        except OSError:
            hw["disk_free_mb"] = 0

        return hw
