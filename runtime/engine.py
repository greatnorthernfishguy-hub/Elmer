"""
Elmer Engine — Substrate Processing Orchestrator

The ElmerEngine is the central runtime that ties together sockets,
pipelines, the NG ecosystem, and the substrate bus.  It receives
raw input, produces SubstrateSignals, routes them through the
appropriate sockets and pipelines, and returns enriched output.

Phase 1: Minimal orchestration — receive text, create signal, route
through sockets, return result.  Full pipeline chaining in Phase 2.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 initial creation.
#   What: ElmerEngine with start/stop lifecycle, process_text entry
#         point, and NG ecosystem integration.
#   Why:  Single orchestration point for all Elmer processing.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.config import ElmerConfig, load_config
from core.socket_manager import SocketManager
from core.comprehension import ComprehensionSocket
from core.monitoring import MonitoringSocket
from ng_ecosystem import NGEcosystem, SubstrateSignal, SignalType
from runtime.graph_encoder import GraphEncoder
from runtime.signal_decoder import SignalDecoder

logger = logging.getLogger("elmer.engine")


class ElmerEngine:
    """Central runtime orchestrator for the Elmer substrate.

    Lifecycle: configure → start() → process_text()* → stop()

    Usage:
        engine = ElmerEngine()
        engine.start()
        result = engine.process_text("Hello, what can you tell me?")
        health = engine.health()
        engine.stop()
    """

    def __init__(self, config: Optional[ElmerConfig] = None) -> None:
        self._config = config or load_config()
        self._socket_manager = SocketManager(
            max_sockets=self._config.sockets.max_sockets,
        )
        self._graph_encoder = GraphEncoder()
        self._signal_decoder = SignalDecoder()
        self._ecosystem: Optional[NGEcosystem] = None
        self._started = False
        self._start_time = 0.0
        self._process_count = 0

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> Dict[str, Any]:
        """Initialize and start the Elmer engine.

        Registers default sockets, connects them, and initializes the
        NG ecosystem.

        Returns:
            Startup report with socket connection results and tier info.
        """
        if self._started:
            return {"status": "already_started"}

        logger.info("Starting Elmer engine v%s", self._config.version)

        # Register default sockets
        self._socket_manager.register(ComprehensionSocket())
        self._socket_manager.register(MonitoringSocket())

        # Connect all sockets
        connect_results = self._socket_manager.connect_all()

        # Initialize NG ecosystem
        eco_config = self._config.ng_ecosystem
        try:
            self._ecosystem = NGEcosystem.get_instance(
                module_id=eco_config.module_id,
                state_path=eco_config.state_path or None,
                config={
                    "peer_bridge": {
                        "enabled": eco_config.peer_bridge_enabled,
                        "sync_interval": eco_config.peer_sync_interval,
                    },
                    "tier3_upgrade": {
                        "enabled": eco_config.tier3_upgrade_enabled,
                        "poll_interval": eco_config.tier3_poll_interval,
                    },
                },
            )
            eco_tier = self._ecosystem.tier
        except Exception as exc:
            logger.warning("NG ecosystem init failed (standalone mode): %s", exc)
            self._ecosystem = None
            eco_tier = 0

        self._started = True
        self._start_time = time.time()

        report = {
            "status": "started",
            "version": self._config.version,
            "sockets": connect_results,
            "ecosystem_tier": eco_tier,
            "hardware": self._socket_manager.detect_hardware(),
        }
        logger.info("Elmer engine started: %s", report)
        return report

    def stop(self) -> None:
        """Gracefully shut down the engine."""
        if not self._started:
            return

        logger.info("Stopping Elmer engine...")
        self._socket_manager.disconnect_all()

        if self._ecosystem:
            self._ecosystem.shutdown()

        self._started = False
        logger.info("Elmer engine stopped")

    # -----------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process raw text through the Elmer substrate.

        Creates a SENSORY SubstrateSignal, routes it through sockets,
        optionally encodes for NG substrate learning, and returns the
        decoded result.

        Args:
            text: Raw input text.

        Returns:
            Dict with processing results, including any substrate
            learning outcomes and signal metadata.
        """
        if not self._started:
            raise RuntimeError("Engine not started. Call start() first.")

        self._process_count += 1

        # Create initial sensory signal
        signal = SubstrateSignal.create(
            source_socket="engine:input",
            signal_type=SignalType.SENSORY,
            payload={"text": text},
            confidence=1.0,
            priority=5,
            metadata={"process_id": self._process_count},
        )

        # Route through socket manager
        processed = self._socket_manager.route_signal(signal)

        # Encode for NG substrate (if ecosystem available)
        graph_encoding = None
        eco_result = None
        if self._ecosystem:
            graph_encoding = self._graph_encoder.encode(processed)
            if graph_encoding and "embedding" in graph_encoding:
                import numpy as np
                embedding = np.array(graph_encoding["embedding"])
                eco_result = self._ecosystem.get_context(embedding)

        # Decode result
        result = self._signal_decoder.decode(processed)
        result["process_id"] = self._process_count
        if eco_result:
            result["ecosystem"] = eco_result

        return result

    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Return comprehensive health report.

        Returns:
            Dict with engine status, socket health, ecosystem tier,
            and hardware info.
        """
        socket_health = self._socket_manager.health_report()

        eco_stats = None
        if self._ecosystem:
            eco_stats = self._ecosystem.stats()

        return {
            "status": "healthy" if self._started else "offline",
            "version": self._config.version,
            "uptime": time.time() - self._start_time if self._started else 0.0,
            "process_count": self._process_count,
            "sockets": socket_health,
            "ecosystem": eco_stats,
        }
