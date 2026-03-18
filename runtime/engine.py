"""
Elmer Engine — Substrate Processing Orchestrator  (PRD §9)

Central runtime that ties together sockets, pipelines, the NG ecosystem,
and the autonomic monitor.  Receives raw input, builds GraphSnapshots,
routes through sockets, chains pipelines, and returns SocketOutputs.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — §5.2/§7/§8/§9 compliant rewrite.
#   What: ElmerEngine using GraphSnapshot routing, autonomic-aware
#         context, pipeline chaining, and NG ecosystem integration.
#   Why:  PRD v0.2.0 §9 mandates orchestration via GraphSnapshot/
#         SocketOutput flow with autonomic modulation.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.base_socket import GraphSnapshot, SocketOutput
from core.comprehension import ComprehensionSocket
from core.config import ElmerConfig, load_config
from core.monitoring import MonitoringSocket
from core.socket_manager import SocketManager
import ng_autonomic
from ng_ecosystem import NGEcosystem
from core.substrate_signal import SubstrateSignal
from pipelines.health import HealthPipeline
from pipelines.identity import IdentityPipeline
from pipelines.inference import InferencePipeline
from pipelines.memory import MemoryPipeline
from pipelines.sensory import SensoryPipeline
from runtime.graph_encoder import GraphEncoder
from runtime.signal_decoder import SignalDecoder

logger = logging.getLogger("elmer.engine")


class ElmerEngine:
    """Central runtime orchestrator for the Elmer substrate.

    Lifecycle: configure → start() → process_text()* → stop()

    Ref: PRD §9
    """

    def __init__(self, config: Optional[ElmerConfig] = None) -> None:
        self._config = config or load_config()
        self._socket_manager = SocketManager(
            max_sockets=self._config.sockets.max_sockets,
            model_dir="models",
        )
        self._graph_encoder = GraphEncoder()
        self._signal_decoder = SignalDecoder()
        # Autonomic state read via ng_autonomic.read_state() (read-only, PRD §7)

        # Pipelines  (PRD §8)
        self._sensory = SensoryPipeline()
        self._inference = InferencePipeline()
        self._health = HealthPipeline()
        self._memory = MemoryPipeline()
        self._identity = IdentityPipeline()

        self._ecosystem: Optional[NGEcosystem] = None
        self._started = False
        self._start_time = 0.0
        self._process_count = 0

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> Dict[str, Any]:
        """Initialize and start the engine.

        Registers default sockets, loads them, and connects NG ecosystem.

        Returns:
            Startup report dict.
        """
        if self._started:
            return {"status": "already_started"}

        logger.info("Starting Elmer engine v%s", self._config.version)

        # Register and load default sockets
        self._socket_manager.register(ComprehensionSocket())
        self._socket_manager.register(MonitoringSocket())
        load_results = self._socket_manager.load_all()

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
            logger.warning("NG ecosystem init failed (standalone): %s", exc)
            self._ecosystem = None
            eco_tier = 0

        self._started = True
        self._start_time = time.time()

        report = {
            "status": "started",
            "version": self._config.version,
            "sockets": load_results,
            "ecosystem_tier": eco_tier,
            "hardware": SocketManager.detect_hardware(),
        }
        logger.info("Elmer engine started: %s", report)
        return report

    def stop(self) -> None:
        """Graceful shutdown."""
        if not self._started:
            return

        logger.info("Stopping Elmer engine...")
        self._socket_manager.unload_all()

        if self._ecosystem:
            self._ecosystem.shutdown()

        self._started = False
        logger.info("Elmer engine stopped")

    # -----------------------------------------------------------------
    # Processing  (PRD §9)
    # -----------------------------------------------------------------

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process raw text through the full Elmer substrate.

        Flow:
          1. Sensory pipeline → observation signal
          2. GraphEncoder → GraphSnapshot
          3. SocketManager routes to ComprehensionSocket + MonitoringSocket
          4. Inference pipeline → coherence signal
          5. Memory pipeline → store
          6. Identity pipeline → enrich
          7. NG ecosystem recording (if available)
          8. SignalDecoder → output dict

        Ref: PRD §9

        Args:
            text: Raw input text.

        Returns:
            Processing result dict.
        """
        if not self._started:
            raise RuntimeError("Engine not started. Call start() first.")

        self._process_count += 1

        # Read autonomic state  (PRD §7)
        autonomic = ng_autonomic.read_state()
        context = {
            "autonomic_state": autonomic.get("state", "PARASYMPATHETIC"),
            "autonomic_intensity": autonomic.get("threat_level", "none"),
            "process_id": self._process_count,
        }

        # 1. Sensory pipeline  (PRD §8)
        sensory_signal = self._sensory.process(text)

        # 2. Build GraphSnapshot from signal  (PRD §5.2.2)
        snapshot = self._graph_encoder.signal_to_snapshot(sensory_signal)

        # 3. Route through sockets  (PRD §5.2)
        socket_outputs = self._socket_manager.route(snapshot, context)

        # 4. Inference pipeline on first socket output  (PRD §8)
        if socket_outputs:
            inference_signal = self._inference.process(socket_outputs[0].signal)
        else:
            inference_signal = self._inference.process(sensory_signal)

        # 5. Memory store  (PRD §8)
        self._memory.store(inference_signal)

        # 6. Identity enrichment  (PRD §8)
        final_signal = self._identity.process(inference_signal)

        # 7. NG ecosystem recording  (PRD §7)
        eco_result = None
        if self._ecosystem:
            try:
                encoding = self._graph_encoder.encode(final_signal)
                if encoding and "embedding" in encoding:
                    import numpy as np
                    embedding = np.array(encoding["embedding"])
                    self._ecosystem.record_outcome(
                        embedding=embedding,
                        target_id="elmer:substrate_input",
                        success=True,
                        metadata={"process_id": self._process_count},
                    )
                    eco_result = self._ecosystem.get_context(embedding)
            except Exception as exc:
                logger.warning("NG ecosystem recording failed: %s", exc)

        # 8. Decode to output  (PRD §9)
        result = self._signal_decoder.decode(final_signal)
        result["process_id"] = self._process_count
        result["autonomic_state"] = autonomic.get("state", "PARASYMPATHETIC")
        if eco_result:
            result["ecosystem"] = eco_result
        if socket_outputs:
            result["socket_outputs"] = [
                self._signal_decoder.decode(o.signal) for o in socket_outputs
            ]

        return result

    # -----------------------------------------------------------------
    # Health  (PRD §14)
    # -----------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Return comprehensive health report."""
        socket_health = self._socket_manager.health_report()
        health_signal = self._health.check()

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
            "health_signal": health_signal.to_dict(),
            "autonomic_state": ng_autonomic.read_state().get("state", "PARASYMPATHETIC"),
            "pipelines": {
                "sensory": self._sensory.stats(),
                "inference": self._inference.stats(),
                "health": self._health.stats(),
                "memory": self._memory.stats(),
                "identity": self._identity.stats(),
            },
        }
