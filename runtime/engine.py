"""
Elmer Engine — Substrate Processing Orchestrator  (PRD §9)

Central runtime that ties together sockets, pipelines, the NG ecosystem,
and the autonomic monitor.  Receives raw input, builds GraphSnapshots,
routes through sockets, chains pipelines, and returns SocketOutputs.

# ---- Changelog ----
# [2026-03-25] Claude Code (Opus 4.6) — Dynamic brain switching
# What: BrainSwitcher dynamically swaps between frozen ElmerBrain and
#   living ProtoUniBrain based on VPS resource availability.
# Why: Can't run both brains simultaneously (16GB VPS). Dynamic switching
#   lets ProtoUniBrain wake up when resources allow, fall back to
#   ElmerBrain when the VPS is busy. User activity resets idle timer.
# How: BrainSwitcher monitors RAM/CPU/idle in a background thread.
#   Starts with ElmerBrain (safe). Upgrades to ProtoUniBrain when:
#   RAM > 6.5GB free, CPU load < 2.0, idle > 120s. Downgrades when
#   resources tighten or user becomes active. 5-min cooldown between
#   switches. Integrated into process_text (notify_input), stop(),
#   and health() for full lifecycle management.
# [2026-03-24] Claude Code (Opus 4.6) — TuningSocket wiring (homeostasis audit)
# What: Register TuningSocket in start(), wire NGLite ref, apply tuning
#   recommendations in process_text() after socket routing.
# Why: Elmer was observation-only for substrate health. The 2026-03-23 tuning
#   (zero firing in 1,931 steps) was manual. TuningSocket closes the loop —
#   Elmer can now adjust substrate parameters (success_boost, failure_penalty,
#   novelty_threshold, pruning_threshold, receptor params) via NGLite's
#   validated update_tunable() API.
# How: TuningSocket registered alongside Comprehension + Monitoring + Myelination.
#   NGLite ref set after ecosystem init. _apply_tuning() reads socket outputs
#   for tuning recommendations and calls ng_lite.update_tunable(). Pattern B
#   feedback via pending health comparison on subsequent cycles.
# [2026-03-23] Claude (Opus 4.6) — MyelinationSocket wiring (punchlist #53 v0.4)
# What: Register MyelinationSocket in start(), wire bridge ref, apply
#   myelination recommendations in process_text() after socket routing.
# Why: Elmer is the oligodendrocyte — decides which tracts get myelinated.
#   MyelinationSocket extracts pathway patterns, engine applies decisions.
# How: MyelinationSocket registered alongside Comprehension + Monitoring.
#   Bridge ref set after ecosystem init. _apply_myelination() reads
#   socket outputs for myelination recommendations and calls
#   myelinate_tract()/demyelinate_tract() on Elmer's own peer bridge.
# -------------------
# [2026-03-19] Claude Code (Opus 4.6) — Cricket rim integration
# What: Load constitutional embeddings and pass to NGEcosystem/NGLite
#   on init. Elmer's extraction bucket now enforces the rim —
#   constitutional nodes in the topology prevent learning in forbidden
#   semantic space.
# Why: Cricket Design v0.1 — Cricket IS the bucket. Elmer defines the
#   bucket's shape, Cricket defines the material. The rim is universal
#   and lives in the vendored ng_lite.py. Elmer wires the embeddings.
# How: _load_constitutional_embeddings() reads data/constitutional_embeddings.json.
#   Embeddings passed to NGEcosystem via config["ng_lite"]["constitutional_embeddings"].
#   Autonomic write path for rim violations added to process_text().
# -------------------
# [2026-02-28] Claude (Opus 4.6) — §5.2/§7/§8/§9 compliant rewrite.
#   What: ElmerEngine using GraphSnapshot routing, autonomic-aware
#         context, pipeline chaining, and NG ecosystem integration.
#   Why:  PRD v0.2.0 §9 mandates orchestration via GraphSnapshot/
#         SocketOutput flow with autonomic modulation.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from core.base_socket import GraphSnapshot, SocketOutput
from core.comprehension import ComprehensionSocket
from core.config import ElmerConfig, load_config
from core.myelination import MyelinationSocket
from core.monitoring import MonitoringSocket
from core.tuning import TuningSocket
from core.socket_manager import SocketManager
try:
    from core.neural_comprehension import NeuralComprehensionSocket
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False
try:
    from core.brain_socket import BrainSocket
    from core.proto_brain_socket import ProtoUniBrainSocket
    _BRAIN_AVAILABLE = True
except ImportError:
    _BRAIN_AVAILABLE = False
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
        # PRD §5.2.3: Use neural sockets when available and configured
        use_neural = self._config.sockets.neural_mode and _NEURAL_AVAILABLE
        if use_neural:
            logger.info("Neural mode enabled — registering NeuralComprehensionSocket")
            self._socket_manager.register(NeuralComprehensionSocket())
        else:
            logger.info("Heuristic mode — registering ComprehensionSocket")
            self._socket_manager.register(ComprehensionSocket())
        self._socket_manager.register(MonitoringSocket())
        self._myelination_socket = MyelinationSocket()
        self._socket_manager.register(self._myelination_socket)
        self._tuning_socket = TuningSocket()
        self._socket_manager.register(self._tuning_socket)
        # UniAI brain switcher — dynamically swaps between frozen ElmerBrain
        # and living ProtoUniBrain based on resource availability
        self._brain_switcher = None
        if _BRAIN_AVAILABLE:
            try:
                from core.brain_switcher import BrainSwitcher
                self._brain_switcher = BrainSwitcher(self._socket_manager)
                self._brain_switcher.start()
                logger.info("BrainSwitcher started (ElmerBrain active, "
                            "ProtoUniBrain on standby)")
            except Exception as exc:
                logger.warning("BrainSwitcher init failed: %s", exc)
        else:
            logger.info("Brain sockets not available (torch/transformers not installed)")
        load_results = self._socket_manager.load_all()

        # Initialize NG ecosystem with Cricket rim (constitutional embeddings)
        eco_config = self._config.ng_ecosystem
        constitutional = self._load_constitutional_embeddings()
        try:
            ng_lite_config: Dict[str, Any] = {}
            if constitutional:
                ng_lite_config["constitutional_embeddings"] = constitutional
                logger.info("Cricket rim: %d constitutional embeddings loaded", len(constitutional))

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
                    "ng_lite": ng_lite_config,
                },
            )
            eco_tier = self._ecosystem.tier
            # Wire MyelinationSocket to Elmer's own peer bridge for event counts
            if hasattr(self._ecosystem, '_peer_bridge') and self._ecosystem._peer_bridge:
                self._myelination_socket.set_bridge_ref(self._ecosystem._peer_bridge)
                logger.info("MyelinationSocket wired to peer bridge")
            # Wire TuningSocket to Elmer's local NGLite for parameter adjustment
            if hasattr(self._ecosystem, '_graph') and self._ecosystem._graph:
                self._tuning_socket.set_ng_lite_ref(self._ecosystem._graph)
                logger.info("TuningSocket wired to NGLite substrate")
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
        if self._brain_switcher:
            self._brain_switcher.stop()
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

        # Notify brain switcher of user activity
        if self._brain_switcher:
            self._brain_switcher.notify_input()

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

        # 3b. Apply myelination recommendations from MyelinationSocket
        self._apply_myelination(socket_outputs)

        # 3c. Apply tuning recommendations from TuningSocket
        self._apply_tuning(socket_outputs)

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
                    outcome = self._ecosystem.dual_record_outcome(
                        content=text,
                        embedding=embedding,
                        target_id="elmer:substrate_input",
                        success=True,
                        metadata={"process_id": self._process_count},
                    )

                    # Cricket rim: if record_outcome hit a constitutional node,
                    # escalate to SYMPATHETIC. This is the sole exception to
                    # Elmer's read-only autonomic rule (CLAUDE.md §2).
                    if outcome.get("constitutional"):
                        logger.warning(
                            "Cricket rim activated — constitutional node %s, "
                            "escalating to SYMPATHETIC",
                            outcome.get("node_id"),
                        )
                        ng_autonomic.write_state({
                            "state": "SYMPATHETIC",
                            "threat_level": "constitutional",
                            "source": "cricket_rim",
                            "module": "elmer",
                            "node_id": outcome.get("node_id"),
                        })

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
    # Myelination  (punchlist #53 v0.4)
    # -----------------------------------------------------------------

    def _apply_myelination(self, socket_outputs: List[SocketOutput]) -> None:
        """Apply myelination recommendations from the MyelinationSocket.

        Reads the MyelinationSocket's output and calls myelinate/demyelinate
        on Elmer's own peer bridge.  This is Elmer managing its own tracts —
        not reaching into another module's bridge (Law 1 compliant).
        """
        if not self._ecosystem or not hasattr(self._ecosystem, '_peer_bridge'):
            return
        bridge = self._ecosystem._peer_bridge
        if bridge is None or not hasattr(bridge, 'myelinate_tract'):
            return

        for output in socket_outputs:
            meta = output.signal.metadata
            if meta.get("socket") != "elmer:myelination":
                continue

            recs = meta.get("myelination_recommendations", {})

            for peer_id in recs.get("myelinate", []):
                if not bridge.is_myelinated(peer_id):
                    if bridge.myelinate_tract(peer_id):
                        logger.info("Myelination: upgraded tract →%s", peer_id)

            for peer_id in recs.get("demyelinate", []):
                if bridge.is_myelinated(peer_id):
                    if bridge.demyelinate_tract(peer_id):
                        logger.info("Myelination: downgraded tract →%s", peer_id)

    # -----------------------------------------------------------------
    # Tuning  (Phase 4 — Elmer outward)
    # -----------------------------------------------------------------

    def _apply_tuning(self, socket_outputs: List[SocketOutput]) -> None:
        """Apply tuning recommendations from the TuningSocket.

        Reads the TuningSocket's output, applies parameter adjustments via
        NGLite.update_tunable(), and records outcomes back to the substrate
        for Pattern B learning.
        """
        if not self._ecosystem or not hasattr(self._ecosystem, '_graph'):
            return
        ng_lite = self._ecosystem._graph
        if ng_lite is None or not hasattr(ng_lite, 'update_tunable'):
            return

        for output in socket_outputs:
            meta = output.signal.metadata
            if meta.get("socket") != "elmer:tuning":
                continue
            if meta.get("frozen"):
                continue

            recs = meta.get("tuning_recommendations", [])
            health_before = meta.get("health_snapshot", {}).get("overall_health", 1.0)

            for rec in recs:
                key = rec["key"]
                proposed = rec["proposed"]
                try:
                    result = ng_lite.update_tunable(key, proposed)
                    logger.info(
                        "Tuning applied: %s %.6f → %.6f (%s)",
                        key, result["old_value"], result["new_value"],
                        rec["reason"],
                    )

                    # Record pending outcome for competence feedback.
                    # Health_after comes on the next cycle — the socket
                    # resolves it automatically via _resolve_pending_outcomes().
                    self._tuning_socket.record_pending_outcome(
                        key=key,
                        old_value=result["old_value"],
                        new_value=result["new_value"],
                        health_before=health_before,
                    )

                except KeyError as exc:
                    logger.warning("Tuning rejected: %s", exc)
                except Exception as exc:
                    logger.error("Tuning error for %s: %s", key, exc)

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

        brain_status = None
        if self._brain_switcher:
            brain_status = self._brain_switcher.status()

        return {
            "status": "healthy" if self._started else "offline",
            "brain": brain_status,
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

    # -----------------------------------------------------------------
    # Cricket Rim
    # -----------------------------------------------------------------

    @staticmethod
    def _load_constitutional_embeddings() -> List[Dict[str, Any]]:
        """Load constitutional embeddings from data/constitutional_embeddings.json.

        Returns the embeddings list for passing to NGLite via config.
        Returns empty list if file not found (module runs without rim —
        constitutional enforcement degrades gracefully).
        """
        paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "constitutional_embeddings.json"),
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    return data.get("embeddings", [])
                except Exception as exc:
                    logger.warning("Failed to load constitutional embeddings from %s: %s", path, exc)
        logger.info("No constitutional embeddings found — Cricket rim inactive")
        return []
