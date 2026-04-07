"""
Elmer Engine — Substrate Processing Orchestrator  (PRD §9)

Central runtime that ties together sockets, pipelines, the NG ecosystem,
and the autonomic monitor.  Receives raw input, builds GraphSnapshots,
routes through sockets, chains pipelines, and returns SocketOutputs.

# ---- Changelog ----
# [2026-03-28] Claude Code (Opus 4.6) — Encoder v2 wiring corrected
# What: Removed context['graph'] pass-through. Brain sockets read Elmer's
#   own local NG-Lite directly via ecosystem ref set during start().
# Why: Passing the central graph through context dicts is a direct object
#   reference across module boundaries. The brain is part of Elmer.
#   Elmer reads its own substrate. The River brought the topology here.
#   The encoder reads what's already present. Law 1 compliant.
# How: Engine passes ecosystem to BrainSwitcher. BrainSwitcher calls
#   set_ecosystem_ref() on each socket. Sockets read self._ecosystem._graph.
# [2026-03-28] Claude Code (Opus 4.6) — Brain buffer + async drain thread
# What: Brain socket processing decoupled from fan-out via buffer + drain thread.
#   process_text() drops snapshot into buffer (instant). Background drain thread
#   picks up latest snapshot at brain's own pace. KISS observes synchronously (cheap).
# Why: Lenia step (~65s on CPU) blocked the fan-out RPC, exceeding the TS plugin's
#   30s afterTurn timeout. TS stopped sending afterTurn calls. Pipeline froze.
#   Buffer/drain pattern: fan-out returns instantly, brains cook in background.
# How: _drop_to_brain_buffer() replaces synchronous _process_brain_sockets_kissed()
#   in process_text(). _start_brain_drain() launches persistent thread on brain load.
#   Thread loops: grab latest snapshot → brain sockets process → sleep if empty.
#   Only latest snapshot kept — brain reads current state, not history.
# [2026-03-29] Claude Code (Opus 4.6) — skip_brains parameter for fan-out context
# What: engine.start(skip_brains=True) skips BrainSwitcher while starting all
#   lightweight sockets (Comprehension, Monitoring, Myelination, Tuning) + KISS.
# Why: Fan-out context can't load brain models (OOM). But engine.start() was
#   being skipped entirely, leaving _started=False and process_text() unreachable.
#   Now fan-out gets full pipeline + KISS without brains. Brains load standalone.
# How: skip_brains param gates only the BrainSwitcher block. Everything else runs.
# [2026-03-27] Claude Code (Opus 4.6) — Set _started before ecosystem init
# What: Move self._started = True before NGEcosystem.get_instance().
# Why: In the fan-out background thread, NGEcosystem.get_instance() can hang
#   on filesystem locks or bridge init contention. With _started gated after
#   ecosystem init, the engine never reports ready, and every _module_on_message
#   falls through to the legacy pipeline — brains, KISS, and sockets never run.
# How: _started set after socket registration, before ecosystem init.
#   All ecosystem-dependent code in process_text() is already guarded by
#   `if self._ecosystem:` — graceful at Tier 0/1 without ecosystem.
#   Ecosystem init still runs — if it succeeds, engine upgrades to Tier 2/3.
#   If it hangs, engine operates at Tier 0/1 with full socket + KISS capability.
# [2026-03-26] Claude Code (Opus 4.6) — KISS Phase 1 (NuWave Layer 1)
# What: KISSFilter integrated into engine. Delta gate + sparse extract
#   on brain socket input path. Phase 1 is observe-only — logs what
#   WOULD be filtered without actually skipping brain processing.
# Why: NuWave compound optimization. Brain sockets cost ~65s per Lenia
#   step on CPU. Redundant inputs waste that entire cycle. KISS reduces
#   input so every Lenia step processes meaningful data.
# How: KISSFilter tracks last snapshot features. Cosine similarity for
#   delta gate. Sparse diff for change detection. Stats to /tmp/elmer_kiss.log
#   and health() endpoint. Brain socket path split in Phase 1.1.
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
# [2026-03-26] Claude Code Opus — Punchlist #44: Wire TuningSocket bridge ref
# What: Wire peer bridge ref to TuningSocket for absorption rate metrics
# Why: Punchlist #44 — TuningSocket needs bridge stats to compute absorption
#   rate, which drives relevance_threshold tuning decisions.
# How: Same wiring pattern as MyelinationSocket.set_bridge_ref().
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
from core.kiss import KISSFilter, KISSConfig
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
    from core.brain_switcher import BrainSwitcher
    _BRAIN_AVAILABLE = True
except Exception as _brain_exc:
    _BRAIN_AVAILABLE = False
    with open("/tmp/elmer_eager.log", "a") as _f:
        from datetime import datetime
        import traceback
        _f.write(f"[{datetime.now()}] _BRAIN_AVAILABLE=False: {_brain_exc}\n")
        _f.write(traceback.format_exc() + "\n")
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
        # KISS — input optimization for brain sockets (NuWave Layer 1)
        self._kiss = KISSFilter()
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

        # Brain buffer — process_text() drops snapshots here, the brain
        # drain thread picks them up at its own pace. Only the latest
        # snapshot matters (the brain reads current state, not history).
        # This decouples the fan-out RPC timeout (~30s) from Lenia step
        # time (~65s+). The fan-out returns instantly. The brain cooks.
        import threading
        self._brain_buffer_lock = threading.Lock()
        self._brain_latest_snapshot = None  # latest GraphSnapshot
        self._brain_latest_context = None   # latest context dict
        self._brain_drain_thread = None
        self._brain_drain_running = False

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self, skip_brains: bool = False) -> Dict[str, Any]:
        """Initialize and start the engine.

        Registers default sockets, loads them, and connects NG ecosystem.

        Args:
            skip_brains: If True, skip BrainSwitcher/BrainSocket/ProtoUniBrainSocket
                registration. Used in fan-out context where brain models would OOM.
                Lightweight sockets (Comprehension, Monitoring, Myelination, Tuning)
                and KISS still run.

        Returns:
            Startup report dict.
        """
        if self._started:
            return {"status": "already_started"}

        logger.info("Starting Elmer engine v%s (skip_brains=%s)", self._config.version, skip_brains)

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
        # and living ProtoUniBrain based on resource availability.
        # Skipped in fan-out context (skip_brains=True) to avoid OOM from
        # loading two 0.5B transformer models in-process with 7 other modules.
        self._brain_switcher = None
        if _BRAIN_AVAILABLE and not skip_brains:
            try:
                self._brain_switcher = BrainSwitcher(
                    self._socket_manager,
                    brain_socket_cls=BrainSocket,
                    proto_brain_socket_cls=ProtoUniBrainSocket,
                    ecosystem=self._ecosystem,
                )
                self._brain_switcher.start()
                self._start_brain_drain()
                logger.info("BrainSwitcher started (ElmerBrain active, "
                            "ProtoUniBrain on standby, drain thread running)")
            except Exception as exc:
                with open("/tmp/elmer_eager.log", "a") as _f:
                    from datetime import datetime
                    import traceback
                    _f.write(f"[{datetime.now()}] BrainSwitcher FAILED: {exc}\n")
                    _f.write(traceback.format_exc() + "\n")
                logger.warning("BrainSwitcher init failed: %s", exc)
        elif skip_brains:
            logger.info("Brain sockets skipped (fan-out context)")
        else:
            logger.info("Brain sockets not available (torch/transformers not installed)")
        load_results = self._socket_manager.load_all()

        # Mark engine as started BEFORE ecosystem init.
        # Sockets, pipelines, and brain sockets are ready — the engine
        # can process messages at Tier 0/1. Ecosystem init adds Tier 2/3
        # connectivity (peer bridge, tract bridge, Cricket rim) but may
        # hang when run from the fan-out's background thread due to
        # filesystem locks or bridge initialization contention.
        # All ecosystem-dependent code in process_text() is already
        # guarded by `if self._ecosystem:` — safe to proceed without it.
        self._started = True
        self._start_time = time.time()

        # Initialize NG ecosystem with Cricket rim (constitutional embeddings)
        # This upgrades the engine from Tier 0 → Tier 2/3 if successful.
        eco_config = self._config.ng_ecosystem
        eco_tier = 0
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
            # Wire TuningSocket to peer bridge for absorption rate (#44)
            if hasattr(self._ecosystem, '_peer_bridge') and self._ecosystem._peer_bridge:
                self._tuning_socket.set_bridge_ref(self._ecosystem._peer_bridge)
                logger.info("TuningSocket wired to peer bridge for absorption metrics")
        except Exception as exc:
            logger.warning("NG ecosystem init failed (standalone): %s", exc)
            self._ecosystem = None
            eco_tier = 0

        report = {
            "status": "started",
            "version": self._config.version,
            "sockets": load_results,
            "ecosystem_tier": eco_tier,
            "hardware": SocketManager.detect_hardware(),
        }
        logger.info("Elmer engine started: %s", report)
        return report

    def set_tonic_engine(self, tonic_engine):
        """Pass tonic engine ref to BrainSwitcher for hot-swap."""
        if self._brain_switcher:
            self._brain_switcher.set_tonic_engine(tonic_engine)

    def load_brains(self) -> None:
        """Load brain sockets into an already-running engine.

        Called after a delay in fan-out context — all other modules have
        settled, GC has run, there's headroom for the ~1.3GB brain models.
        The BrainSwitcher registers both sockets with the existing socket
        manager. Next process_text() call will route to them + Lenia steps.
        """
        if not self._started:
            raise RuntimeError("Engine not started. Call start() first.")
        if self._brain_switcher:
            logger.info("Brains already loaded — skipping")
            return
        if not _BRAIN_AVAILABLE:
            logger.warning("Brain classes not available — cannot load")
            return

        self._brain_switcher = BrainSwitcher(
            self._socket_manager,
            brain_socket_cls=None,  # frozen brain disabled — proto solo
            proto_brain_socket_cls=ProtoUniBrainSocket,
            ecosystem=self._ecosystem,
        )
        self._brain_switcher.start()
        self._start_brain_drain()
        logger.info("BrainSwitcher loaded (delayed) — both brains active, drain thread running")

    def stop(self) -> None:
        """Graceful shutdown."""
        if not self._started:
            return

        logger.info("Stopping Elmer engine...")
        self._stop_brain_drain()
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

        # Provide live substrate objects for v2 encoder (if ecosystem initialized)
        # The brain sockets use these to read hyperedges, fired nodes, predictions
        # directly from the live NGLite graph — no serialization boundary.
        # Brain sockets read Elmer's local substrate directly via
        # ecosystem ref. The River deposited topology into Elmer's NG-Lite.
        # The encoder reads what's already here — no pass-through needed.

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

        # 3d. Drop snapshot into brain buffer (NuWave Layer 1)
        #     Brain sockets are expensive (~65s Lenia step on CPU).
        #     The brain drain thread picks up the latest snapshot at its
        #     own pace — no blocking the fan-out RPC. KISS observes here
        #     (synchronous, cheap), brain processing happens async.
        self._drop_to_brain_buffer(snapshot, context)

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
                    embedding = encoding["embedding"]
                    outcome = self._ecosystem.dual_record_outcome(
                        content=text,
                        embedding=embedding,
                        target_id="elmer:substrate_input",
                        success=True,
                        metadata={"process_id": str(self._process_count)},
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
    # Brain Buffer — Async Brain Processing (NuWave Layer 1)
    # -----------------------------------------------------------------

    def _drop_to_brain_buffer(self, snapshot: GraphSnapshot, context: dict) -> None:
        """Drop snapshot into the brain buffer. Returns instantly.

        The brain drain thread picks up the latest snapshot at its own
        pace. Only the most recent snapshot is kept — the brain reads
        current state, not history. This is biologically correct: the
        brainstem reads the current state of the world when it's ready,
        not each individual signal as it arrives.

        KISS observation runs here (synchronous, cheap). Brain processing
        runs on the drain thread (async, expensive).
        """
        # KISS observation — synchronous, runs every call
        self._observe_kiss(snapshot, context)

        # Drop latest snapshot for the brain drain thread
        with self._brain_buffer_lock:
            self._brain_latest_snapshot = snapshot
            self._brain_latest_context = context

    def _start_brain_drain(self) -> None:
        """Start the background thread that drains the brain buffer.

        Called after brains are loaded. The thread loops: check buffer,
        if there's a new snapshot, process it through brain sockets
        (KISS filter + forward pass + Lenia step). Then sleep briefly
        and check again.
        """
        if self._brain_drain_running:
            return

        import threading

        self._brain_drain_running = True

        def _drain_loop():
            import time as _time
            _cycle_interval = 60.0  # seconds between brain cycles

            while self._brain_drain_running:
                snapshot = None
                context = None

                # Check for conversation-driven snapshot first
                with self._brain_buffer_lock:
                    if self._brain_latest_snapshot is not None:
                        snapshot = self._brain_latest_snapshot
                        context = self._brain_latest_context
                        self._brain_latest_snapshot = None
                        self._brain_latest_context = None

                if snapshot is not None:
                    # Conversation active — process all sockets with snapshot
                    try:
                        self._process_brain_sockets_kissed(snapshot, context, [])
                    except Exception as exc:
                        logger.warning("Brain drain error: %s", exc)
                    _time.sleep(1.0)
                else:
                    # No conversation — run brain sockets anyway.
                    # ProtoUniBrain reads from the River (BTF tract) directly.
                    # The River always has fresh topology from the Tonic.
                    try:
                        from core.base_socket import GraphSnapshot
                        autonomic = ng_autonomic.read_state()
                        empty_snapshot = GraphSnapshot(nodes=[], edges=[], metadata={})
                        empty_context = {
                            "autonomic_state": autonomic.get("state", "PARASYMPATHETIC"),
                            "autonomic_intensity": autonomic.get("threat_level", "none"),
                        }
                        self._process_brain_sockets_kissed(
                            empty_snapshot, empty_context, []
                        )
                    except Exception as exc:
                        logger.debug("Brain idle cycle error: %s", exc)
                    _time.sleep(_cycle_interval)

        self._brain_drain_thread = threading.Thread(
            target=_drain_loop,
            name="elmer-brain-drain",
            daemon=True,
        )
        self._brain_drain_thread.start()
        logger.info("Brain drain thread started — brains process at their own pace")

    def _stop_brain_drain(self) -> None:
        """Stop the brain drain thread."""
        self._brain_drain_running = False
        if self._brain_drain_thread:
            self._brain_drain_thread.join(timeout=5)
            self._brain_drain_thread = None

    def _observe_kiss(self, snapshot: GraphSnapshot, context: dict) -> None:
        """Run KISS observation on the snapshot. Synchronous, cheap.

        Logs what KISS would filter — runs every process_text() call
        regardless of whether brains are loaded.
        """
        try:
            import numpy as _np
            nodes = snapshot.nodes
            edges = snapshot.edges
            meta = snapshot.metadata

            if nodes:
                node_feat = [_np.mean([n.get('voltage', 0.0) for n in nodes]),
                             _np.mean([n.get('firing_rate', 0.0) for n in nodes]),
                             _np.mean([n.get('excitability', 0.5) for n in nodes])]
            else:
                node_feat = [0.0, 0.0, 0.5]

            if edges:
                synapse_feat = [_np.mean([e.get('weight', 0.5) for e in edges]),
                                _np.mean([e.get('age', 0.0) for e in edges])]
            else:
                synapse_feat = [0.5, 0.0]

            n_nodes = max(len(nodes), 1)
            n_edges = len(edges)
            max_edges = n_nodes * (n_nodes - 1)
            topo_feat = [n_edges / max_edges if max_edges > 0 else 0.0,
                         meta.get('clustering_coefficient', 0.0),
                         min(meta.get('connected_components', 1) / max(n_nodes, 1), 1.0)]

            if nodes:
                recent = [n.get('recent_firing', 0.0) for n in nodes]
                stdp = [n.get('stdp_timing', 0.0) for n in nodes]
                temporal_feat = [_np.mean(recent), _np.std(recent),
                                 _np.mean(stdp), _np.std(stdp)]
            else:
                temporal_feat = [0.0, 0.0, 0.0, 0.0]

            identity = meta.get('identity_embedding', _np.zeros(384))

            features = {
                'node_features': node_feat,
                'synapse_features': synapse_feat,
                'topo_features': topo_feat,
                'temporal_features': temporal_feat,
                'identity_embedding': identity,
            }

            kiss_result = self._kiss.filter(features)

            stats = self._kiss.stats.to_dict()
            if kiss_result:
                stats["kiss_mode"] = kiss_result.get("kiss_mode", "full")
                stats["kiss_meta"] = kiss_result.get("kiss_meta", {})
            else:
                stats["kiss_mode"] = "skipped"
                stats["kiss_meta"] = {"reason": "delta_below_threshold"}

            with open("/tmp/elmer_kiss.log", "a") as f:
                from datetime import datetime
                import json
                f.write(f"[{datetime.now()}] {json.dumps(stats)}\n")

        except Exception as exc:
            logger.debug("KISS observation failed: %s", exc)

    def _process_brain_sockets_kissed(
        self,
        snapshot: GraphSnapshot,
        context: dict,
        socket_outputs: List[SocketOutput],
    ) -> None:
        """Process snapshot through brain sockets. Called from drain thread.

        Runs the brain sockets (frozen + proto w/ Lenia) on the snapshot.
        Logs the competence delta between frozen and proto outputs.
        This is the expensive path (~65s per Lenia step on CPU). Runs
        on the drain thread, not the fan-out thread.
        """
        if not self._brain_switcher or self._brain_switcher.active_brain == "none":
            return

        frozen_output = None
        proto_output = None

        try:
            # Process frozen brain first
            frozen_sock = self._socket_manager.get_socket("elmer:brain")
            if frozen_sock and hasattr(frozen_sock, 'process'):
                try:
                    frozen_output = frozen_sock.process(snapshot, context)
                    logger.info(
                        "Brain drain: elmer:brain processed (%.1fs)",
                        frozen_output.processing_time if hasattr(frozen_output, 'processing_time') else 0,
                    )
                except Exception as exc:
                    logger.warning("Brain drain: elmer:brain error: %s", exc)

            # Process proto brain
            proto_sock = self._socket_manager.get_socket("elmer:proto_unibrain")
            if proto_sock and hasattr(proto_sock, 'process'):
                try:
                    proto_output = proto_sock.process(snapshot, context)
                    logger.info(
                        "Brain drain: elmer:proto_unibrain processed (%.1fs)",
                        proto_output.processing_time if hasattr(proto_output, 'processing_time') else 0,
                    )
                except Exception as exc:
                    logger.warning("Brain drain: elmer:proto_unibrain error: %s", exc)

            # Log competence metrics
            if proto_output:
                self._log_competence_delta(frozen_output, proto_output)

        except Exception as exc:
            logger.warning("Brain drain processing error: %s", exc)



    def _log_competence_delta(self, frozen_output, proto_output: SocketOutput) -> None:
        """Log intrinsic quality metrics for the living brain.

        Quality is measured by what the model DOES, not by distance
        from something frozen. Frozen output may be None (proto solo mode).
        """
        try:
            import json, os, math
            from datetime import datetime

            delta_path = os.path.expanduser("~/.elmer/competence_delta.jsonl")
            os.makedirs(os.path.dirname(delta_path), exist_ok=True)

            proto_sig = proto_output.signal
            frozen_sig = frozen_output.signal if frozen_output else None

            proto_vals = {
                'coherence': proto_sig.coherence_score,
                'health': proto_sig.health_score,
                'anomaly': proto_sig.anomaly_level,
                'novelty': proto_sig.novelty,
                'confidence': proto_sig.confidence,
                'severity': proto_sig.severity,
                'identity_coherence': proto_sig.identity_coherence,
                'pruning_pressure': proto_sig.pruning_pressure,
                'topology_health': proto_sig.topology_health,
            }
            frozen_vals = {}
            if frozen_sig:
                frozen_vals = {
                    'coherence': frozen_sig.coherence_score,
                    'health': frozen_sig.health_score,
                    'anomaly': frozen_sig.anomaly_level,
                    'novelty': frozen_sig.novelty,
                    'confidence': frozen_sig.confidence,
                    'severity': frozen_sig.severity,
                    'identity_coherence': frozen_sig.identity_coherence,
                    'pruning_pressure': frozen_sig.pruning_pressure,
                    'topology_health': frozen_sig.topology_health,
                }

            vals = list(proto_vals.values())
            n = len(vals)

            # Intrinsic quality metrics
            output_norm = round(math.sqrt(sum(v ** 2 for v in vals)), 6)
            mean_val = sum(vals) / n
            variance = round(sum((v - mean_val) ** 2 for v in vals) / n, 6)
            val_range = round(max(vals) - min(vals), 6)

            # Entropy: how differentiated are the signals?
            # Treat as soft probability distribution via softmax-like normalization
            shifted = [v - min(vals) + 0.001 for v in vals]  # avoid log(0)
            total = sum(shifted)
            probs = [s / total for s in shifted]
            entropy = round(-sum(p * math.log(p + 1e-12) for p in probs), 6)

            # Extremity: how many signals are saturated?
            saturated = sum(1 for v in vals if v < 0.05 or v > 0.95)

            # Lenia metadata
            proto_meta = proto_sig.metadata or {}
            lenia_step = proto_meta.get('lenia_step', 0)
            lenia_delta_norm = proto_meta.get('lenia_delta_norm', 0)
            lenia_time_ms = proto_meta.get('lenia_time_ms', 0)

            entry = {
                'timestamp': datetime.now().isoformat(),
                'lenia_step': lenia_step,
                # Intrinsic quality (what the model DOES)
                'output_norm': output_norm,
                'entropy': entropy,
                'variance': variance,
                'signal_range': val_range,
                'saturated_signals': saturated,
                # Raw signals (for trajectory analysis)
                'proto_signals': {k: round(v, 6) for k, v in proto_vals.items()},
                # Frozen reference (NOT a target — just context)
                'frozen_signals': {k: round(v, 6) for k, v in frozen_vals.items()},
                # Lenia state
                'lenia_delta_norm': lenia_delta_norm,
                'lenia_time_ms': round(lenia_time_ms, 1),
            }

            with open(delta_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

            logger.info(
                "Living brain: norm=%.3f entropy=%.3f var=%.4f range=%.3f sat=%d step=%d delta=%.6f",
                output_norm, entropy, variance, val_range, saturated, lenia_step, lenia_delta_norm,
            )

        except Exception as exc:
            logger.debug("Competence logging failed: %s", exc)

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
            "kiss": self._kiss.stats.to_dict(),
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
