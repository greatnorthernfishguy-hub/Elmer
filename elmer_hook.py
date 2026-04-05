"""
Elmer OpenClaw Hook — Substrate Processing Skill Adapter  (PRD §7, §9)

Subclasses OpenClawAdapter to integrate Elmer's cognitive substrate
into the OpenClaw AI assistant framework.  This is the entry point
that OpenClaw discovers and calls.

on_message():
    Raw text → SensoryPipeline → ComprehensionSocket → InferencePipeline
    → Memory store → NG ecosystem learning → enriched context response.
    Autonomic state (§7) modulates processing priority and pruning.

get_context():
    Query text → NG ecosystem context → memory recall → identity
    → cross-module recommendations.

stats() / health():
    Full telemetry from engine, sockets, pipelines, ecosystem,
    and autonomic state.

# ---- Changelog ----
# [2026-03-28] Claude Code (Opus 4.6) — #109 Pulse loop: Elmer alive between conversations
#   What: Added _pulse_loop() daemon thread following the Tonic pattern.
#   Why:  #109 — organs must run continuously, not only on fan-out messages.
#         Elmer's brainstem function (drain tracts, monitor topology) must
#         happen between conversations, not just during them.
#   How:  _shutdown_event + _in_conversation flag. Daemon thread drains
#         River tracts via _eco._peer_bridge.drain(), reads autonomic state,
#         and runs engine.process_text() with synthetic health signal when
#         engine is started. on_conversation_started/ended swap intervals
#         (30s resting / 10s conversation). Does not touch _module_on_message.
# [2026-03-29] Claude Code (Opus 4.6) — Fan-out context: engine without brains
# What: Refined NEUROGRAPH_FANOUT_CONTEXT guard. Instead of skipping __init__
#   entirely (leaving _started=False forever), call engine.start(skip_brains=True).
#   Lightweight sockets + KISS run. process_text() is reachable. _started=True.
# Why: Fan-out was calling _module_on_message but _started was always False,
#   so every call fell through to legacy _process_message. Engine, KISS, and
#   brain socket observation never ran. Builder CC confirmed: start engine
#   without brains in fan-out, skip only the brain thread.
# How: _in_fanout flag. Fan-out path calls engine.start(skip_brains=True)
#   synchronously. Standalone path uses background thread with full brains.
# [2026-03-26] Claude Code (Opus 4.6) — Eager start + route through engine
# What: Engine starts in __init__ (brainstem always on). _module_on_message
#   now delegates to engine.process_text() instead of running parallel pipelines.
# Why: Brain sockets (frozen ElmerBrain + ProtoUniBrain w/ Lenia) were wired
#   into the engine but fan-out bypassed it. Proto needs real data + every cycle.
# How: self.start() in __init__, _module_on_message → engine.process_text().
#   Legacy _process_message retained for direct OC skill calls.
# [2026-03-23] Claude Code (Opus 4.6) — Wire _module_on_message (#101)
# What: Added _module_on_message override that delegates to _process_message.
# Why:  _process_message was never called from the OpenClawAdapter lifecycle.
#   Elmer's domain processing was dead. Punchlist #101 fan-out needs this.
# How:  _module_on_message(text, embedding) calls _process_message(text, embedding, result).
# [2026-03-19] Claude Code (Opus 4.6) — Migrate to BAAI/bge-base-en-v1.5 (#45)
# What: fastembed model all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5 (768-dim).
# Why: Ecosystem-wide embedding migration. Punchlist #45.
# How: TextEmbedding() model string + docstring update.
# -------------------
# [2026-02-28] Claude (Opus 4.6) — §6.1/§7/§9 compliant rewrite.
#   What: ElmerHook with §6.1 SubstrateSignal format, §7 autonomic
#         awareness (read-only), full pipeline chaining, and §14
#         threshold-aware health reporting.
#   Why:  PRD v0.2.0 §7 mandates autonomic integration; §6.1 mandates
#         flat scored signal fields; §9 mandates full pipeline chain.
# -------------------
"""

from __future__ import annotations

# Auto-update on startup — pull latest code + sync vendored files
try:
    from ng_updater import auto_update; auto_update()
except Exception:
    pass  # Never prevent module startup

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from core.config import ElmerConfig, load_config
import ng_autonomic
from core.substrate_signal import COHERENCE_HEALTHY, SubstrateSignal
from openclaw_adapter import OpenClawAdapter
from pipelines.health import HealthPipeline
from pipelines.identity import IdentityPipeline
from pipelines.inference import InferencePipeline
from pipelines.memory import MemoryPipeline
from pipelines.sensory import SensoryPipeline
from runtime.engine import ElmerEngine

logger = logging.getLogger("elmer.hook")


class ElmerHook(OpenClawAdapter):
    """OpenClaw skill hook for the Elmer cognitive substrate.

    Extends OpenClawAdapter with:
      - §6.1 SubstrateSignal format (flat scored fields)
      - §7 Autonomic modulation (read-only)
      - §8 Full pipeline chaining (sensory → inference → memory → identity)
      - §14 Threshold-aware health reporting

    Ref: PRD §7, §9
    """

    MODULE_ID = "elmer"
    SKIP_ECOSYSTEM = True
    SKILL_NAME = "Elmer Cognitive Substrate"
    WORKSPACE_ENV = "ELMER_WORKSPACE_DIR"
    DEFAULT_WORKSPACE = "~/.openclaw/elmer"

    _instance: Optional["ElmerHook"] = None

    def __init__(
        self,
        config: Optional[ElmerConfig] = None,
    ) -> None:
        self._elmer_config = config or load_config()

        super().__init__()

        self._engine = ElmerEngine(config=self._elmer_config)
        # Autonomic state read via ng_autonomic.read_state() (read-only, PRD §7)

        # Pipelines  (PRD §8)
        self._sensory = SensoryPipeline()
        self._inference = InferencePipeline()
        self._health = HealthPipeline()
        self._memory = MemoryPipeline()
        self._identity = IdentityPipeline()

        # Pin Elmer's modules under names the fan-out won't stash.
        # The fan-out clears generic prefixes (core, pipelines, runtime, etc.)
        # from sys.modules between each module's _module_on_message call.
        # Pinning them under _elmer_pinned_* keeps them alive so lazy imports
        # inside process_text() can find them.
        import sys as _sys
        for mod_name in list(_sys.modules.keys()):
            if any(mod_name == p or mod_name.startswith(p + ".")
                   for p in ("core", "pipelines", "runtime")):
                _sys.modules[f"_elmer_pinned_{mod_name}"] = _sys.modules[mod_name]

        self._started = False

        # --- #109 Pulse loop infrastructure ---
        self._shutdown_event = threading.Event()
        self._in_conversation = False
        self._resting_interval = 30.0
        self._conversation_interval = 10.0

        # [2026-03-29] Fan-out context: start engine WITHOUT brains.
        # Brain sockets (~1.3GB for two 0.5B models) cause OOM when loaded
        # in-process with NeuroGraph + 7 other module hooks during init spikes.
        # Engine starts with lightweight sockets (Comprehension, Monitoring,
        # Myelination, Tuning) + KISS observation. process_text() runs.
        # Standalone Elmer loads brains in background thread.
        import os
        self._in_fanout = os.environ.get("NEUROGRAPH_FANOUT_CONTEXT") == "1"

        if self._in_fanout:
            # Fan-out context: start engine immediately, skip brains
            try:
                self._engine.start(skip_brains=True)
                self._started = True
                logger.info("Elmer: engine started (fan-out context — no brains, KISS active)")
            except Exception as exc:
                logger.warning("Elmer: fan-out engine start failed: %s", exc)

            # Delayed brain load — wait for all modules to settle + GC,
            # then load brains into the already-running engine.
            # Builder CC confirmed: ~9.5GB free at steady state, brains
            # add ~1.3GB. Elmer loads last, so by the time this fires
            # all other modules have initialized. 60s delay for safety.
            def _delayed_brain_load():
                import gc, time
                try:
                    time.sleep(60)  # Let all modules settle
                    gc.collect()
                    with open("/tmp/elmer_eager.log", "a") as _f:
                        _f.write(f"[{__import__('datetime').datetime.now()}] Delayed brain load starting (fan-out context)...\n")
                    self._engine.load_brains()
                    with open("/tmp/elmer_eager.log", "a") as _f:
                        _f.write(f"[{__import__('datetime').datetime.now()}] Delayed brain load SUCCEEDED\n")
                except Exception as exc:
                    with open("/tmp/elmer_eager.log", "a") as _f:
                        import traceback
                        _f.write(f"[{__import__('datetime').datetime.now()}] Delayed brain load FAILED: {exc}\n")
                        _f.write(traceback.format_exc() + "\n")
                    logger.warning("Delayed brain load failed: %s", exc)
            t = threading.Thread(target=_delayed_brain_load, name="elmer-delayed-brains", daemon=True)
            t.start()
        else:
            # Standalone: start engine with brains in background thread
            def _deferred_start():
                try:
                    with open("/tmp/elmer_eager.log", "a") as _f:
                        _f.write(f"[{__import__('datetime').datetime.now()}] Eager start beginning (background)...\n")
                    self.start()
                    with open("/tmp/elmer_eager.log", "a") as _f:
                        _f.write(f"[{__import__('datetime').datetime.now()}] Eager start SUCCEEDED\n")
                except Exception as exc:
                    with open("/tmp/elmer_eager.log", "a") as _f:
                        import traceback
                        _f.write(f"[{__import__('datetime').datetime.now()}] Eager start FAILED: {exc}\n")
                        _f.write(traceback.format_exc() + "\n")
                    logger.warning("Eager start failed (will retry on first message): %s", exc)
            t = threading.Thread(target=_deferred_start, name="elmer-eager-start", daemon=True)
            t.start()

        # --- #109 Start pulse loop daemon thread ---
        self._pulse_thread = threading.Thread(
            target=self._pulse_loop, name="elmer-pulse", daemon=True
        )
        self._pulse_thread.start()
        logger.info("Elmer: pulse loop started (resting=%.0fs, conversation=%.0fs)",
                     self._resting_interval, self._conversation_interval)

    # -----------------------------------------------------------------
    # #109 Pulse loop — organ alive between conversations
    # -----------------------------------------------------------------

    def _pulse_loop(self):
        """Continuous brainstem function — drain tracts, monitor topology.

        Follows the Tonic pattern: daemon thread with shutdown event wait.
        Each cycle does real domain work regardless of conversation state.
        """
        while not self._shutdown_event.is_set():
            try:
                self._pulse_cycle()
            except Exception as exc:
                logger.debug("Pulse cycle error: %s", exc)
            interval = (
                self._conversation_interval
                if self._in_conversation
                else self._resting_interval
            )
            self._shutdown_event.wait(timeout=interval)

    def _pulse_cycle(self):
        """One pulse cycle — drain River tracts, read autonomic, run topology pass."""
        # 1. Drain tracts from peer bridge (topology deltas from the River)
        #    _drain_all() reads JSONL tract files (where topology deltas land)
        #    and populates _peer_events. bridge.drain() is the mmap path (empty
        #    when no mmap tracts are set up). Without _drain_all(), JSONL tracts
        #    accumulate undrained forever.
        drained = 0
        try:
            bridge = getattr(self._eco, '_peer_bridge', None) if hasattr(self, '_eco') else None
            if bridge and hasattr(bridge, '_drain_all'):
                bridge._drain_all()
                # Process newly drained events through the engine
                peer_events = getattr(bridge, '_peer_events', [])
                if peer_events:
                    drained = len(peer_events)
                    for event in peer_events:
                        text = self._extract_pulse_text(event)
                        if text and self._started:
                            try:
                                self._engine.process_text(text)
                            except Exception as exc:
                                logger.debug("Pulse drain process error: %s", exc)
        except Exception as exc:
            logger.debug("Pulse drain error: %s", exc)

        # 2. Read autonomic state for context
        try:
            autonomic_state = ng_autonomic.read_state().get("state", "PARASYMPATHETIC")
        except Exception:
            autonomic_state = "PARASYMPATHETIC"

        # 3. Lightweight topology pass if engine is started
        if self._started:
            try:
                self._engine.process_text(
                    f"pulse:autonomic={autonomic_state},drained={drained}"
                )
            except Exception as exc:
                logger.debug("Pulse topology pass error: %s", exc)

    @staticmethod
    def _extract_pulse_text(event):
        """Extract processable text from a topology delta event.

        Mirrors elmer_service.py _extract_text_from_delta — raw experience
        from the River converted to text for the sensory pipeline (Law 7).
        """
        if not isinstance(event, dict):
            return None
        parts = []
        fired = event.get('fired_node_ids', [])
        if fired:
            parts.append(f"fired:{len(fired)} nodes")
        fired_he = event.get('fired_hyperedge_ids', [])
        if fired_he:
            parts.append(f"hyperedges:{len(fired_he)} active")
        pruned = event.get('synapses_pruned', 0)
        sprouted = event.get('synapses_sprouted', 0)
        if pruned or sprouted:
            parts.append(f"structural:+{sprouted}/-{pruned}")
        confirmed = event.get('predictions_confirmed', 0)
        surprised = event.get('predictions_surprised', 0)
        if confirmed or surprised:
            parts.append(f"predictions:confirmed={confirmed},surprised={surprised}")
        return " ".join(parts) if parts else None

    def on_conversation_started(self):
        """Mode swap: shorter pulse interval during active conversation."""
        self._in_conversation = True
        logger.debug("Elmer: conversation started — pulse interval %.0fs",
                      self._conversation_interval)

    def on_conversation_ended(self):
        """Mode swap: longer pulse interval between conversations."""
        self._in_conversation = False
        logger.debug("Elmer: conversation ended — pulse interval %.0fs",
                      self._resting_interval)

    # -----------------------------------------------------------------
    # Singleton
    # -----------------------------------------------------------------

    @classmethod
    def get_instance(cls, config: Optional[ElmerConfig] = None) -> "ElmerHook":
        if cls._instance is None:
            cls._instance = cls(config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        if cls._instance is not None:
            cls._instance.stop()
        cls._instance = None

    # -----------------------------------------------------------------
    # Embedding (OpenClawAdapter abstract method)
    # -----------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Embed text via ng_embed (centralized ecosystem embedding).

        Ecosystem standard: Snowflake/snowflake-arctic-embed-m-v1.5 (768-dim).
        ONNX Runtime, no torch dependency.
        """
        try:
            from ng_embed import embed
            return embed(text)
        except Exception:
            return self._hash_embed(text)

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> Dict[str, Any]:
        """Start the Elmer engine and all subsystems.

        Called eagerly from __init__ — Elmer is the brainstem,
        it should always be on.  Brain sockets (frozen ElmerBrain +
        living ProtoUniBrain with Lenia dynamics) need every cycle
        they can get.
        """
        if self._started:
            return {"status": "already_started"}
        result = self._engine.start()
        self._started = True
        logger.info("ElmerHook started (eager — brainstem always on)")
        return result

    def stop(self) -> None:
        """Stop the engine, pulse loop, and save state."""
        self._shutdown_event.set()
        if not self._started:
            return
        self._engine.stop()
        # Save ecosystem state if available (OpenClawAdapter may not define save())
        if hasattr(super(), 'save'):
            super().save()
        self._started = False
        logger.info("ElmerHook stopped")

    # -----------------------------------------------------------------
    # OpenClawAdapter overrides  (PRD §7, §8)
    # -----------------------------------------------------------------

    def _module_on_message(self, text: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Route through the engine — pipelines + sockets + brain sockets.

        The engine runs the full processing chain: sensory → GraphSnapshot →
        socket routing (including frozen ElmerBrain + living ProtoUniBrain
        with Lenia dynamics) → inference → memory → identity → ecosystem.
        """
        # Restore pinned modules — fan-out stashes core/pipelines/runtime
        # between calls. Re-inject them so the engine can function.
        import sys as _sys
        for mod_name in list(_sys.modules.keys()):
            if mod_name.startswith("_elmer_pinned_"):
                real_name = mod_name[len("_elmer_pinned_"):]
                _sys.modules[real_name] = _sys.modules[mod_name]

        # If engine isn't ready yet (background start still loading),
        # fall back to lightweight pipeline processing
        if not self._started:
            result: Dict[str, Any] = {}
            self._process_message(text, embedding, result)
            return result

        try:
            engine_result = self._engine.process_text(text)

            return {
                "elmer": {
                    "pipelines_active": True,
                    "autonomic_state": engine_result.get("autonomic_state", "PARASYMPATHETIC"),
                    "process_id": engine_result.get("process_id"),
                    "socket_outputs": engine_result.get("socket_outputs"),
                },
                "_substrate_target_id": "elmer:substrate_input",
                "_substrate_success": True,
            }

        except Exception as exc:
            logger.warning("Engine processing failed: %s", exc)
            return {
                "elmer": {"pipelines_active": False, "error": str(exc)},
                "_substrate_target_id": "elmer:substrate_input",
                "_substrate_success": False,
            }

    def _process_message(
        self,
        text: str,
        embedding: np.ndarray,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Legacy pipeline path — retained for direct OpenClaw skill calls.

        Fan-out messages now go through _module_on_message → engine.
        This path is only hit if OpenClaw invokes the skill directly.
        """
        try:
            autonomic_state = ng_autonomic.read_state().get("state", "PARASYMPATHETIC")
            sensory_signal = self._sensory.process(text)
            inference_signal = self._inference.process(sensory_signal)
            self._memory.store(inference_signal)

            result["elmer"] = {
                "sensory_signal": sensory_signal.signal_id,
                "inference_signal": inference_signal.signal_id,
                "pipelines_active": True,
                "autonomic_state": autonomic_state,
                "coherence_score": inference_signal.coherence_score,
                "coherence_status": inference_signal.coherence_status,
            }

        except Exception as exc:
            logger.warning("Pipeline processing failed: %s", exc)
            result["elmer"] = {"pipelines_active": False, "error": str(exc)}

        elmer_data = result.get("elmer", {})
        status = elmer_data.get("coherence_status", "healthy")
        result["_substrate_target_id"] = f"elmer:health:{status}"
        result["_substrate_success"] = elmer_data.get("pipelines_active", False)

        return result

    def _enrich_context(
        self,
        text: str,
        embedding: np.ndarray,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Add Elmer-specific context (memory, identity, autonomic)."""
        try:
            # Memory recall
            memory_signal = self._memory.recall(text)
            context["memory"] = memory_signal.metadata

            # Identity context
            identity_signal = self._identity.query()
            context["identity"] = identity_signal.metadata.get("identity", {})

            # Autonomic awareness  (PRD §7)
            autonomic_state = ng_autonomic.read_state().get("state", "PARASYMPATHETIC")
            context["autonomic_state"] = autonomic_state

        except Exception as exc:
            logger.debug("Context enrichment failed: %s", exc)

        return context

    def _derive_target(self, text: str) -> str:
        return "elmer:substrate_input"

    # -----------------------------------------------------------------
    # Health  (PRD §14)
    # -----------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Comprehensive health report with §14 threshold status."""
        engine_health = self._engine.health() if self._started else {"status": "offline"}
        health_signal = self._health.check()

        return {
            "module": "elmer",
            "version": self._elmer_config.version,
            "started": self._started,
            "engine": engine_health,
            "pipelines": {
                "sensory": self._sensory.stats(),
                "inference": self._inference.stats(),
                "health": self._health.stats(),
                "memory": self._memory.stats(),
                "identity": self._identity.stats(),
            },
            "autonomic_state": ng_autonomic.read_state().get("state", "PARASYMPATHETIC"),
            "coherence_status": health_signal.coherence_status,
            "ecosystem": self.stats(),
        }
