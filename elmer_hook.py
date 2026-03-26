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

        self._started = False

        # Eager start — Elmer is the brainstem, it should always be on.
        # Brain sockets load in a background thread because the 1.5B model
        # takes minutes to load on CPU. The fan-out can't block that long.
        import threading
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
        """Stop the engine and save state."""
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
