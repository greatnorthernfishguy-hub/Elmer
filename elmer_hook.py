"""
Elmer OpenClaw Hook — Substrate Processing Skill Adapter

Subclasses OpenClawAdapter to integrate Elmer's cognitive substrate
into the OpenClaw AI assistant framework.  This is the entry point
that OpenClaw discovers and calls.

on_message():
    Raw text → SensoryPipeline → ComprehensionSocket → InferencePipeline
    → NG ecosystem learning → enriched context response.

get_context():
    Query text → NG ecosystem context retrieval → cross-module
    recommendations → enriched context for prompt injection.

stats():
    Full telemetry from engine, sockets, pipelines, and ecosystem.

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 initial creation.
#   What: ElmerHook subclassing OpenClawAdapter with ElmerEngine
#         integration, pipeline routing, and health endpoint.
#   Why:  Standard OpenClaw skill interface for the Elmer substrate.
#         Enables "just works" integration — install and go.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from openclaw_adapter import OpenClawAdapter
from runtime.engine import ElmerEngine
from core.config import ElmerConfig, load_config
from pipelines.sensory import SensoryPipeline
from pipelines.inference import InferencePipeline
from pipelines.health import HealthPipeline
from pipelines.memory import MemoryPipeline
from pipelines.identity import IdentityPipeline

logger = logging.getLogger("elmer.hook")


class ElmerHook(OpenClawAdapter):
    """OpenClaw skill hook for the Elmer cognitive substrate.

    Extends OpenClawAdapter with Elmer-specific processing:
      - Full pipeline routing (sensory → inference → memory)
      - Engine lifecycle management
      - Health monitoring via MonitoringSocket
      - Identity consistency via IdentityPipeline

    Usage (OpenClaw auto-discovery via et_module.json entry_point):
        from elmer_hook import ElmerHook

        hook = ElmerHook()
        result = hook.on_message("What patterns have you observed?")
        context = hook.get_context("Tell me about X")
        health = hook.health()
        print(hook.stats())
    """

    _instance: Optional["ElmerHook"] = None

    def __init__(
        self,
        config: Optional[ElmerConfig] = None,
        embedder_fn: Optional[Any] = None,
    ) -> None:
        self._elmer_config = config or load_config()

        super().__init__(
            module_id=self._elmer_config.module_id,
            embedder_fn=embedder_fn,
            state_path=self._elmer_config.ng_ecosystem.state_path or None,
            config={
                "peer_bridge": {
                    "enabled": self._elmer_config.ng_ecosystem.peer_bridge_enabled,
                    "sync_interval": self._elmer_config.ng_ecosystem.peer_sync_interval,
                },
                "tier3_upgrade": {
                    "enabled": self._elmer_config.ng_ecosystem.tier3_upgrade_enabled,
                    "poll_interval": self._elmer_config.ng_ecosystem.tier3_poll_interval,
                },
            },
        )

        # Initialize engine
        self._engine = ElmerEngine(config=self._elmer_config)

        # Initialize pipelines
        self._sensory = SensoryPipeline()
        self._inference = InferencePipeline()
        self._health = HealthPipeline()
        self._memory = MemoryPipeline()
        self._identity = IdentityPipeline()

        self._started = False

    # -----------------------------------------------------------------
    # Singleton
    # -----------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        config: Optional[ElmerConfig] = None,
    ) -> "ElmerHook":
        """Return the singleton ElmerHook instance."""
        if cls._instance is None:
            cls._instance = cls(config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (testing only)."""
        if cls._instance is not None:
            cls._instance.stop()
        cls._instance = None

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> Dict[str, Any]:
        """Start the Elmer engine and all subsystems."""
        if self._started:
            return {"status": "already_started"}

        result = self._engine.start()
        self._started = True
        logger.info("ElmerHook started")
        return result

    def stop(self) -> None:
        """Stop the engine and save state."""
        if not self._started:
            return
        self._engine.stop()
        self.save()
        self._started = False
        logger.info("ElmerHook stopped")

    # -----------------------------------------------------------------
    # OpenClawAdapter overrides
    # -----------------------------------------------------------------

    def _process_message(
        self,
        text: str,
        embedding: np.ndarray,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Elmer-specific message processing through pipelines.

        Routes: sensory → inference → memory store.
        """
        try:
            # Sensory pipeline: raw text → structured signal
            sensory_signal = self._sensory.process(text)

            # Inference pipeline: comprehend + reason
            inference_signal = self._inference.process(sensory_signal)

            # Memory pipeline: store for future recall
            self._memory.store(inference_signal)

            result["elmer"] = {
                "sensory_signal": sensory_signal.signal_id,
                "inference_signal": inference_signal.signal_id,
                "pipelines_active": True,
            }

        except Exception as exc:
            logger.warning("Pipeline processing failed: %s", exc)
            result["elmer"] = {"pipelines_active": False, "error": str(exc)}

        return result

    def _enrich_context(
        self,
        text: str,
        embedding: np.ndarray,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Add Elmer-specific context (memory recall, identity)."""
        try:
            # Memory recall
            memory_signal = self._memory.recall(text)
            context["memory"] = memory_signal.payload

            # Identity context
            identity_signal = self._identity.query()
            context["identity"] = identity_signal.payload

        except Exception as exc:
            logger.debug("Context enrichment failed: %s", exc)

        return context

    def _derive_target(self, text: str) -> str:
        """Elmer uses signal-type based targeting."""
        return "elmer:substrate_input"

    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Comprehensive health report from all Elmer subsystems."""
        engine_health = self._engine.health() if self._started else {"status": "offline"}

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
            "ecosystem": self.stats(),
        }
