"""
OpenClaw Adapter — E-T Systems OpenClaw Skill Integration

Concrete implementation of NGEcosystemAdapter for OpenClaw AI assistant
skills.  Provides on_message / get_context / stats vocabulary that maps
directly to the OpenClaw skill lifecycle.

Every E-T module that ships as an OpenClaw skill subclasses this adapter
(or uses it directly) to get full NG-Lite → Peer → SNN learning with
zero boilerplate.

Vendored from NeuroGraph canonical source.

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Vendored for Elmer module.
#   What: OpenClawAdapter concrete class implementing NGEcosystemAdapter.
#   Why:  Standard adapter for OpenClaw skills.  Elmer's hook subclasses
#         this to add substrate signal processing.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ng_ecosystem import NGEcosystem, NGEcosystemAdapter

logger = logging.getLogger("openclaw_adapter")

__version__ = "1.0.0"


class OpenClawAdapter(NGEcosystemAdapter):
    """Concrete OpenClaw skill adapter over NGEcosystem.

    Provides the standard on_message / get_context / stats API that
    OpenClaw skill hooks expect.  Handles embedding generation so the
    skill code never touches raw vectors.

    Usage:
        from openclaw_adapter import OpenClawAdapter

        adapter = OpenClawAdapter(module_id="trollguard")
        result = adapter.on_message("suspicious prompt injection attempt")
        context = adapter.get_context("what threats have we seen?")
        print(adapter.stats())

    Subclassing:
        Override _process_message() and _enrich_context() to add
        module-specific behavior while keeping the ecosystem plumbing.
    """

    def __init__(
        self,
        module_id: str,
        embedder_fn: Optional[Callable[[str], np.ndarray]] = None,
        state_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            module_id: Unique module identifier (must match et_module.json).
            embedder_fn: Function that takes text and returns an embedding
                        vector (np.ndarray).  If None, a simple hash-based
                        fallback is used (suitable for testing only).
            state_path: Path for NGLite state persistence.
            config: NGEcosystem config overrides.
        """
        self.module_id = module_id
        self._embedder_fn = embedder_fn or self._default_embedder
        self._eco = NGEcosystem.get_instance(
            module_id=module_id,
            state_path=state_path,
            config=config,
        )
        self._message_count = 0
        self._last_message_time = 0.0

    # -----------------------------------------------------------------
    # NGEcosystemAdapter interface
    # -----------------------------------------------------------------

    def on_message(self, text: str) -> Dict[str, Any]:
        """Process an incoming message through the ecosystem.

        Embeds the text, records the outcome, and returns enriched
        context including cross-module recommendations.

        Args:
            text: Raw message text.

        Returns:
            Dict with status, tier, recommendations, and novelty score.
        """
        if not text or not text.strip():
            return {"status": "skipped", "reason": "empty_input", "tier": self._eco.tier}

        self._message_count += 1
        self._last_message_time = time.time()

        try:
            embedding = self._embedder_fn(text)
            target_id = self._derive_target(text)

            # Record outcome (always success for ingestion)
            self._eco.record_outcome(
                embedding=embedding,
                target_id=target_id,
                success=True,
                metadata={"text_preview": text[:200], "message_count": self._message_count},
            )

            # Get context for response enrichment
            context = self._eco.get_context(embedding)

            result = {
                "status": "ingested",
                "tier": context["tier"],
                "tier_name": context["tier_name"],
                "novelty": context["novelty"],
                "recommendations": context["recommendations"],
                "message_count": self._message_count,
            }

            # Hook for subclass enrichment
            enriched = self._process_message(text, embedding, result)
            return enriched if enriched is not None else result

        except Exception as exc:
            logger.warning("[%s] on_message failed: %s", self.module_id, exc)
            return {"status": "error", "reason": str(exc), "tier": self._eco.tier}

    def get_context(self, text: str) -> Dict[str, Any]:
        """Retrieve cross-module context for prompt enrichment.

        Args:
            text: Query text to find context for.

        Returns:
            Dict with recommendations, novelty, tier info, and any
            module-specific enrichment.
        """
        if not text or not text.strip():
            return {"tier": self._eco.tier, "recommendations": [], "novelty": 1.0}

        try:
            embedding = self._embedder_fn(text)
            context = self._eco.get_context(embedding)

            # Hook for subclass enrichment
            enriched = self._enrich_context(text, embedding, context)
            return enriched if enriched is not None else context

        except Exception as exc:
            logger.warning("[%s] get_context failed: %s", self.module_id, exc)
            return {"tier": self._eco.tier, "recommendations": [], "novelty": 1.0}

    def stats(self) -> Dict[str, Any]:
        """Return adapter + ecosystem telemetry."""
        eco_stats = self._eco.stats()
        return {
            "adapter_version": __version__,
            "module_id": self.module_id,
            "message_count": self._message_count,
            "last_message_time": self._last_message_time,
            **eco_stats,
        }

    # -----------------------------------------------------------------
    # Subclass hooks (override in module-specific adapters)
    # -----------------------------------------------------------------

    def _process_message(
        self,
        text: str,
        embedding: np.ndarray,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Override to add module-specific message processing.

        Called after ecosystem ingestion.  Return a modified result dict
        or None to use the default.
        """
        return None

    def _enrich_context(
        self,
        text: str,
        embedding: np.ndarray,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Override to add module-specific context enrichment.

        Called after ecosystem context retrieval.  Return a modified
        context dict or None to use the default.
        """
        return None

    def _derive_target(self, text: str) -> str:
        """Derive a target_id from message text.

        Default: generic "message" target.  Override for domain-specific
        targeting (e.g., threat classes, model selections).
        """
        return f"{self.module_id}:message"

    # -----------------------------------------------------------------
    # Ecosystem delegation
    # -----------------------------------------------------------------

    @property
    def ecosystem(self) -> NGEcosystem:
        """Direct access to the underlying NGEcosystem instance."""
        return self._eco

    def save(self) -> None:
        """Persist ecosystem state."""
        self._eco.save()

    def shutdown(self) -> None:
        """Graceful shutdown."""
        self._eco.shutdown()

    # -----------------------------------------------------------------
    # Default embedder (testing fallback)
    # -----------------------------------------------------------------

    @staticmethod
    def _default_embedder(text: str) -> np.ndarray:
        """Hash-based embedding for testing when no real embedder is available.

        NOT suitable for production — produces deterministic vectors with
        no semantic meaning.  Install sentence-transformers or use Ollama
        for real embeddings.
        """
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h * 12, dtype=np.float32)[:384]
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec
