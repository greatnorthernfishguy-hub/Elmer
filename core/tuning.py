"""
TuningSocket — Elmer's hypothalamus.

Reads substrate health metrics and proposes parameter adjustments to the
local NG-Lite instance.  Uses Pattern B (implicit substrate authority):
the substrate's own learned topology informs tuning decisions.

Biological analog: the hypothalamus monitors internal conditions (temperature,
hunger, hormone levels) and adjusts autonomic outputs to maintain homeostasis.
It doesn't think — it regulates.

Competence Model — Continuous, Not Tiered:

    There are no Apprentice/Journeyman/Master gates.  Competence is a
    continuous [0, 1] score per parameter, derived from outcome history.
    It works like real skill acquisition:

    - You start cautious (competence ≈ 0): tight healthy ranges, tiny steps.
    - Each successful tuning decision nudges competence upward.
    - Each regression nudges it downward (faster — trust is hard to earn,
      easy to lose).
    - Competence scales two things:
        1. Step size: how bold the adjustment can be.
        2. Range tolerance: how far from bootstrap center is acceptable.
    - At high competence the system trusts its own judgment — wider
      operating ranges, larger adjustments, less reliance on the
      bootstrap scaffolding.
    - Competence is per-parameter, not global.  You can be confident
      about pruning_threshold and cautious about success_boost.

    The bootstrap ranges (_BOOTSTRAP_RANGES) are training wheels.  They
    never fully disappear (hard bounds in TUNABLE_PARAMS enforce that),
    but competence makes them progressively irrelevant.  The substrate
    replaces the scaffolding with learned authority.

    Competence persists across cycles via the tuning history.  On restart,
    competence rebuilds from zero — but rebuilds fast because the substrate
    topology already encodes the successful tuning patterns.

# ---- Changelog ----
# [2026-03-26] Claude Code Opus — Punchlist #44: Adaptive relevance thresholds
#   What: Added relevance_threshold to TuningSocket's monitored parameters
#   Why: Punchlist #44 — peer bridge relevance_threshold should adapt based on
#     event volume and absorption quality, not remain static at 0.30
#   How: TuningSocket gets a bridge ref (same pattern as MyelinationSocket).
#     _extract_absorption_rate() computes the fraction of peer events that pass
#     the relevance threshold. Bootstrap range for absorption_rate: 0.10–0.60.
#     Too few events absorbed → lower threshold. Too many → raise it.
# [2026-03-24] Claude Code (Opus 4.6) — Continuous competence model
#   What: Replaced static healthy ranges and fixed step size with continuous
#     per-parameter competence scoring.  Competence [0,1] derived from
#     outcome history scales step size (2%→15%) and range tolerance.
#     No tier gates — authority accumulates through demonstrated accuracy.
#     Regression penalized 2x vs. improvement reward (trust asymmetry).
#   Why:  Josh directive: "I want substrate and competence based loosening.
#     Think of it like guided building of skill and self confidence."
#     Static scaffolding is bootstrap only — the substrate should earn its
#     own authority through outcomes, not be handed tiers.
#   How:  _competence dict per parameter.  _update_competence() on each
#     outcome.  _effective_step() and _effective_range() scale with
#     competence.  Bootstrap ranges are the floor, not the ceiling.
# [2026-03-24] Claude Code (Opus 4.6) — Initial creation (homeostasis audit)
#   What: TuningSocket implementing ElmerSocket interface.
#   Why:  Elmer was observation-only.  This socket closes the loop.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from core.base_socket import (
    ElmerSocket,
    GraphSnapshot,
    HardwareRequirements,
    SocketHealth,
    SocketOutput,
)
from core.substrate_signal import SubstrateSignal

logger = logging.getLogger("elmer.tuning")

# Bootstrap healthy ranges — training wheels.
# At competence=0, these define what "healthy" means.
# At competence=1, tolerance expands by _RANGE_EXPANSION_FACTOR in each
# direction, so the system accepts a wider operating envelope.
_BOOTSTRAP_RANGES = {
    "avg_weight":       (0.20, 0.65),
    "node_utilization": (0.10, 0.80),
    "synapse_density":  (1.0,  8.0),
    "absorption_rate":  (0.10, 0.60),   # Punchlist #44 — fraction of peer events absorbed
}

# How much the healthy range expands at full competence.
# At competence=1, each bound moves outward by this fraction of the
# bootstrap range width.  E.g., avg_weight (0.20, 0.65) with 0.5
# expansion → (0.20 - 0.225, 0.65 + 0.225) = (-0.025, 0.875),
# clamped to (0.0, 1.0).
_RANGE_EXPANSION_FACTOR = 0.5

# Step size bounds as fraction of current value.
# At competence=0: cautious.  At competence=1: confident.
_STEP_FRACTION_MIN = 0.02   # 2% at zero competence
_STEP_FRACTION_MAX = 0.15   # 15% at full competence

# Competence update rates.  Asymmetric by design: trust is hard to earn,
# easy to lose.  A single bad outcome shouldn't destroy confidence, but
# regressions carry more weight than improvements.
_COMPETENCE_GAIN = 0.05     # per successful outcome
_COMPETENCE_LOSS = 0.10     # per regression (2x gain — trust asymmetry)

# Minimum outcomes before competence can exceed this threshold.
# Prevents a lucky first adjustment from producing overconfidence.
_MIN_OUTCOMES_FOR_CONFIDENCE = 5


class TuningSocket(ElmerSocket):
    """Elmer's homeostatic tuning extraction bucket.

    Observes substrate health metrics and produces parameter adjustment
    recommendations.  Competence is continuous and per-parameter — earned
    through demonstrated accuracy, not granted by tier gates.
    """

    SOCKET_ID = "elmer:tuning"
    SOCKET_TYPE = "tuning"

    def __init__(self) -> None:
        super().__init__()
        self._ng_lite_ref = None  # Set by engine after registration
        self._bridge_ref = None   # Set by engine for absorption rate (#44)
        self._last_health_snapshot: Dict[str, float] = {}
        self._tuning_history: List[Dict[str, Any]] = []
        self._history_max = 500

        # Per-parameter competence: continuous [0, 1].
        # Starts at 0 (no evidence).  Grows with successful outcomes.
        self._competence: Dict[str, float] = {}

        # Per-parameter outcome counts for the minimum-outcomes gate.
        self._outcome_counts: Dict[str, int] = {}

        # Punchlist #44: track absorption metrics across cycles
        self._last_peer_events_cached = 0
        self._last_drain_count = 0

    @property
    def socket_id(self) -> str:
        return self.SOCKET_ID

    @property
    def socket_type(self) -> str:
        return self.SOCKET_TYPE

    def declare_requirements(self) -> HardwareRequirements:
        return HardwareRequirements(
            min_memory_mb=64,
            gpu_required=False,
            cpu_cores=1,
            disk_mb=0,
        )

    def load(self, model_path: str) -> bool:
        if self._loaded:
            return True
        self._loaded = True
        self._load_time = time.time()
        logger.info("TuningSocket loaded")
        return True

    def unload(self) -> None:
        self._loaded = False
        logger.info("TuningSocket unloaded")

    def set_ng_lite_ref(self, ng_lite) -> None:
        """Engine provides a reference to the local NGLite instance.

        This is NOT a Law 1 violation — Elmer reads and tunes its own
        substrate, not another module's.
        """
        self._ng_lite_ref = ng_lite

    def set_bridge_ref(self, bridge) -> None:
        """Engine provides a reference to the peer/tract bridge for absorption metrics.

        Punchlist #44: needed to compute absorption rate for
        relevance_threshold tuning.  Same pattern as MyelinationSocket.
        NOT a Law 1 violation — Elmer reads its own bridge stats.
        """
        self._bridge_ref = bridge

    # -------------------------------------------------------------------
    # Competence
    # -------------------------------------------------------------------

    def get_competence(self, key: str) -> float:
        """Return current competence for a parameter.  0 = no evidence."""
        raw = self._competence.get(key, 0.0)
        # Gate: cap effective competence until minimum outcomes reached
        outcomes = self._outcome_counts.get(key, 0)
        if outcomes < _MIN_OUTCOMES_FOR_CONFIDENCE:
            cap = outcomes / _MIN_OUTCOMES_FOR_CONFIDENCE * 0.5
            return min(raw, cap)
        return raw

    def _update_competence(self, key: str, improved: bool) -> float:
        """Update competence for a parameter based on outcome.

        Returns the new competence value.
        """
        current = self._competence.get(key, 0.0)
        self._outcome_counts[key] = self._outcome_counts.get(key, 0) + 1

        if improved:
            # Competence grows toward 1.0, decelerating as it approaches
            new = current + _COMPETENCE_GAIN * (1.0 - current)
        else:
            # Competence drops toward 0.0, faster than it grew
            new = current - _COMPETENCE_LOSS * current

        new = max(0.0, min(1.0, new))
        self._competence[key] = new

        logger.info(
            "Competence %s: %.3f → %.3f (%s, %d outcomes)",
            key, current, new,
            "improved" if improved else "regressed",
            self._outcome_counts[key],
        )
        return new

    def _effective_step_fraction(self, key: str) -> float:
        """Step size fraction scaled by competence.

        Low competence → small cautious steps.
        High competence → larger confident steps.
        """
        c = self.get_competence(key)
        return _STEP_FRACTION_MIN + c * (_STEP_FRACTION_MAX - _STEP_FRACTION_MIN)

    def _effective_range(self, metric: str) -> Tuple[float, float]:
        """Healthy range for a metric, widened by competence.

        At competence=0, returns bootstrap range.
        At competence=1, range expands by _RANGE_EXPANSION_FACTOR in
        each direction.  The substrate earns a wider operating envelope.
        """
        if metric not in _BOOTSTRAP_RANGES:
            return (0.0, 1.0)

        lo, hi = _BOOTSTRAP_RANGES[metric]
        width = hi - lo

        # Use the max competence across all parameters as the metric-level
        # competence (metrics aren't tied to a single parameter).
        c = max(
            (self.get_competence(k) for k in self._competence),
            default=0.0,
        )

        expansion = width * _RANGE_EXPANSION_FACTOR * c
        effective_lo = max(0.0, lo - expansion)
        effective_hi = min(
            float("inf") if metric == "synapse_density" else 1.0,
            hi + expansion,
        )
        return (effective_lo, effective_hi)

    # -------------------------------------------------------------------
    # Main processing
    # -------------------------------------------------------------------

    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Analyze substrate health and propose tuning adjustments."""
        if not self._loaded:
            raise RuntimeError("TuningSocket not loaded")

        t0 = time.time()
        self._process_count += 1

        # Freeze tuning during SYMPATHETIC — don't adjust in a crisis
        autonomic = context.get("autonomic_state", "PARASYMPATHETIC")
        if autonomic == "SYMPATHETIC":
            elapsed = time.time() - t0
            signal = SubstrateSignal.create(
                signal_type="health",
                description="Tuning frozen — SYMPATHETIC state active",
                coherence_score=1.0,
                health_score=1.0,
                anomaly_level=0.0,
                novelty=0.0,
                confidence=0.0,
                severity=0.0,
                temporal_window=elapsed,
                metadata={
                    "socket": self.SOCKET_ID,
                    "tuning_recommendations": [],
                    "frozen": True,
                    "reason": "sympathetic",
                },
            )
            return SocketOutput(
                signal=signal,
                graph_delta=None,
                confidence=0.0,
                processing_time=elapsed,
            )

        # Resolve any pending outcomes from previous cycle
        self._resolve_pending_outcomes()

        # Collect health metrics
        health = self._extract_health(snapshot)
        self._last_health_snapshot = health

        # Determine what needs adjustment
        recommendations = self._diagnose_and_recommend(health)

        elapsed = time.time() - t0
        self._total_latency += elapsed

        issues = len(recommendations)
        confidence = max(0.1, 1.0 - (issues * 0.2))

        # Include competence snapshot in metadata for observability
        competence_snapshot = {
            k: {"competence": self.get_competence(k), "outcomes": self._outcome_counts.get(k, 0)}
            for k in self._competence
        }

        signal = SubstrateSignal.create(
            signal_type="health",
            description=(
                f"Tuning assessment: {issues} adjustment(s) recommended"
                if issues > 0
                else "Substrate parameters within healthy range"
            ),
            coherence_score=health.get("coherence", 1.0),
            health_score=health.get("overall_health", 1.0),
            anomaly_level=min(1.0, issues * 0.25),
            novelty=0.0,
            confidence=confidence,
            severity=min(1.0, issues * 0.2),
            temporal_window=elapsed,
            metadata={
                "socket": self.SOCKET_ID,
                "tuning_recommendations": recommendations,
                "health_snapshot": health,
                "competence": competence_snapshot,
                "frozen": False,
            },
        )

        return SocketOutput(
            signal=signal,
            graph_delta=None,
            confidence=confidence,
            processing_time=elapsed,
        )

    def health(self) -> SocketHealth:
        return self._make_health("healthy")

    # -------------------------------------------------------------------
    # Health extraction
    # -------------------------------------------------------------------

    def _extract_health(self, snapshot: GraphSnapshot) -> Dict[str, float]:
        """Extract health metrics from the substrate snapshot."""
        nodes = snapshot.nodes
        edges = snapshot.edges
        node_count = len(nodes)
        edge_count = len(edges)

        weights = [e.get("weight", 0.5) for e in edges if "weight" in e]
        avg_weight = sum(weights) / len(weights) if weights else 0.5

        active_nodes = sum(
            1 for n in nodes
            if n.get("activation_count", 0) > 0
        )
        node_utilization = active_nodes / node_count if node_count > 0 else 0.0

        synapse_density = edge_count / node_count if node_count > 0 else 0.0

        w_lo, w_hi = self._effective_range("avg_weight")
        w_center = (w_lo + w_hi) / 2
        w_span = max(w_hi - w_lo, 0.01)
        weight_health = 1.0 - min(1.0, abs(avg_weight - w_center) / w_span)

        overall_health = (weight_health + min(1.0, node_utilization / 0.5)) / 2

        return {
            "node_count": float(node_count),
            "edge_count": float(edge_count),
            "avg_weight": avg_weight,
            "node_utilization": node_utilization,
            "synapse_density": synapse_density,
            "weight_health": weight_health,
            "overall_health": overall_health,
            "coherence": snapshot.metadata.get("coherence", 1.0),
            "absorption_rate": self._extract_absorption_rate(),
        }

    def _extract_absorption_rate(self) -> float:
        """Estimate peer event absorption rate from bridge stats.

        Punchlist #44.  Absorption rate = fraction of peer events that
        would pass the current relevance_threshold.  Approximated by
        comparing cached peer event growth vs drain/sync count growth.

        If no bridge is available or no data yet, returns the midpoint
        of the bootstrap range (neutral — no tuning signal).
        """
        if self._bridge_ref is None or not hasattr(self._bridge_ref, 'get_stats'):
            return 0.35  # Neutral midpoint

        stats = self._bridge_ref.get_stats()
        peer_cached = stats.get("peer_events_cached", 0)

        # Use drain_count (tract bridge) or sync_count (peer bridge)
        drain_count = stats.get("drain_count", stats.get("sync_count", 0))

        # On first call, just record baseline
        if self._last_drain_count == 0 and drain_count == 0:
            self._last_peer_events_cached = peer_cached
            self._last_drain_count = drain_count
            return 0.35  # No data yet

        # Compute delta since last check
        events_delta = peer_cached - self._last_peer_events_cached
        drain_delta = drain_count - self._last_drain_count

        # Update tracking
        self._last_peer_events_cached = peer_cached
        self._last_drain_count = drain_count

        if drain_delta <= 0 or events_delta <= 0:
            return 0.35  # No new activity — neutral

        # Approximate absorption: events cached per drain cycle relative
        # to a baseline.  Higher peer_events_cached growth per drain
        # means more events are passing the threshold (high absorption).
        # Normalize: assume ~50 events per drain is "normal" throughput.
        events_per_drain = events_delta / max(drain_delta, 1)
        absorption = min(1.0, events_per_drain / 50.0)

        return absorption

    # -------------------------------------------------------------------
    # Diagnosis and recommendation
    # -------------------------------------------------------------------

    def _diagnose_and_recommend(
        self, health: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Compare health metrics against competence-scaled ranges."""
        recommendations: List[Dict[str, Any]] = []

        if self._ng_lite_ref is None:
            return recommendations
        if not hasattr(self._ng_lite_ref, 'get_tunables'):
            logger.warning("NGLite instance lacks update_tunable() — skipping tuning")
            return recommendations

        tunables = self._ng_lite_ref.get_tunables()
        avg_weight = health.get("avg_weight", 0.5)
        node_util = health.get("node_utilization", 0.5)
        syn_density = health.get("synapse_density", 3.0)

        # --- Weight saturation / starvation ---
        w_lo, w_hi = self._effective_range("avg_weight")
        if avg_weight > w_hi and "success_boost" in tunables:
            rec = self._propose_adjustment(
                "success_boost", tunables["success_boost"], direction=-1,
                reason=f"avg_weight {avg_weight:.3f} > {w_hi:.3f} (saturating)",
            )
            if rec:
                recommendations.append(rec)
            if "failure_penalty" in tunables:
                rec = self._propose_adjustment(
                    "failure_penalty", tunables["failure_penalty"], direction=1,
                    reason=f"avg_weight {avg_weight:.3f} > {w_hi:.3f} (strengthen forgetting)",
                )
                if rec:
                    recommendations.append(rec)

        elif avg_weight < w_lo and "success_boost" in tunables:
            rec = self._propose_adjustment(
                "success_boost", tunables["success_boost"], direction=1,
                reason=f"avg_weight {avg_weight:.3f} < {w_lo:.3f} (too weak)",
            )
            if rec:
                recommendations.append(rec)
            if "failure_penalty" in tunables:
                rec = self._propose_adjustment(
                    "failure_penalty", tunables["failure_penalty"], direction=-1,
                    reason=f"avg_weight {avg_weight:.3f} < {w_lo:.3f} (reduce forgetting)",
                )
                if rec:
                    recommendations.append(rec)

        # --- Node utilization ---
        u_lo, u_hi = self._effective_range("node_utilization")
        if node_util < u_lo and "novelty_threshold" in tunables:
            rec = self._propose_adjustment(
                "novelty_threshold", tunables["novelty_threshold"], direction=-1,
                reason=f"node_utilization {node_util:.3f} < {u_lo:.3f} (dead substrate)",
            )
            if rec:
                recommendations.append(rec)
        elif node_util > u_hi and "novelty_threshold" in tunables:
            rec = self._propose_adjustment(
                "novelty_threshold", tunables["novelty_threshold"], direction=1,
                reason=f"node_utilization {node_util:.3f} > {u_hi:.3f} (noisy)",
            )
            if rec:
                recommendations.append(rec)

        # --- Synapse density ---
        d_lo, d_hi = self._effective_range("synapse_density")
        if syn_density < d_lo and "pruning_threshold" in tunables:
            rec = self._propose_adjustment(
                "pruning_threshold", tunables["pruning_threshold"], direction=-1,
                reason=f"synapse_density {syn_density:.2f} < {d_lo:.2f} (too sparse)",
            )
            if rec:
                recommendations.append(rec)
        elif syn_density > d_hi and "pruning_threshold" in tunables:
            rec = self._propose_adjustment(
                "pruning_threshold", tunables["pruning_threshold"], direction=1,
                reason=f"synapse_density {syn_density:.2f} > {d_hi:.2f} (over-connected)",
            )
            if rec:
                recommendations.append(rec)

        # --- Peer event absorption rate (Punchlist #44) ---
        absorption = health.get("absorption_rate", 0.35)
        a_lo, a_hi = self._effective_range("absorption_rate")
        if absorption < a_lo and "relevance_threshold" in tunables:
            # Too few events absorbed — threshold is too high, lower it
            rec = self._propose_adjustment(
                "relevance_threshold", tunables["relevance_threshold"], direction=-1,
                reason=f"absorption_rate {absorption:.3f} < {a_lo:.3f} (starved — threshold too high)",
            )
            if rec:
                recommendations.append(rec)
        elif absorption > a_hi and "relevance_threshold" in tunables:
            # Too many events flooding in — threshold is too low, raise it
            rec = self._propose_adjustment(
                "relevance_threshold", tunables["relevance_threshold"], direction=1,
                reason=f"absorption_rate {absorption:.3f} > {a_hi:.3f} (flooded — threshold too low)",
            )
            if rec:
                recommendations.append(rec)

        return recommendations

    def _propose_adjustment(
        self,
        key: str,
        tunable_info: Dict[str, float],
        direction: int,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Propose a competence-scaled adjustment to a parameter.

        Step size scales with competence: cautious when new, bolder
        as accuracy is demonstrated.
        """
        current = tunable_info["value"]
        lo = tunable_info["min"]
        hi = tunable_info["max"]

        step_fraction = self._effective_step_fraction(key)
        step = max(current * step_fraction, 0.001)
        proposed = current + (step * direction)
        proposed = max(lo, min(hi, proposed))

        if abs(proposed - current) < 1e-8:
            return None

        return {
            "key": key,
            "current": current,
            "proposed": proposed,
            "direction": "increase" if direction > 0 else "decrease",
            "reason": reason,
            "step_fraction": step_fraction,
            "competence": self.get_competence(key),
        }

    # -------------------------------------------------------------------
    # Outcome tracking and competence feedback (Pattern B)
    # -------------------------------------------------------------------

    def record_pending_outcome(
        self, key: str, old_value: float, new_value: float, health_before: float,
    ) -> None:
        """Record a pending tuning decision awaiting next-cycle health comparison.

        Called by the engine immediately after applying a recommendation.
        The outcome resolves on the next process() cycle when we can
        compare health_before vs health_after.
        """
        self._last_health_snapshot[f"_pending_{key}"] = {
            "old_value": old_value,
            "new_value": new_value,
            "health_before": health_before,
        }

    def _resolve_pending_outcomes(self) -> None:
        """Resolve pending tuning outcomes by comparing pre/post health.

        Called at the start of each process() cycle.  If there are pending
        outcomes from the previous cycle, compare health_before (stored)
        against current health (just measured on previous cycle's snapshot).
        """
        health_after = self._last_health_snapshot.get("overall_health")
        if health_after is None:
            return

        pending_keys = [
            k for k in list(self._last_health_snapshot.keys())
            if k.startswith("_pending_")
        ]

        for pkey in pending_keys:
            pending = self._last_health_snapshot.pop(pkey)
            key = pkey[len("_pending_"):]
            health_before = pending["health_before"]
            improved = health_after >= health_before

            # Update competence based on outcome
            new_competence = self._update_competence(key, improved)

            # Record in history
            entry = {
                "key": key,
                "old_value": pending["old_value"],
                "new_value": pending["new_value"],
                "health_before": health_before,
                "health_after": health_after,
                "improved": improved,
                "competence_after": new_competence,
                "timestamp": time.time(),
            }
            self._tuning_history.append(entry)
            if len(self._tuning_history) > self._history_max:
                self._tuning_history = self._tuning_history[-self._history_max:]

            logger.info(
                "Tuning outcome: %s %.6f→%.6f, health %.3f→%.3f (%s), "
                "competence=%.3f",
                key, pending["old_value"], pending["new_value"],
                health_before, health_after,
                "improved" if improved else "regressed",
                new_competence,
            )
