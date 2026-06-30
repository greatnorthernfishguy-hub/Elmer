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
# [2026-06-30] Claude Code (Sonnet 4.6) — #329 deposit side: Elmer health assessments → Commons
#   What: _deposit_health_to_commons() salience-gates SubstrateSignal fields (coherence_score,
#         health_score, anomaly_level, novelty, confidence, severity) to Commons on each pulse.
#         Salience signal = coherence deviation from COHERENCE_HEALTHY (0.70): degraded substrate
#         → anomaly deposit with full detail; stable → nominal aggregate. Gate initialized in
#         __init__ via ng_salience_gate.SalienceGate("elmer", _elmer_health_surprise, ...).
#         Module-level _elmer_health_surprise() defines the surprise fn. Called in _pulse_cycle()
#         after _bucket_commons_substrate(). Also updated Darwin agg_fields + NG agg_fields
#         to restore total_nodes/total_synapses (same session).
#   Why:  THC and Immunis need Elmer's health signal for Triad self-regulation. Flood concern
#         is solved by the salience gate (same pattern as NG's metrics gate). LAW-7 concern:
#         health metrics are telemetry observations, same class as NG's graph counts (cleared
#         by Josh #320). Elmer CLAUDE.md §2 read-only rule applies to autonomic STATE writes,
#         not Commons deposits of health observations.
#   How:  _elmer_health_surprise at module level. Gate init in __init__ after _commons_seen.
#         _deposit_health_to_commons() method + _pulse_cycle() call. Fail-soft throughout.
# [2026-06-29] Claude Code (Sonnet 4.6) — #329 Commons bucket migration (bucket side, now complete)
#   What: _bucket_commons_substrate() in _pulse_cycle() buckets metrics:neurograph:* deposits from
#         the Commons → _commons_substrate_novelty EWMA. Exposed in health() as
#         'commons_substrate_novelty'. Mirrors THC's _bucket_commons_novelty() exactly.
#         _commons_seen dedup set bounded to 4096→trim to 2048.
#   Why:  Elmer's only Commons connection was _arousal() (#328). Without a bucket path for
#         NG topology metrics, Elmer has no live view of the shared substrate health since
#         the tract feed died 2026-06-07. This restores that observation path via the Commons.
#   How:  _pulse_cycle() → _bucket_commons_substrate() → commons.bucket_recent(limit=50,
#         with_metadata=True) → filter metrics:neurograph:* → _surprise_from_substrate_metric()
#         → EWMA update. Fail-soft throughout. _bucket_commons_substrate() mirrors
#         THC's _bucket_commons_novelty() (same limit/dedup/EWMA shape). The
#         _surprise_from_substrate_metric() reads ng_salience_gate.py's actual fields
#         (anomaly→meta["signal"], nominal→agg["predictions_surprised"]/total) — also
#         mirrors THC's _surprise_from_metric() exactly (healing_collective_hook.py:518).
# [2026-06-26] Claude Code (Sonnet 4.6) — Fix LAW 7 pulse violation + restore dead _on_river_events (LAW 3)
#   What: (1) _pulse_cycle() called process_text(f"pulse:autonomic={autonomic_state},drained={drained}")
#         every 30s — a synthetic pre-classified label deposited into the substrate, violating LAW 7.
#         Fixed: pulse now just calls _drain_river(); ProtoUniBrain Lenia is sustained by the
#         engine's brain drain idle branch (engine.py _drain_loop, empty_snapshot every 60s) which
#         was already running autonomously and never needed pulse injection (engine.py lines 741-758).
#         (2) _on_river_events() override added: _drain_river() (base class) calls it for each event
#         batch but ElmerHook never overrode it, so ALL BTF events silently hit the base no-op since
#         NEW-5 (2026-05-25), permanently dead-ending #154's raw River absorption path. Override now
#         deposits raw BTF event embeddings directly via eco.record_outcome() — no sensory pipeline,
#         no label wrapping (LAW 7). Events without embeddings are skipped.
#   Why:  "pulse:autonomic=PARASYMPATHETIC,drained=3" is a pre-classification label, not raw
#         experience (ARCHITECTURE.md §7, LAW 7 — substrate receives raw unclassified experience).
#         Dead _on_river_events = LAW 3 (restore not rebuild — #154's intent existed, override missing).
#   How:  _pulse_cycle → single self._drain_river() call. _on_river_events() loops events, handles
#         typed BTF (embedding_as_numpy()) and dicts; records via eco.record_outcome(). Substrate
#         _ecosystem accessed via getattr to fail-soft if engine not yet started.
# [2026-06-22] Claude Code (Opus 4.8) — #328 Step 2: Elmer reads arousal from the Commons
#   What: New _arousal() helper buckets get_commons().read_arousal(); the 4 ng_autonomic.read_state()
#         reads (_pulse_cycle, _process_message, _enrich_context, health) now call self._arousal().
#   Why: #328 reader sweep — autonomic is deposit (Immunis) + bucket (everyone). Elmer reads, never
#         writes (its Cricket-rim WRITE already became a Commons depositor in Step 3 B). Fail-soft.
#   How: read_arousal direct-lookup (vagus never missed). ng_autonomic import now unused (harmless).
# [2026-05-25] Claude Code (Sonnet 4.6) — NEW-5: Replace dead manual drain with _drain_river()
#   What: Replaced manual _eco._peer_bridge._drain_all() block in _pulse_cycle() with
#         self._drain_river(). Removed dead _eco.record_outcome() call that also silently
#         failed because _eco is None (SKIP_ECOSYSTEM=True).
#   Why:  self._eco is None (SKIP_ECOSYSTEM=True), so the bridge was always None and
#         the entire drain block was dead code — tracts were not being drained. The
#         _eco.record_outcome() call also silently failed for the same reason.
#         _drain_river() uses self._tract_bridge correctly (set up by OpenClawAdapter).
#   How:  Drop manual block entirely; call self._drain_river() which hits _tract_bridge.
# [2026-04-29] Claude Code (Sonnet 4.6) — Wire CC Tonic to BrainSwitcher for body sharing (#159)
#   What: Added CC Tonic registration in _delayed_brain_load() after Syl's wiring.
#   Why:  neurograph_rpc.py bootstrap attempt searched _memory._modules (empty at boot)
#         — always a no-op. _delayed_brain_load() fires 60s post-startup after
#         BrainSwitcher loads brains — correct timing for hot-swap offer.
#   How:  sys.modules.get('cc_ng_host') finds CC's TonicEngine; self._engine.set_tonic_engine()
#         registers with BrainSwitcher. Independent try-except from Syl's wiring.
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
# [2026-04-22] Claude Code (Sonnet 4.6) — Guard _deferred_start against post-stop() runs
#   What: _deferred_start() checks _shutdown_event before calling self.start().
#   Why:  Thread spawned in __init__ fires after stop() in tests → see engine.py #205 fix.
#   How:  `if self._shutdown_event.is_set(): return` at thread entry.
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

# [2026-04-18] Claude Code (Sonnet 4.6) — Punchlist #154: Fix Law 7 violation in pulse loop
#   What: Replace text-extraction loop with raw embedding deposit to substrate.
#         Remove _extract_pulse_text staticmethod (dead after this change).
#   Why:  _pulse_cycle was calling process_text() with text extracted from topology
#         deltas — classifying River events before substrate deposit, violating Law 7.
#         Substrate must receive raw experience (embeddings), not extracted labels.
#   How:  Snapshot _peer_events before drain, iterate new events, skip events without
#         embeddings, deposit raw float32 embedding via record_outcome().
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


def _elmer_health_surprise(m: dict) -> float:
    """Salience signal for Elmer's health gate — deviation of coherence from healthy baseline.

    COHERENCE_HEALTHY = 0.70 (core/substrate_signal.py).
    Returns 0.0 when fully healthy; climbs continuously as coherence degrades;
    ~1.0 at the critical threshold (0.15). This drives salience-gate granularity:
    anomaly deposits when health is noticeably degraded, nominal aggregates otherwise.
    """
    coherence = float(m.get("coherence_score", 0.70))
    return max(0.0, (COHERENCE_HEALTHY - coherence) / COHERENCE_HEALTHY)


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

        # --- #329 Commons bucket + deposit: NG metrics in / Elmer health out ---
        # Bucket: mirrors THC's _substrate_novelty pattern. Reads metrics:neurograph:*
        # from Commons on each pulse — live substrate-health feed even when tracts dark.
        # Deposit: salience-gated health assessments out → THC diagnosis + Immunis correlation.
        self._commons_seen: set = set()
        self._commons_substrate_novelty: float = 1.0  # EWMA, starts high (unknown=novel)
        try:
            from ng_salience_gate import SalienceGate as _SG
            self._health_gate = _SG(
                "elmer", _elmer_health_surprise,
                agg_fields=("coherence_score", "health_score", "anomaly_level",
                            "novelty", "confidence", "severity"),
            )
        except Exception:
            self._health_gate = None

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

                    # Wire Tonic engine to BrainSwitcher — direct reference,
                    # no threads, no timers, no hunting through globals.
                    try:
                        import sys as _sys
                        main_mod = _sys.modules.get('__main__')
                        mem = getattr(main_mod, '_memory', None)
                        logger.info("Tonic wiring: _memory=%s", type(mem).__name__ if mem else None)
                        if mem:
                            tonic = getattr(mem, '_tonic_thread', None)
                            logger.info("Tonic wiring: _tonic_thread=%s", type(tonic).__name__ if tonic else None)
                            if tonic:
                                tonic_engine = getattr(tonic, '_latent_engine', None)
                                logger.info("Tonic wiring: _latent_engine=%s", type(tonic_engine).__name__ if tonic_engine else None)
                                if tonic_engine:
                                    self._engine.set_tonic_engine(tonic_engine)
                                    logger.info("Tonic wired to BrainSwitcher — hot-swap active")
                                else:
                                    logger.warning("Tonic wiring: _latent_engine is None")
                            else:
                                logger.warning("Tonic wiring: _tonic_thread is None")
                        else:
                            logger.warning("Tonic wiring: _memory is None")
                    except Exception as exc:
                        logger.warning("Tonic wiring failed: %s", exc)

                    # Register CC's Tonic for body sharing (#159) — independent of Syl's.
                    # cc_ng_host is already imported in neurograph_rpc.py; sys.modules lookup
                    # is safe in namespace-isolated fan-out context.
                    try:
                        import sys as _sys
                        _cc_mod = _sys.modules.get('cc_ng_host')
                        _cc_st = getattr(_cc_mod, '_STATE', None)
                        _cc_ng = getattr(_cc_st, 'cc_ng', None) if _cc_st else None
                        _cc_tt = getattr(_cc_ng, '_tonic_thread', None) if _cc_ng else None
                        _cc_eng = getattr(_cc_tt, '_latent_engine', None) if _cc_tt else None
                        if _cc_eng is not None:
                            self._engine.set_tonic_engine(_cc_eng)
                            logger.info("CC Tonic registered with BrainSwitcher — body sharing live (#159)")
                        else:
                            logger.debug("CC Tonic not running — single-engine mode")
                    except Exception as _ccte:
                        logger.debug("CC Tonic BrainSwitcher registration failed (non-fatal): %s", _ccte)

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
                if self._shutdown_event.is_set():
                    return  # stop() was called before thread ran — bail cleanly
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

    def _arousal(self) -> str:
        """#328 Step 2: current arousal from the Commons (the vagus bucket). Elmer reads, never writes.
        Immunis (sole authority) deposits autonomic:arousal; Elmer buckets it via read_arousal,
        replacing ng_autonomic.read_state() file reads. Fail-soft → PARASYMPATHETIC.
        """
        try:
            from commons import get_commons
            _c = get_commons()
            return _c.read_arousal() if _c is not None else "PARASYMPATHETIC"
        except Exception:
            return "PARASYMPATHETIC"

    def _pulse_cycle(self):
        """One pulse cycle — drain River tracts, absorb raw events into substrate.

        ProtoUniBrain Lenia runs autonomously on the engine's brain drain idle
        branch (engine.py _drain_loop empty_snapshot path, 60s interval) — no
        pulse injection needed. The classified label removed here was LAW 7 violation.
        """
        # Drain tracts. Calls _on_river_events() for LAW 7-compliant absorption.
        self._drain_river()
        # Bucket NG substrate metrics from the Commons (#329).
        self._bucket_commons_substrate()
        # Deposit Elmer's substrate health to Commons for THC/Immunis (#329).
        self._deposit_health_to_commons()

    def _bucket_commons_substrate(self) -> None:
        """Bucket NG substrate metrics from the Commons → _commons_substrate_novelty EWMA.

        Mirrors THC's _bucket_commons_novelty(). Reads metrics:neurograph:* deposits
        (salience-gated by NG's _metrics_gate, every ~2s on the autonomous pulse).
        Updates _commons_substrate_novelty — an EWMA of substrate surprise visible to
        health() and the operator. Fail-soft: bucket failures never break the pulse.
        """
        try:
            from commons import get_commons
            commons = get_commons()
        except Exception:
            return
        if commons is None:
            return
        try:
            recs = commons.bucket_recent(limit=50, with_metadata=True)
        except Exception as exc:
            logger.debug("Elmer Commons substrate bucket failed: %s", exc)
            return
        for target_id, _w, _r, meta in recs:
            if not target_id.startswith("metrics:neurograph:") or target_id in self._commons_seen:
                continue
            self._commons_seen.add(target_id)
            surprise = self._surprise_from_substrate_metric(meta)
            if surprise is not None:
                self._commons_substrate_novelty = (
                    0.8 * self._commons_substrate_novelty + 0.2 * surprise
                )
        if len(self._commons_seen) > 4096:
            self._commons_seen = set(list(self._commons_seen)[-2048:])

    def _deposit_health_to_commons(self) -> None:
        """Deposit Elmer's substrate health assessment to Commons (salience-gated).

        THC buckets from elmer:* for diagnosis. Immunis buckets for threat correlation.
        The salience gate (keyed on coherence deviation from COHERENCE_HEALTHY = 0.70)
        emits granular anomaly deposits when health degrades, nominal aggregates when stable.
        Fail-soft: gate failures never break the pulse.
        """
        if not self._started or self._health_gate is None:
            return
        health_signal = self._health.check()
        metrics = {
            "coherence_score": health_signal.coherence_score,
            "health_score": health_signal.health_score,
            "anomaly_level": health_signal.anomaly_level,
            "novelty": health_signal.novelty,
            "confidence": health_signal.confidence,
            "severity": health_signal.severity,
        }
        try:
            self._health_gate.observe(metrics)
        except Exception as exc:
            logger.debug("Elmer health deposit failed: %s", exc)

    @staticmethod
    def _surprise_from_substrate_metric(meta) -> "Optional[float]":
        """Extract NG's surprise ratio from a bucketed salience-gate metric deposit.

        Mirrors THC's _surprise_from_metric() exactly. Producer is ng_salience_gate.py:
          anomaly  → meta["signal"] (the surprise ratio from the gate's own signal)
          nominal  → predictions_surprised / (predictions_confirmed + predictions_surprised)
        """
        if not isinstance(meta, dict):
            return None
        if meta.get("salience") == "anomaly" and "signal" in meta:
            return float(meta["signal"])
        agg = meta.get("aggregate") if meta.get("salience") == "nominal" else None
        if isinstance(agg, dict):
            c = agg.get("predictions_confirmed", 0)
            s = agg.get("predictions_surprised", 0)
            t = c + s
            return (s / t) if t else 0.0
        return None

    def _on_river_events(self, events: list) -> None:
        """Absorb raw River topology into Elmer's substrate. Restores #154.

        Law 7: BTF event embeddings are raw experience from peer module deposits.
        Record directly — no sensory pipeline classification before deposit.
        Events without embeddings are skipped (hash fallback is not a substitute
        for a missing raw signal — would deposit noise, not experience).
        """
        if not self._started or not events:
            return
        eco = getattr(self._engine, '_ecosystem', None)
        if eco is None:
            return
        for event in events:
            try:
                # Handle typed BTF objects and legacy dicts
                if hasattr(event, 'embedding_as_numpy'):
                    emb = event.embedding_as_numpy()
                elif isinstance(event, dict):
                    raw = event.get('embedding')
                    emb = np.asarray(raw, dtype=np.float32) if raw is not None else None
                else:
                    emb = None
                if emb is None or not len(emb):
                    continue
                target_id = (event.get('target_id', 'river:event') if isinstance(event, dict)
                             else getattr(event, 'target_id', 'river:event'))
                success = (event.get('success', True) if isinstance(event, dict)
                           else bool(getattr(event, 'success', True)))
                eco.record_outcome(
                    embedding=np.asarray(emb, dtype=np.float32),
                    target_id=target_id,
                    success=success,
                )
            except Exception as exc:
                logger.debug("River event absorption error: %s", exc)

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
            autonomic_state = self._arousal()
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
            autonomic_state = self._arousal()
            context["autonomic_state"] = autonomic_state

        except Exception as exc:
            logger.debug("Context enrichment failed: %s", exc)

        return context

    def _derive_target(self, text: str) -> str:
        return "elmer:substrate_input"

    # -----------------------------------------------------------------
    # Health  (PRD §14)
    # -----------------------------------------------------------------

    def _module_stats(self) -> Dict[str, Any]:
        """Elmer-specific stats — sockets, ProtoUniBrain, pipelines, autonomic."""
        if not self._started:
            return {"started": False}
        h = self._engine.health()
        sockets = h.get("sockets") or {}
        brain = h.get("brain") or {}
        pipelines = h.get("pipelines") or {}
        return {
            "started": True,
            "autonomic_state": h.get("autonomic_state", "PARASYMPATHETIC"),
            "process_count": h.get("process_count", 0),
            "proto_unibrain": sockets.get("elmer:proto_unibrain", {}).get("status", "offline"),
            "sockets": {sid: s.get("status", "offline") for sid, s in sockets.items()},
            "brain": brain,
            "pipelines": {k: v.get("processed", 0) for k, v in pipelines.items() if isinstance(v, dict)},
        }

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
            "autonomic_state": self._arousal(),
            "coherence_status": health_signal.coherence_status,
            "commons_substrate_novelty": round(self._commons_substrate_novelty, 4),
            "ecosystem": self.stats(),
        }
