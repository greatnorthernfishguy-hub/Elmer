"""
ProtoUniBrainSocket — The Living Brain

Same surgery as BrainSocket, but the transformer body is UNFROZEN.
Lenia dynamics run on every process() call — the model learns from
real substrate state, not random tensors.

Runs alongside the frozen BrainSocket. The frozen one is the stable
reference. This one is alive and learning.

# ---- Changelog ----
# [2026-03-29] Claude Code (Opus 4.6) — Unified encoder, no v1/v2 split
#   What: Single GraphEncoder reads whatever data is available. No branching.
#   Why:  Same encoder for frozen and living brain. Reads topology deltas,
#         live graph, or flat snapshots — richest first. Encoder params
#         unfrozen on proto — Lenia evolves the mesh.
# [2026-03-28] Claude Code (Opus 4.6) — v2 reads Elmer's own substrate
#   What: Proto socket reads Elmer's local NG-Lite directly via ecosystem ref.
#   Why:  Same as BrainSocket — the River deposited topology here. Read it
#         directly. No context dict pass-through. Law 1. The encoder is part
#         of the extraction bucket mesh. It reads the substrate's own organization.
#   How:  set_ecosystem_ref() wired by engine. process() reads local substrate.
#         v2 encoder params are UNFROZEN — Lenia evolves the mesh.
# [2026-03-25] Claude Code (Opus 4.6) — Initial implementation
#   What: ProtoUniBrain socket with live Lenia dynamics
#   Why:  The whole point of UniAI — a brain that learns from use
# -------------------
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Dict

import numpy as np

from core.base_socket import (
    ElmerSocket,
    GraphSnapshot,
    HardwareRequirements,
    SocketHealth,
    SocketOutput,
)
from core.substrate_signal import SubstrateSignal

logger = logging.getLogger("elmer.proto_brain")

_torch = None
_brain_module = None
_lenia_module = None
_encoder_module = None


def _lazy_imports():
    global _torch, _brain_module, _lenia_module
    if _torch is None:
        import torch
        _torch = torch
    if _brain_module is None:
        import importlib.util
        import sys as _sys
        spec = importlib.util.spec_from_file_location(
            "elmer_brain_surgery",
            os.path.expanduser("~/UniAI/surgery/elmer_brain_surgery.py"),
        )
        _brain_module = importlib.util.module_from_spec(spec)
        _sys.modules["elmer_brain_surgery"] = _brain_module
        spec.loader.exec_module(_brain_module)
    if _encoder_module is None:
        try:
            import importlib.util as ilu2
            import sys as _sys2
            spec = ilu2.spec_from_file_location(
                "graph_encoder",
                os.path.expanduser("~/UniAI/surgery/graph_encoder.py"),
            )
            _encoder_module = ilu2.module_from_spec(spec)
            _sys2.modules["graph_encoder"] = _encoder_module
            spec.loader.exec_module(_encoder_module)
        except Exception:
            _encoder_module = None
    if _lenia_module is None:
        import importlib.util as ilu
        import sys as _sys
        spec = ilu.spec_from_file_location(
            "lenia_engine",
            os.path.expanduser("~/UniAI/lenia/engine.py"),
        )
        _lenia_module = ilu.module_from_spec(spec)
        _sys.modules["lenia_engine"] = _lenia_module
        spec.loader.exec_module(_lenia_module)


class ProtoUniBrainSocket(ElmerSocket):
    """The living brain. Unfrozen transformer body + Lenia dynamics.

    On every process() call:
      1. Convert GraphSnapshot to substrate state
      2. Forward pass through the transformer (captures activations)
      3. Apply one Lenia dynamics step (reshapes weights)
      4. Return SubstrateSignal

    The model learns from doing the work. Activation flow IS the
    training signal. Lenia dynamics ARE the learning rule.
    """

    def __init__(self, model_path: str = None, ecosystem=None):
        super().__init__()
        self._model_path = model_path or os.path.expanduser(
            "~/UniAI/models/elmer_brain_v0.1.pt"
        )
        self._brain = None  # unified ElmerBrain with GraphEncoder
        self._lenia = None
        self._config = None
        self._lenia_steps = 0
        self._ecosystem = ecosystem

    @property
    def socket_id(self) -> str:
        return "elmer:proto_unibrain"

    @property
    def socket_type(self) -> str:
        return "proto_brain"

    def declare_requirements(self) -> HardwareRequirements:
        return HardwareRequirements(
            min_memory_mb=6000,
            gpu_required=False,
            cpu_cores=2,
            disk_mb=5500,
        )

    def load(self, model_path: str) -> bool:
        """Load the model with UNFROZEN body and attach Lenia dynamics."""
        try:
            _lazy_imports()
            from transformers import AutoModelForCausalLM

            # Load checkpoint config
            checkpoint = _torch.load(
                self._model_path,
                map_location="cpu",
                weights_only=False,
            )
            self._config = checkpoint['config']
            del checkpoint
            gc.collect()

            # Load base model directly (memory-efficient path)
            logger.info("ProtoUniBrain: loading %s...", self._config['base_model'])
            base = AutoModelForCausalLM.from_pretrained(
                self._config['base_model'],
                dtype=_torch.float32,
                low_cpu_mem_usage=True,
            )
            body = base.model
            body.embed_tokens = None

            # Load v1 brain to get transformer body + decoder
            v1_brain = _brain_module.ElmerBrain(body, self._config['hidden_size'])
            del base
            gc.collect()

            # Build unified brain with GraphEncoder
            if _encoder_module is not None:
                self._brain = _encoder_module.ElmerBrain(
                    v1_brain.transformer_body,
                    v1_brain.decoder,
                    self._config['hidden_size'],
                )
                logger.info("ProtoUniBrain: unified GraphEncoder loaded")
            else:
                self._brain = v1_brain
                logger.info("ProtoUniBrain: unified encoder not available, using v1")

            # UNFREEZE the transformer body — this is the key difference
            for param in self._brain.transformer_body.parameters():
                param.requires_grad = True

            # Encoder params also unfrozen — Lenia evolves the mesh
            if hasattr(self._brain, 'encoder'):
                for param in self._brain.encoder.parameters():
                    param.requires_grad = True

            # Attach Lenia dynamics engine
            self._lenia = _lenia_module.LeniaEngine(
                self._brain.transformer_body,
                _lenia_module.LeniaConfig(
                    kernel_radius=5,
                    kernel_sigma=0.8,
                    growth_mu=0.12,
                    growth_sigma=0.02,
                    growth_scale=0.005,
                    max_weight_delta=0.05,
                    activation_coupling=2.0,
                ),
            )
            self._lenia.register_hooks()

            self._brain.eval()
            self._loaded = True

            trainable = sum(
                p.numel() for p in self._brain.parameters() if p.requires_grad
            )
            logger.info(
                "ProtoUniBrain loaded: %d trainable params, Lenia attached",
                trainable,
            )
            return True

        except Exception as exc:
            logger.error("ProtoUniBrain load failed: %s", exc)
            self._loaded = False
            return False

    def unload(self) -> None:
        if self._lenia:
            self._lenia.remove_hooks()
        self._brain = None
        self._lenia = None
        self._config = None
        self._loaded = False
        gc.collect()
        logger.info("ProtoUniBrain unloaded")

    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Process snapshot AND apply Lenia dynamics.

        This is where the model learns. Every call reshapes the weights.
        """
        if not self._loaded or self._brain is None:
            raise RuntimeError("ProtoUniBrain not loaded")

        start = time.time()
        self._process_count += 1

        try:
            # 1. Gather all available data sources
            autonomic = context.get('autonomic_state', 'PARASYMPATHETIC')
            graph = None
            latest_delta = None
            if self._ecosystem:
                graph = getattr(self._ecosystem, '_graph', None)
                latest_delta = self._drain_latest_delta()

            # Single path — encoder reads whatever's available
            with _torch.no_grad():
                if hasattr(self._brain, 'encoder') and hasattr(self._brain.encoder, 'topology_proj'):
                    output = self._brain(
                        topology_delta=latest_delta,
                        graph=graph,
                        snapshot=snapshot,
                        autonomic_state=autonomic,
                    )
                else:
                    substrate_state = self._snapshot_to_substrate(snapshot)
                    output = self._brain(substrate_state)

            # 2. Lenia dynamics step — THE LEARNING
            lenia_metrics = self._lenia.step()
            self._lenia_steps += 1

            # 4. Build signal from output
            signals = output['signals'][0].tolist()
            actions = output['actions'][0].tolist()
            signal_names = output['signal_names']
            action_names = output['action_names']
            sig = {n: v for n, v in zip(signal_names, signals)}
            act = {n: v for n, v in zip(action_names, actions)}

            elapsed = time.time() - start
            self._total_latency += elapsed

            substrate_signal = SubstrateSignal.create(
                signal_type="coherence",
                description="ProtoUniBrain living assessment",
                coherence_score=sig['coherence'],
                health_score=sig['health'],
                anomaly_level=sig['anomaly'],
                novelty=sig['novelty'],
                confidence=sig['confidence'],
                severity=sig['severity'],
                identity_coherence=sig['identity_coherence'],
                pruning_pressure=sig['pruning_pressure'],
                topology_health=sig['topology_health'],
                metadata={
                    "socket": "elmer:proto_unibrain",
                    "actions": act,
                    "inference_time_ms": elapsed * 1000,
                    "lenia_step": self._lenia_steps,
                    "lenia_delta_norm": lenia_metrics['total_delta_norm'],
                    "lenia_time_ms": lenia_metrics['time_ms'],
                    "model": "proto_unibrain_1.5b",
                },
            )

            return SocketOutput(
                signal=substrate_signal,
                confidence=sig['confidence'],
                processing_time=elapsed,
            )

        except Exception as exc:
            self._error_count += 1
            elapsed = time.time() - start
            self._total_latency += elapsed
            logger.error("ProtoUniBrain process error: %s", exc)

            return SocketOutput(
                signal=SubstrateSignal.create(
                    signal_type="health",
                    description=f"ProtoUniBrain error: {exc}",
                    health_score=0.0,
                    confidence=0.0,
                    severity=1.0,
                    metadata={
                        "socket": "elmer:proto_unibrain",
                        "error": str(exc),
                        "lenia_step": self._lenia_steps,
                    },
                ),
                confidence=0.0,
                processing_time=elapsed,
            )

    def set_ecosystem_ref(self, ecosystem):
        """Set reference to Elmer's own ecosystem. Called by engine after init."""
        self._ecosystem = ecosystem

    def _drain_latest_delta(self):
        """Read the latest topology delta from already-drained peer events.

        Reads _peer_events on the tract bridge — data that the ecosystem's
        own drain cycle already absorbed into the local substrate. Same
        pattern Bunyan uses. Does NOT call bridge.drain() (that would
        compete with the ecosystem's drain cycle and steal data).

        Returns the most recent topology_delta, or None if none available.
        """
        if not self._ecosystem:
            return None
        bridge = getattr(self._ecosystem, '_peer_bridge', None)
        if bridge is None:
            return None
        peer_events = getattr(bridge, '_peer_events', [])
        if not peer_events:
            return None
        # Most recent topology delta
        for entry in reversed(peer_events):
            if isinstance(entry, dict) and entry.get('type') == 'topology_delta':
                return entry
        return None

    def health(self) -> SocketHealth:
        h = self._make_health("healthy" if self._loaded else "offline")
        return h

    def get_lenia_summary(self) -> Dict[str, Any]:
        """Get Lenia dynamics state for monitoring."""
        if self._lenia:
            return self._lenia.get_summary()
        return {}

    # -----------------------------------------------------------------
    # Feature extraction (same as BrainSocket)
    # -----------------------------------------------------------------

    def _snapshot_to_substrate(self, snapshot: GraphSnapshot) -> dict:
        """Convert GraphSnapshot to substrate_state dict."""
        nodes = snapshot.nodes
        edges = snapshot.edges
        meta = snapshot.metadata

        if nodes:
            voltages = [n.get('voltage', 0.0) for n in nodes]
            firing_rates = [n.get('firing_rate', 0.0) for n in nodes]
            excitabilities = [n.get('excitability', 0.5) for n in nodes]
            node_feat = [
                np.mean(voltages),
                np.mean(firing_rates),
                np.mean(excitabilities),
            ]
        else:
            node_feat = [0.0, 0.0, 0.5]

        if edges:
            weights = [e.get('weight', 0.5) for e in edges]
            ages = [e.get('age', 0.0) for e in edges]
            synapse_feat = [np.mean(weights), np.mean(ages)]
        else:
            synapse_feat = [0.5, 0.0]

        n_nodes = max(len(nodes), 1)
        n_edges = len(edges)
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0
        clustering = meta.get('clustering_coefficient', 0.0)
        components = meta.get('connected_components', 1)
        norm_components = min(components / max(n_nodes, 1), 1.0)
        topo_feat = [density, clustering, norm_components]

        if nodes:
            recent = [n.get('recent_firing', 0.0) for n in nodes]
            stdp = [n.get('stdp_timing', 0.0) for n in nodes]
            temporal_feat = [
                np.mean(recent), np.std(recent),
                np.mean(stdp), np.std(stdp),
            ]
        else:
            temporal_feat = [0.0, 0.0, 0.0, 0.0]

        identity = meta.get('identity_embedding', None)
        if identity is not None:
            identity = np.array(identity, dtype=np.float32)[:384]
            if len(identity) < 384:
                identity = np.pad(identity, (0, 384 - len(identity)))
        else:
            identity = np.zeros(384, dtype=np.float32)

        return {
            'node_features': _torch.tensor([node_feat], dtype=_torch.float32),
            'synapse_features': _torch.tensor([synapse_feat], dtype=_torch.float32),
            'topo_features': _torch.tensor([topo_feat], dtype=_torch.float32),
            'temporal_features': _torch.tensor([temporal_feat], dtype=_torch.float32),
            'identity_embedding': _torch.from_numpy(identity.reshape(1, -1)),
        }
