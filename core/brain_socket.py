"""
BrainSocket — ElmerBrain as an Elmer Processing Socket

Wraps the surgically modified Qwen2.5-0.5B (ElmerBrain) as a standard
ElmerSocket. Reads GraphSnapshot, extracts substrate features, runs
through the transformer body, outputs SubstrateSignal.

This is the frozen cognitive core — the stable reference brain.
ProtoUniBrain (Phase 0) will be a second instance with Lenia dynamics
on the unfrozen transformer body, running alongside this one.

# ---- Changelog ----
# [2026-03-25] Claude Code (Opus 4.6) — Initial implementation
#   What: BrainSocket wrapping ElmerBrain (~/UniAI/models/elmer_brain_v0.1.pt)
#   Why:  UniAI Phase 0 — socket-ify the surgically modified transformer
#         so it can run alongside existing Elmer sockets and process
#         real substrate state.
# -------------------
"""

from __future__ import annotations

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

logger = logging.getLogger("elmer.brain_socket")

# Torch is heavy — import lazily so Elmer can still start without it
_torch = None
_brain_module = None


def _lazy_imports():
    """Import torch and brain module on first use."""
    global _torch, _brain_module
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


class BrainSocket(ElmerSocket):
    """ElmerBrain wrapped as a standard Elmer socket.

    Extracts substrate features from GraphSnapshot, runs inference
    through the surgically modified transformer, and produces a
    SubstrateSignal with the 9 standard fields + 3 Elmer extensions.
    """

    def __init__(self, model_path: str = None):
        super().__init__()
        self._model_path = model_path or os.path.expanduser(
            "~/UniAI/models/elmer_brain_v0.1.pt"
        )
        self._brain = None
        self._config = None

    @property
    def socket_id(self) -> str:
        return "elmer:brain"

    @property
    def socket_type(self) -> str:
        return "brain"

    def declare_requirements(self) -> HardwareRequirements:
        return HardwareRequirements(
            min_memory_mb=1500,  # ~1.4GB model + overhead
            gpu_required=False,
            cpu_cores=2,
            disk_mb=1500,
        )

    def load(self, model_path: str) -> bool:
        """Load the ElmerBrain checkpoint."""
        try:
            _lazy_imports()
            checkpoint = _torch.load(
                self._model_path,
                map_location="cpu",
                weights_only=False,
            )
            self._config = checkpoint['config']

            # Reconstruct the model
            # We need a fresh Qwen2 body to load state into
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                self._config['base_model'],
                dtype=_torch.float32,
                low_cpu_mem_usage=True,
            )
            body = base_model.model
            body.embed_tokens = None

            self._brain = _brain_module.ElmerBrain(body, self._config['hidden_size'])
            self._brain.load_state_dict(checkpoint['model_state_dict'])
            self._brain.eval()

            # Freeze everything for the stable reference brain
            for param in self._brain.parameters():
                param.requires_grad = False

            self._loaded = True
            logger.info(
                "BrainSocket loaded: %s (%d params)",
                self._model_path,
                self._config['parameter_counts']['total'],
            )
            return True

        except Exception as exc:
            logger.error("BrainSocket load failed: %s", exc)
            self._loaded = False
            return False

    def unload(self) -> None:
        self._brain = None
        self._config = None
        self._loaded = False
        logger.info("BrainSocket unloaded")

    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Process a GraphSnapshot through ElmerBrain."""
        if not self._loaded or self._brain is None:
            raise RuntimeError("BrainSocket not loaded")

        start = time.time()
        self._process_count += 1

        try:
            # Extract substrate features from GraphSnapshot
            substrate_state = self._snapshot_to_substrate(snapshot)

            # Run inference
            with _torch.no_grad():
                output = self._brain(substrate_state)

            # Map decoder output to SubstrateSignal
            signals = output['signals'][0].tolist()  # unbatch
            actions = output['actions'][0].tolist()

            signal_names = output['signal_names']
            action_names = output['action_names']

            # Build signal dict from model output
            sig = {name: val for name, val in zip(signal_names, signals)}
            act = {name: val for name, val in zip(action_names, actions)}

            elapsed = time.time() - start
            self._total_latency += elapsed

            substrate_signal = SubstrateSignal.create(
                signal_type="coherence",
                description="ElmerBrain transformer assessment",
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
                    "socket": "elmer:brain",
                    "actions": act,
                    "inference_time_ms": elapsed * 1000,
                    "model": "elmer_brain_v0.1",
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
            logger.error("BrainSocket process error: %s", exc)

            # Return a degraded signal rather than crashing the pipeline
            return SocketOutput(
                signal=SubstrateSignal.create(
                    signal_type="health",
                    description=f"ElmerBrain error: {exc}",
                    health_score=0.0,
                    confidence=0.0,
                    severity=1.0,
                    metadata={"socket": "elmer:brain", "error": str(exc)},
                ),
                confidence=0.0,
                processing_time=elapsed,
            )

    def health(self) -> SocketHealth:
        return self._make_health("healthy" if self._loaded else "offline")

    # -----------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------

    def _snapshot_to_substrate(self, snapshot: GraphSnapshot) -> dict:
        """Convert GraphSnapshot to the substrate_state dict ElmerBrain expects.

        Extracts aggregate statistics from the snapshot's nodes and edges
        to populate the five feature groups:
        - node_features: voltage, firing_rate, excitability
        - synapse_features: weight_mean, age_mean
        - topo_features: density, clustering, connected_components
        - temporal_features: recent_firing_mean, recent_firing_std,
                            stdp_timing_mean, stdp_timing_std
        - identity_embedding: 384-dim (from metadata or zero-filled)
        """
        nodes = snapshot.nodes
        edges = snapshot.edges
        meta = snapshot.metadata

        # Node features — aggregate across all nodes
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

        # Synapse features — aggregate across all edges
        if edges:
            weights = [e.get('weight', 0.5) for e in edges]
            ages = [e.get('age', 0.0) for e in edges]
            synapse_feat = [np.mean(weights), np.mean(ages)]
        else:
            synapse_feat = [0.5, 0.0]

        # Topological features
        n_nodes = max(len(nodes), 1)
        n_edges = len(edges)
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0
        clustering = meta.get('clustering_coefficient', 0.0)
        components = meta.get('connected_components', 1)
        # Normalize components to [0,1] range
        norm_components = min(components / max(n_nodes, 1), 1.0)
        topo_feat = [density, clustering, norm_components]

        # Temporal features
        if nodes:
            recent = [n.get('recent_firing', 0.0) for n in nodes]
            stdp = [n.get('stdp_timing', 0.0) for n in nodes]
            temporal_feat = [
                np.mean(recent), np.std(recent),
                np.mean(stdp), np.std(stdp),
            ]
        else:
            temporal_feat = [0.0, 0.0, 0.0, 0.0]

        # Identity embedding — 384-dim ecosystem standard
        identity = meta.get('identity_embedding', None)
        if identity is not None:
            identity = np.array(identity, dtype=np.float32)[:384]
            if len(identity) < 384:
                identity = np.pad(identity, (0, 384 - len(identity)))
        else:
            identity = np.zeros(384, dtype=np.float32)

        # Convert to tensors
        return {
            'node_features': _torch.tensor([node_feat], dtype=_torch.float32),
            'synapse_features': _torch.tensor([synapse_feat], dtype=_torch.float32),
            'topo_features': _torch.tensor([topo_feat], dtype=_torch.float32),
            'temporal_features': _torch.tensor([temporal_feat], dtype=_torch.float32),
            'identity_embedding': _torch.from_numpy(identity.reshape(1, -1)),
        }
