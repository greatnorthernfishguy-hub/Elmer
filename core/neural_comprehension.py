"""
Neural Comprehension Socket — Transformer-Powered Graph Reasoning  (PRD §5.2.3)

Drop-in replacement for ComprehensionSocket that uses ProtoUniBrain's living
transformer body instead of heuristics. Same interface, same socket contract,
but transformer-class pattern recognition under the hood.

Brain is wired externally by BrainSwitcher after ProtoUniBrain loads — this
socket never self-loads any model. Falls back to heuristics until brain arrives.

# ---- Changelog ----
# [2026-04-19] Claude Code (Sonnet 4.6) — Fix _neural_process brain call + signal unpacking
#   What: Removed GraphFeatures conversion; call self._brain(snapshot=snapshot) directly.
#         Unpack signals tensor via signal_names; fix key names (coherence not coherence_score).
#   Why:  graph_encoder.ElmerBrain expects snapshot=, not a GraphFeatures positional arg.
#         output['signals'] is a tensor, not a dict — .get() fails on both counts.
#   How:  Skip _snapshot_to_features(); unpack (1,9) tensor into named dict for SubstrateSignal.
# [2026-04-18] Claude Code (Sonnet 4.6) — Wire to ProtoUniBrain via BrainSwitcher
#   What: Removed self-loading ElmerBrain. load() is now a no-op that starts in
#         heuristic fallback. set_brain(brain) / revoke_brain() called by
#         BrainSwitcher after proto loads/sheds. Zero duplicate transformer load.
#   Why:  ElmerBrain is disabled. ProtoUniBrain is the sole brain. Heuristics are
#         emergency fallback only.
#   How:  BrainSwitcher._wire_neural_comprehension() calls set_brain(proto._brain)
#         after _activate_both() succeeds. Same pattern as Tonic body-sharing.
# [2026-03-20] Claude Code (Opus 4.6) — Initial implementation
#   What: NeuralComprehensionSocket using ElmerBrain for inference
#   Why:  PRD §5.2.3 socket 0 — Comprehension via harvested reasoning engine
#   How:  Loads ElmerBrain, converts GraphSnapshot to GraphFeatures,
#         runs transformer inference, packs output as SocketOutput
# -------------------
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import torch

from core.base_socket import (
    ElmerSocket,
    GraphSnapshot,
    HardwareRequirements,
    SocketHealth,
    SocketOutput,
)
from core.substrate_signal import SubstrateSignal

logger = logging.getLogger("elmer.neural_comprehension")


class NeuralComprehensionSocket(ElmerSocket):
    """Transformer-powered comprehension socket.

    Uses a surgically stripped Qwen2.5-0.5B with graph-native I/O layers
    to assess substrate state. The transformer's attention mechanism reasons
    over graph features the way it was pretrained to reason over language.

    Ref: PRD §5.2.3, §5.4
    """

    SOCKET_ID = "elmer:neural_comprehension"
    SOCKET_TYPE = "comprehension"

    def __init__(self):
        super().__init__()
        self._brain: Optional[Any] = None  # ElmerBrain instance
        self._fallback_mode = False

    @property
    def socket_id(self) -> str:
        return self.SOCKET_ID

    @property
    def socket_type(self) -> str:
        return self.SOCKET_TYPE

    def declare_requirements(self) -> HardwareRequirements:
        return HardwareRequirements(
            min_memory_mb=2048,   # ~2GB for 0.5B model in float32
            gpu_required=False,    # CPU inference supported
            cpu_cores=2,
            disk_mb=1024,         # model weights on disk
        )

    def load(self, model_path: str) -> bool:
        """Register socket. Brain wired later via set_brain() from BrainSwitcher.

        Does NOT load its own model. Starts in heuristic fallback until
        ProtoUniBrain's brain is offered via set_brain().
        """
        if self._loaded:
            return True
        self._fallback_mode = True
        self._loaded = True
        self._load_time = time.time()
        logger.info("NeuralComprehensionSocket ready — awaiting ProtoUniBrain body")
        return True

    def set_brain(self, brain) -> None:
        """Wire ProtoUniBrain's brain. Called by BrainSwitcher after proto loads."""
        self._brain = brain
        self._fallback_mode = False
        logger.info("NeuralComprehensionSocket wired to ProtoUniBrain — neural mode active")

    def revoke_brain(self) -> None:
        """Revoke brain (proto shed). Fall back to heuristics."""
        self._brain = None
        self._fallback_mode = True
        logger.info("NeuralComprehensionSocket brain revoked — heuristic fallback")

    def unload(self) -> None:
        """Release the model from memory."""
        if self._brain is not None:
            del self._brain
            self._brain = None
            # Encourage garbage collection of the large model
            import gc
            gc.collect()
        self._loaded = False
        self._fallback_mode = False
        logger.info("NeuralComprehensionSocket unloaded")

    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Run transformer inference on graph state.

        Converts the GraphSnapshot into GraphFeatures, runs through
        the ElmerBrain (encoder → transformer → decoder), and packs
        the SubstrateSignal output.

        Falls back to heuristics if the brain isn't loaded.
        """
        if not self._loaded:
            raise RuntimeError("NeuralComprehensionSocket not loaded")

        t0 = time.time()
        self._process_count += 1

        if self._fallback_mode or self._brain is None:
            return self._heuristic_process(snapshot, context, t0)

        return self._neural_process(snapshot, context, t0)

    def _neural_process(self, snapshot: GraphSnapshot, context: dict, t0: float) -> SocketOutput:
        """Process using the transformer brain."""
        try:
            # graph_encoder.ElmerBrain prefers a live graph object (per-node/hyperedge
            # detail) over a flat GraphSnapshot (12 scalar averages). Engine puts the
            # live NG-Lite instance in context["live_graph"]; fall back to snapshot= only
            # if it's absent (e.g. in tests or before ecosystem initializes).
            live_graph = context.get('live_graph')
            with torch.no_grad():
                if live_graph is not None:
                    output = self._brain(graph=live_graph)
                else:
                    output = self._brain(snapshot=snapshot)

            elapsed = time.time() - t0
            self._total_latency += elapsed

            # output['signals'] is a tensor (batch, n_fields); unpack via signal_names.
            signals_tensor = output['signals']       # (1, 9)
            signal_names = output.get('signal_names', [])
            signals = {name: signals_tensor[0, i].item()
                       for i, name in enumerate(signal_names)}

            # top_action from actions tensor
            actions_tensor = output['actions']       # (1, n_actions)
            action_names = output.get('action_names', [])
            top_idx = int(actions_tensor[0].argmax().item())
            top_action = action_names[top_idx] if top_idx < len(action_names) else 'observe'
            action_probs = {name: actions_tensor[0, i].item()
                            for i, name in enumerate(action_names)}

            signal = SubstrateSignal.create(
                signal_type="observation",
                description=f"Neural comprehension: {top_action}",
                coherence_score=signals.get('coherence', 0.5),
                health_score=signals.get('health', 0.5),
                anomaly_level=signals.get('anomaly', 0.0),
                novelty=signals.get('novelty', 0.5),
                confidence=signals.get('confidence', 0.5),
                severity=signals.get('severity', 0.0),
                temporal_window=elapsed,
                identity_coherence=signals.get('identity_coherence', None),
                pruning_pressure=signals.get('pruning_pressure', None),
                topology_health=signals.get('topology_health', None),
                metadata={
                    "socket": self.socket_id,
                    "mode": "neural",
                    "top_action": top_action,
                    "action_probs": action_probs,
                    "inference_ms": elapsed * 1000,
                    "node_count": len(snapshot.nodes),
                    "edge_count": len(snapshot.edges),
                },
            )

            return SocketOutput(
                signal=signal,
                graph_delta=None,
                confidence=signals.get('confidence', 0.5),
                processing_time=elapsed,
            )

        except Exception as exc:
            logger.error("Neural inference failed: %s — falling back", exc)
            return self._heuristic_process(snapshot, {}, time.time())

    def _heuristic_process(self, snapshot: GraphSnapshot, context: dict, t0: float) -> SocketOutput:
        """Fallback heuristic processing (same as original ComprehensionSocket)."""
        node_count = len(snapshot.nodes)
        edge_count = len(snapshot.edges)

        max_edges = max(node_count * (node_count - 1) / 2, 1)
        coherence = min(edge_count / max_edges, 1.0) if node_count > 1 else 1.0
        novelty = 1.0 / (1.0 + node_count * 0.1)

        elapsed = time.time() - t0
        self._total_latency += elapsed

        signal = SubstrateSignal.create(
            signal_type="observation",
            description=f"Comprehension (heuristic): {node_count} nodes, {edge_count} edges",
            coherence_score=coherence,
            health_score=1.0,
            anomaly_level=0.0,
            novelty=novelty,
            confidence=0.8,
            severity=0.0,
            temporal_window=elapsed,
            metadata={
                "socket": self.socket_id,
                "mode": "heuristic_fallback",
                "node_count": node_count,
                "edge_count": edge_count,
            },
        )

        return SocketOutput(
            signal=signal,
            graph_delta=None,
            confidence=0.8,
            processing_time=elapsed,
        )

    def _snapshot_to_features(self, snapshot: GraphSnapshot) -> 'GraphFeatures':
        """Convert a GraphSnapshot into GraphFeatures for the encoder."""
        from graph_io import GraphFeatures

        nodes = snapshot.nodes
        edges = snapshot.edges
        n_nodes = len(nodes)
        n_edges = len(edges)

        # Extract node features
        if nodes:
            voltages = torch.tensor(
                [n.get('voltage', 0.5) for n in nodes], dtype=torch.float32
            )
            firing_rates = torch.tensor(
                [n.get('firing_rate', 0.1) for n in nodes], dtype=torch.float32
            )
            excitability = torch.tensor(
                [0.1 if n.get('constitutional', False) else 0.8 for n in nodes],
                dtype=torch.float32,
            )
        else:
            voltages = torch.zeros(1)
            firing_rates = torch.zeros(1)
            excitability = torch.zeros(1)

        # Extract edge/synapse features
        if edges:
            weights = torch.tensor(
                [e.get('weight', 0.5) for e in edges], dtype=torch.float32
            )
            ages = torch.tensor(
                [e.get('age', 0.0) for e in edges], dtype=torch.float32
            )
        else:
            weights = torch.zeros(1)
            ages = torch.zeros(1)

        # Topology
        max_possible = max(n_nodes * (n_nodes - 1) / 2, 1)
        density = n_edges / max_possible

        # Identity embedding from metadata if available
        meta = snapshot.metadata
        id_emb = torch.zeros(384)
        if 'identity_embedding' in meta:
            raw = meta['identity_embedding']
            if isinstance(raw, (list, tuple)):
                id_emb = torch.tensor(raw[:384], dtype=torch.float32)
                if len(id_emb) < 384:
                    id_emb = torch.cat([id_emb, torch.zeros(384 - len(id_emb))])

        return GraphFeatures(
            node_voltages=voltages,
            node_firing_rates=firing_rates,
            node_excitability=excitability,
            synapse_weights=weights,
            synapse_ages=ages,
            density=torch.tensor([density]),
            clustering=torch.tensor([meta.get('clustering', 0.0)]),
            n_components=torch.tensor([float(meta.get('n_components', 1))]),
            n_nodes=torch.tensor([float(n_nodes)]),
            n_synapses=torch.tensor([float(n_edges)]),
            n_hyperedges=torch.tensor([float(meta.get('n_hyperedges', 0))]),
            recent_firings=torch.tensor(meta.get('recent_firings', [0]*15), dtype=torch.float32)[:15],
            stdp_delta_mean=torch.tensor([meta.get('stdp_delta_mean', 0.0)]),
            identity_embedding=id_emb,
        )

    def health(self) -> SocketHealth:
        mode = "neural" if not self._fallback_mode else "heuristic_fallback"
        return self._make_health("healthy" if self._loaded else "offline")
