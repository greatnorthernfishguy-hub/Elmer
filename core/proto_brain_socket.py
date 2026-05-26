"""
ProtoUniBrainSocket — The Living Brain

Same surgery as BrainSocket, but the transformer body is UNFROZEN.
Lenia dynamics run on every process() call — the model learns from
real substrate state, not random tensors.

Runs alongside the frozen BrainSocket. The frozen one is the stable
reference. This one is alive and learning.

# ---- Changelog ----
# [2026-05-25] CC — Lenia Phase 1: activation-coupled dynamics, body collapse reset (#252)
#   What: (1) LeniaConfig: added noise_scale=0.2.
#         (2) save/restore: _initial_mean_abs persisted alongside _initial_norms.
#         (3) _reset_body_to_pretrained(): reloads body from cached Qwen2.5-0.5B
#             safetensors, clears Lenia baselines (_initial_norms + _initial_mean_abs).
#         (4) _check_and_reset_body(): detects collapse (std≈0 on >50% of body
#             weight matrices) at startup, triggers reset if found.
#   Why:  Ring kernel + mass conservation created permanent constant-field fixed point.
#         168/168 body matrices had std=0. h_mean≈7, h_norm>>420 (bias-dominated).
#         Lenia burned 0.82 delta-norm/step, all reversed by conservation. Same failure
#         mode as TID free-model domination — attractor wins by physics, not merit.
#   How:  Collapse auto-detected on load; pretrained weights injected; Lenia re-baselines
#         from healthy pretrained stats on first post-reset step.
# [2026-05-25] CC — Fix #251: handle ENTRY_TOPOLOGY entries (type=2)
#   What: _read_river_delta now reads BOTH ENTRY_OUTCOME and ENTRY_TOPOLOGY entries.
#         Topology entries decoded via msgpack, keys normalized to encoder dict format
#         (fired_node_ids→fired_nodes, predictions_confirmed→predictions.confirmed, etc.).
#         Returns dict {'topology': dict|None, 'outcomes': List[np.ndarray]|None}.
#         Call site updated to unpack and pass topology_delta= + outcome_embeddings=.
#   Why:  After the River drain fix (#249), NeuroGraph started depositing TOPOLOGY
#         entries (type=2) not OUTCOME (type=1). Our #249 fix read the right bytes but
#         filtered for the wrong type — still got idle_token. Topology entries carry
#         richer substrate data (node IDs, synapse events, prediction accuracy).
#   How:  Normalize at consumption boundary per LAW 7. Encoder dict path unchanged.
# [2026-05-25] CC — Fix #249: restore River data flow to proto encoder
#   What: (1) _read_river_delta: fixed magic scan b"BT"→b"TB", changed filter from
#         entry_type==2 (topology) to entry_type==ENTRY_OUTCOME (1, 768-dim embeddings).
#         Added cursor-based reading (proto_unibrain.cursor, same JSON format as bridge).
#         Returns List[np.ndarray] instead of PyTopologyEntry.
#         (2) process(): encoder call changed topology_delta= → outcome_embeddings=.
#         (3) logger.debug→logger.error on River read failure so silent bugs surface.
#   Why:  River has never delivered real data to the encoder. Tract carries OUTCOME
#         entries (NeuroGraph learning events, 768-dim embeddings) — encoder was
#         looking for TOPOLOGY entries that don't exist in this tract. BTF magic is
#         0x54,0x42 ("TB") not 0x42,0x54 ("BT") — magic scan never matched.
#         GraphEncoder (in graph_encoder.py) also updated: outcome_proj linear added,
#         outcome_embeddings path prepends projected embeddings to position sequence.
#   How:  See graph_encoder.py #249 changelog for encoder side.
# [2026-05-25] CC — Fresh start: fix adapter init + CRISPR detection (post-collapse analysis)
#   What: (1) DecoderAdapter init: eye_() → normal_(std=0.02). Prevents Lenia mass-conservation
#             collapse (identity diagonal=1.0 is 50x body weight scale → Lenia shrank it to 0).
#         (2) _check_adapter_pressure: measures decoder signal saturation via probe (adapter →
#             decoder → sat_frac) instead of identity deviation. Identity deviation was a proxy
#             that fired on the wrong condition — Lenia had moved the adapter off-identity into
#             an equally degenerate zero attractor without triggering the stuck counter.
#         (3) _apply_crispr_kick: full re-init to N(0,0.02) instead of additive noise. Landing
#             at body scale lets Lenia treat the adapter as a peer, not an outlier.
#         (4) Removed stale constants _ADAPTER_STUCK_EPSILON + _ADAPTER_KICK_SCALE.
#         (5) proto_weights.pt + proto_lenia_state.pt deleted for clean restart.
#   Why:  Post-mortem: adapter diagonal collapsed 1.0→0.0 over 8580 Lenia steps. Off-diagonal
#         stayed at exactly 0.0 (std=0.00000000). Near-zero adapter output → decoder bias-
#         dominated → constant signals. CRISPR never fired (deviation stayed > 0.005 once
#         Lenia started moving). Punchlist #240 concern about co-evolution compounding drift
#         was exactly correct.
# [2026-05-23] CC — Fix log_weight_stats() silent failure: parameters() → named_parameters() (#247)
#   What: Line 774: self._brain.parameters() → self._brain.named_parameters().
#   Why:  parameters() returns bare tensors; unpacking into (name, param) always raised
#         ValueError, caught silently. weight_stats.jsonl has never been written.
#   How:  One-word fix.
# [2026-05-23] CC — Morphogenesis-inspired DecoderAdapter CRISPR pressure/kick
#   What: Added _check_adapter_pressure(), _apply_crispr_kick(), _log_kick_event().
#         Tracks consecutive steps where adapter weight deviation from identity < 0.005.
#         After 200 stuck steps, injects structured noise (off-diagonal full-scale,
#         diagonal 10x smaller) to break identity equilibrium. Kick scale escalates
#         with kick count. Events logged to competence_delta.jsonl.
#         adapter_deviation / adapter_stuck_steps / adapter_kick_count added to metadata.
#   Why:  DecoderAdapter frozen at identity (step 9357+): anomaly=0.999998 constant.
#         Lenia has no gradient to work with when the adapter is pure passthrough.
#         Kick gives Lenia surface area — breaks symmetry so dynamics can learn.
#   How:  _check_adapter_pressure() called in process() outside body_lock, after
#         Lenia step. Adapter weights are separate from transformer_body (Tonic-safe).
# [2026-05-22] CC — Fix _read_river_delta() tract path: was reading elmer.tract (wrong inbox)
#   What: Changed tract_path from neurograph/elmer.tract to neurograph/proto_unibrain.tract.
#   Why:  Proto was consuming Elmer's NeuroGraph inbox, not its own. ~8G accumulated unread.
#         Tail-read (32KB) means no catch-up flood on fix.
#   How:  One-line path correction.
# [2026-05-13] CC — DecoderAdapter: identity-init bridge between evolved body and frozen decoder
#   What: Added DecoderAdapter (hidden_size×hidden_size linear, no bias, identity init).
#         Sits between transformer body output and frozen SignalDecoder. Governed by Lenia.
#   Why:  After ~38k Lenia steps the body hidden states drifted out-of-distribution for
#         the frozen decoder → all SubstrateSignal outputs saturated (anomaly=1 constant).
#         Adapter absorbs drift; decoder contract stays stable. Option B from design discussion.
#   How:  DecoderAdapter assigned to self._brain.decoder_adapter — PyTorch auto-registers
#         it in state_dict, so it persists in proto_weights.pt at zero extra save/load code.
#         Passed as decoder= to LeniaEngine so Rust governs its evolution.
# [2026-04-19] CC (punchlist #172) -- Remove low_cpu_mem_usage=True from from_pretrained
#   What: Removed low_cpu_mem_usage=True from AutoModelForCausalLM.from_pretrained()
#   Why:  Triggers accelerate init_empty_weights() meta-device path; strict=False
#         load_state_dict leaves some params on meta; after ~7k steps a hidden-state
#         traversal hits an unmaterialized param. Model is 1GB, VPS has 15GB -- no
#         memory justification for the flag.
#   How:  Deleted the flag. Full CPU load, same result, no meta-device risk.
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
_splat_adapter_module = None

# Persistence paths — proto brain survives restarts
_PROTO_WEIGHTS_PATH = os.path.expanduser("~/.elmer/proto_weights.pt")
_PROTO_LENIA_STATE_PATH = os.path.expanduser("~/.elmer/proto_lenia_state.pt")
_DELTA_LOG_PATH = os.path.expanduser("~/.elmer/competence_delta.jsonl")
_WEIGHT_STATS_LOG_PATH = os.path.expanduser("~/.elmer/weight_stats.jsonl")

# Adapter pressure monitoring (Morphogenesis-inspired CRISPR kick)
_ADAPTER_STUCK_THRESHOLD = 200     # consecutive saturated steps before kick
_ADAPTER_SAT_THRESHOLD = 0.75      # fraction of decoder signals pegged at 0/1 = stuck


def _lazy_imports():
    global _torch, _brain_module, _lenia_module, _encoder_module, _splat_adapter_module
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
    if _splat_adapter_module is None:
        try:
            import importlib.util as ilu3
            import sys as _sys3
            spec = ilu3.spec_from_file_location(
                "splat_adapter",
                os.path.expanduser("~/UniAI/lenia/splat_adapter.py"),
            )
            _splat_adapter_module = ilu3.module_from_spec(spec)
            _sys3.modules["splat_adapter"] = _splat_adapter_module
            spec.loader.exec_module(_splat_adapter_module)
        except Exception as e:
            logger.warning("SplatAdapter import failed (non-fatal): %s", e)
            _splat_adapter_module = None


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
        self._splat_adapter = None
        self._config = None
        self._lenia_steps = 0
        self._splat_step_count = 0
        self._ecosystem = ecosystem
        self._body_lock = None
        self._adapter_stuck_steps = 0
        self._adapter_kick_count = 0
        self._adapter_sat_frac = 0.0

    def set_body_lock(self, lock) -> None:
        """Set the shared body access lock (from BrainSwitcher)."""
        self._body_lock = lock

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
            import torch.nn as _nn
            from transformers import AutoModelForCausalLM

            class DecoderAdapter(_nn.Module):
                """Identity-init bridge between evolved body and frozen decoder.
                Absorbs hidden-state distribution drift as Lenia reshapes the body.
                Lenia-governed (registered as decoder= in LeniaEngine). Starts as
                pure passthrough; diverges only as the body drifts.
                """
                def __init__(self, hidden_size: int):
                    super().__init__()
                    self.proj = _nn.Linear(hidden_size, hidden_size, bias=False)
                    _nn.init.normal_(self.proj.weight, std=0.02)

                def forward(self, x):
                    return self.proj(x)

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

            # Decoder adapter — absorbs body↔decoder distribution drift.
            # Identity init: pure passthrough at start, Lenia governs evolution.
            # Assigning as nn.Module attribute auto-registers it in state_dict.
            self._brain.decoder_adapter = DecoderAdapter(self._config['hidden_size'])
            for param in self._brain.decoder_adapter.parameters():
                param.requires_grad = True
            logger.info(
                "ProtoUniBrain: DecoderAdapter created (%d×%d, N(0,0.02) init)",
                self._config['hidden_size'], self._config['hidden_size'],
            )

            # Attach Lenia dynamics engine — body + encoder (eyes) + decoder adapter.
            # The frozen SignalDecoder is NOT included — it's the stable output vocabulary.
            self._lenia = _lenia_module.LeniaEngine(
                self._brain.transformer_body,
                _lenia_module.LeniaConfig(
                    kernel_radius=5,
                    kernel_sigma=0.8,
                    growth_mu=0.12,
                    growth_sigma=0.02,
                    growth_scale=0.005,
                    max_weight_delta=0.05,
                    activation_coupling=0.5,
                    noise_scale=0.2,
                ),
                encoder=getattr(self._brain, 'encoder', None),
                decoder=self._brain.decoder_adapter,
            )
            self._lenia.register_hooks()

            # Attach SplatAdapter — myelin layer via Gaussian splat dynamics.
            # Uses the same ng_tract.LeniaEngine as the dense Lenia engine.
            # Phase 1: dynamics run, metrics logged. Contribution to forward pass in Phase 2.
            if _splat_adapter_module is not None:
                try:
                    rust_engine = self._lenia._rust_engine  # the ng_tract.LeniaEngine instance
                    splat_cfg = _splat_adapter_module.SplatAdapterConfig(
                        myelin_splats_per_layer=64,
                        init_sigma=0.15,
                        init_amp=0.05,
                    )
                    self._splat_adapter = _splat_adapter_module.SplatAdapter(splat_cfg, rust_engine)
                    # Register splat populations for transformer body layers
                    n_registered = 0
                    for name, param in self._brain.transformer_body.named_parameters():
                        if param.dim() == 2 and 'weight' in name:
                            rows, cols = param.shape
                            self._splat_adapter.register_layer(name, rows, cols)
                            n_registered += 1
                    # Register forward hooks — myelin contribution added on every forward pass
                    self._splat_adapter.register_forward_hooks(self._brain.transformer_body)
                    logger.info(
                        "ProtoUniBrain: SplatAdapter registered %d layers (%d myelin splats each)",
                        n_registered, splat_cfg.myelin_splats_per_layer
                    )
                except Exception as e:
                    logger.warning("SplatAdapter init failed (non-fatal): %s", e)
                    self._splat_adapter = None

            self._brain.eval()
            self._loaded = True

            trainable = sum(
                p.numel() for p in self._brain.parameters() if p.requires_grad
            )
            logger.info(
                "ProtoUniBrain loaded: %d trainable params, Lenia attached",
                trainable,
            )

            # Restore previously evolved weights (if any)
            self._restore_evolved_weights()

            # Detect and repair body collapse from prior Lenia failure (Phase 0 fixed points)
            self._check_and_reset_body()

            return True

        except Exception as exc:
            logger.error("ProtoUniBrain load failed: %s", exc)
            self._loaded = False
            return False

    def unload(self) -> None:
        if self._loaded and self._brain is not None:
            self.save_evolved_weights()
            self.log_weight_stats()
        if self._lenia:
            self._lenia.remove_hooks()
        if self._splat_adapter is not None:
            self._splat_adapter.remove_hooks()
        self._brain = None
        self._lenia = None
        self._splat_adapter = None
        self._config = None
        self._loaded = False
        gc.collect()
        logger.info("ProtoUniBrain unloaded")

    def process(self, snapshot: GraphSnapshot, context: dict) -> SocketOutput:
        """Read River, run forward pass, deposit raw hidden state back to River.

        Law 7: raw in, raw out. The model deposits its full hidden state
        to the River as binary. No decoder in the loop. No signal schema.
        The decoder bucket extracts signals separately.

        The loop closes: River -> encoder -> transformer -> River.
        """
        if not self._loaded or self._brain is None:
            raise RuntimeError("ProtoUniBrain not loaded")

        start = time.time()
        self._process_count += 1

        try:
            autonomic = context.get('autonomic_state', 'PARASYMPATHETIC')

            # Read latest topology delta from the River (BTF tract)
            latest_delta = self._read_river_delta()

            # Serialize body access with Tonic — same physical memory.
            # Both run in separate threads on the same tensor memory.
            # One lock ensures they take turns — no concurrent reads/writes.
            import contextlib
            _body_ctx = self._body_lock if self._body_lock is not None else contextlib.nullcontext()

            with _body_ctx:
                # Forward pass — get raw hidden state, NOT decoder output
                with _torch.no_grad():
                    _topo = latest_delta.get("topology") if isinstance(latest_delta, dict) else None
                    _outcomes = latest_delta.get("outcomes") if isinstance(latest_delta, dict) else None
                    inputs_embeds = self._brain.encoder(
                        topology_delta=_topo,
                        outcome_embeddings=_outcomes,
                        autonomic_state=autonomic,
                    )
                    body_output = self._brain.transformer_body(
                        input_ids=None,
                        inputs_embeds=inputs_embeds,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    # Raw hidden state — full sequence, no pooling, no decoder
                    raw_hidden = body_output.last_hidden_state  # (1, seq_len, 896)
                    # All layer hidden states for per-layer measurement
                    all_hidden = body_output.hidden_states  # tuple of (1, seq_len, 896) x 25

                # Lenia dynamics step — THE LEARNING
                # Must hold lock: writes to the same weight tensors Tonic reads
                lenia_metrics = self._lenia.step()
                self._lenia_steps += 1

            # Splat adapter step — runs OUTSIDE the body lock.
            # Splat arrays are separate from transformer weight tensors.
            # No shared memory with Tonic — no lock needed.
            if self._splat_adapter is not None and self._lenia_steps % 5 == 0:
                try:
                    # Feed per-layer activation magnitudes from this forward pass
                    if all_hidden and len(all_hidden) > 0:
                        layer_activations = {}
                        for name, _ in self._brain.transformer_body.named_parameters():
                            if 'weight' in name:
                                # Approximate activation for this layer from hidden norms
                                # Layer index from name: "layers.N.xxx"
                                parts = name.split('.')
                                layer_idx = None
                                for p in parts:
                                    try:
                                        layer_idx = int(p)
                                        break
                                    except ValueError:
                                        pass
                                if layer_idx is not None and layer_idx < len(all_hidden):
                                    lh = all_hidden[layer_idx].squeeze(0)
                                    layer_activations[name] = float(lh.norm() / max(lh.numel() ** 0.5, 1.0))
                        self._splat_adapter.update_activations(layer_activations)
                    splat_metrics = self._splat_adapter.step()
                    self._splat_step_count += 1
                    if self._splat_step_count % 10 == 0:
                        stats = self._splat_adapter.get_stats()
                        logger.debug(
                            "SplatAdapter step %d: myelin_active=%d drift=%.6f high_amp=%d",
                            self._splat_step_count,
                            splat_metrics.get('myelin_active', 0),
                            splat_metrics.get('total_drift', 0.0),
                            splat_metrics.get('high_amp_splats', 0),
                        )
                except Exception as e:
                    logger.warning("SplatAdapter step failed (non-fatal): %s", e)

            # Adapter pressure check — outside lock, reads/writes adapter weights only.
            # Measures identity deviation; applies CRISPR kick when stuck too long.
            self._check_adapter_pressure(raw_hidden)

            # Deposit raw hidden state to River — outside lock, no body access
            self._deposit_to_river(raw_hidden)

            # Periodic persistence
            if self._lenia_steps % 5 == 0 and self._lenia_steps > 0:
                self.save_evolved_weights()
                self.log_weight_stats()

            elapsed = time.time() - start
            self._total_latency += elapsed

            # Raw statistics from the hidden state. No scaling, no clamping.
            # The brain reports what it produced. Buckets scale to their needs.
            h = raw_hidden.squeeze(0)  # (seq_len, 896)
            h_mean = float(h.mean())
            h_std = float(h.std())
            h_norm = float(h.norm())
            pos_norms = h.norm(dim=-1)  # (seq_len,)
            pos_mean = float(pos_norms.mean())
            pos_std = float(pos_norms.std())
            pos_probs = _torch.softmax(pos_norms, dim=0)
            pos_entropy = float(-(_torch.log(pos_probs + 1e-12) * pos_probs).sum())
            near_zero = float((h.abs() < 0.01).float().mean())

            # Per-layer statistics — finer grained than whole-tensor aggregate
            layer_stats = []
            if all_hidden:
                for i, lh in enumerate(all_hidden):
                    lh_sq = lh.squeeze(0)  # (seq_len, 896)
                    layer_stats.append({
                        "layer": i,
                        "mean": round(float(lh_sq.mean()), 6),
                        "std": round(float(lh_sq.std()), 6),
                        "norm": round(float(lh_sq.norm()), 4),
                    })

            # Summary: how much are layers diverging from each other?
            if len(layer_stats) > 1:
                means = [ls["mean"] for ls in layer_stats]
                stds = [ls["std"] for ls in layer_stats]
                norms = [ls["norm"] for ls in layer_stats]
                layer_mean_spread = round(max(means) - min(means), 6)
                layer_std_spread = round(max(stds) - min(stds), 6)
                layer_norm_spread = round(max(norms) - min(norms), 4)
            else:
                layer_mean_spread = 0.0
                layer_std_spread = 0.0
                layer_norm_spread = 0.0

            raw_stats = {
                "h_mean": round(h_mean, 6),
                "h_std": round(h_std, 6),
                "h_norm": round(h_norm, 4),
                "pos_entropy": round(pos_entropy, 4),
                "near_zero_frac": round(near_zero, 6),
                "layer_mean_spread": layer_mean_spread,
                "layer_std_spread": layer_std_spread,
                "layer_norm_spread": layer_norm_spread,
                "n_layers": len(layer_stats),
            }

            # Normalize signal fields to (0, 1) at the boundary.
            # Raw values preserved in metadata["raw"] for bucket consumers.
            # tanh(x / scale): healthy state maps to ~0.3-0.7, saturates
            # gracefully under explosion rather than emitting garbage values.
            import math as _math
            def _sig(x, scale): return float(_math.tanh(abs(x) / max(scale, 1e-9)))

            return SocketOutput(
                signal=SubstrateSignal.create(
                    signal_type="coherence",
                    description="ProtoUniBrain living deposit",
                    coherence_score=_sig(pos_entropy, 4.0),   # entropy of pos dist
                    health_score=_sig(h_std, 2.0),             # hidden state spread
                    anomaly_level=_sig(h_mean, 1.0),           # mean deviation from 0
                    novelty=_sig(layer_std_spread, 1.0),        # layer differentiation
                    confidence=_sig(h_norm, 50.0),              # output magnitude
                    severity=float(near_zero),                  # already [0,1]
                    identity_coherence=_sig(pos_mean, 30.0),    # per-position norm
                    pruning_pressure=float(near_zero),          # already [0,1]
                    topology_health=_sig(layer_norm_spread, 10.0),  # layer norm spread
                    metadata={
                        "socket": "elmer:proto_unibrain",
                        "inference_time_ms": elapsed * 1000,
                        "lenia_step": self._lenia_steps,
                        "lenia_delta_norm": lenia_metrics['total_delta_norm'],
                        "lenia_time_ms": lenia_metrics.get('time_ms', 0),
                        "adapter_sat_frac": round(self._adapter_sat_frac, 4),
                        "adapter_stuck_steps": self._adapter_stuck_steps,
                        "adapter_kick_count": self._adapter_kick_count,
                        "model": "proto_unibrain",
                        "raw": raw_stats,
                    },
                ),
                confidence=_sig(h_norm, 50.0),
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

    def _deposit_to_river(self, hidden_state):
        """Deposit raw hidden state to the River as BTF experience entry.

        Full sequence tensor, no pooling, no reduction. Law 7.
        The decoder bucket and any other consumer reads this.
        """
        try:
            import ng_tract
            # (1, seq_len, hidden_size) -> (seq_len, hidden_size) -> raw bytes
            raw = hidden_state.squeeze(0).detach().cpu().numpy()
            content_bytes = raw.tobytes()

            # Deposit to latent tract
            tract_dir = os.path.expanduser("~/.et_modules/tracts/proto_unibrain")
            os.makedirs(tract_dir, exist_ok=True)
            tract_path = os.path.join(tract_dir, "latent.tract")

            ng_tract.deposit_experience(
                source="proto_unibrain",
                content_type="latent_state",
                content=content_bytes,
                tract_paths=[tract_path],
            )
        except Exception as exc:
            logger.debug("River deposit failed: %s", exc)

    # -----------------------------------------------------------------
    # Adapter pressure monitoring (Morphogenesis-inspired)
    # -----------------------------------------------------------------

    def _check_adapter_pressure(self, raw_hidden) -> None:
        """Detect and break DecoderAdapter output saturation.

        Runs a quick probe: raw_hidden → adapter → decoder → check what
        fraction of the 9 signal outputs are pegged at 0 or 1 (Sigmoid
        saturation). If >= _ADAPTER_SAT_THRESHOLD of signals are saturated
        for _ADAPTER_STUCK_THRESHOLD consecutive steps, apply a CRISPR kick:
        re-initialize the adapter to N(0, 0.02) so Lenia starts from a
        realistic scale rather than an identity or zero attractor.

        Measures the real problem (decoder sees constant bias-dominated input)
        rather than a proxy (identity deviation), which was the v1 failure mode.
        """
        adapter = getattr(self._brain, 'decoder_adapter', None)
        if adapter is None:
            return
        try:
            with _torch.no_grad():
                adapted = adapter(raw_hidden)
                probe = self._brain.decoder(adapted)
            signals = probe['signals']  # (1, 9), Sigmoid → [0, 1]
            sat_frac = float(((signals > 0.98) | (signals < 0.02)).float().mean())
            self._adapter_sat_frac = sat_frac

            if sat_frac >= _ADAPTER_SAT_THRESHOLD:
                self._adapter_stuck_steps += 1
            else:
                self._adapter_stuck_steps = max(0, self._adapter_stuck_steps - 5)

            if self._adapter_stuck_steps >= _ADAPTER_STUCK_THRESHOLD:
                w = adapter.proj.weight
                sat_before = sat_frac
                self._apply_crispr_kick(w, w.shape[0])
                self._adapter_stuck_steps = 0
                self._adapter_kick_count += 1
                self._log_kick_event(sat_before)
        except Exception as exc:
            logger.debug("Adapter pressure check failed: %s", exc)

    def _apply_crispr_kick(self, weight: 'torch.Tensor', dim: int) -> None:
        """Re-initialize adapter to N(0, 0.02) — same scale as transformer body.

        Full re-init rather than additive noise. This avoids Lenia re-collapsing
        a nudged-but-still-degenerate weight — we want to land in the same
        distribution the body occupies so Lenia's mass conservation law treats
        the adapter as a peer, not an outlier to shrink.
        """
        with _torch.no_grad():
            _torch.nn.init.normal_(weight, std=0.02)
        logger.info(
            "ProtoUniBrain: CRISPR kick #%d — adapter re-init to N(0,0.02) "
            "(sat_frac_before=%.4f, stuck_steps_reset)",
            self._adapter_kick_count + 1,
            self._adapter_sat_frac,
        )

    def _log_kick_event(self, sat_before: float) -> None:
        """Append CRISPR kick event to competence_delta.jsonl."""
        os.makedirs(os.path.dirname(_DELTA_LOG_PATH), exist_ok=True)
        try:
            from datetime import datetime
            import json
            entry = {
                "timestamp": datetime.now().isoformat(),
                "event": "crispr_kick",
                "lenia_step": self._lenia_steps,
                "kick_number": self._adapter_kick_count + 1,
                "sat_frac_before": round(sat_before, 4),
            }
            with open(_DELTA_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("CRISPR kick log failed: %s", exc)

    def set_ecosystem_ref(self, ecosystem):
        """Set reference to Elmer's own ecosystem. Called by engine after init."""
        self._ecosystem = ecosystem

    # -----------------------------------------------------------------
    # Weight persistence — proto brain survives fan-out cycles
    # -----------------------------------------------------------------

    def _reset_body_to_pretrained(self) -> None:
        """Reset all transformer body weights to the pretrained Qwen2.5-0.5B checkpoint.

        Called when body collapse is detected (all/most weight matrices std≈0).
        Lenia baselines are cleared so Phase 1 measures from the healthy pretrained values.
        """
        from safetensors.torch import load_file
        pretrained_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/"
            "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors"
        )
        if not os.path.exists(pretrained_path):
            logger.error("ProtoUniBrain: pretrained checkpoint not found at %s — cannot reset", pretrained_path)
            return

        pretrained = load_file(pretrained_path)
        restored = 0
        for name, param in self._brain.transformer_body.named_parameters():
            hf_key = f"model.{name}"
            if hf_key in pretrained:
                param.data.copy_(pretrained[hf_key].to(param.dtype))
                restored += 1
        for name, buf in self._brain.transformer_body.named_buffers():
            hf_key = f"model.{name}"
            if hf_key in pretrained:
                buf.data.copy_(pretrained[hf_key].to(buf.dtype))

        # Clear Lenia baselines so they're re-measured from healthy pretrained values
        self._lenia._initial_norms.clear()
        if hasattr(self._lenia, '_initial_mean_abs'):
            self._lenia._initial_mean_abs.clear()

        logger.info(
            "ProtoUniBrain: body reset to pretrained (%d params restored). Lenia baselines cleared.",
            restored,
        )

    def _check_and_reset_body(self) -> None:
        """Detect body weight collapse and reset to pretrained if >50% of matrices have std≈0."""
        collapsed = sum(
            1 for _, p in self._brain.transformer_body.named_parameters()
            if p.dim() >= 2 and p.requires_grad and p.float().std().item() < 1e-6
        )
        total = sum(
            1 for _, p in self._brain.transformer_body.named_parameters()
            if p.dim() >= 2 and p.requires_grad
        )
        if total > 0 and collapsed / total > 0.5:
            logger.warning(
                "ProtoUniBrain: body collapse detected (%d/%d matrices std≈0). Resetting to pretrained.",
                collapsed, total,
            )
            self._reset_body_to_pretrained()

    def _restore_evolved_weights(self) -> bool:
        """Restore previously evolved weights + Lenia state from disk.

        Returns True if restored, False if starting fresh.
        """
        os.makedirs(os.path.dirname(_PROTO_WEIGHTS_PATH), exist_ok=True)

        if not os.path.exists(_PROTO_WEIGHTS_PATH):
            logger.info("ProtoUniBrain: no persisted weights — starting fresh")
            return False

        try:
            saved = _torch.load(_PROTO_WEIGHTS_PATH, map_location="cpu", weights_only=False)
            self._brain.load_state_dict(saved['model_state_dict'], strict=False)
            self._lenia_steps = saved.get('lenia_steps', 0)
            logger.info(
                "ProtoUniBrain: restored evolved weights (step %d, saved %s)",
                self._lenia_steps,
                saved.get('timestamp', 'unknown'),
            )

            # Restore Lenia engine state (baselines for Phase 1 activation-coupled dynamics)
            if os.path.exists(_PROTO_LENIA_STATE_PATH) and self._lenia:
                lenia_state = _torch.load(_PROTO_LENIA_STATE_PATH, map_location="cpu", weights_only=False)
                self._lenia._initial_norms = lenia_state.get('initial_norms', {})
                self._lenia._initial_mean_abs = lenia_state.get('initial_mean_abs', {})
                self._lenia.state.step_count = lenia_state.get('step_count', 0)
                logger.info("ProtoUniBrain: restored Lenia state (baselines for %d params)",
                            len(self._lenia._initial_norms))

            return True
        except Exception as exc:
            logger.warning("ProtoUniBrain: weight restore failed (%s) — starting fresh", exc)
            return False

    def save_evolved_weights(self) -> None:
        """Persist current evolved weights + Lenia state to disk.

        Called periodically during process() and on unload().
        """
        os.makedirs(os.path.dirname(_PROTO_WEIGHTS_PATH), exist_ok=True)

        try:
            from datetime import datetime
            _torch.save({
                'model_state_dict': self._brain.state_dict(),
                'lenia_steps': self._lenia_steps,
                'timestamp': datetime.now().isoformat(),
                'process_count': self._process_count,
            }, _PROTO_WEIGHTS_PATH)

            # Save Lenia state (baselines for Phase 1 activation-coupled dynamics)
            if self._lenia:
                _torch.save({
                    'initial_norms': self._lenia._initial_norms,
                    'initial_mean_abs': self._lenia._initial_mean_abs,
                    'step_count': self._lenia.state.step_count,
                    'weight_deltas': {k: v for k, v in self._lenia.state.weight_deltas.items()},
                }, _PROTO_LENIA_STATE_PATH)

            logger.info("ProtoUniBrain: weights saved (step %d)", self._lenia_steps)
        except Exception as exc:
            logger.error("ProtoUniBrain: weight save failed: %s", exc)

    def log_weight_stats(self) -> None:
        """Append current weight distribution stats to the log."""
        os.makedirs(os.path.dirname(_WEIGHT_STATS_LOG_PATH), exist_ok=True)

        try:
            from datetime import datetime
            import json

            stats = {
                'timestamp': datetime.now().isoformat(),
                'lenia_step': self._lenia_steps,
                'process_count': self._process_count,
                'layers': {},
            }

            total_near_zero = 0
            total_params = 0
            for name, param in self._brain.named_parameters():
                if param.requires_grad and param.dim() >= 2:
                    near_zero = (param.abs() < 0.01).sum().item()
                    total = param.numel()
                    total_near_zero += near_zero
                    total_params += total
                    stats['layers'][name] = {
                        'mean': round(param.mean().item(), 8),
                        'std': round(param.std().item(), 6),
                        'sparsity': round(near_zero / total, 4),
                    }

            stats['global_sparsity'] = round(total_near_zero / max(total_params, 1), 4)

            with open(_WEIGHT_STATS_LOG_PATH, 'a') as f:
                f.write(json.dumps(stats) + '\n')

        except Exception as exc:
            logger.debug("ProtoUniBrain: weight stats log failed: %s", exc)

    def _read_river_delta(self):
        """Read new substrate entries from the River (BTF tract).

        Handles both entry types deposited by NeuroGraph:
        - ENTRY_OUTCOME (type=1): 768-dim embedding → outcome_embeddings
        - ENTRY_TOPOLOGY (type=2): msgpack compact topology delta → topology_delta dict

        Uses a cursor to read only entries deposited since last call.
        Returns dict with keys 'topology' (dict|None) and 'outcomes' (List[np.ndarray]|None).

        # ---- Changelog ----
        # [2026-05-25] CC — Fix #251: handle ENTRY_TOPOLOGY entries (type=2) which NeuroGraph
        #   now deposits after the River drain fix (#249 other CC). Topology entries carry
        #   msgpack-encoded compact delta (fired_node_ids, synapses_pruned/sprouted,
        #   predictions_confirmed/surprised). Normalized to encoder's dict format at this
        #   boundary (LAW 7: extraction happens at consumption). Returns dict with both
        #   'topology' and 'outcomes' keys so caller can pass each to encoder correctly.
        # [2026-05-25] CC — Fix #249: magic bytes b"BT"→b"TB", ENTRY_OUTCOME filter,
        #   cursor-based reading, logger.debug→logger.error.
        # -------------------
        """
        try:
            import json as _json
            import msgpack as _msgpack
            import ng_tract
            tract_path = os.path.expanduser(
                "~/.et_modules/tracts/neurograph/proto_unibrain.tract"
            )
            if not os.path.exists(tract_path):
                return None

            file_size = os.path.getsize(tract_path)
            if file_size == 0:
                return None

            # Cursor tracks bytes already consumed — same JSON format as bridge cursors
            cursor_path = os.path.expanduser(
                "~/.et_modules/tracts/neurograph/proto_unibrain.cursor"
            )
            offset = 0
            if os.path.exists(cursor_path):
                try:
                    with open(cursor_path) as _cf:
                        _cd = _json.load(_cf)
                    offset = int(_cd.get("offset", 0))
                except Exception:
                    offset = 0

            if offset >= file_size:
                return None  # nothing new

            # Read bytes deposited since last cursor position
            with open(tract_path, "rb") as f:
                f.seek(offset)
                data = f.read()

            if not data:
                return None

            # If reading mid-file, scan forward to first valid BTF entry (magic: TB = 0x54, 0x42)
            if offset > 0:
                magic_pos = data.find(b"TB")
                if magic_pos > 0:
                    data = data[magic_pos:]

            reader = ng_tract.TractReader(data)
            embeddings = []
            last_topo_dict = None
            topo_count = 0
            outcome_count = 0

            for entry in reader:
                if not hasattr(entry, "entry_type"):
                    continue
                if entry.entry_type == ng_tract.ENTRY_OUTCOME:
                    if hasattr(entry, "embedding_as_numpy"):
                        emb = entry.embedding_as_numpy()
                        if emb is not None and len(emb) > 0:
                            embeddings.append(emb)
                            outcome_count += 1
                elif entry.entry_type == ng_tract.ENTRY_TOPOLOGY:
                    # Decode msgpack and normalize to encoder's dict format.
                    # Topology entries carry compact delta: fired_node_ids, synapses_pruned/sprouted,
                    # predictions_confirmed/surprised. Normalize at this boundary so the encoder's
                    # existing dict path (_read_from_delta, _read_global) works unchanged.
                    try:
                        raw = entry.raw()
                        raw_dict = _msgpack.unpackb(raw, raw=False)
                        last_topo_dict = {
                            "fired_nodes": [{"node_id": nid} for nid in raw_dict.get("fired_node_ids", [])],
                            "fired_hyperedges": [{"he_id": hid} for hid in raw_dict.get("fired_hyperedge_ids", [])],
                            "predictions": {
                                "confirmed": raw_dict.get("predictions_confirmed", 0),
                                "surprised": raw_dict.get("predictions_surprised", 0),
                            },
                            "structural": {
                                "synapses_pruned": raw_dict.get("synapses_pruned", 0),
                                "synapses_sprouted": raw_dict.get("synapses_sprouted", 0),
                            },
                            "timestep": raw_dict.get("timestep", 0),
                            "salience": [],
                        }
                        topo_count += 1
                    except Exception:
                        pass

            # Advance cursor to current file end
            total = outcome_count + topo_count
            try:
                with open(cursor_path, "w") as _cf:
                    _json.dump({"offset": file_size, "ts": time.time(), "entries": total}, _cf)
            except Exception:
                pass

            result = {
                "topology": last_topo_dict,
                "outcomes": embeddings if embeddings else None,
            }
            return result if (last_topo_dict is not None or embeddings) else None

        except Exception as exc:
            logger.error("River read failed: %s", exc)
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

