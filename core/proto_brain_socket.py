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

# Persistence paths — proto brain survives restarts
_PROTO_WEIGHTS_PATH = os.path.expanduser("~/.elmer/proto_weights.pt")
_PROTO_LENIA_STATE_PATH = os.path.expanduser("~/.elmer/proto_lenia_state.pt")
_DELTA_LOG_PATH = os.path.expanduser("~/.elmer/competence_delta.jsonl")
_WEIGHT_STATS_LOG_PATH = os.path.expanduser("~/.elmer/weight_stats.jsonl")


def _lazy_imports():
    global _torch, _brain_module, _lenia_module, _encoder_module
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
        self._body_lock = None

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

            # Attach Lenia dynamics engine — operates on EVERYTHING:
            # body (transformer) + encoder (the eyes).
            # Decoder is decoupled — it's a River bucket, not a model layer.
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
                ),
                encoder=getattr(self._brain, 'encoder', None),
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

            # Restore previously evolved weights (if any)
            self._restore_evolved_weights()

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
        self._brain = None
        self._lenia = None
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
                    inputs_embeds = self._brain.encoder(
                        topology_delta=latest_delta,
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

    def set_ecosystem_ref(self, ecosystem):
        """Set reference to Elmer's own ecosystem. Called by engine after init."""
        self._ecosystem = ecosystem

    # -----------------------------------------------------------------
    # Weight persistence — proto brain survives fan-out cycles
    # -----------------------------------------------------------------

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

            # Restore Lenia engine state (initial norms for mass conservation)
            if os.path.exists(_PROTO_LENIA_STATE_PATH) and self._lenia:
                lenia_state = _torch.load(_PROTO_LENIA_STATE_PATH, map_location="cpu", weights_only=False)
                self._lenia._initial_norms = lenia_state.get('initial_norms', {})
                self._lenia.state.step_count = lenia_state.get('step_count', 0)
                logger.info("ProtoUniBrain: restored Lenia state (norms for %d params)",
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

            # Save Lenia state separately (initial norms critical for mass conservation)
            if self._lenia:
                _torch.save({
                    'initial_norms': self._lenia._initial_norms,
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
            for name, param in self._brain.parameters():
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
        """Read the latest topology delta from the River (BTF tract).

        Reads the tail of the BTF tract file deposited by NeuroGraph.
        Returns a PyTopologyEntry for the encoder's BTF fast path,
        or None if no tract data available.

        The River IS the input. No bridges, no _peer_events.
        """
        try:
            import ng_tract
            tract_path = os.path.expanduser(
                "~/.et_modules/tracts/neurograph/elmer.tract"
            )
            if not os.path.exists(tract_path):
                return None

            # Read the last chunk of the tract — recent entries only.
            # The tract grows continuously; we only need the latest delta.
            chunk_size = 32768  # 32KB — enough for several topology entries
            file_size = os.path.getsize(tract_path)
            if file_size == 0:
                return None

            offset = max(0, file_size - chunk_size)
            with open(tract_path, "rb") as f:
                if offset > 0:
                    f.seek(offset)
                data = f.read()

            # If we seeked into the middle, find the first valid entry
            # by scanning for the BTF magic bytes (0x42, 0x54)
            if offset > 0:
                magic_pos = data.find(b"BT")
                if magic_pos >= 0:
                    data = data[magic_pos:]
                else:
                    # No valid entry in tail — read more
                    with open(tract_path, "rb") as f:
                        data = f.read()

            reader = ng_tract.TractReader(data)
            last_topo = None
            for entry in reader:
                if entry.entry_type == 2:  # topology
                    last_topo = entry
            return last_topo

        except Exception as exc:
            logger.debug("River read failed: %s", exc)
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

