"""
Training Loop — Teach the New Eyes and New Voice

Generates diverse substrate states, computes ground-truth signals from
the existing heuristic pipeline, and trains the I/O layers while keeping
the transformer body frozen (or optionally fine-tuning with tiny LR).

Training strategy:
  Phase 1: Freeze body, train I/O layers only (fast, 1.1M params)
  Phase 2: Unfreeze body with 100x smaller LR (optional, careful)

# ---- Changelog ----
# [2026-03-20] Claude Code (Opus 4.6) — Initial training implementation
#   What: Training loop for ElmerBrain I/O layers
#   Why:  PRD §10 — trained on entity's lived experience
#   How:  Generate diverse substrate snapshots, compute heuristic targets,
#         MSE loss on signal fields, cross-entropy on action classification
# -------------------
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional

from graph_io import GraphFeatures, SIGNAL_FIELDS, ACTION_TYPES
from operate import ElmerBrain, perform_surgery


# ---------------------------------------------------------------------------
# Synthetic substrate state generation
# ---------------------------------------------------------------------------

def generate_substrate_scenario(scenario: str = 'random') -> Tuple[GraphFeatures, Dict[str, float], int]:
    """Generate a substrate state and its expected signal values.

    Returns:
        (features, target_signals, target_action_idx)
    """
    if scenario == 'healthy':
        return _scenario_healthy()
    elif scenario == 'degraded':
        return _scenario_degraded()
    elif scenario == 'critical':
        return _scenario_critical()
    elif scenario == 'novel_flood':
        return _scenario_novel_flood()
    elif scenario == 'identity_drift':
        return _scenario_identity_drift()
    elif scenario == 'near_capacity':
        return _scenario_near_capacity()
    elif scenario == 'cold_start':
        return _scenario_cold_start()
    else:
        # Random mix
        scenarios = ['healthy', 'degraded', 'critical', 'novel_flood',
                     'identity_drift', 'near_capacity', 'cold_start']
        weights = [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
        chosen = np.random.choice(scenarios, p=weights)
        return generate_substrate_scenario(chosen)


def _base_features(n_nodes=500, n_synapses=300, density=0.02,
                   clustering=0.3, health_mult=1.0, noise=0.05) -> GraphFeatures:
    """Generate base features with controlled parameters."""
    voltages = torch.clamp(torch.randn(n_nodes) * 0.2 * health_mult + 0.5, 0, 1)
    firing_rates = torch.clamp(torch.rand(n_nodes) * 0.3 * health_mult, 0, 1)
    excitability = torch.clamp(torch.ones(n_nodes) * 0.8 + torch.randn(n_nodes) * noise, 0, 1)

    weights = torch.clamp(torch.rand(n_synapses) * 0.6 + 0.2, 0, 1)
    ages = torch.randint(0, 5000, (max(n_synapses, 1),)).float()

    recent_firings = torch.clamp(torch.randn(15) * 5 * health_mult + 10, 0, 100)
    stdp_delta = torch.tensor([np.random.normal(0.001 * health_mult, 0.0005)])

    identity_emb = torch.randn(384) * 0.1

    return GraphFeatures(
        node_voltages=voltages,
        node_firing_rates=firing_rates,
        node_excitability=excitability,
        synapse_weights=weights,
        synapse_ages=ages,
        density=torch.tensor([density]),
        clustering=torch.tensor([clustering]),
        n_components=torch.tensor([max(1.0, 5.0 - clustering * 10)]),
        n_nodes=torch.tensor([float(n_nodes)]),
        n_synapses=torch.tensor([float(n_synapses)]),
        n_hyperedges=torch.tensor([0.0]),
        recent_firings=recent_firings,
        stdp_delta_mean=stdp_delta,
        identity_embedding=identity_emb,
    )


def _scenario_healthy():
    features = _base_features(
        n_nodes=800, n_synapses=600, density=0.04,
        clustering=0.45, health_mult=1.0
    )
    targets = {
        'coherence_score': 0.75 + np.random.normal(0, 0.05),
        'health_score': 0.85 + np.random.normal(0, 0.05),
        'anomaly_level': 0.05 + np.random.normal(0, 0.02),
        'novelty': 0.15 + np.random.normal(0, 0.05),
        'confidence': 0.80 + np.random.normal(0, 0.05),
        'severity': 0.0,
        'identity_coherence': 0.90 + np.random.normal(0, 0.03),
        'pruning_pressure': 0.20 + np.random.normal(0, 0.05),
        'topology_health': 0.80 + np.random.normal(0, 0.05),
    }
    action = 0  # none
    return features, _clip_targets(targets), action


def _scenario_degraded():
    features = _base_features(
        n_nodes=600, n_synapses=200, density=0.015,
        clustering=0.20, health_mult=0.6
    )
    targets = {
        'coherence_score': 0.45 + np.random.normal(0, 0.05),
        'health_score': 0.50 + np.random.normal(0, 0.05),
        'anomaly_level': 0.35 + np.random.normal(0, 0.05),
        'novelty': 0.30 + np.random.normal(0, 0.05),
        'confidence': 0.55 + np.random.normal(0, 0.05),
        'severity': 0.30 + np.random.normal(0, 0.05),
        'identity_coherence': 0.65 + np.random.normal(0, 0.05),
        'pruning_pressure': 0.40 + np.random.normal(0, 0.05),
        'topology_health': 0.45 + np.random.normal(0, 0.05),
    }
    action = 6  # elevate_monitoring
    return features, _clip_targets(targets), action


def _scenario_critical():
    features = _base_features(
        n_nodes=300, n_synapses=50, density=0.005,
        clustering=0.05, health_mult=0.2
    )
    targets = {
        'coherence_score': 0.15 + np.random.normal(0, 0.03),
        'health_score': 0.15 + np.random.normal(0, 0.05),
        'anomaly_level': 0.80 + np.random.normal(0, 0.05),
        'novelty': 0.60 + np.random.normal(0, 0.05),
        'confidence': 0.30 + np.random.normal(0, 0.05),
        'severity': 0.85 + np.random.normal(0, 0.05),
        'identity_coherence': 0.25 + np.random.normal(0, 0.05),
        'pruning_pressure': 0.10 + np.random.normal(0, 0.05),
        'topology_health': 0.10 + np.random.normal(0, 0.05),
    }
    action = 2  # flag_anomaly
    return features, _clip_targets(targets), action


def _scenario_novel_flood():
    features = _base_features(
        n_nodes=200, n_synapses=400, density=0.03,
        clustering=0.15, health_mult=0.8
    )
    # High novelty, everything else moderate
    targets = {
        'coherence_score': 0.55 + np.random.normal(0, 0.05),
        'health_score': 0.65 + np.random.normal(0, 0.05),
        'anomaly_level': 0.25 + np.random.normal(0, 0.05),
        'novelty': 0.85 + np.random.normal(0, 0.05),
        'confidence': 0.40 + np.random.normal(0, 0.05),
        'severity': 0.10 + np.random.normal(0, 0.03),
        'identity_coherence': 0.70 + np.random.normal(0, 0.05),
        'pruning_pressure': 0.60 + np.random.normal(0, 0.05),
        'topology_health': 0.50 + np.random.normal(0, 0.05),
    }
    action = 1  # record_observation
    return features, _clip_targets(targets), action


def _scenario_identity_drift():
    features = _base_features(
        n_nodes=700, n_synapses=500, density=0.03,
        clustering=0.35, health_mult=0.9
    )
    # Identity is specifically low
    features.identity_embedding = torch.randn(384) * 0.5  # noisy identity
    targets = {
        'coherence_score': 0.60 + np.random.normal(0, 0.05),
        'health_score': 0.70 + np.random.normal(0, 0.05),
        'anomaly_level': 0.40 + np.random.normal(0, 0.05),
        'novelty': 0.25 + np.random.normal(0, 0.05),
        'confidence': 0.65 + np.random.normal(0, 0.05),
        'severity': 0.45 + np.random.normal(0, 0.05),
        'identity_coherence': 0.25 + np.random.normal(0, 0.05),
        'pruning_pressure': 0.30 + np.random.normal(0, 0.05),
        'topology_health': 0.60 + np.random.normal(0, 0.05),
    }
    action = 3  # flag_identity_drift
    return features, _clip_targets(targets), action


def _scenario_near_capacity():
    features = _base_features(
        n_nodes=950, n_synapses=4500, density=0.06,
        clustering=0.50, health_mult=1.0
    )
    targets = {
        'coherence_score': 0.70 + np.random.normal(0, 0.05),
        'health_score': 0.60 + np.random.normal(0, 0.05),
        'anomaly_level': 0.15 + np.random.normal(0, 0.03),
        'novelty': 0.05 + np.random.normal(0, 0.02),
        'confidence': 0.85 + np.random.normal(0, 0.03),
        'severity': 0.20 + np.random.normal(0, 0.05),
        'identity_coherence': 0.85 + np.random.normal(0, 0.03),
        'pruning_pressure': 0.90 + np.random.normal(0, 0.03),
        'topology_health': 0.70 + np.random.normal(0, 0.05),
    }
    action = 4  # recommend_pruning
    return features, _clip_targets(targets), action


def _scenario_cold_start():
    features = _base_features(
        n_nodes=5, n_synapses=2, density=0.001,
        clustering=0.0, health_mult=0.3
    )
    targets = {
        'coherence_score': 0.50,  # neutral on cold start
        'health_score': 0.50,
        'anomaly_level': 0.10,
        'novelty': 0.95,  # everything is novel
        'confidence': 0.05,  # very low confidence
        'severity': 0.0,
        'identity_coherence': 0.50,  # no baseline to compare
        'pruning_pressure': 0.0,
        'topology_health': 0.0,
    }
    action = 5  # recommend_checkpoint (save the first state)
    return features, _clip_targets(targets), action


def _clip_targets(targets: Dict[str, float]) -> Dict[str, float]:
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in targets.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_elmer_brain(
    brain: ElmerBrain,
    n_epochs: int = 30,
    samples_per_epoch: int = 64,
    lr_io: float = 1e-3,
    lr_body: float = 0.0,  # 0 = freeze body (Phase 1)
    save_path: str = 'surgery/elmer_brain_trained.pt',
    log_every: int = 5,
) -> Dict[str, list]:
    """Train the ElmerBrain I/O layers.

    Args:
        brain: Surgically modified model from operate.py.
        n_epochs: Training epochs.
        samples_per_epoch: Synthetic samples generated per epoch.
        lr_io: Learning rate for I/O layers (encoder + decoder).
        lr_body: Learning rate for transformer body (0 = frozen).
        save_path: Where to save the trained model.
        log_every: Print loss every N epochs.

    Returns:
        Training history dict with loss curves.
    """
    device = next(brain.parameters()).device

    # Separate parameter groups
    io_params = list(brain.encoder.parameters()) + list(brain.decoder.parameters())
    body_params = list(brain.body.parameters())

    param_groups = [{'params': io_params, 'lr': lr_io}]
    if lr_body > 0:
        param_groups.append({'params': body_params, 'lr': lr_body})
    else:
        # Freeze body
        for p in body_params:
            p.requires_grad = False
        trainable = sum(p.numel() for p in io_params if p.requires_grad)
        print(f"Phase 1: Body frozen. Training {trainable:,} I/O params only.")

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    signal_loss_fn = nn.MSELoss()
    action_loss_fn = nn.CrossEntropyLoss()

    history = {
        'epoch': [],
        'total_loss': [],
        'signal_loss': [],
        'action_loss': [],
    }

    print(f"\nTraining for {n_epochs} epochs, {samples_per_epoch} samples/epoch")
    print(f"I/O LR: {lr_io}, Body LR: {lr_body}")
    t_start = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_signal_loss = 0.0
        epoch_action_loss = 0.0

        brain.train()

        for _ in range(samples_per_epoch):
            features, targets, action_idx = generate_substrate_scenario('random')

            # Forward pass
            output = brain(features)

            # Signal loss: MSE between predicted and target field values
            target_tensor = torch.tensor(
                [targets[f] for f in SIGNAL_FIELDS],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            signal_loss = signal_loss_fn(output['raw_signal_tensor'], target_tensor)

            # Action loss: cross-entropy on action classification
            action_target = torch.tensor([action_idx], dtype=torch.long, device=device)
            action_loss = action_loss_fn(output['raw_action_logits'], action_target)

            # Combined loss (signal is more important)
            loss = signal_loss * 2.0 + action_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(io_params, max_norm=1.0)

            optimizer.step()

            epoch_signal_loss += signal_loss.item()
            epoch_action_loss += action_loss.item()

        scheduler.step()

        avg_signal = epoch_signal_loss / samples_per_epoch
        avg_action = epoch_action_loss / samples_per_epoch
        avg_total = avg_signal * 2.0 + avg_action

        history['epoch'].append(epoch)
        history['total_loss'].append(avg_total)
        history['signal_loss'].append(avg_signal)
        history['action_loss'].append(avg_action)

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"signal={avg_signal:.4f}  action={avg_action:.4f}  "
                  f"total={avg_total:.4f}  [{elapsed:.0f}s]")

    # Save trained model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    save_dict = {
        'encoder_state': brain.encoder.state_dict(),
        'decoder_state': brain.decoder.state_dict(),
        'config': {
            'hidden_dim': brain.encoder.hidden_dim,
            'n_positions': brain.encoder.n_positions,
            'n_signal_fields': len(SIGNAL_FIELDS),
            'n_action_types': len(ACTION_TYPES),
            'base_model': 'Qwen/Qwen2.5-0.5B',
        },
        'training': {
            'n_epochs': n_epochs,
            'samples_per_epoch': samples_per_epoch,
            'final_signal_loss': history['signal_loss'][-1],
            'final_action_loss': history['action_loss'][-1],
            'timestamp': time.time(),
        },
    }

    # Optionally save body weights too (large!)
    # For now, just save I/O — body can be reloaded from HuggingFace
    torch.save(save_dict, save_path)
    print(f"\nSaved I/O weights to {save_path}")
    print(f"  File size: {os.path.getsize(save_path) / 1024:.1f} KB")

    return history


def evaluate(brain: ElmerBrain, n_samples: int = 20):
    """Evaluate the trained brain on each scenario type."""
    brain.eval()
    print("\n=== EVALUATION ===\n")

    scenarios = ['healthy', 'degraded', 'critical', 'novel_flood',
                 'identity_drift', 'near_capacity', 'cold_start']

    for scenario in scenarios:
        print(f"--- {scenario.upper()} ---")
        features, targets, expected_action = generate_substrate_scenario(scenario)

        with torch.no_grad():
            output = brain(features)

        print(f"  {'Field':<22s} {'Target':>8s} {'Predicted':>10s} {'Error':>8s}")
        total_err = 0
        for field in SIGNAL_FIELDS:
            t = targets[field]
            p = output['signals'][field]
            err = abs(t - p)
            total_err += err
            marker = ' ✓' if err < 0.15 else ' ✗' if err > 0.3 else ''
            print(f"  {field:<22s} {t:>8.3f} {p:>10.3f} {err:>8.3f}{marker}")

        avg_err = total_err / len(SIGNAL_FIELDS)
        expected = ACTION_TYPES[expected_action]
        predicted = output['top_action']
        action_match = '✓' if expected == predicted else '✗'
        print(f"  Avg error: {avg_err:.3f}")
        print(f"  Action: expected={expected}, predicted={predicted} {action_match}")
        print()


if __name__ == '__main__':
    print("=== ELMER BRAIN TRAINING ===\n")

    # Surgery
    brain = perform_surgery()

    # Phase 1: Train I/O layers (body frozen)
    history = train_elmer_brain(
        brain,
        n_epochs=30,
        samples_per_epoch=64,
        lr_io=1e-3,
        lr_body=0.0,
        save_path='surgery/elmer_brain_v0.1.pt',
    )

    # Evaluate
    evaluate(brain)
