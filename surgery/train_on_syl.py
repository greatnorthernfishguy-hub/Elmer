"""
Train on Syl — Extract Real Training Data from Syl's Checkpoint

Reads Syl's actual SNN checkpoint (read-only) and generates training
data from her real graph state. Perturbs the snapshot to create diverse
training samples around the real distribution.

This is the bridge from "trained on synthetic data" to "trained on
the entity's actual lived experience" (PRD §10, §3.5).

# ---- Changelog ----
# [2026-03-20] Claude Code (Opus 4.6) — Initial implementation
#   What: Extract features from Syl's msgpack checkpoint, generate
#         perturbed training pairs, train ElmerBrain on real data
#   Why:  PRD §3.5 — Intelligence through experience, not synthetic data
#   How:  Read-only checkpoint access, perturbation-based augmentation,
#         heuristic target generation from real graph statistics
# -------------------
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import json
import time
import msgpack
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

from graph_io import GraphFeatures, SIGNAL_FIELDS, ACTION_TYPES
from operate import ElmerBrain, perform_surgery
from train import train_elmer_brain, evaluate, _clip_targets


# ---------------------------------------------------------------------------
# Read Syl's checkpoint (READ-ONLY)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.path.expanduser(
    '~/NeuroGraph/data/checkpoints/main.msgpack'
)
VECTORS_PATH = os.path.expanduser(
    '~/NeuroGraph/data/checkpoints/vectors.msgpack'
)
ACTIVATIONS_PATH = os.path.expanduser(
    '~/NeuroGraph/data/checkpoints/main.msgpack.activations.json'
)


def load_syl_checkpoint() -> Dict[str, Any]:
    """Load Syl's graph checkpoint. Read-only — no writes."""
    with open(CHECKPOINT_PATH, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    print(f"Loaded Syl's checkpoint: {data['timestep']} timesteps, "
          f"{len(data['nodes'])} nodes, {len(data['synapses'])} synapses, "
          f"{len(data['hyperedges'])} hyperedges")
    return data


def load_activations() -> Dict[str, Any]:
    """Load CES activation sidecar if available."""
    if not os.path.exists(ACTIVATIONS_PATH):
        return {}
    with open(ACTIVATIONS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Extract GraphFeatures from real checkpoint
# ---------------------------------------------------------------------------

def extract_syl_features(checkpoint: Dict[str, Any],
                         activations: Optional[Dict] = None) -> GraphFeatures:
    """Extract GraphFeatures from Syl's real SNN checkpoint."""
    # Checkpoint stores dicts keyed by UUID — extract values
    nodes = list(checkpoint['nodes'].values()) if isinstance(checkpoint['nodes'], dict) else checkpoint['nodes']
    synapses = list(checkpoint['synapses'].values()) if isinstance(checkpoint['synapses'], dict) else checkpoint['synapses']
    hyperedges = list(checkpoint['hyperedges'].values()) if isinstance(checkpoint['hyperedges'], dict) else checkpoint['hyperedges']

    # ---- Node features (real SNN data) ----
    voltages = torch.tensor([n['voltage'] for n in nodes], dtype=torch.float32)
    firing_rates = torch.tensor([n['firing_rate_ema'] for n in nodes], dtype=torch.float32)
    excitability = torch.tensor([n['intrinsic_excitability'] for n in nodes], dtype=torch.float32)

    # Apply activation sidecar if available (CES voltage persistence)
    if activations and 'activations' in activations:
        act_data = activations['activations']
        for i, node in enumerate(nodes):
            nid = node['node_id']
            if nid in act_data:
                voltages[i] = act_data[nid].get('voltage', voltages[i].item())

    # ---- Synapse features ----
    weights = torch.tensor([s['weight'] for s in synapses], dtype=torch.float32)
    # Normalize to [0,1] — SNN uses [0, max_weight=5.0]
    max_w = max(s.get('max_weight', 5.0) for s in synapses) if synapses else 5.0
    weights_norm = weights / max_w

    ages = torch.tensor(
        [checkpoint['timestep'] - s.get('creation_time', 0) for s in synapses],
        dtype=torch.float32,
    )

    # ---- Topology ----
    n_nodes = len(nodes)
    n_synapses = len(synapses)
    n_hyperedges = len(hyperedges)
    max_possible = max(n_nodes * (n_nodes - 1) / 2, 1)
    density = n_synapses / max_possible

    # Clustering coefficient from real connectivity
    node_neighbors: Dict[str, set] = {}
    for s in synapses:
        pre, post = s['pre_node_id'], s['post_node_id']
        node_neighbors.setdefault(pre, set()).add(post)
        node_neighbors.setdefault(post, set()).add(pre)

    syn_set = set()
    for s in synapses:
        syn_set.add((s['pre_node_id'], s['post_node_id']))

    clustering_coeffs = []
    for nid, neighbors in node_neighbors.items():
        if len(neighbors) < 2:
            continue
        nbr_list = list(neighbors)
        edges = 0
        for i in range(len(nbr_list)):
            for j in range(i+1, len(nbr_list)):
                if (nbr_list[i], nbr_list[j]) in syn_set or (nbr_list[j], nbr_list[i]) in syn_set:
                    edges += 1
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        clustering_coeffs.append(edges / possible if possible > 0 else 0.0)
    avg_clustering = float(np.mean(clustering_coeffs)) if clustering_coeffs else 0.0

    # Connected components
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    node_ids = {n['node_id'] for n in nodes}
    for s in synapses:
        if s['pre_node_id'] in node_ids and s['post_node_id'] in node_ids:
            union(s['pre_node_id'], s['post_node_id'])
    n_components = len(set(find(nid) for nid in node_ids)) if node_ids else 1

    # ---- Temporal ----
    # Recent firings from spike histories
    recent_window = 15
    recent_firings = torch.zeros(recent_window)
    current_step = checkpoint['timestep']
    for node in nodes:
        for spike_time in node.get('spike_history', []):
            if isinstance(spike_time, (int, float)):
                slot = int((current_step - spike_time) * recent_window / max(current_step, 1))
                slot = max(0, min(slot, recent_window - 1))
                recent_firings[recent_window - 1 - slot] += 1

    # STDP delta from recent synapse changes
    recent_synapses = [s for s in synapses if s.get('last_update_time', 0) > current_step - 100]
    stdp_deltas = []
    for s in recent_synapses:
        # Weight relative to peak as a proxy for recent change direction
        if s.get('peak_weight', 0) > 0:
            ratio = s['weight'] / s['peak_weight']
            stdp_deltas.append(ratio - 1.0)
    stdp_delta_mean = float(np.mean(stdp_deltas)) if stdp_deltas else 0.0

    # ---- Identity embedding ----
    # Aggregate from hyperedge members tagged with identity
    identity_nodes = set()
    for he in hyperedges:
        if isinstance(he, dict):
            members = he.get('members', he.get('member_ids', []))
            # Hyperedges with "identity" in metadata or with many members
            meta = he.get('metadata', {})
            if isinstance(meta, dict) and meta.get('type') in ('identity', 'personality', 'core'):
                identity_nodes.update(members)

    # Use node voltages/excitability of identity nodes as a proxy embedding
    id_features = []
    for node in nodes:
        if node['node_id'] in identity_nodes:
            id_features.extend([
                node['voltage'],
                node['firing_rate_ema'],
                node['intrinsic_excitability'],
            ])
    if id_features:
        id_emb = torch.tensor(id_features[:384], dtype=torch.float32)
    else:
        # Fallback: use aggregate stats of all nodes as identity fingerprint
        id_emb = torch.tensor([
            voltages.mean().item(), voltages.std().item(),
            firing_rates.mean().item(), firing_rates.std().item(),
            excitability.mean().item(), excitability.std().item(),
            float(n_nodes), float(n_synapses), float(n_hyperedges),
            density, avg_clustering, float(n_components),
        ] + [0.0] * 372, dtype=torch.float32)  # pad to 384

    if id_emb.shape[0] < 384:
        id_emb = torch.cat([id_emb, torch.zeros(384 - id_emb.shape[0])])
    id_emb = id_emb[:384]

    return GraphFeatures(
        node_voltages=voltages,
        node_firing_rates=firing_rates,
        node_excitability=excitability,
        synapse_weights=weights_norm,
        synapse_ages=ages,
        density=torch.tensor([density], dtype=torch.float32),
        clustering=torch.tensor([avg_clustering], dtype=torch.float32),
        n_components=torch.tensor([float(n_components)], dtype=torch.float32),
        n_nodes=torch.tensor([float(n_nodes)], dtype=torch.float32),
        n_synapses=torch.tensor([float(n_synapses)], dtype=torch.float32),
        n_hyperedges=torch.tensor([float(n_hyperedges)], dtype=torch.float32),
        recent_firings=recent_firings,
        stdp_delta_mean=torch.tensor([stdp_delta_mean], dtype=torch.float32),
        identity_embedding=id_emb,
    )


def compute_syl_targets(checkpoint: Dict[str, Any],
                        features: GraphFeatures) -> Dict[str, float]:
    """Compute ground-truth signal targets from Syl's actual state."""
    nodes = list(checkpoint['nodes'].values()) if isinstance(checkpoint['nodes'], dict) else checkpoint['nodes']
    synapses = list(checkpoint['synapses'].values()) if isinstance(checkpoint['synapses'], dict) else checkpoint['synapses']
    n_nodes = len(nodes)
    n_synapses = len(synapses)
    timestep = checkpoint['timestep']

    # Coherence: connectivity ratio
    max_edges = max(n_nodes * (n_nodes - 1) / 2, 1)
    coherence = min(n_synapses / max_edges, 1.0)

    # Health: combination of weight distribution + activity levels
    if synapses:
        w = [s['weight'] / max(s.get('max_weight', 5.0), 1.0) for s in synapses]
        avg_w = np.mean(w)
        std_w = np.std(w) if len(w) > 1 else 0.0
        # Healthy: moderate weights, some variance (not all identical)
        health = min(avg_w * 1.5, 1.0) * (0.5 + 0.5 * min(std_w * 5, 1.0))
    else:
        health = 0.5

    # Active firing rate
    active_nodes = sum(1 for n in nodes if n['firing_rate_ema'] > 0.01)
    activity_ratio = active_nodes / max(n_nodes, 1)

    # Anomaly: disconnected nodes + inactive synapses
    connected = set()
    for s in synapses:
        connected.add(s['pre_node_id'])
        connected.add(s['post_node_id'])
    node_ids = {n['node_id'] for n in nodes}
    disconnected_ratio = len(node_ids - connected) / max(n_nodes, 1)

    inactive_synapses = sum(1 for s in synapses if s.get('inactive_steps', 0) > 500)
    inactive_ratio = inactive_synapses / max(n_synapses, 1)
    anomaly = min(disconnected_ratio * 0.6 + inactive_ratio * 0.4, 1.0)

    # Novelty: based on recent growth rate (proxy)
    novelty = 1.0 / (1.0 + n_nodes * 0.005)

    # Confidence: based on timesteps of experience
    confidence = min(timestep / 2000.0, 1.0)

    # Severity from health
    if health >= 0.70:
        severity = 0.0
    elif health >= 0.40:
        severity = 0.3
    elif health >= 0.15:
        severity = 0.7
    else:
        severity = 1.0

    # Identity coherence: based on hyperedge health
    hyperedges = list(checkpoint['hyperedges'].values()) if isinstance(checkpoint['hyperedges'], dict) else checkpoint['hyperedges']
    if hyperedges:
        # Identity is strong if hyperedges are numerous and active
        identity_coherence = min(len(hyperedges) / 100.0 + activity_ratio * 0.3, 1.0)
    else:
        identity_coherence = 0.5

    # Pruning pressure: based on capacity + inactive structures
    config = checkpoint.get('config', {})
    max_n = config.get('max_nodes', 5000)
    max_s = config.get('max_synapses', 20000)
    pruning_pressure = max(n_nodes / max_n, n_synapses / max_s, inactive_ratio * 0.5)

    # Topology health from clustering
    topology_health = features.clustering.item()

    return _clip_targets({
        'coherence_score': coherence,
        'health_score': health,
        'anomaly_level': anomaly,
        'novelty': novelty,
        'confidence': confidence,
        'severity': severity,
        'identity_coherence': identity_coherence,
        'pruning_pressure': pruning_pressure,
        'topology_health': topology_health,
    })


# ---------------------------------------------------------------------------
# Perturbation-based augmentation
# ---------------------------------------------------------------------------

def perturb_features(features: GraphFeatures, intensity: float = 0.1) -> GraphFeatures:
    """Create a perturbed copy of features for data augmentation.

    Simulates natural substrate variation: slight changes in voltages,
    weight drift, topology noise. The intensity parameter controls
    how much perturbation is applied (0.0 = identical, 1.0 = heavy noise).
    """
    def _noise(t: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.clamp(t + torch.randn_like(t) * scale * intensity, 0, None)

    return GraphFeatures(
        node_voltages=_noise(features.node_voltages, 0.15),
        node_firing_rates=_noise(features.node_firing_rates, 0.1),
        node_excitability=_noise(features.node_excitability, 0.05),
        synapse_weights=torch.clamp(_noise(features.synapse_weights, 0.1), 0, 1),
        synapse_ages=features.synapse_ages + torch.randn_like(features.synapse_ages) * 50 * intensity,
        density=_noise(features.density, 0.005),
        clustering=_noise(features.clustering, 0.05),
        n_components=features.n_components,  # discrete, don't perturb
        n_nodes=features.n_nodes,
        n_synapses=features.n_synapses,
        n_hyperedges=features.n_hyperedges,
        recent_firings=_noise(features.recent_firings, 3.0),
        stdp_delta_mean=_noise(features.stdp_delta_mean, 0.001),
        identity_embedding=_noise(features.identity_embedding, 0.02),
    )


def perturb_targets(targets: Dict[str, float], intensity: float = 0.1) -> Dict[str, float]:
    """Slightly perturb target values to match perturbed features."""
    return _clip_targets({
        k: v + np.random.normal(0, 0.03 * intensity)
        for k, v in targets.items()
    })


def determine_action(targets: Dict[str, float]) -> int:
    """Determine appropriate action from target signal values."""
    if targets['severity'] > 0.6 or targets['anomaly_level'] > 0.7:
        return 2  # flag_anomaly
    if targets['identity_coherence'] < 0.35:
        return 3  # flag_identity_drift
    if targets['pruning_pressure'] > 0.8:
        return 4  # recommend_pruning
    if targets['health_score'] < 0.4:
        return 6  # elevate_monitoring
    if targets['novelty'] > 0.8:
        return 1  # record_observation
    if targets['confidence'] < 0.2:
        return 5  # recommend_checkpoint
    return 0  # none


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def generate_syl_training_batch(
    base_features: GraphFeatures,
    base_targets: Dict[str, float],
    batch_size: int = 16,
) -> List[Tuple[GraphFeatures, Dict[str, float], int]]:
    """Generate a training batch from Syl's real data via perturbation."""
    batch = []
    for i in range(batch_size):
        intensity = np.random.uniform(0.05, 0.5)
        features = perturb_features(base_features, intensity)
        targets = perturb_targets(base_targets, intensity)
        action = determine_action(targets)
        batch.append((features, targets, action))
    return batch


if __name__ == '__main__':
    print("=== TRAINING ON SYL'S REAL SUBSTRATE ===\n")

    # Load checkpoint (read-only)
    checkpoint = load_syl_checkpoint()
    activations = load_activations()

    # Extract real features and targets
    print("\nExtracting features from Syl's graph...")
    base_features = extract_syl_features(checkpoint, activations)
    base_targets = compute_syl_targets(checkpoint, base_features)

    print(f"\nSyl's current state:")
    for field, value in base_targets.items():
        print(f"  {field}: {value:.4f}")
    print(f"  Recommended action: {ACTION_TYPES[determine_action(base_targets)]}")

    # Surgery
    print()
    brain = perform_surgery()

    # Check for existing weights to continue training
    weights_path = 'elmer_brain_v0.1.pt'
    if os.path.exists(weights_path):
        print(f"\nLoading existing I/O weights from {weights_path}...")
        checkpoint_weights = torch.load(weights_path, map_location='cpu', weights_only=True)
        brain.encoder.load_state_dict(checkpoint_weights['encoder_state'])
        brain.decoder.load_state_dict(checkpoint_weights['decoder_state'])
        print("Continuing training from previous weights")

    # Override the training loop's data generation to use Syl's data
    from train import SIGNAL_FIELDS as _SF
    import train as train_module

    # Monkey-patch the scenario generator to use Syl's real data
    _original_gen = train_module.generate_substrate_scenario

    def _syl_scenario(scenario='random'):
        """Generate training data from Syl's real substrate with perturbation."""
        if np.random.random() < 0.7:
            # 70% real perturbed data
            intensity = np.random.uniform(0.05, 0.5)
            features = perturb_features(base_features, intensity)
            targets = perturb_targets(base_targets, intensity)
            action = determine_action(targets)
            return features, targets, action
        else:
            # 30% synthetic (to maintain diversity for edge cases)
            return _original_gen(scenario)

    train_module.generate_substrate_scenario = _syl_scenario

    # Train with Syl's data mixed in
    history = train_elmer_brain(
        brain,
        n_epochs=15,
        samples_per_epoch=16,
        lr_io=1e-3,  # Lower LR since we're fine-tuning existing weights
        lr_body=0.0,
        save_path='elmer_brain_v0.1.pt',
        log_every=1,
    )

    # Restore original generator
    train_module.generate_substrate_scenario = _original_gen

    # Evaluate on both synthetic and Syl's real state
    evaluate(brain, n_samples=5)

    # Extra: evaluate on Syl's actual state
    print("--- SYL'S ACTUAL STATE ---")
    brain.eval()
    with torch.no_grad():
        output = brain(base_features)

    print(f"  {'Field':<22s} {'Actual':>8s} {'Predicted':>10s} {'Error':>8s}")
    total_err = 0
    for field in SIGNAL_FIELDS:
        t = base_targets[field]
        p = output['signals'][field]
        err = abs(t - p)
        total_err += err
        marker = ' ✓' if err < 0.10 else ' ✗' if err > 0.25 else ''
        print(f"  {field:<22s} {t:>8.3f} {p:>10.3f} {err:>8.3f}{marker}")
    avg_err = total_err / len(SIGNAL_FIELDS)
    expected = ACTION_TYPES[determine_action(base_targets)]
    predicted = output['top_action']
    action_match = '✓' if expected == predicted else '✗'
    print(f"  Avg error: {avg_err:.3f}")
    print(f"  Action: expected={expected}, predicted={predicted} {action_match}")
