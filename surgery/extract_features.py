"""
Feature Extractor — Pull Graph State from NG-Lite into GraphFeatures

Bridges the NG-Lite data structures (NGLiteNode, NGLiteSynapse) into
the tensor format that GraphStateEncoder expects.

# ---- Changelog ----
# [2026-03-20] Claude Code (Opus 4.6) — Initial implementation
#   What: Extract real graph state from NG-Lite into GraphFeatures tensors
#   Why:  Training data pipeline for ElmerBrain I/O layers
#   How:  Read nodes, synapses, compute topology metrics, pack into tensors
# -------------------
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple

from graph_io import GraphFeatures


def extract_features_from_ng_lite(ng_lite, identity_embedding: Optional[np.ndarray] = None) -> GraphFeatures:
    """Extract GraphFeatures from a live NG-Lite instance.

    Args:
        ng_lite: An NGLite instance (from the vendored ng_lite.py).
        identity_embedding: Optional 384-dim identity vector. If None,
            attempts to compute one from identity-tagged nodes.

    Returns:
        GraphFeatures ready for the GraphStateEncoder.
    """
    nodes = list(ng_lite.nodes.values())
    synapses = list(ng_lite.synapses.values())

    now = time.time()

    # ---- Node-level features ----
    if nodes:
        # Voltages: approximate from recency of activation
        # (NG-Lite doesn't track voltage directly — that's full SNN territory)
        # Use activation recency as a proxy: recently fired = high voltage
        voltages = []
        firing_rates = []
        excitabilities = []
        for node in nodes:
            # Recency proxy: exponential decay from last activation
            dt = now - node.last_activation if node.last_activation > 0 else 1e6
            voltage = np.exp(-dt / 3600.0)  # 1-hour time constant
            voltages.append(voltage)

            # Firing rate: activation count normalized by time alive
            age = max(now - (node.metadata.get('created', now) if isinstance(node.metadata, dict) else now), 1.0)
            rate = node.activation_count / (age / 3600.0)  # activations per hour
            firing_rates.append(min(rate, 100.0) / 100.0)  # normalize to [0,1]

            # Excitability: constitutional nodes are "frozen" = low excitability
            exc = 0.1 if node.constitutional else 0.8
            excitabilities.append(exc)

        node_voltages = torch.tensor(voltages, dtype=torch.float32)
        node_firing_rates = torch.tensor(firing_rates, dtype=torch.float32)
        node_excitability = torch.tensor(excitabilities, dtype=torch.float32)
    else:
        node_voltages = torch.zeros(1)
        node_firing_rates = torch.zeros(1)
        node_excitability = torch.zeros(1)

    # ---- Synapse-level features ----
    if synapses:
        weights = [s.weight for s in synapses]
        ages = [(now - s.last_updated) if s.last_updated > 0 else 0.0 for s in synapses]

        synapse_weights = torch.tensor(weights, dtype=torch.float32)
        synapse_ages = torch.tensor(ages, dtype=torch.float32)
    else:
        synapse_weights = torch.zeros(1)
        synapse_ages = torch.zeros(1)

    # ---- Topology metrics ----
    n_nodes = len(nodes)
    n_synapses = len(synapses)
    max_possible = max(n_nodes * (n_nodes - 1) / 2, 1)
    density = n_synapses / max_possible

    # Clustering: approximate from local connectivity
    # For each node, what fraction of its neighbors are also connected to each other
    node_neighbors: Dict[str, set] = {}
    for s in synapses:
        node_neighbors.setdefault(s.source_id, set()).add(s.target_id)
        node_neighbors.setdefault(s.target_id, set()).add(s.source_id)

    clustering_coeffs = []
    for node_id, neighbors in node_neighbors.items():
        if len(neighbors) < 2:
            clustering_coeffs.append(0.0)
            continue
        # Count edges between neighbors
        neighbor_edges = 0
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                key1 = (neighbor_list[i], neighbor_list[j])
                key2 = (neighbor_list[j], neighbor_list[i])
                if key1 in ng_lite.synapses or key2 in ng_lite.synapses:
                    neighbor_edges += 1
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        clustering_coeffs.append(neighbor_edges / possible if possible > 0 else 0.0)

    avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0.0

    # Connected components: approximate with union-find
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

    all_node_ids = {n.node_id for n in nodes}
    for s in synapses:
        if s.source_id in all_node_ids and s.target_id in all_node_ids:
            union(s.source_id, s.target_id)
    components = len(set(find(nid) for nid in all_node_ids)) if all_node_ids else 1

    # Hyperedges: NG-Lite doesn't track these, but full SNN does.
    # Use 0 for Tier 1/2, would be populated at Tier 3.
    n_hyperedges = 0

    # ---- Temporal features ----
    # Recent firings: count activations in the last N steps
    # NG-Lite tracks _history as a list of dicts
    history = getattr(ng_lite, '_history', [])
    window_size = 15
    recent_firings = torch.zeros(window_size)
    if history:
        # Bin recent history entries into window slots
        recent = history[-min(len(history), window_size * 10):]
        for i, entry in enumerate(recent):
            slot = i * window_size // max(len(recent), 1)
            slot = min(slot, window_size - 1)
            recent_firings[slot] += 1

    # STDP delta: approximate from recent weight changes in history
    stdp_deltas = []
    for entry in history[-50:]:
        if isinstance(entry, dict) and 'weight_delta' in entry:
            stdp_deltas.append(entry['weight_delta'])
    stdp_delta_mean = torch.tensor([np.mean(stdp_deltas) if stdp_deltas else 0.0])

    # ---- Identity embedding ----
    if identity_embedding is not None:
        id_emb = torch.tensor(identity_embedding, dtype=torch.float32)
    else:
        # Try to compute from identity-tagged nodes
        id_embeddings = []
        for node in nodes:
            if node.embedding is not None and isinstance(node.metadata, dict):
                if node.metadata.get('identity', False) or node.metadata.get('hyperedge_member', False):
                    id_embeddings.append(node.embedding)
        if id_embeddings:
            avg = np.mean(id_embeddings, axis=0)
            id_emb = torch.tensor(avg, dtype=torch.float32)
        else:
            # No identity info available — zero vector
            id_emb = torch.zeros(384)

    # Ensure identity embedding is 384-dim
    if id_emb.shape[0] > 384:
        id_emb = id_emb[:384]
    elif id_emb.shape[0] < 384:
        id_emb = torch.cat([id_emb, torch.zeros(384 - id_emb.shape[0])])

    return GraphFeatures(
        node_voltages=node_voltages,
        node_firing_rates=node_firing_rates,
        node_excitability=node_excitability,
        synapse_weights=synapse_weights,
        synapse_ages=synapse_ages,
        density=torch.tensor([density], dtype=torch.float32),
        clustering=torch.tensor([avg_clustering], dtype=torch.float32),
        n_components=torch.tensor([float(components)], dtype=torch.float32),
        n_nodes=torch.tensor([float(n_nodes)], dtype=torch.float32),
        n_synapses=torch.tensor([float(n_synapses)], dtype=torch.float32),
        n_hyperedges=torch.tensor([float(n_hyperedges)], dtype=torch.float32),
        recent_firings=recent_firings,
        stdp_delta_mean=stdp_delta_mean,
        identity_embedding=id_emb,
    )


def extract_training_targets(ng_lite, features: GraphFeatures) -> Dict[str, float]:
    """Compute ground-truth SubstrateSignal targets from graph state.

    These come from the SAME heuristics that the current stub sockets use,
    so the trained model learns to replicate (and eventually surpass) them.
    """
    nodes = list(ng_lite.nodes.values())
    synapses = list(ng_lite.synapses.values())
    n_nodes = len(nodes)
    n_synapses = len(synapses)

    # Coherence: ratio of edges to possible edges (same as ComprehensionSocket)
    max_edges = max(n_nodes * (n_nodes - 1) / 2, 1)
    coherence = min(n_synapses / max_edges, 1.0) if n_nodes > 1 else 1.0

    # Health: weighted combination of connectivity + weight distribution
    if synapses:
        avg_weight = np.mean([s.weight for s in synapses])
        weight_std = np.std([s.weight for s in synapses]) if len(synapses) > 1 else 0.0
        # Healthy: moderate average weight, not too uniform
        health = min(avg_weight * 1.2, 1.0) * (1.0 - 0.5 * (1.0 - min(weight_std * 3, 1.0)))
    else:
        health = 0.5

    # Anomaly: disconnected nodes + extreme weight values
    connected_ids = set()
    for s in synapses:
        connected_ids.add(s.source_id)
        connected_ids.add(s.target_id)
    node_ids = {n.node_id for n in nodes}
    disconnected = len(node_ids - connected_ids)
    anomaly = disconnected / max(n_nodes, 1)

    # Add anomaly from extreme weights
    if synapses:
        extreme_weights = sum(1 for s in synapses if s.weight > 0.95 or s.weight < 0.05)
        anomaly = min(anomaly + extreme_weights / max(len(synapses), 1) * 0.3, 1.0)

    # Novelty: inverse of node count density
    novelty = 1.0 / (1.0 + n_nodes * 0.01)

    # Confidence: based on amount of data
    total_outcomes = getattr(ng_lite, '_total_outcomes', 0)
    confidence = min(total_outcomes / 100.0, 1.0)  # saturates at 100 outcomes

    # Severity: from health thresholds (PRD §14)
    if health >= 0.70:
        severity = 0.0
    elif health >= 0.40:
        severity = 0.3
    elif health >= 0.15:
        severity = 0.7
    else:
        severity = 1.0

    # Identity coherence: placeholder — needs identity baseline to compare against
    identity_coherence = 0.8 if n_nodes > 10 else 0.5

    # Pruning pressure: how close to capacity
    max_nodes = ng_lite.config.get('max_nodes', 1000)
    max_synapses = ng_lite.config.get('max_synapses', 5000)
    node_pressure = n_nodes / max_nodes
    syn_pressure = n_synapses / max_synapses
    pruning_pressure = max(node_pressure, syn_pressure)

    # Topology health: clustering quality
    topology_health = features.clustering.item()

    return {
        'coherence_score': float(np.clip(coherence, 0, 1)),
        'health_score': float(np.clip(health, 0, 1)),
        'anomaly_level': float(np.clip(anomaly, 0, 1)),
        'novelty': float(np.clip(novelty, 0, 1)),
        'confidence': float(np.clip(confidence, 0, 1)),
        'severity': float(np.clip(severity, 0, 1)),
        'identity_coherence': float(np.clip(identity_coherence, 0, 1)),
        'pruning_pressure': float(np.clip(pruning_pressure, 0, 1)),
        'topology_health': float(np.clip(topology_health, 0, 1)),
    }
