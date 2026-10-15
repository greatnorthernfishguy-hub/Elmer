"""
Graph-Native I/O Layers for Surgically Modified Transformer

New Eyes (GraphStateEncoder): Projects graph-state features into the
transformer's hidden dimension (896 for Qwen2.5-0.5B).

New Voice (GraphSignalDecoder): Projects transformer hidden states into
SubstrateSignal-compatible output fields.

These replace the token embedding and vocabulary prediction head that
were designed for language. The transformer body between them is unchanged.

# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane mean-centre) — #1108: the identity input is 768-d end to end
#   What: IDENTITY_DIM = 768 (= ng_embed.EMBEDDING_DIM, the Snowflake model the identity vector comes from);
#         identity_proj = Linear(IDENTITY_DIM, hidden). forward() takes the identity vector at exactly IDENTITY_DIM
#         (an empty / absent one = zeros, "no spine", as before) and raises ValueError on any other length — no
#         pad, no truncate. A checkpoint whose identity_proj is 384 wide (every encoder trained so far:
#         elmer_brain_v0.1.pt, NeuroGraph tonic_brain.pt) is migrated on load_state_dict by a pre-hook: its 384
#         trained input columns are kept as columns 0..383 and columns 384..767 start at zero, logged at WARNING.
#   Why:  #1108 / DIM_GUARD.md §2.2: the CC Tonic feeds the 768-d Snowflake spine vector and _pad_or_truncate cut it
#         to 384 silently, every tick. Why the migration is exact rather than a re-initialisation: the 384 columns
#         were trained on proxy features (train_on_syl.py: identity-node voltage/excitability stats padded to 384;
#         train.py / operate.py: noise), never on an embedding, so they carry no learned meaning for Snowflake dims in
#         either layout; zero new columns keep today's outputs bit-for-bit (proven in tests/test_identity_768.py)
#         while the input itself is no longer truncated. Using dims 384..767 needs identity_proj retrained on 768-d
#         spine vectors (train_tonic.py) — logged at every migration, listed in MEAN_CENTRE.md as open.
#   How:  IDENTITY_DIM constant; _check_identity(); _migrate_identity_proj (nn.Module load_state_dict pre-hook).
# [2026-03-20] Claude Code (Opus 4.6) — Initial implementation
#   What: GraphStateEncoder and GraphSignalDecoder for Elmer model surgery
#   Why:  PRD §5.4 — harvest reasoning engine, new eyes, new voice
#   How:  Encoder projects graph features to hidden_dim via learned linear
#         layers. Decoder projects hidden states to SubstrateSignal fields.
# -------------------
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional


# ---- Graph State Encoder ("New Eyes") ----
# PRD §5.4.1 operation 2: Replace token embedding with graph-state encoder
#
# The transformer's embed_tokens maps discrete token IDs to dense vectors.
# We replace that with a continuous projection from graph-state features:
#   - Node activation vectors (voltage, firing rate, excitability)
#   - Synapse weight statistics (mean, std, sparsity)
#   - Topological features (density, clustering, connected components)
#   - Temporal features (recent firing patterns, STDP timing)
#
# The encoder must output shape (batch, seq_len, hidden_dim) to match
# what the transformer layers expect from the embedding layer.

# How many raw graph features we extract per "token position"
GRAPH_FEATURE_DIM = 64

# The identity-conditioning vector's width = the active embedding model's dimension (ng_embed.EMBEDDING_DIM,
# Snowflake/snowflake-arctic-embed-m-v1.5). Pinned to ng_embed by tests/test_identity_768.py. #1108.
IDENTITY_DIM = 768
# Width of identity_proj in every encoder trained before #1108 (migrated on load, see _migrate_identity_proj).
LEGACY_IDENTITY_DIM = 384


@dataclass
class GraphFeatures:
    """Raw graph state features extracted from an NG-Lite snapshot.

    Each field is a 1-D tensor. The encoder combines them into
    the feature vector the transformer sees.
    """
    # Node-level (aggregated stats across active nodes)
    node_voltages: torch.Tensor       # [n_sample_nodes] current voltages
    node_firing_rates: torch.Tensor   # [n_sample_nodes] recent firing rate
    node_excitability: torch.Tensor   # [n_sample_nodes] homeostatic state

    # Synapse-level (aggregated stats)
    synapse_weights: torch.Tensor     # [n_sample_synapses] weight values
    synapse_ages: torch.Tensor        # [n_sample_synapses] timesteps since creation

    # Topology-level (scalar features)
    density: torch.Tensor             # [1] edge_count / max_possible_edges
    clustering: torch.Tensor          # [1] average local clustering coefficient
    n_components: torch.Tensor        # [1] number of connected components
    n_nodes: torch.Tensor             # [1] total node count
    n_synapses: torch.Tensor          # [1] total synapse count
    n_hyperedges: torch.Tensor        # [1] total hyperedge count

    # Temporal (recent dynamics)
    recent_firings: torch.Tensor      # [window_size] firing count per recent step
    stdp_delta_mean: torch.Tensor     # [1] mean weight change from recent STDP

    # Identity (embedding of identity-bearing hyperedge members)
    identity_embedding: torch.Tensor  # [embed_dim] aggregated identity vector


class GraphStateEncoder(nn.Module):
    """New Eyes — projects graph state into transformer hidden space.

    Takes a GraphFeatures struct (or raw feature tensor) and produces
    a (batch, seq_len, hidden_dim) tensor that feeds directly into
    the transformer layers, replacing the token embedding.

    The "sequence" dimension is synthetic — we create multiple "positions"
    from different aspects of the graph state so the transformer's
    attention mechanism can relate them to each other. This is the key
    insight: attention between graph features IS graph reasoning.

    Positions:
      0: Global topology summary
      1: Node dynamics summary
      2: Synapse dynamics summary
      3: Temporal dynamics summary
      4: Identity embedding
      5-N: Sampled node neighborhoods (if available)
    """

    MIN_SEQ_LEN = 5  # topology + nodes + synapses + temporal + identity

    def __init__(self, hidden_dim: int = 896, n_positions: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_positions = max(n_positions, self.MIN_SEQ_LEN)

        # Each "aspect" of the graph gets its own projection into hidden_dim
        # This lets each aspect have its own learned representation
        self.topology_proj = nn.Linear(6, hidden_dim)   # density, clustering, components, nodes, synapses, hyperedges
        self.node_proj = nn.Linear(32, hidden_dim)       # aggregated node stats (padded/truncated to 32)
        self.synapse_proj = nn.Linear(32, hidden_dim)    # aggregated synapse stats (padded/truncated to 32)
        self.temporal_proj = nn.Linear(16, hidden_dim)   # recent firing window + stdp stats
        self.identity_proj = nn.Linear(IDENTITY_DIM, hidden_dim)  # identity embedding (#1108: 768, ng_embed's dim)
        self._register_load_state_dict_pre_hook(self._migrate_identity_proj)

        # Extra positions for neighborhood samples
        self.neighborhood_proj = nn.Linear(GRAPH_FEATURE_DIM, hidden_dim)

        # Learned position embeddings for the synthetic sequence
        self.position_embedding = nn.Embedding(self.n_positions, hidden_dim)

        # Layer norm to match what the transformer expects post-embedding
        self.layer_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _migrate_identity_proj(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                               error_msgs):
        """load_state_dict pre-hook (#1108): a 384-wide identity_proj (pre-#1108 checkpoint) becomes IDENTITY_DIM
        wide — trained columns 0..383 kept, columns 384..767 zero — so the outputs are exactly the old ones."""
        key = prefix + "identity_proj.weight"
        w = state_dict.get(key)
        if w is None or w.dim() != 2 or w.shape[1] == IDENTITY_DIM:
            return
        if w.shape[1] != LEGACY_IDENTITY_DIM:
            return  # anything else: let load_state_dict report the size mismatch
        new = torch.zeros(w.shape[0], IDENTITY_DIM, dtype=w.dtype, device=w.device)
        new[:, :LEGACY_IDENTITY_DIM] = w
        state_dict[key] = new
        import logging
        logging.getLogger("elmer.graph_io").warning(
            "GraphStateEncoder: migrated a %d-wide identity_proj to %d (columns %d..%d zero): the identity vector is "
            "now taken at full width, but only its first %d dims are weighted until identity_proj is retrained on "
            "%d-d identity vectors (#1108)", LEGACY_IDENTITY_DIM, IDENTITY_DIM, LEGACY_IDENTITY_DIM,
            IDENTITY_DIM - 1, LEGACY_IDENTITY_DIM, IDENTITY_DIM)

    @staticmethod
    def _check_identity(t: torch.Tensor) -> torch.Tensor:
        """The identity vector at exactly IDENTITY_DIM (empty = zeros); any other length raises (#1108)."""
        t = t.reshape(-1)
        if t.numel() == 0:
            return torch.zeros(IDENTITY_DIM, device=t.device)
        if t.numel() != IDENTITY_DIM:
            raise ValueError(
                f"GraphStateEncoder: identity_embedding is {t.numel()}-d but the encoder takes {IDENTITY_DIM}-d "
                f"(the active embedding model's dimension); it is not padded or truncated (#1108)")
        return t

    def _pad_or_truncate(self, t: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad with zeros or truncate to target length."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if len(t) >= target_len:
            return t[:target_len]
        return torch.cat([t, torch.zeros(target_len - len(t), device=t.device)])

    def _aggregate_stats(self, t: torch.Tensor, target_len: int) -> torch.Tensor:
        """Compute statistical summary of a variable-length tensor."""
        if t.numel() == 0:
            return torch.zeros(target_len, device=t.device)
        stats = torch.tensor([
            t.mean(),
            t.std() if t.numel() > 1 else torch.tensor(0.0),
            t.min(),
            t.max(),
            t.median(),
            float(t.numel()),
            t.sum(),
            (t > 0).float().mean(),  # fraction positive
        ], device=t.device)
        return self._pad_or_truncate(stats, target_len)

    def forward(self, features: GraphFeatures) -> torch.Tensor:
        """Encode graph features into transformer input.

        Args:
            features: GraphFeatures from the substrate snapshot.

        Returns:
            Tensor of shape (1, n_positions, hidden_dim) ready for
            the transformer layers.
        """
        device = next(self.parameters()).device

        # Position 0: Topology summary
        topo = torch.stack([
            features.density.squeeze(),
            features.clustering.squeeze(),
            features.n_components.squeeze().float(),
            features.n_nodes.squeeze().float(),
            features.n_synapses.squeeze().float(),
            features.n_hyperedges.squeeze().float(),
        ]).to(device)
        h_topo = self.topology_proj(topo.unsqueeze(0))  # (1, hidden_dim)

        # Position 1: Node dynamics
        node_stats = torch.cat([
            self._aggregate_stats(features.node_voltages, 10),
            self._aggregate_stats(features.node_firing_rates, 11),
            self._aggregate_stats(features.node_excitability, 11),
        ]).to(device)
        h_nodes = self.node_proj(node_stats.unsqueeze(0))

        # Position 2: Synapse dynamics
        syn_stats = torch.cat([
            self._aggregate_stats(features.synapse_weights, 16),
            self._aggregate_stats(features.synapse_ages, 16),
        ]).to(device)
        h_synapses = self.synapse_proj(syn_stats.unsqueeze(0))

        # Position 3: Temporal dynamics
        temporal = torch.cat([
            self._pad_or_truncate(features.recent_firings, 15),
            features.stdp_delta_mean.squeeze().unsqueeze(0),
        ]).to(device)
        h_temporal = self.temporal_proj(temporal.unsqueeze(0))

        # Position 4: Identity embedding (#1108: full width, never padded / truncated)
        identity = self._check_identity(features.identity_embedding).to(device)
        h_identity = self.identity_proj(identity.unsqueeze(0))

        # Stack into sequence: (1, n_positions, hidden_dim)
        positions = [h_topo, h_nodes, h_synapses, h_temporal, h_identity]

        # Fill remaining positions with zero (placeholder for neighborhood samples)
        while len(positions) < self.n_positions:
            positions.append(torch.zeros(1, self.hidden_dim, device=device))

        hidden = torch.stack(positions, dim=1)  # (1, n_positions, hidden_dim)

        # Add position embeddings
        pos_ids = torch.arange(self.n_positions, device=device)
        hidden = hidden + self.position_embedding(pos_ids).unsqueeze(0)

        # Layer norm
        hidden = self.layer_norm(hidden)

        return hidden


# ---- Graph Signal Decoder ("New Voice") ----
# PRD §5.4.1 operation 3: Replace vocabulary prediction head with
# graph-signal decoder that outputs SubstrateSignal-compatible fields.
#
# Instead of 151,936 logits over a text vocabulary, we output:
#   - coherence_score (0-1)
#   - health_score (0-1)
#   - anomaly_level (0-1)
#   - novelty (0-1)
#   - confidence (0-1)
#   - severity (0-1)
#   - identity_coherence (0-1)
#   - pruning_pressure (0-1)
#   - topology_health (0-1)
#
# Plus action logits for recommended substrate operations.

SIGNAL_FIELDS = [
    'coherence_score',
    'health_score',
    'anomaly_level',
    'novelty',
    'confidence',
    'severity',
    'identity_coherence',
    'pruning_pressure',
    'topology_health',
]

ACTION_TYPES = [
    'none',                  # no action needed
    'record_observation',    # record to substrate
    'flag_anomaly',          # anomaly detected
    'flag_identity_drift',   # identity coherence concern
    'recommend_pruning',     # topology needs pruning
    'recommend_checkpoint',  # save state recommended
    'elevate_monitoring',    # increase monitoring frequency
]


class GraphSignalDecoder(nn.Module):
    """New Voice — projects transformer hidden states into SubstrateSignal.

    Takes the transformer's final hidden states (batch, seq_len, hidden_dim)
    and produces SubstrateSignal field values and action recommendations.

    Architecture:
      1. Pool across the sequence dimension (attention-weighted)
      2. Project to signal fields via sigmoid (bounded 0-1)
      3. Project to action logits via separate head
    """

    def __init__(self, hidden_dim: int = 896):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Attention pooling: learn which positions matter most
        self.pool_query = nn.Parameter(torch.randn(hidden_dim))
        self.pool_scale = hidden_dim ** -0.5

        # Normalize the transformer's output before projecting —
        # the frozen body produces hidden states with large magnitudes
        # that saturate sigmoid without this.
        self.pre_norm = nn.LayerNorm(hidden_dim)

        # Signal head: hidden_dim -> 9 SubstrateSignal fields
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, len(SIGNAL_FIELDS)),
            nn.Sigmoid(),  # all fields bounded [0, 1]
        )

        # Action head: hidden_dim -> action logits
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, len(ACTION_TYPES)),
        )

        # Initialize final layers with small weights to avoid sigmoid saturation
        self._init_small(self.signal_head[-2])  # Linear before Sigmoid
        self._init_small(self.action_head[-1])

    @staticmethod
    def _init_small(layer: nn.Module):
        """Initialize a Linear layer with small weights for stable early training."""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """Decode transformer output into SubstrateSignal fields.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) from transformer.

        Returns:
            Dict with 'signals' (field values) and 'actions' (recommended ops).
        """
        # Normalize hidden states from the frozen body
        hidden_states = self.pre_norm(hidden_states)

        # Attention-weighted pooling across sequence
        # scores: (batch, seq_len)
        scores = torch.matmul(hidden_states, self.pool_query) * self.pool_scale
        weights = torch.softmax(scores, dim=1)
        # pooled: (batch, hidden_dim)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

        # Signal fields
        signal_values = self.signal_head(pooled)  # (batch, 9)

        # Action logits
        action_logits = self.action_head(pooled)  # (batch, n_actions)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Pack into dict
        signals = {}
        for i, field_name in enumerate(SIGNAL_FIELDS):
            signals[field_name] = signal_values[0, i].item()

        actions = {}
        for i, action_name in enumerate(ACTION_TYPES):
            actions[action_name] = action_probs[0, i].item()

        return {
            'signals': signals,
            'actions': actions,
            'top_action': ACTION_TYPES[action_logits[0].argmax().item()],
            'raw_signal_tensor': signal_values,
            'raw_action_logits': action_logits,
        }
