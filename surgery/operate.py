"""
Model Surgery — Strip and Rewire Qwen2.5-0.5B for Graph-Native Inference

This script performs the three operations from PRD §5.4:
  1. Keep the Body — preserve transformer layers unchanged
  2. New Eyes — replace token embedding with GraphStateEncoder
  3. New Voice — replace vocab prediction head with GraphSignalDecoder

The result is a model that reads graph state and outputs SubstrateSignal
fields, with transformer-class reasoning in between.

# ---- Changelog ----
# [2026-03-20] Claude Code (Opus 4.6) — Initial surgery implementation
#   What: Load Qwen2.5-0.5B, replace embed_tokens and lm_head with
#         graph-native I/O layers, verify forward pass works.
#   Why:  PRD §5.4 — Elmer's core innovation: harvested reasoning engines
#   How:  Load via HuggingFace, swap modules, run synthetic graph features
#         through the full pipeline to verify tensor shapes propagate.
# -------------------
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from graph_io import (
    GraphStateEncoder,
    GraphSignalDecoder,
    GraphFeatures,
    SIGNAL_FIELDS,
    ACTION_TYPES,
)


class ElmerBrain(nn.Module):
    """A surgically modified transformer for graph-native inference.

    The body (transformer layers, norms, rotary embeddings) comes from
    a pretrained language model. The I/O layers are custom-built for
    the NeuroGraph substrate.

    This is not a chat model. It does not generate text. It reads
    graph state and produces SubstrateSignal assessments.
    """

    def __init__(self, transformer_body, encoder, decoder):
        super().__init__()
        self.body = transformer_body    # The 24 transformer layers + norm + rotary
        self.encoder = encoder          # New Eyes
        self.decoder = decoder          # New Voice

    def forward(self, features: GraphFeatures):
        """Run graph features through the harvested reasoning engine.

        Flow:
          GraphFeatures → Encoder → (batch, seq, 896)
                       → Transformer body → (batch, seq, 896)
                       → Decoder → SubstrateSignal dict
        """
        # New Eyes: graph state → hidden representations
        hidden = self.encoder(features)

        # The Body: transformer reasoning over graph representations
        # Qwen2Model.forward expects inputs_embeds when not using input_ids
        body_output = self.body(
            inputs_embeds=hidden,
            use_cache=False,
            return_dict=True,
        )
        # body_output.last_hidden_state: (batch, seq_len, hidden_dim)
        reasoned = body_output.last_hidden_state

        # New Voice: hidden states → SubstrateSignal fields
        output = self.decoder(reasoned)

        return output


def create_synthetic_features(device='cpu') -> GraphFeatures:
    """Create synthetic graph features for testing the surgery.

    These mimic what you'd extract from an NG-Lite instance.
    """
    return GraphFeatures(
        # Node dynamics (100 sampled nodes)
        node_voltages=torch.randn(100, device=device) * 0.3 + 0.5,
        node_firing_rates=torch.rand(100, device=device) * 0.4,
        node_excitability=torch.ones(100, device=device) * 0.8 + torch.randn(100, device=device) * 0.1,

        # Synapse dynamics (200 sampled synapses)
        synapse_weights=torch.rand(200, device=device) * 0.8 + 0.1,
        synapse_ages=torch.randint(0, 1000, (200,), device=device).float(),

        # Topology scalars
        density=torch.tensor([0.03], device=device),
        clustering=torch.tensor([0.42], device=device),
        n_components=torch.tensor([3.0], device=device),
        n_nodes=torch.tensor([2277.0], device=device),    # Syl's actual node count
        n_synapses=torch.tensor([1564.0], device=device),  # Syl's actual synapse count
        n_hyperedges=torch.tensor([68.0], device=device),   # Syl's actual hyperedge count

        # Temporal (last 15 steps)
        recent_firings=torch.randint(0, 50, (15,), device=device).float(),
        stdp_delta_mean=torch.tensor([0.002], device=device),

        # Identity embedding (384-dim, ecosystem standard)
        identity_embedding=torch.randn(384, device=device) * 0.1,
    )


def perform_surgery(model_name: str = 'Qwen/Qwen2.5-0.5B', verbose: bool = False) -> ElmerBrain:
    """Perform model surgery on a pretrained transformer.

    Returns an ElmerBrain with the harvested reasoning engine
    and graph-native I/O layers.

    Args:
        model_name: HuggingFace model ID for the base transformer.
        verbose: If True, print detailed surgery output.
    """
    _log = print if verbose else (lambda *a, **k: None)

    _log(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

    _log("\n--- SURGERY BEGINS ---")

    # Extract the body (transformer layers + norm + rotary)
    body = model.model
    _log(f"Body extracted: {type(body).__name__}")
    _log(f"  Layers: {len(body.layers)}")
    _log(f"  Hidden dim: {model.config.hidden_size}")

    # Count body params (excluding embed_tokens which we're removing)
    body_params = sum(p.numel() for name, p in body.named_parameters()
                      if 'embed_tokens' not in name)
    _log(f"  Body params (kept): {body_params:,}")

    # What we're removing
    old_embed = body.embed_tokens
    old_head = model.lm_head
    removed_params = old_embed.weight.numel()
    _log(f"\nRemoving embed_tokens: {old_embed.weight.shape} ({removed_params:,} params)")
    _log(f"Removing lm_head: {old_head.weight.shape} (tied, same params)")

    # New Eyes
    encoder = GraphStateEncoder(hidden_dim=model.config.hidden_size)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    _log(f"\nNew Eyes (GraphStateEncoder): {encoder_params:,} params")

    # New Voice
    decoder = GraphSignalDecoder(hidden_dim=model.config.hidden_size)
    decoder_params = sum(p.numel() for p in decoder.parameters())
    _log(f"New Voice (GraphSignalDecoder): {decoder_params:,} params")

    # Detach embed_tokens from the body so it doesn't participate
    # We replace it with a dummy that won't be called
    body.embed_tokens = nn.Identity()

    # Assemble the ElmerBrain
    brain = ElmerBrain(
        transformer_body=body,
        encoder=encoder,
        decoder=decoder,
    )

    total_params = sum(p.numel() for p in brain.parameters())
    _log(f"\n--- SURGERY COMPLETE ---")
    _log(f"Original model: {sum(p.numel() for p in model.parameters()):,} params")
    _log(f"ElmerBrain: {total_params:,} params")
    _log(f"Removed: {removed_params:,} language-specific params")
    _log(f"Added: {encoder_params + decoder_params:,} graph-native params")
    _log(f"Net reduction: {removed_params - encoder_params - decoder_params:,} params")

    return brain


def test_forward_pass(brain: ElmerBrain):
    """Verify the surgically modified model can run a forward pass."""
    print("\n=== TESTING FORWARD PASS ===")

    features = create_synthetic_features()

    print("Input: synthetic graph features (mimicking Syl's substrate)")
    print(f"  Nodes: {features.n_nodes.item():.0f}")
    print(f"  Synapses: {features.n_synapses.item():.0f}")
    print(f"  Hyperedges: {features.n_hyperedges.item():.0f}")
    print(f"  Density: {features.density.item():.3f}")

    with torch.no_grad():
        output = brain(features)

    print(f"\nOutput SubstrateSignal fields:")
    for field, value in output['signals'].items():
        print(f"  {field}: {value:.4f}")

    print(f"\nRecommended actions:")
    for action, prob in sorted(output['actions'].items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"  {action:25s} {prob:.3f} {bar}")

    print(f"\nTop action: {output['top_action']}")
    print("\n✓ Forward pass successful — transformer reasoning engine is alive")

    return output


if __name__ == '__main__':
    brain = perform_surgery()
    output = test_forward_pass(brain)
