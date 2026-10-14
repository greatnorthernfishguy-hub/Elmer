# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane mean-centre) — new file: #1108 identity 768 / #1110 fail-loud embed
# What: the GraphStateEncoder takes the identity vector at IDENTITY_DIM (= ng_embed.EMBEDDING_DIM) and refuses other
#   widths; a pre-#1108 checkpoint (384-wide identity_proj) migrates on load with outputs bit-identical to the old
#   truncating encoder; GraphEncoder._embed fails loud instead of returning a zero vector.
# Why: #1108 (Tonic / ElmerBrain identity truncated to 384), #1110 (zero-vector embed fallback).
# How: build a legacy encoder in-test (the pre-#1108 forward, a 384-wide identity_proj), save its state, load it into
#   the current encoder, compare forwards.
# -------------------
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "surgery"))
sys.path.insert(0, ROOT)

import graph_io  # noqa: E402
from graph_io import GraphFeatures, GraphStateEncoder, IDENTITY_DIM, LEGACY_IDENTITY_DIM  # noqa: E402


def _features(identity):
    g = torch.Generator().manual_seed(0)
    r = lambda n: torch.rand(n, generator=g)  # noqa: E731
    return GraphFeatures(
        node_voltages=r(20), node_firing_rates=r(20), node_excitability=r(20),
        synapse_weights=r(30), synapse_ages=r(30),
        density=torch.tensor([0.1]), clustering=torch.tensor([0.2]), n_components=torch.tensor([3.0]),
        n_nodes=torch.tensor([20.0]), n_synapses=torch.tensor([30.0]), n_hyperedges=torch.tensor([2.0]),
        recent_firings=r(15), stdp_delta_mean=torch.tensor([0.01]), identity_embedding=identity,
    )


class _LegacyEncoder(GraphStateEncoder):
    """The pre-#1108 encoder: a 384-wide identity_proj fed `_pad_or_truncate(identity, 384)`."""

    def __init__(self, hidden_dim=64):
        super().__init__(hidden_dim=hidden_dim)
        self.identity_proj = torch.nn.Linear(LEGACY_IDENTITY_DIM, hidden_dim)
        self._load_state_dict_pre_hooks.clear()

    @staticmethod
    def _check_identity(t):
        t = t.reshape(-1)
        if len(t) >= LEGACY_IDENTITY_DIM:
            return t[:LEGACY_IDENTITY_DIM]
        return torch.cat([t, torch.zeros(LEGACY_IDENTITY_DIM - len(t))])


def test_identity_dim_is_the_embedding_models():
    import ng_embed
    assert IDENTITY_DIM == ng_embed.EMBEDDING_DIM == 768
    assert GraphStateEncoder(hidden_dim=32).identity_proj.in_features == IDENTITY_DIM


def test_a_384_wide_checkpoint_migrates_with_identical_outputs(caplog):
    torch.manual_seed(1)
    legacy = _LegacyEncoder(hidden_dim=64).eval()
    state = {k: v.clone() for k, v in legacy.state_dict().items()}
    assert state["identity_proj.weight"].shape == (64, LEGACY_IDENTITY_DIM)
    enc = GraphStateEncoder(hidden_dim=64).eval()
    with caplog.at_level("WARNING", logger="elmer.graph_io"):
        enc.load_state_dict(state)  # strict: the pre-hook makes the shapes fit
    assert any("migrated a 384-wide identity_proj" in r.getMessage() for r in caplog.records)
    w = enc.identity_proj.weight
    assert w.shape == (64, IDENTITY_DIM)
    assert torch.equal(w[:, :LEGACY_IDENTITY_DIM], state["identity_proj.weight"])
    assert torch.count_nonzero(w[:, LEGACY_IDENTITY_DIM:]) == 0
    ident = torch.randn(IDENTITY_DIM)
    with torch.no_grad():
        a = legacy(_features(ident))   # old: truncates the 768-d vector to 384
        b = enc(_features(ident))      # new: takes all 768, columns 384.. are zero
    assert torch.equal(a, b), "migration must keep the outputs bit-identical"
    # and a migrated encoder saves at full width (the migration is permanent)
    assert enc.state_dict()["identity_proj.weight"].shape == (64, IDENTITY_DIM)


def test_identity_of_another_width_is_refused_not_truncated():
    enc = GraphStateEncoder(hidden_dim=32).eval()
    with torch.no_grad():
        enc(_features(torch.randn(IDENTITY_DIM)))
        enc(_features(torch.zeros(0)))  # absent identity = zeros, as before
        with pytest.raises(ValueError, match="not padded or truncated"):
            enc(_features(torch.randn(LEGACY_IDENTITY_DIM)))
        with pytest.raises(ValueError):
            enc(_features(torch.randn(1024)))


def test_graph_encoder_embed_fails_loud(monkeypatch, caplog):
    import ng_embed
    from runtime.graph_encoder import GraphEncoder

    def boom(*a, **k):
        raise ng_embed.EmbeddingUnavailableError("embedding model unavailable")

    monkeypatch.setattr(ng_embed, "embed", boom)
    with caplog.at_level("ERROR", logger="elmer.graph_encoder"):
        with pytest.raises(ng_embed.EmbeddingUnavailableError):
            GraphEncoder()._embed("anything")
    assert any("embedding failed" in r.getMessage() for r in caplog.records)
