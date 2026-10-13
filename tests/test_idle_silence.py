"""
Tests for E1 (2026-10-12): protoUniBrain's idle cycles are silence for Flow-Lenia F1.

# ---- Changelog ----
# [2026-10-12] Claude Code (Opus 5.5, lane f1-idle-silence; Josh 2026-10-12 "E1 for now") — created
#   What: process() opens the engine's capture_context(traffic=...) with traffic=False when the River delivered
#         nothing this cycle (the encoder's idle filler) and traffic=True when it delivered entries; an engine
#         whose capture_context() has no traffic keyword still gets its old call (every own forward counts).
#   Why:  design 2026-10-11-proto-flow-lenia-design.md §4.4 ("With no traffic, rho relaxes toward 1 ... wake
#         silence"); drift analysis 2026-10-12 C8 / E1.
#   How:  fake `commons` / `ng_embed` modules (as test_river_feed.py), a fake brain and a recording fake engine;
#         torch only for the hidden-state tensors process() summarises. No model load.
# -------------------
"""

import contextlib
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import core.proto_brain_socket as pbs  # noqa: E402
from core.base_socket import GraphSnapshot  # noqa: E402
from tests.test_river_feed import _FakeCommons  # noqa: E402


@pytest.fixture
def medium(monkeypatch):
    m = _FakeCommons()
    commons_mod = types.ModuleType("commons")
    commons_mod.get_commons = lambda config=None: m
    monkeypatch.setitem(sys.modules, "commons", commons_mod)
    embed_mod = types.ModuleType("ng_embed")
    embed_mod.embed = lambda text: np.ones(768, dtype=np.float32)
    monkeypatch.setitem(sys.modules, "ng_embed", embed_mod)
    monkeypatch.setattr(pbs, "_torch", torch)
    return m


class _Brain:
    def __init__(self):
        self.seen = []

        def encoder(topology_delta=None, outcome_embeddings=None, autonomic_state=None):
            self.seen.append(0 if outcome_embeddings is None else len(outcome_embeddings))
            return torch.zeros(1, 3, 8)

        def body(**kw):
            h = torch.linspace(0, 1, 24).reshape(1, 3, 8)
            return types.SimpleNamespace(last_hidden_state=h, hidden_states=(h, h))

        self.encoder = encoder
        self.transformer_body = body


class _Engine:
    flow_f1_active = True

    def __init__(self):
        self.captures = []

    @contextlib.contextmanager
    def capture_context(self, traffic=True):
        self.captures.append(traffic)
        yield

    def step(self, arousal=None):
        return {"total_delta_norm": 0.0, "time_ms": 0.0}

    def drain_flow_events(self):
        return []


class _OldEngine(_Engine):
    @contextlib.contextmanager
    def capture_context(self):              # engine.py before 2026-10-12: no traffic keyword
        self.captures.append("old")
        yield


def _socket(engine):
    s = pbs.ProtoUniBrainSocket()
    s._loaded = True
    s._brain = _Brain()
    s._lenia = engine
    s._splat_adapter = None
    s._check_adapter_pressure = lambda h: None
    s._deposit_to_river = lambda h: None
    s.save_evolved_weights = lambda: None
    s.log_weight_stats = lambda: None
    return s


def test_idle_cycle_is_captured_as_silence_and_river_cycle_as_traffic(medium):
    eng = _Engine()
    s = _socket(eng)
    snap = GraphSnapshot.empty()
    s.process(snap, {})                                   # empty River: the encoder's idle filler
    medium.add("cc:experience:a", 100.0, {"text": "a real turn"}, None)
    s.process(snap, {})                                   # one River entry
    s.process(snap, {})                                   # nothing new again
    assert s._brain.seen == [0, 1, 0]
    assert eng.captures == [False, True, False]
    assert s._error_count == 0


def test_engine_without_traffic_keyword_keeps_the_old_call(medium):
    eng = _OldEngine()
    s = _socket(eng)
    s.process(GraphSnapshot.empty(), {})
    assert eng.captures == ["old"]
    assert s._error_count == 0
