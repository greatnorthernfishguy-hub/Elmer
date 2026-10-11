"""
Tests for protoUniBrain's River feed (the Commons bucket), its ET_TRACTS_DIR-scoped deposits, and the
ELMER_WORKSPACE_DIR-scoped instance files (#1075 / #1079).

# ---- Changelog ----
# [2026-10-11] Claude Code (Opus 5.5, lane river-feed-fix) — created
#   What: _read_river_delta() buckets the Commons (watermark, oldest-first order, raw text embedded at the bucket,
#         embedding fallback, bounded); deposits use the canonical deposit_experience(raw, source, tract_paths)
#         shape under ET_TRACTS_DIR; instance_paths defaults are byte-identical when the variables are unset.
#   Why:  the River moved to the Commons (no tracts/neurograph/<peer>.tract writer exists); LAW 5 path scoping.
#   How:  fake `commons`, `ng_embed` and `ng_tract` modules in sys.modules; no torch, no model load.
# -------------------
"""

import os
import sys
import types

import numpy as np
import pytest

from core import instance_paths
from core.proto_brain_socket import ProtoUniBrainSocket, _RIVER_BUCKET_LIMIT


class _FakeCommons:
    """bucket_recent over a list of (target_id, ts, metadata, embedding), newest first, like commons.Commons."""

    def __init__(self):
        self.rows = []
        self.calls = []

    def add(self, tid, ts, meta, emb):
        self.rows.append((tid, ts, meta, emb))

    def bucket_recent(self, limit=50, since=0.0, with_metadata=False, with_embedding=False):
        self.calls.append({"limit": limit, "since": since, "with_embedding": with_embedding})
        out = []
        for tid, ts, meta, emb in sorted(self.rows, key=lambda r: r[1], reverse=True):
            if ts <= since:
                break
            out.append((tid, 0.5, f"recency@{ts:.3f}", meta, emb))
            if len(out) >= limit:
                break
        return out


@pytest.fixture
def river(monkeypatch):
    medium = _FakeCommons()
    commons_mod = types.ModuleType("commons")
    commons_mod.get_commons = lambda config=None: medium
    monkeypatch.setitem(sys.modules, "commons", commons_mod)
    embedded = []
    embed_mod = types.ModuleType("ng_embed")

    def embed(text):
        embedded.append(text)
        v = np.zeros(768, dtype=np.float32)
        v[len(text) % 768] = 1.0
        return v

    embed_mod.embed = embed
    monkeypatch.setitem(sys.modules, "ng_embed", embed_mod)
    return medium, embedded


def test_empty_river_is_idle(river):
    sock = ProtoUniBrainSocket()
    assert sock._read_river_delta() is None
    assert sock._river_since == 0.0


def test_new_deposits_reach_encoder_oldest_first_and_watermark_advances(river):
    medium, embedded = river
    medium.add("cc:experience:a", 100.0, {"text": "first"}, None)
    medium.add("cc:experience:b", 200.0, {"text": "second turn"}, None)
    sock = ProtoUniBrainSocket()
    d = sock._read_river_delta()
    assert d["topology"] is None
    assert len(d["outcomes"]) == 2
    assert embedded == ["first", "second turn"]          # oldest first, raw text embedded at the bucket
    assert all(v.shape == (768,) and v.dtype == np.float32 for v in d["outcomes"])
    assert sock._river_since == pytest.approx(200.0)
    assert medium.calls[-1]["with_embedding"] is True
    # nothing new -> idle; the watermark is passed to the bucket
    assert sock._read_river_delta() is None
    assert medium.calls[-1]["since"] == pytest.approx(200.0)
    medium.add("cc:experience:c", 300.0, {"text": "third"}, None)
    d = sock._read_river_delta()
    assert len(d["outcomes"]) == 1 and embedded[-1] == "third"
    assert (sock._river_reads, sock._river_entries) == (2, 3)


def test_textless_deposit_uses_its_embedding_and_experience_halves_are_joined(river):
    medium, embedded = river
    emb = np.arange(768, dtype=np.float64)
    medium.add("topology:n1", 10.0, {"kind": "topology_delta", "node_id": "n1"}, emb)
    medium.add("experience:x", 20.0, {"kind": "experience", "user_text": "hi", "assistant_text": "hello"}, None)
    medium.add("metrics:y", 30.0, {}, None)                # no text, no embedding -> skipped
    sock = ProtoUniBrainSocket()
    d = sock._read_river_delta()
    assert len(d["outcomes"]) == 2
    np.testing.assert_array_equal(d["outcomes"][0], emb.astype(np.float32))
    assert embedded == ["hi\n\nhello"]
    assert sock._river_since == pytest.approx(30.0)


def test_bucket_is_bounded(river):
    medium, _ = river
    for i in range(_RIVER_BUCKET_LIMIT + 10):
        medium.add(f"cc:experience:{i}", float(i + 1), {"text": f"t{i}"}, None)
    sock = ProtoUniBrainSocket()
    d = sock._read_river_delta()
    assert len(d["outcomes"]) == _RIVER_BUCKET_LIMIT
    assert medium.calls[-1]["limit"] == _RIVER_BUCKET_LIMIT


def test_river_read_failure_is_soft(monkeypatch):
    commons_mod = types.ModuleType("commons")

    def boom(config=None):
        raise RuntimeError("no medium")

    commons_mod.get_commons = boom
    monkeypatch.setitem(sys.modules, "commons", commons_mod)
    assert ProtoUniBrainSocket()._read_river_delta() is None


def test_tracts_root_follows_et_tracts_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("ET_TRACTS_DIR", raising=False)
    assert instance_paths.tracts_root() == os.path.expanduser("~/.et_modules/tracts")
    monkeypatch.setenv("ET_TRACTS_DIR", str(tmp_path / "tracts"))
    assert instance_paths.tracts_root() == str(tmp_path / "tracts")


def test_instance_file_default_unchanged_and_scoped_when_set(monkeypatch, tmp_path):
    monkeypatch.delenv("ELMER_WORKSPACE_DIR", raising=False)
    assert instance_paths.instance_file("~/.elmer/competence_delta.jsonl", "competence_delta.jsonl") == \
        os.path.expanduser("~/.elmer/competence_delta.jsonl")
    monkeypatch.setenv("ELMER_WORKSPACE_DIR", str(tmp_path / "ws"))
    assert instance_paths.instance_file("~/.elmer/competence_delta.jsonl", "competence_delta.jsonl") == \
        str(tmp_path / "ws" / "competence_delta.jsonl")


def test_brain_switcher_lock_path_scoping(monkeypatch, tmp_path):
    from core.brain_switcher import BrainSwitcher
    monkeypatch.delenv("ELMER_WORKSPACE_DIR", raising=False)
    get = BrainSwitcher._get_lock_file_path
    assert get(None) == os.path.expanduser("~/.et_modules/elmer/proto_body.lock")
    monkeypatch.setenv("ELMER_WORKSPACE_DIR", str(tmp_path))
    assert get(None) == str(tmp_path / "proto_body.lock")


def test_latent_deposit_uses_canonical_shape_under_et_tracts_dir(monkeypatch, tmp_path):
    calls = []
    nt = types.ModuleType("ng_tract")

    def deposit_experience(raw, source, tract_paths):   # the canonical wheel's positional shape
        calls.append((raw, source, tract_paths))

    nt.deposit_experience = deposit_experience
    monkeypatch.setitem(sys.modules, "ng_tract", nt)
    monkeypatch.setenv("ET_TRACTS_DIR", str(tmp_path))

    class _H:
        def squeeze(self, _):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((2, 3), dtype=np.float32)

    sock = ProtoUniBrainSocket()
    sock._deposit_to_river(_H())
    assert len(calls) == 1
    raw, source, paths = calls[0]
    assert raw == np.ones((2, 3), dtype=np.float32).tobytes()
    assert source == "proto_unibrain"
    assert paths == [str(tmp_path / "proto_unibrain" / "latent.tract")]


def test_latent_deposit_failure_warns_once(monkeypatch, tmp_path, caplog):
    nt = types.ModuleType("ng_tract")

    def deposit_experience(raw, source, tract_paths):
        raise TypeError("bad shape")

    nt.deposit_experience = deposit_experience
    monkeypatch.setitem(sys.modules, "ng_tract", nt)
    monkeypatch.setenv("ET_TRACTS_DIR", str(tmp_path))

    class _H:
        def squeeze(self, _):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones(1, dtype=np.float32)

    sock = ProtoUniBrainSocket()
    with caplog.at_level("DEBUG", logger="elmer.proto_brain"):
        sock._deposit_to_river(_H())
        sock._deposit_to_river(_H())
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1
