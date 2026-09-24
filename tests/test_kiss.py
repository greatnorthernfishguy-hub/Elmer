"""Tests for KISS (core/kiss.py) — KISS-03 repair.

# ---- Changelog ----
# [2026-09-24] T3 Code / Claude Sonnet 5 (Tier-M) — first real KISS tests
#   What: Added the first Elmer test coverage for core.kiss. Prior to this file,
#         nothing in tests/ imported core.kiss, so the 2026-09-23 changelog's
#         "all 102 tests pass unchanged" claim exercised none of the changed code.
#   Why:  chief-003 kiss03 ruling ordered a repair lane with real tests (docs
#         f95963c1). Charter §3: "tests pass" only counts when the tests
#         actually exercise the changed path.
#   How:  (a) drive KISSFilter.filter() through every gate and assert on
#         kiss_mode/kiss_meta["reason"]; (b) a regression test for the removed
#         dangling _gate_graph_sparse_extract call, which fails on 79ae336;
#         (c) a step-by-step behavior-preservation comparison against the
#         pre-refactor a331de3 core/kiss.py, dynamically loaded via `git show`.
# -------------------
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.kiss import KISSFilter, KISSConfig, KISSStats

REPO_ROOT = Path(__file__).resolve().parent.parent
PRE_REFACTOR_REV = "a331de3"


def _snapshot(rng: np.random.Generator, base: dict | None = None,
              perturb_keys: tuple = (), perturb_amount: float = 0.3) -> dict:
    """Build a real numpy feature dict using the keys named in filter()'s docstring.

    If `base` is given, returns a copy with `perturb_keys` shifted by
    `perturb_amount` (and all other keys identical) — for constructing
    controlled small/large/no-op deltas relative to a previous snapshot.
    """
    if base is None:
        snap = {
            "node_features": rng.uniform(0.4, 0.6, size=8).astype(np.float32),
            "synapse_features": rng.uniform(0.4, 0.6, size=8).astype(np.float32),
            "topo_features": rng.uniform(0.4, 0.6, size=6).astype(np.float32),
            "temporal_features": rng.uniform(0.4, 0.6, size=4).astype(np.float32),
            "identity_embedding": rng.uniform(0.4, 0.6, size=8).astype(np.float32),
        }
    else:
        snap = {k: v.copy() for k, v in base.items()}
    for key in perturb_keys:
        snap[key] = snap[key] + perturb_amount
    return snap


ALL_KEYS = ("node_features", "synapse_features", "topo_features",
            "temporal_features", "identity_embedding")


# ---------------------------------------------------------------------------
# (a) Gate coverage — every gate, asserted on kiss_mode + kiss_meta["reason"]
# ---------------------------------------------------------------------------

class TestGateCoverage:
    def test_warmup_gate(self):
        """Gate 1: every message up to and including warmup_messages passes full."""
        rng = np.random.default_rng(1)
        kiss = KISSFilter(KISSConfig(warmup_messages=3))
        for i in range(1, 4):
            result = kiss.filter(_snapshot(rng))
            assert result["kiss_mode"] == "full"
            assert result["kiss_meta"]["reason"] == "warmup"
            assert result["kiss_meta"]["message_num"] == i
        assert kiss.stats.warmup_passed == 3

    def test_forced_refresh_gate(self):
        """Gate 2: forces a full snapshot once messages_since_full reaches force_full_every,
        even on the very first message (warmup disabled)."""
        rng = np.random.default_rng(2)
        kiss = KISSFilter(KISSConfig(warmup_messages=0, force_full_every=1))
        result = kiss.filter(_snapshot(rng))
        assert result["kiss_mode"] == "full"
        assert result["kiss_meta"]["reason"] == "forced_refresh"
        assert kiss.stats.forced_full == 1

    def test_delta_gate_skip(self):
        """Gate 3: a near-identical snapshot (cosine similarity >= delta_threshold)
        is skipped — filter() returns None and delta_skipped increments."""
        rng = np.random.default_rng(3)
        kiss = KISSFilter(KISSConfig(warmup_messages=1))
        snap0 = _snapshot(rng)
        kiss.filter(snap0)  # warmup
        assert kiss.stats.delta_skipped == 0

        result = kiss.filter(dict(snap0))  # identical -> similarity 1.0
        assert result is None
        assert kiss.stats.delta_skipped == 1

    def test_major_change_gate(self):
        """Gate 4 (sparse extract): when more than half the feature keys changed
        past sparse_min_delta, a full snapshot is sent with reason major_change."""
        rng = np.random.default_rng(4)
        kiss = KISSFilter(KISSConfig(warmup_messages=1))
        snap0 = _snapshot(rng)
        kiss.filter(snap0)  # warmup

        # 3 of 5 keys changed -> majority -> major_change
        snap_major = _snapshot(rng, base=snap0,
                                perturb_keys=("node_features", "synapse_features", "topo_features"))
        result = kiss.filter(snap_major)
        assert result["kiss_mode"] == "full"
        assert result["kiss_meta"]["reason"] == "major_change"
        assert result["kiss_meta"]["changed"] == 3
        assert result["kiss_meta"]["total"] == 5

    def test_sparse_change_gate(self):
        """Gate 4 (sparse extract): when a minority of feature keys changed,
        a sparse result is sent with reason sparse_change and changed_features."""
        rng = np.random.default_rng(5)
        kiss = KISSFilter(KISSConfig(warmup_messages=1))
        snap0 = _snapshot(rng)
        kiss.filter(snap0)  # warmup

        # 1 of 5 keys changed -> minority -> sparse_change
        snap_sparse = _snapshot(rng, base=snap0, perturb_keys=("node_features",))
        result = kiss.filter(snap_sparse)
        assert result["kiss_mode"] == "sparse"
        assert result["kiss_meta"]["reason"] == "sparse_change"
        assert result["kiss_meta"]["changed"] == 1
        assert result["kiss_meta"]["changed_features"] == ["node_features"]

    def test_first_message_fallback_reachable_with_warmup_disabled(self):
        """The `first_message` fallback at the tail of filter() is reachable, but
        only when warmup_messages=0 AND force_full_every is not yet due: warmup
        normally sets _last_raw on message 1, so by the time warmup stops firing
        _last_raw is already populated and the Gate 4 sparse-extract branch always
        fires instead of falling through. With warmup off, _last_raw is None on
        the very first call, so nothing before the fallback matches."""
        rng = np.random.default_rng(6)
        kiss = KISSFilter(KISSConfig(warmup_messages=0, force_full_every=1_000_000))
        result = kiss.filter(_snapshot(rng))
        assert result["kiss_mode"] == "full"
        assert result["kiss_meta"]["reason"] == "first_message"


# ---------------------------------------------------------------------------
# (b) Regression test for the removed dangling Gate 4 call — fails on 79ae336
# ---------------------------------------------------------------------------

class TestNoDanglingGateCall:
    def test_filter_does_not_raise_past_warmup(self):
        """Feeds more than warmup_messages distinct snapshots through filter().
        On 79ae336, the first post-warmup, non-skipped call raises AttributeError
        from the dangling self._gate_graph_sparse_extract(...) call at
        core/kiss.py:278 — this test fails on that commit and passes after the
        repair removes it."""
        rng = np.random.default_rng(11)
        kiss = KISSFilter(KISSConfig(warmup_messages=5, force_full_every=1000))

        results = []
        prev = None
        for _ in range(10):
            # Perturb 2 keys each round so the delta gate doesn't skip and
            # Gate 4 (sparse extract) is actually reached.
            snap = _snapshot(rng, base=prev, perturb_keys=("node_features", "topo_features")) \
                if prev is not None else _snapshot(rng)
            result = kiss.filter(snap)  # must not raise AttributeError
            results.append(result)
            prev = snap

        assert kiss.stats.total_received == 10
        # At least one post-warmup call must have actually reached the gate
        # that used to be dangling (sparse or major), proving the path ran.
        reasons = [r["kiss_meta"]["reason"] for r in results if r is not None]
        assert any(r in ("sparse_change", "major_change") for r in reasons)


# ---------------------------------------------------------------------------
# (c) Behavior preservation against the pre-refactor a331de3 base
# ---------------------------------------------------------------------------

def _load_pre_refactor_kiss_module():
    """Load a331de3's core/kiss.py as an isolated module via `git show`, without
    touching sys.modules['core.kiss'] (which must keep pointing at the current,
    repaired implementation for every other test in this file)."""
    src = subprocess.run(
        ["git", "show", f"{PRE_REFACTOR_REV}:core/kiss.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout

    tmp_dir = tempfile.mkdtemp(prefix="kiss_prerefactor_")
    tmp_path = Path(tmp_dir) / "kiss_prerefactor.py"
    tmp_path.write_text(src)

    spec = importlib.util.spec_from_file_location("kiss_prerefactor", tmp_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["kiss_prerefactor"] = module
    spec.loader.exec_module(module)
    return module


def _deterministic_sequence(rng: np.random.Generator, n: int = 34) -> list:
    """A fixed sequence covering repeats, small changes, large changes, and
    enough length to cross a forced refresh boundary under the shared config
    used by both filters in this test (force_full_every=10)."""
    seq = []
    prev = None
    for i in range(n):
        if i % 7 == 0 and prev is not None:
            snap = dict(prev)  # exact repeat -> should trigger delta skip
        elif i % 5 == 0:
            snap = _snapshot(rng, base=prev, perturb_keys=ALL_KEYS[:4], perturb_amount=0.35) \
                if prev is not None else _snapshot(rng)  # large change -> major_change
        elif i % 3 == 0 and prev is not None:
            snap = _snapshot(rng, base=prev, perturb_keys=("node_features",), perturb_amount=0.25)  # small/sparse change
        else:
            snap = _snapshot(rng, base=prev, perturb_keys=("synapse_features", "temporal_features"), perturb_amount=0.02) \
                if prev is not None else _snapshot(rng)  # tiny change -> likely delta skip
        seq.append(snap)
        prev = snap
    return seq


class TestBehaviorPreservation:
    def test_matches_pre_refactor_step_by_step(self):
        pre = _load_pre_refactor_kiss_module()

        shared_config_kwargs = dict(
            delta_threshold=0.99,
            sparse_min_delta=0.01,
            warmup_messages=5,
            force_full_every=10,
        )
        new_filter = KISSFilter(KISSConfig(**shared_config_kwargs))
        old_filter = pre.KISSFilter(pre.KISSConfig(**shared_config_kwargs))

        rng = np.random.default_rng(99)
        sequence = _deterministic_sequence(rng, n=34)

        reasons_seen = set()
        for i, snap in enumerate(sequence):
            new_result = new_filter.filter(dict(snap))
            old_result = old_filter.filter(dict(snap))

            assert new_result == old_result, f"step {i}: {new_result!r} != {old_result!r}"
            if new_result is not None:
                reasons_seen.add(new_result["kiss_meta"]["reason"])

        # Sanity: the deterministic sequence actually exercised a spread of gates
        # on both filters, not just warmup.
        assert {"warmup", "forced_refresh"} <= reasons_seen or "forced_refresh" in reasons_seen
        assert len(reasons_seen) >= 3

        assert new_filter.stats.to_dict() == old_filter.stats.to_dict()
