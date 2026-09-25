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
#
# [2026-09-24] T3 Code / Claude Sonnet 5 (Tier-M) — Revision 1: differential
#   coverage gap + tautological assertion + numpy-identity masking, all fixed
#   What: (a) replaced the pseudo-random 34-step differential sequence, which
#         (per the zone manager's instrumentation) hit sparse_change and
#         first_message zero times, with a fixed-point sequence that reaches
#         every reason the main config can produce (warmup, sparse_change —
#         including a run crossing the force_full_every boundary —,
#         forced_refresh, major_change, delta-skip) plus a second config/
#         sequence dedicated to first_message; (b) replaced the tautological
#         sanity assertion (`{"warmup","forced_refresh"} <= reasons_seen or
#         "forced_refresh" in reasons_seen`, which reduces to just the
#         right-hand clause) with an exact-set assertion; (c) replaced
#         `new_result == old_result` with explicit kiss_mode/kiss_meta
#         equality plus per-key np.array_equal on snapshot, and gave each
#         filter its own independent deep copy of every input snapshot.
#   Why:  the zone manager's revision request identified that (1) the exact
#         code path the refactor moved — the sparse branch of
#         _gate_sparse_extract — was never behaviorally checked against
#         a331de3, letting a real regression (adding a stray
#         `self._messages_since_full = 0` to that branch) pass unnoticed;
#         (2) plain dict/list equality on a dict containing numpy arrays
#         short-circuits on object identity (PyObject_RichCompareBool)
#         before calling ndarray.__eq__ elementwise, so two filters handed
#         shared array objects could "match" without ever comparing array
#         contents. Charter §3: name the test that runs each changed line —
#         a green suite that never reaches a line doesn't count.
#   How:  see _main_sequence, _first_message_snapshot, and
#         _assert_results_equal below. Verified empirically (scratch run,
#         not committed) that the ZM's proposed mutation — inserting
#         `self._messages_since_full = 0` after `self.stats.sparse_passed
#         += 1` in the sparse branch of _gate_sparse_extract — makes
#         test_matches_pre_refactor_main_sequence fail at the step where the
#         forced-refresh boundary is crossed (new filter reports
#         sparse_change where old reports forced_refresh), then restored
#         core/kiss.py and confirmed byte-identical via md5sum.
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


def _shift(snap: dict, keys: tuple, amount: float) -> dict:
    """Independent copy of `snap` with `keys` shifted by a fixed amount."""
    out = {k: v.copy() for k, v in snap.items()}
    for key in keys:
        out[key] = out[key] + amount
    return out


def _deep(snap: dict) -> dict:
    """An independent deep copy — every array is a distinct object, so a
    filter mutating or the two filters returning references to the SAME
    underlying array can never masquerade as agreement (see Revision 1)."""
    return {k: v.copy() for k, v in snap.items()}


def _main_sequence(rng: np.random.Generator) -> list:
    """Fixed-point (non-cumulative) sequence for the main config
    (warmup_messages=5, force_full_every=8). Alternates between a small
    number of pre-built reference snapshots rather than perturbing forward
    from the previous step, so magnitudes don't drift with sequence length
    (see Revision 1 changelog note on cumulative-perturbation drift).

    Deliberately reaches every reason the main config can produce, AND
    crosses a force_full_every boundary twice — once before major_change
    (to reach forced_refresh at all) and once after it (to prove
    major_change's own `_messages_since_full = 0` reset actually happened:
    with the reset removed, the second boundary is crossed one step early,
    which this sequence's step-by-step comparison catches — see Revision 1
    mutation table, row 8):
      - warmup            x5  (the 5 warmup_messages)
      - sparse_change      (two runs of alternating single-key changes,
                             one before and one after major_change, each
                             crossing a force_full_every=8 boundary)
      - forced_refresh     (both boundary crossings)
      - major_change        (a majority of keys changed at once)
      - delta-skip          (an exact repeat of the previous snapshot)
    `first_message` is NOT reachable here — see `_first_message_snapshot`.
    """
    A = _snapshot(rng)
    B = _shift(A, ("node_features",), 0.4)
    C = _shift(A, ("node_features", "synapse_features", "topo_features"), 0.4)  # major vs A
    E = _shift(C, ("temporal_features",), 0.4)                                  # sparse vs C

    warmup_fillers = [_snapshot(rng) for _ in range(4)]
    seq = warmup_fillers + [A]                       # 5 warmup messages
    seq += [B, A, B, A, B, A, B, A]                   # 8 alternating sparse steps,
    #                                                    the 8th crosses force_full_every=8
    #                                                    -> forced_refresh
    seq += [C]                                        # major_change vs A
    seq += [C, E, C, E, C, E, C, E]                   # exact repeat (skip), then 7 more
    #                                                    alternating sparse steps; the 8th
    #                                                    post-major-change message crosses
    #                                                    force_full_every=8 again -> forced_refresh
    #                                                    (only IF major_change's own reset fired)
    return seq


def _first_message_snapshot(rng: np.random.Generator) -> dict:
    """A single snapshot to be fed to a filter built with warmup_messages=0.
    With warmup off, `_last_raw` is None on the very first call, so nothing
    before the tail-of-filter() fallback can match — this is the only way
    to reach the `first_message` reason (see TestGateCoverage)."""
    return _snapshot(rng)


MAIN_CONFIG_KWARGS = dict(
    delta_threshold=0.99,
    sparse_min_delta=0.01,
    warmup_messages=5,
    force_full_every=8,
)
FIRST_MESSAGE_CONFIG_KWARGS = dict(
    delta_threshold=0.99,
    sparse_min_delta=0.01,
    warmup_messages=0,
    force_full_every=1_000_000,
)


def _assert_results_equal(new_result, old_result, step):
    """Compare two filter() outputs field by field. `kiss_mode` and
    `kiss_meta` are compared with plain equality (safe — no numpy arrays
    inside kiss_meta). `snapshot` is compared key by key with
    np.array_equal, because dict/plain equality on a dict containing numpy
    arrays short-circuits on object identity (PyObject_RichCompareBool)
    before ever calling ndarray.__eq__ elementwise — a real divergence in
    array contents could pass a `==` comparison undetected if the two
    filters happened to be handed shared array objects. Callers must pass
    each filter its own independently-copied input (see `_deep`) so that
    identity can never do that masking here either."""
    assert (new_result is None) == (old_result is None), \
        f"step {step}: one filter skipped and the other didn't (new={new_result!r}, old={old_result!r})"
    if new_result is None:
        return
    assert new_result["kiss_mode"] == old_result["kiss_mode"], \
        f"step {step}: kiss_mode differs: {new_result['kiss_mode']!r} != {old_result['kiss_mode']!r}"
    assert new_result["kiss_meta"] == old_result["kiss_meta"], \
        f"step {step}: kiss_meta differs: {new_result['kiss_meta']!r} != {old_result['kiss_meta']!r}"
    assert new_result["snapshot"].keys() == old_result["snapshot"].keys(), \
        f"step {step}: snapshot key sets differ"
    for key in new_result["snapshot"]:
        assert np.array_equal(new_result["snapshot"][key], old_result["snapshot"][key]), \
            f"step {step}: snapshot[{key!r}] differs between new and old"


class TestBehaviorPreservation:
    def test_matches_pre_refactor_main_sequence(self):
        """Runs the fixed-point main-config sequence through both the
        current KISSFilter and the dynamically-loaded a331de3 module,
        comparing every step and the final stats. Deliberately reaches
        warmup, sparse_change (including a run crossing the
        force_full_every boundary), forced_refresh, major_change, and a
        delta-skip — the exact reason set the original Revision 0 sequence
        missed (it hit sparse_change and first_message zero times, per the
        ZM's instrumentation)."""
        pre = _load_pre_refactor_kiss_module()
        new_filter = KISSFilter(KISSConfig(**MAIN_CONFIG_KWARGS))
        old_filter = pre.KISSFilter(pre.KISSConfig(**MAIN_CONFIG_KWARGS))

        rng = np.random.default_rng(2024)
        sequence = _main_sequence(rng)
        assert len(sequence) >= 15

        reasons_seen = set()
        skip_count = 0
        for i, snap in enumerate(sequence):
            # Each filter gets its OWN deep copy — object identity between
            # the two filters' inputs can never mask a real divergence.
            new_result = new_filter.filter(_deep(snap))
            old_result = old_filter.filter(_deep(snap))

            _assert_results_equal(new_result, old_result, i)
            if new_result is None:
                skip_count += 1
            else:
                reasons_seen.add(new_result["kiss_meta"]["reason"])

        # Exact-set assertion — replaces the Revision 0 tautological check
        # (`{"warmup","forced_refresh"} <= reasons_seen or "forced_refresh"
        # in reasons_seen`, which reduces to just the right-hand clause).
        assert reasons_seen == {"warmup", "sparse_change", "forced_refresh", "major_change"}
        assert skip_count == 1  # the deliberate exact-repeat step

        assert new_filter.stats.to_dict() == old_filter.stats.to_dict()

    def test_matches_pre_refactor_first_message(self):
        """A second config (warmup_messages=0) reaches the `first_message`
        fallback the main sequence above cannot reach (see
        TestGateCoverage.test_first_message_fallback_reachable_with_warmup_disabled).
        Revision 0's differential sequence hit this reason zero times."""
        pre = _load_pre_refactor_kiss_module()
        new_filter = KISSFilter(KISSConfig(**FIRST_MESSAGE_CONFIG_KWARGS))
        old_filter = pre.KISSFilter(pre.KISSConfig(**FIRST_MESSAGE_CONFIG_KWARGS))

        rng = np.random.default_rng(555)
        snap = _first_message_snapshot(rng)

        new_result = new_filter.filter(_deep(snap))
        old_result = old_filter.filter(_deep(snap))

        _assert_results_equal(new_result, old_result, 0)
        assert new_result["kiss_meta"]["reason"] == "first_message"
        assert new_filter.stats.to_dict() == old_filter.stats.to_dict()
