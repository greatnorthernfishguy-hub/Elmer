"""
Tests for atomic saves (2026-10-14): an interrupted save never leaves a torn proto_weights.pt.

# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane atomic-saves; Josh 2026-10-14 "anything that can't risk a failed save,
#   be atomic") — created
#   What: (1) core.atomic_io: a failing write_fn leaves the previous file byte-identical and no temp behind; the
#             final file keeps the old permission bits; stale temps of dead writers are reported, never deleted.
#         (2) SIGKILL proof: a forked child process saves in a loop through the REAL ProtoUniBrainSocket.save_evolved_weights
#             (paths pointed at tmp_path); the parent SIGKILLs it at random points, repeatedly. After every kill
#             both files load with torch.load and hold one complete save (body version == lenia version or one
#             behind). A control child doing the old in-place torch.save tears the file under the same kills,
#             so the harness can see a torn write.
#   Why:  #1095: a daemon restart caught torch.save of ~/.elmer/proto_weights.pt mid-write; ~18 h of state lost.
#   How:  os.fork of the test process (torch already imported) + os.kill(SIGKILL); ~8 MB tensors.
# -------------------
"""

import os
import random
import signal
import stat
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")

from core import atomic_io  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_failed_write_leaves_previous_file(tmp_path):
    p = tmp_path / "state.pt"
    atomic_io.torch_save_atomic({"v": 1}, str(p), torch)
    before = p.read_bytes()

    def boom(tmp):
        with open(tmp, "wb") as f:
            f.write(b"half a file")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        atomic_io.atomic_write(str(p), boom)
    assert p.read_bytes() == before
    assert torch.load(str(p), weights_only=False) == {"v": 1}
    assert [n for n in os.listdir(tmp_path) if ".tmp-" in n] == []


def test_complete_write_replaces_and_keeps_mode(tmp_path):
    p = tmp_path / "state.pt"
    atomic_io.torch_save_atomic({"v": 1}, str(p), torch)
    os.chmod(p, 0o640)
    atomic_io.torch_save_atomic({"v": 2}, str(p), torch)
    assert torch.load(str(p), weights_only=False) == {"v": 2}
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o640


def test_stale_temps_reported_not_deleted(tmp_path):
    p = tmp_path / "state.pt"
    atomic_io.atomic_write_bytes(str(p), b"x")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = tmp_path / f"state.pt.tmp-{dead.pid}-deadbeef"
    stale.write_bytes(b"torn")
    live = tmp_path / f"state.pt.tmp-{os.getpid()}-cafecafe"
    live.write_bytes(b"mine")
    found = atomic_io.find_stale_temps(str(p))
    assert found == [str(stale)]
    atomic_io.warn_stale_temps(str(p))
    assert stale.exists() and live.exists()


class _Brain:
    def __init__(self):
        self.w = {"layer%d" % i: torch.randn(256, 1024) for i in range(8)}   # ~8 MB
    def state_dict(self):
        return self.w


class _LeniaState:
    step_count = 0
    weight_deltas = {}


class _Lenia:
    flow_f1_active = False
    def __init__(self):
        self.state = _LeniaState()
        self._initial_norms = {}
        self._initial_mean_abs = {}
    def flow_state_dict(self):
        return {"counters": {"flow_steps": self.state.step_count}, "mass": torch.ones(64)}


def _child_save_loop(out, mode, ready_fd):
    """Runs in a forked child: the REAL save_evolved_weights, in a loop, until SIGKILLed."""
    import core.proto_brain_socket as pbs
    pbs._torch = torch
    pbs._PROTO_WEIGHTS_PATH = os.path.join(out, "proto_weights.pt")
    pbs._PROTO_LENIA_STATE_PATH = os.path.join(out, "proto_lenia_state.pt")
    if mode == "inplace":   # the pre-fix writer, for the control
        pbs.torch_save_atomic = lambda obj, path, t=None: torch.save(obj, path)
    s = pbs.ProtoUniBrainSocket.__new__(pbs.ProtoUniBrainSocket)
    s._brain = _Brain(); s._lenia = _Lenia(); s._process_count = 0
    os.write(ready_fd, b"r")                          # the parent starts its random kill timer now
    v = int(time.time() * 1000) % 10_000_000          # strictly increasing across rounds
    while True:
        v += 1
        s._lenia_steps = v
        s._lenia.state.step_count = v
        s._brain.w["layer0"][0, 0] = float(v)
        s.save_evolved_weights()


def _verify(out):
    w = os.path.join(out, "proto_weights.pt")
    ls = os.path.join(out, "proto_lenia_state.pt")
    try:
        saved = torch.load(w, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        return None                                      # killed before the first save finished: nothing to tear
    except Exception:
        return False
    try:
        body_v = saved["lenia_steps"]
        assert float(saved["model_state_dict"]["layer0"][0, 0]) == float(body_v)
        assert saved["flow_lenia_f1"]["counters"]["flow_steps"] == body_v
        if os.path.exists(ls):
            lstate = torch.load(ls, map_location="cpu", weights_only=False)
            assert lstate["step_count"] <= body_v       # the baselines are this save's or an earlier one's
    except Exception:
        return False
    return True


def _run_kills(tmp_path, mode, kills, seed):
    import core.proto_brain_socket  # noqa: F401  (import once in the parent; forks inherit it)
    rng = random.Random(seed)
    out = tmp_path / ("out-" + mode)
    out.mkdir()
    results = []
    import io
    torch.save(_Brain().state_dict(), io.BytesIO())      # warm torch's lazy init in the parent, before forking
    for _ in range(kills):
        r_fd, w_fd = os.pipe()
        pid = os.fork()
        if pid == 0:                                     # child
            try:
                os.close(r_fd)
                _child_save_loop(str(out), mode, w_fd)
            finally:
                os._exit(3)
        os.close(w_fd)
        assert os.read(r_fd, 1) == b"r"
        os.close(r_fd)
        time.sleep(rng.uniform(0.05, 0.8))
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        temp_left = any(".tmp-" in n for n in os.listdir(out))
        results.append((_verify(str(out)), temp_left))
        for n in os.listdir(out):      # the next round starts like a restart: stale temps don't matter
            if ".tmp-" in n:
                os.remove(out / n)
    return results


def test_sigkill_during_proto_save_never_tears(tmp_path):
    res = _run_kills(tmp_path, "atomic", kills=25, seed=1095)
    assert all(ok is not False for ok, _ in res), res
    assert sum(1 for ok, _ in res if ok) >= 15, res          # most rounds had a previous good file to protect
    assert sum(1 for _, t in res if t) >= 5, res             # and many kills landed mid-write (a temp was left)


def test_control_inplace_save_tears_under_same_kills(tmp_path):
    res = _run_kills(tmp_path, "inplace", kills=25, seed=1095)
    assert any(ok is False for ok, _ in res), res            # the harness does see torn files


def test_failed_restore_sets_files_aside(tmp_path, monkeypatch):
    import core.proto_brain_socket as pbs
    w = tmp_path / "proto_weights.pt"
    ls = tmp_path / "proto_lenia_state.pt"
    w.write_bytes(b"PK\x03\x04 torn")                    # unloadable
    torch.save({"step_count": 1}, str(ls))
    monkeypatch.setattr(pbs, "_torch", torch)
    monkeypatch.setattr(pbs, "_PROTO_WEIGHTS_PATH", str(w))
    monkeypatch.setattr(pbs, "_PROTO_LENIA_STATE_PATH", str(ls))
    s = pbs.ProtoUniBrainSocket.__new__(pbs.ProtoUniBrainSocket)
    s._brain = _Brain(); s._lenia = _Lenia()
    assert s._restore_evolved_weights() is False
    names = sorted(os.listdir(tmp_path))
    assert not w.exists() and not ls.exists()
    assert any(n.startswith("proto_weights.pt.unloadable-") for n in names), names
    assert any(n.startswith("proto_lenia_state.pt.unloadable-") for n in names), names
