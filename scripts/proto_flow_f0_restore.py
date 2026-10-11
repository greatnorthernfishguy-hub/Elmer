"""
proto_flow_f0_restore.py — F0 of the Flow-Lenia design: restore protoUniBrain's body and initialise F1.

Run OFFLINE, isolated, on a COPY (never on the live ~/.elmer): HOME must point at a scratch home whose .elmer/
holds a copy of proto_weights.pt + proto_lenia_state.pt. It drives the real ProtoUniBrainSocket:

  1. load() the copy (encoder + DecoderAdapter are the CC's own, never stepped since 09-24)
  2. _reset_body_to_pretrained() — the existing restore, unchanged (body = Qwen2.5-0.5B pretrained)
  3. build the persisted probe of the CC's own encoder-shaped traffic: the encoder's idle sequence plus windows
     of the real River OUTCOME entries from a copied tract (--traffic), saved to <state>/probe_inputs.pt
  4. hunger-probe baselines on the restored body (UniAI lenia/flow_probe.py): vault / drive-domain LM loss,
     residual ratio, Tonic cos(out,in); token ids persisted to <state>/probe_tokens.pt
  5. engine F0 capture: usage0 + functional sketch over the probe, 48 graphs, mass = 2^32 per unit
  6. save_evolved_weights(): proto_weights.pt now carries 'flow_lenia_f1' with the body
The kill-switch file is NOT created (dormant by default; Josh enables explicitly).

usage (inside run-isolated.sh, UNIAI_LENIA_FLOW_F1=1; three separate processes, so no one process holds the
body and the borrowed LM at once):
  python proto_flow_f0_restore.py --stage restore --traffic <tract copy> --tokenizer <qwen tok dir> --out r.json
  python ~/UniAI/lenia/flow_probe.py --body ~/.elmer/proto_weights.pt --state-dir ~/.elmer/proto_flow_lenia \
      --pretrained <Qwen2.5-0.5B snapshot>
  python proto_flow_f0_restore.py --stage init --out i.json

# ---- Changelog ----
# [2026-10-11] Claude Code (Opus 5.5, lane flow-lenia-f1) — created
#   What: the F0 restore + F1 initialisation as an operator tool on a copy (design §8 F0, §9 R1).
#   Why:  docs superpowers/specs/2026-10-11-proto-flow-lenia-design.md: "Archive the current body;
#         _reset_body_to_pretrained(); capture usage0 on the restored body over a persisted probe of real CC
#         encoder traffic; record both probe baselines." Restoring is Josh's call; this only prepares the files.
#   How:  the socket's own load / _reset_body_to_pretrained / save_evolved_weights; UniAI's engine F0 capture
#         and flow_probe are loaded by path the way the socket loads engine.py.
# -------------------
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_outcomes(tract_path):
    import ng_tract
    data = open(tract_path, "rb").read()
    out = []
    for e in ng_tract.TractReader(data):
        if getattr(e, "entry_type", None) == ng_tract.ENTRY_OUTCOME and hasattr(e, "embedding_as_numpy"):
            v = e.embedding_as_numpy()
            if v is not None and len(v) > 0:
                out.append(v)
    return out


def probe_windows(n):
    """Windows over the n real entries, in deposit order: sizes 1, 2, 4, 8, 16, 32 with 50 % overlap."""
    wins = []
    for w in (1, 2, 4, 8, 16, 32):
        if w > n:
            break
        for i in range(0, n - w + 1, max(1, w // 2)):
            wins.append((i, i + w))
    return wins


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("restore", "init"), required=True,
                    help="restore: reset body + persist probe inputs/tokens + save; then run lenia/flow_probe.py on "
                         "the saved body (its own process) for the baselines; init: F0 capture with those baselines")
    ap.add_argument("--traffic", help="(restore) copy of a River tract with the CC's OUTCOME entries")
    ap.add_argument("--tokenizer", help="(restore) Qwen2.5 tokenizer dir")
    ap.add_argument("--docs", default=os.path.expanduser("~/docs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--idle-copies", type=int, default=8, help="idle-sequence forwards in the probe")
    a = ap.parse_args(argv)
    assert os.environ.get("UNIAI_LENIA_FLOW_F1") == "1", "run with UNIAI_LENIA_FLOW_F1=1"
    home = os.path.expanduser("~")
    assert home != "/home/josh", "refusing to run on the live HOME: point HOME at a scratch copy"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    torch.set_num_threads(1)
    from core.proto_brain_socket import ProtoUniBrainSocket

    t0 = time.time()
    rep = {"stage": a.stage}
    sock = ProtoUniBrainSocket()
    assert sock.load(sock._model_path), "socket load failed"
    eng = sock._lenia
    assert eng.flow_f1_active, "engine.py has no F1 or UNIAI_LENIA_FLOW_F1 is not set"
    flow = eng._flow
    rep["load_s"] = time.time() - t0
    sock._reset_body_to_pretrained()      # the existing restore (idempotent in stage init)
    os.makedirs(flow.cfg.state_dir, exist_ok=True)

    if a.stage == "restore":
        assert a.traffic and a.tokenizer, "--traffic and --tokenizer are required for --stage restore"
        probe_mod = _load_by_path("uniai_flow_probe", os.path.join(
            os.path.dirname(sys.modules["lenia_engine"].__file__), "flow_probe.py"))
        outcomes = read_outcomes(a.traffic)
        rep["traffic"] = a.traffic
        rep["traffic_outcomes"] = len(outcomes)
        inputs, kinds = [], []
        with torch.no_grad():
            idle = sock._brain.encoder(topology_delta=None, outcome_embeddings=None,
                                       autonomic_state="PARASYMPATHETIC")
            for _ in range(a.idle_copies):
                inputs.append(idle.clone())
                kinds.append("idle")
            for (i, j) in probe_windows(len(outcomes)):
                inputs.append(sock._brain.encoder(topology_delta=None, outcome_embeddings=outcomes[i:j],
                                                  autonomic_state="PARASYMPATHETIC"))
                kinds.append("outcome")
        torch.save({"inputs": inputs, "kinds": kinds, "traffic": a.traffic,
                    "windows": probe_windows(len(outcomes))}, os.path.join(flow.cfg.state_dir, "probe_inputs.pt"))
        rep["probe_inputs"] = {"n": len(inputs), "idle": kinds.count("idle"), "outcome": kinds.count("outcome"),
                               "tokens": int(sum(x.shape[1] for x in inputs))}
        torch.save(probe_mod.make_probe_tokens(a.tokenizer, a.docs),
                   os.path.join(flow.cfg.state_dir, "probe_tokens.pt"))
    else:
        pr = json.load(open(os.path.join(flow.cfg.state_dir, "probe_result.json")))
        assert pr.get("ok") and pr.get("flow_step") is None and pr.get("lenia_steps") is not None, \
            "probe_result.json must be the probe of the restored (stage restore) body"
        base = {k: pr[k] for k in ("vault_loss", "drive_loss", "residual_ratio", "tonic_cos")}
        base["probe_time"] = pr["time"]
        rep["baselines"] = base
        pi = torch.load(os.path.join(flow.cfg.state_dir, "probe_inputs.pt"), weights_only=False)
        rep["f0_capture"] = flow.f0_capture(pi["inputs"], base, pi["kinds"])
        flow.last_probe_seen = float(pr["time"])   # the F0 probe is the baseline, not a brake input
    sock.save_evolved_weights()
    rep["saved"] = {f: os.path.getsize(os.path.join(home, ".elmer", f)) for f in ("proto_weights.pt",
                                                                                   "proto_lenia_state.pt")}
    rep["kill_switch_file_exists"] = os.path.exists(os.path.join(flow.cfg.state_dir, "enabled"))
    rep["wall_s"] = time.time() - t0
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
