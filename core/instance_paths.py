"""
instance_paths — where THIS Elmer instance keeps its tracts and its per-instance runtime files.

Two scopes, each from the environment variable that already owns it (LAW 5):

  - tracts_root()        $ET_TRACTS_DIR, else ~/.et_modules/tracts
                         The same resolution the vendored ng_tract_bridge.py (NGTractBridge.__init__) and
                         openclaw_adapter.py (tract-only deposit) already use, so every tract Elmer writes lands
                         in the one tracts tree its host configured. The CC daemon runs with
                         ET_TRACTS_DIR=~/.claude/plugins/neurograph/tracts (~/.bashrc, the #302 isolation).
  - instance_file(default, name)
                         $ELMER_WORKSPACE_DIR/<name> when ELMER_WORKSPACE_DIR is set, else `default` unchanged.
                         ELMER_WORKSPACE_DIR is Elmer's existing per-instance workspace variable
                         (elmer_hook.ElmerHook.WORKSPACE_ENV, read by the vendored OpenClawAdapter). A host that
                         scopes its Elmer (the CC daemon, via the CC registry's per-module "env") sets it; an
                         unset variable keeps today's paths byte-for-byte, so Syl's VPS is unchanged unless her
                         environment sets it.

# ---- Changelog ----
# [2026-10-11] Claude Code (Opus 5.5, lane river-feed-fix) — created
#   What: tracts_root() and instance_file(), the two path resolutions protoUniBrain, BrainSwitcher and the
#         engine's competence log use instead of hard-coded ~/.et_modules / ~/.elmer paths.
#   Why:  LAW 5. proto_brain_socket.py hard-coded ~/.et_modules/tracts/... and ignored ET_TRACTS_DIR (#1079
#         family); the CC's Elmer wrote Syl-scoped coordination/log paths (#1075:
#         ~/.et_modules/elmer/proto_body.lock, ~/.elmer/competence_delta.jsonl).
#   How:  read the existing variables at call time (no new variable, no caching, so a host that sets env before
#         loading Elmer is honoured); unset = the old default exactly.
# -------------------
"""

from __future__ import annotations

import os


def tracts_root() -> str:
    """The tracts tree this process writes into: $ET_TRACTS_DIR, else ~/.et_modules/tracts."""
    return os.path.expanduser(os.environ.get("ET_TRACTS_DIR") or "~/.et_modules/tracts")


def instance_file(default: str, name: str) -> str:
    """$ELMER_WORKSPACE_DIR/<name> when the instance workspace is set, else `default` (expanded) unchanged."""
    ws = os.environ.get("ELMER_WORKSPACE_DIR")
    if ws:
        return os.path.join(os.path.expanduser(ws), name)
    return os.path.expanduser(default)
