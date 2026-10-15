"""
Atomic file writes for Elmer's persistent state.

A save that is interrupted (daemon restart, crash, OOM kill, suspend hang, power loss) must leave the PREVIOUS
good file in place, never a torn one. Every writer here goes: temp file in the SAME directory -> write -> flush +
fsync -> os.replace onto the final name -> fsync the directory. A reader therefore sees either the old complete
file or the new complete file, never a mix. The final file's bytes and format are exactly what the writer would
have produced in place (torch.save / json / msgpack unchanged), so readers need no change.

# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5) — temps of dead writers clear themselves
#   What: warn_stale_temps -> clear_stale_temps: this target's temp files left by DEAD writers are DELETED (one
#         WARNING naming them), and atomic_write calls it on every write (was: only torch_save_atomic, log only).
#   Why:  Josh 2026-10-15: "Temp files are temporary, so if they are no longer needed, I'm not aware of any reason to
#         keep them around" / "Temp files clearing themselves once done sounds nice." A killed proto save leaves a
#         ~1.44 GB temp each time.
#   How:  same match as before (this basename + ".tmp-<pid>-", writer pid dead and not ours); only the process named
#         in a temp's name ever writes it, so a dead pid means nobody will. A pid now reused by a live process is
#         left alone (conservative). Never raises.
# [2026-10-14] Claude Code (Opus 5.5, lane atomic-saves) — initial helper
#   What: atomic_write(path, write_fn) (write_fn writes the whole file at the temp path it is given),
#         atomic_write_bytes / atomic_write_text, torch_save_atomic(obj, path), and find_stale_temps / warn_stale_temps(path)
#         (2026-10-14: logged, did not delete; this target's temp files left by DEAD processes, e.g. a SIGKILL mid-save).
#   Why:  Josh 2026-10-14: "We should have all saves and checkpoints, anything that can't risk a failed save, be
#         atomic." A daemon restart caught save_evolved_weights() mid-torch.save of ~/.elmer/proto_weights.pt
#         (non-atomic, punch list #1095); the torn file could not load, the proto restarted from the pretrained
#         body and then saved over the torn file: ~18 h of Flow-Lenia F1 + adapter state lost.
#   How:  temp name "<final>.tmp-<pid>-<rand><suffix>" (suffix kept for writers that dispatch on the extension);
#         the final file keeps the old file's permission bits (or the umask default for a new file); on any
#         failure the temp is removed and the final file is untouched. Directory fsync is best-effort (logged).
# -------------------
"""

from __future__ import annotations

import errno
import logging
import os
import re
import uuid
from typing import Any, Callable, Optional

logger = logging.getLogger("elmer.atomic_io")

_TMP_MARK = ".tmp-"


def _target_mode(path: str) -> int:
    """Permission bits for the final file: the existing file's, else the process default (0666 & ~umask)."""
    try:
        return os.stat(path).st_mode & 0o7777
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def _fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(directory: str) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        logger.debug("atomic_io: cannot open %s for fsync (%s)", directory, exc)
        return
    try:
        os.fsync(fd)
    except OSError as exc:  # some filesystems refuse fsync on a directory; the rename itself is still atomic
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
            logger.warning("atomic_io: directory fsync failed for %s (%s)", directory, exc)
    finally:
        os.close(fd)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def find_stale_temps(path: str) -> list:
    """Temp files of THIS target left by DEAD processes (a write killed mid-way), as full paths. Never raises.

    Only names "<basename>.tmp-<pid>-..." in the target's directory are considered; a temp whose pid is this
    process or any live process is not stale. clear_stale_temps deletes what this returns.
    """
    directory = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    pat = re.compile(re.escape(base + _TMP_MARK) + r"(\d+)-")
    stale = []
    try:
        names = os.listdir(directory)
    except OSError:
        return stale
    for name in names:
        m = pat.match(name)
        if not m:
            continue
        pid = int(m.group(1))
        if pid == os.getpid() or _pid_alive(pid):
            continue
        stale.append(os.path.join(directory, name))
    return stale


def clear_stale_temps(path: str) -> list:
    """Delete this target's temp files left by dead writers (interrupted saves). Returns the removed paths. Never raises."""
    removed, total = [], 0
    for p in find_stale_temps(path):
        try:
            size = os.path.getsize(p)
            os.remove(p)
        except OSError as exc:
            logger.debug("atomic_io: could not remove stale temp %s (%s)", p, exc)
            continue
        removed.append(p)
        total += size
    if removed:
        logger.warning("atomic_io: removed %d stale temp file(s) of %s left by interrupted saves (%.0f MB): %s -- "
                       "the real file was intact.", len(removed), os.path.basename(path), total / 1e6,
                       ", ".join(os.path.basename(p) for p in removed))
    return removed


def atomic_write(path: str, write_fn: Callable[[str], Any], *, suffix: str = "") -> Any:
    """Write `path` atomically. `write_fn(tmp_path)` must write the COMPLETE file at tmp_path.

    Returns whatever write_fn returns. On any exception (or interrupt) the temp is removed, `path` is untouched,
    and the exception propagates.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    mode = _target_mode(path)
    clear_stale_temps(path)
    tmp = os.path.join(directory, f"{os.path.basename(path)}{_TMP_MARK}{os.getpid()}-{uuid.uuid4().hex[:8]}{suffix}")
    try:
        result = write_fn(tmp)
        os.chmod(tmp, mode)
        _fsync_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(directory)
    return result


def atomic_write_bytes(path: str, data: bytes) -> None:
    def _w(tmp: str) -> None:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
    atomic_write(path, _w)


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def torch_save_atomic(obj: Any, path: str, torch_module: Optional[Any] = None) -> None:
    """torch.save(obj, path), atomically. Same bytes as an in-place torch.save; torch.load reads it unchanged."""
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415 (lazy: Elmer imports torch lazily)
    atomic_write(path, lambda tmp: torch_module.save(obj, tmp))
