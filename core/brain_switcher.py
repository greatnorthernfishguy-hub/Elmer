"""
BrainSwitcher — Dynamic Resource-Aware Brain Socket Management

Manages both brain sockets on the 16GB VPS:
  - ElmerBrain (frozen, 0.5B, ~2GB, fast) — stable reference
  - ProtoUniBrain (living, 1.5B, ~5.5GB, Lenia) — learning from use

Default: both running (~7.5GB total). Falls back to ElmerBrain-only
if VPS memory gets tight. Monitors resources in background.

Decision factors:
  - Available RAM (need ~6GB free to load ProtoUniBrain)
  - CPU load (need headroom for ~80s Lenia steps)
  - User activity (no recent input = good time for ProtoUniBrain)
  - Autonomic state (SYMPATHETIC = stay light, PARASYMPATHETIC = go heavy)

# ---- Changelog ----
# [2026-03-25] Claude Code (Opus 4.6) — Initial implementation
#   What: Resource-aware brain switching
#   Why:  Can't run both brains simultaneously on 16GB VPS
# -------------------
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("elmer.brain_switcher")


@dataclass
class ResourceThresholds:
    """Thresholds for switching between brains."""
    # Minimum free RAM (MB) to keep ProtoUniBrain loaded
    min_free_ram_mb: int = 2000
    # Maximum CPU load average (1-min) before shedding ProtoUniBrain
    max_cpu_load: float = 3.5  # out of 4 cores
    # Minimum seconds between switch attempts (cooldown)
    switch_cooldown_seconds: float = 300.0
    # How often to check resources (seconds)
    check_interval: float = 60.0


class BrainSwitcher:
    """Manages dynamic brain socket switching based on resource availability.

    Usage:
        switcher = BrainSwitcher(socket_manager)
        switcher.start()          # begins monitoring in background
        switcher.notify_input()   # call on user activity
        switcher.stop()           # cleanup
    """

    def __init__(
        self,
        socket_manager,
        thresholds: ResourceThresholds = None,
    ):
        self._socket_manager = socket_manager
        self._thresholds = thresholds or ResourceThresholds()
        self._active_brain: str = "none"  # "elmer_brain", "proto_unibrain", "none"
        self._last_input_time: float = time.time()
        self._last_switch_time: float = 0.0
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def active_brain(self) -> str:
        return self._active_brain

    def notify_input(self):
        """Call this when user input arrives. Resets idle timer."""
        self._last_input_time = time.time()

    def start(self):
        """Start background resource monitoring."""
        if self._running:
            return
        self._running = True
        # Start both brains — VPS has 16GB, both fit (~7.5GB total)
        self._activate_both()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="brain-switcher",
        )
        self._monitor_thread.start()
        logger.info("BrainSwitcher started (checking every %.0fs)",
                     self._thresholds.check_interval)

    def stop(self):
        """Stop monitoring and unload active brain."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._deactivate_current()
        logger.info("BrainSwitcher stopped")

    # -----------------------------------------------------------------
    # Resource monitoring
    # -----------------------------------------------------------------

    def _monitor_loop(self):
        """Background loop that checks resources and switches brains."""
        while self._running:
            try:
                self._evaluate_and_switch()
            except Exception as exc:
                logger.error("BrainSwitcher monitor error: %s", exc)
            time.sleep(self._thresholds.check_interval)

    def _evaluate_and_switch(self):
        """Check resources and shed ProtoUniBrain if memory gets tight."""
        resources = self._check_resources()

        if self._active_brain == "both":
            # Only shed ProtoUniBrain if resources are genuinely tight
            should_shed = (
                resources['free_ram_mb'] < self._thresholds.min_free_ram_mb
                or resources['cpu_load_1m'] > self._thresholds.max_cpu_load
            )
            if should_shed:
                elapsed = time.time() - self._last_switch_time
                if elapsed < self._thresholds.switch_cooldown_seconds:
                    return
                reason = []
                if resources['free_ram_mb'] < self._thresholds.min_free_ram_mb:
                    reason.append(f"RAM low ({resources['free_ram_mb']}MB)")
                if resources['cpu_load_1m'] > self._thresholds.max_cpu_load:
                    reason.append(f"CPU high ({resources['cpu_load_1m']:.1f})")
                logger.warning(
                    "Resources tight (%s) — shedding ProtoUniBrain, keeping ElmerBrain",
                    ", ".join(reason),
                )
                self._shed_proto_unibrain()

        elif self._active_brain == "elmer_brain":
            # Try to restore ProtoUniBrain if resources recovered
            can_restore = (
                resources['free_ram_mb'] >= 6000
                and resources['cpu_load_1m'] <= 2.0
            )
            if can_restore:
                elapsed = time.time() - self._last_switch_time
                if elapsed < self._thresholds.switch_cooldown_seconds:
                    return
                logger.info(
                    "Resources recovered (RAM: %dMB free, CPU: %.1f) "
                    "— restoring ProtoUniBrain",
                    resources['free_ram_mb'],
                    resources['cpu_load_1m'],
                )
                self._add_proto_unibrain()

    def _check_resources(self) -> Dict[str, Any]:
        """Check current VPS resource availability."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            load = os.getloadavg()
            return {
                'free_ram_mb': mem.available // (1024 * 1024),
                'total_ram_mb': mem.total // (1024 * 1024),
                'ram_percent': mem.percent,
                'cpu_load_1m': load[0],
                'cpu_load_5m': load[1],
            }
        except ImportError:
            # Fallback without psutil
            try:
                with open('/proc/meminfo') as f:
                    lines = f.readlines()
                mem_available = 0
                for line in lines:
                    if line.startswith('MemAvailable:'):
                        mem_available = int(line.split()[1]) // 1024  # kB to MB
                load = os.getloadavg()
                return {
                    'free_ram_mb': mem_available,
                    'total_ram_mb': 0,
                    'ram_percent': 0,
                    'cpu_load_1m': load[0],
                    'cpu_load_5m': load[1],
                }
            except Exception:
                return {
                    'free_ram_mb': 0,
                    'total_ram_mb': 0,
                    'ram_percent': 100,
                    'cpu_load_1m': 99.0,
                    'cpu_load_5m': 99.0,
                }

    # -----------------------------------------------------------------
    # Brain activation
    # -----------------------------------------------------------------

    def _activate_both(self):
        """Load both brains — frozen reference + living Lenia."""
        with self._lock:
            ok_brain = False
            ok_proto = False

            def _flog(msg):
                with open("/tmp/elmer_eager.log", "a") as _f:
                    from datetime import datetime
                    _f.write(f"[{datetime.now()}] SWITCHER: {msg}\n")

            # Ensure Elmer's install dir is importable — the fan-out's
            # namespace isolation may have stashed 'core' from sys.modules
            # by the time this background thread runs.
            import sys as _sys
            _elmer_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _elmer_dir not in _sys.path:
                _sys.path.insert(0, _elmer_dir)

            # Load ElmerBrain (frozen reference) first — lightweight
            try:
                from core.brain_socket import BrainSocket
                brain_socket = BrainSocket()
                _flog(f"BrainSocket created, model_path={brain_socket._model_path}")
                self._socket_manager.register(brain_socket)
                ok_brain = brain_socket.load("models/brain")
                _flog(f"BrainSocket.load() returned {ok_brain}")
                if not ok_brain:
                    logger.error("ElmerBrain failed to load")
                    self._socket_manager.unregister(brain_socket.socket_id)
            except Exception as exc:
                import traceback
                _flog(f"ElmerBrain EXCEPTION: {exc}\n{traceback.format_exc()}")
                logger.error("ElmerBrain activation failed: %s", exc)

            # Load ProtoUniBrain (living, Lenia dynamics)
            try:
                from core.proto_brain_socket import ProtoUniBrainSocket
                proto_socket = ProtoUniBrainSocket()
                _flog(f"ProtoUniBrainSocket created, model_path={proto_socket._model_path}")
                self._socket_manager.register(proto_socket)
                ok_proto = proto_socket.load("models/proto_brain")
                _flog(f"ProtoUniBrainSocket.load() returned {ok_proto}")
                if not ok_proto:
                    logger.warning("ProtoUniBrain failed to load — ElmerBrain only")
                    self._socket_manager.unregister(proto_socket.socket_id)
            except Exception as exc:
                import traceback
                _flog(f"ProtoUniBrain EXCEPTION: {exc}\n{traceback.format_exc()}")
                logger.warning("ProtoUniBrain activation failed: %s", exc)

            if ok_brain and ok_proto:
                self._active_brain = "both"
                logger.info("Both brains active (frozen + living)")
            elif ok_brain:
                self._active_brain = "elmer_brain"
                logger.info("ElmerBrain only (ProtoUniBrain unavailable)")
            else:
                self._active_brain = "none"
                logger.error("No brain sockets loaded")

            self._last_switch_time = time.time()

    def _shed_proto_unibrain(self):
        """Shed ProtoUniBrain to free resources, keep ElmerBrain."""
        with self._lock:
            try:
                self._socket_manager.unregister("elmer:proto_unibrain")
            except (ValueError, KeyError):
                pass
            self._active_brain = "elmer_brain"
            self._last_switch_time = time.time()
            import gc
            gc.collect()
            logger.info("ProtoUniBrain shed — ElmerBrain only")

    def _add_proto_unibrain(self):
        """Restore ProtoUniBrain alongside ElmerBrain."""
        with self._lock:
            try:
                from core.proto_brain_socket import ProtoUniBrainSocket
                proto_socket = ProtoUniBrainSocket()
                self._socket_manager.register(proto_socket)
                if proto_socket.load("models/proto_brain"):
                    self._active_brain = "both"
                    self._last_switch_time = time.time()
                    logger.info("ProtoUniBrain restored — both brains active")
                else:
                    self._socket_manager.unregister(proto_socket.socket_id)
                    logger.warning("ProtoUniBrain restore failed")
            except Exception as exc:
                logger.warning("ProtoUniBrain restore failed: %s", exc)

    def _deactivate_current(self):
        """Unload and unregister all brain sockets."""
        for sid in ["elmer:brain", "elmer:proto_unibrain"]:
            try:
                self._socket_manager.unregister(sid)
            except (ValueError, KeyError):
                pass
        self._active_brain = "none"
        import gc
        gc.collect()

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get current switcher status."""
        resources = self._check_resources()
        idle = time.time() - self._last_input_time
        since_switch = time.time() - self._last_switch_time
        return {
            'active_brain': self._active_brain,
            'idle_seconds': idle,
            'since_last_switch': since_switch,
            'resources': resources,
            'would_shed_proto': (
                resources['free_ram_mb'] < self._thresholds.min_free_ram_mb
                or resources['cpu_load_1m'] > self._thresholds.max_cpu_load
            ),
        }
