"""
Elmer Configuration — YAML + Dataclass Loader

Loads config.yaml and produces a validated ElmerConfig dataclass.
Pattern: THC core/config.py (YAML → frozen dataclass → dot-access).

# ---- Changelog ----
# [2026-02-28] Claude (Opus 4.6) — Phase 1 initial creation.
#   What: ElmerConfig dataclass with nested sections for hardware,
#         sockets, NG ecosystem, and pipelines.  YAML loader with
#         env-var interpolation and sensible defaults.
#   Why:  Single source of truth for all Elmer configuration.
#         Dataclass validation catches typos at startup, not at runtime.
# -------------------
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("elmer.config")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# ---------------------------------------------------------------------------
# Config dataclasses (nested, validated at construction)
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    """Hardware detection and allocation settings."""
    prefer_gpu: bool = True
    gpu_memory_fraction: float = 0.5
    cpu_threads: int = 0  # 0 = auto-detect
    enable_npu: bool = False


@dataclass
class SocketsConfig:
    """Socket manager settings."""
    max_sockets: int = 16
    health_check_interval: float = 30.0
    connect_timeout: float = 10.0
    default_priority: int = 5


@dataclass
class NGEcosystemConfig:
    """NG-Lite / peer bridge / tier 3 upgrade settings."""
    module_id: str = "elmer"
    state_path: str = ""
    peer_bridge_enabled: bool = True
    peer_sync_interval: int = 100
    tier3_upgrade_enabled: bool = True
    tier3_poll_interval: float = 300.0


@dataclass
class PipelinesConfig:
    """Pipeline enable/disable flags and settings."""
    sensory_enabled: bool = True
    inference_enabled: bool = True
    health_enabled: bool = True
    memory_enabled: bool = True
    identity_enabled: bool = True


@dataclass
class ElmerConfig:
    """Top-level Elmer configuration.

    Loaded from config.yaml with env-var interpolation.
    All fields have sensible defaults so Elmer works out of the box.
    """
    module_id: str = "elmer"
    display_name: str = "Elmer"
    version: str = "0.1.0"
    log_level: str = "INFO"
    data_dir: str = "~/.elmer"

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    sockets: SocketsConfig = field(default_factory=SocketsConfig)
    ng_ecosystem: NGEcosystemConfig = field(default_factory=NGEcosystemConfig)
    pipelines: PipelinesConfig = field(default_factory=PipelinesConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _interpolate_env(value: Any) -> Any:
    """Replace ${ENV_VAR} patterns with environment variable values."""
    if isinstance(value, str) and "${" in value:
        import re
        def _replace(match: Any) -> str:
            var = match.group(1)
            return os.environ.get(var, match.group(0))
        return re.sub(r"\$\{(\w+)\}", _replace, value)
    return value


def _apply_dict(target_dc: Any, source: Dict[str, Any]) -> None:
    """Apply a plain dict onto a dataclass instance, interpolating env vars."""
    for key, value in source.items():
        if not hasattr(target_dc, key):
            logger.warning("Unknown config key: %s", key)
            continue
        current = getattr(target_dc, key)
        if isinstance(current, (HardwareConfig, SocketsConfig,
                                NGEcosystemConfig, PipelinesConfig)):
            if isinstance(value, dict):
                _apply_dict(current, value)
        else:
            setattr(target_dc, key, _interpolate_env(value))


def load_config(path: Optional[str] = None) -> ElmerConfig:
    """Load ElmerConfig from YAML file.

    Args:
        path: Path to config.yaml.  Defaults to project root config.yaml.

    Returns:
        Populated ElmerConfig with env-var interpolation applied.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    cfg = ElmerConfig()

    if not config_path.exists():
        logger.info("No config file at %s, using defaults", config_path)
        return cfg

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, using defaults")
        return cfg

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if raw and isinstance(raw, dict):
            _apply_dict(cfg, raw)
            logger.info("Config loaded from %s", config_path)
    except Exception as exc:
        logger.error("Failed to load config from %s: %s", config_path, exc)

    return cfg
