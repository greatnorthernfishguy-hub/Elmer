"""
KISS — Keep Input Simple, Substrate

Phase 1: Delta Gate + Sparse Extract
Input optimization layer between GraphSnapshot creation and brain socket processing.
Reduces redundant data reaching the organism so every datum is nutritious.

KISS does NOT classify input (Law 7). It removes redundancy, not meaning.
It decides what's *different from what the organism already knows.*

# ---- Changelog ----
# [2026-03-26] Claude Code (Opus 4.6) — Phase 1 implementation
#   What: Delta Gate + Sparse Extract for brain sockets
#   Why:  NuWave Layer 1 — compound optimization starts with input efficiency.
#         Proto brain on CPU processes 65s per Lenia step. Every skipped
#         redundant input is 65 seconds saved for meaningful work.
#   How:  KISSFilter tracks last snapshot features. Cosine similarity for
#         delta gate. Sparse diff for changed features. Competence Model
#         governs threshold (starts Apprentice — conservative).
# -------------------
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("elmer.kiss")


@dataclass
class KISSConfig:
    """Configuration for KISS filter.

    Thresholds start conservative (Apprentice) and can tighten
    as the organism's activation patterns stabilize.
    """
    # Delta Gate: cosine similarity threshold — above this = skip
    # 0.99 = very conservative (only skip near-identical inputs)
    # 0.95 = moderate, 0.90 = aggressive
    delta_threshold: float = 0.99  # Apprentice: conservative

    # Sparse Extract: minimum feature change to include in sparse output
    # Features that changed less than this are treated as unchanged
    sparse_min_delta: float = 0.01

    # Minimum messages before KISS starts filtering
    # Let the organism see some raw data first
    warmup_messages: int = 5

    # How often to let a full snapshot through regardless of delta
    # Prevents the organism from drifting on stale state
    force_full_every: int = 20


@dataclass
class KISSStats:
    """Running statistics for KISS filtering."""
    total_received: int = 0
    delta_skipped: int = 0
    sparse_passed: int = 0
    full_passed: int = 0
    forced_full: int = 0
    warmup_passed: int = 0
    avg_delta: float = 0.0
    min_delta: float = 1.0
    max_delta: float = 0.0
    _delta_sum: float = 0.0
    _delta_count: int = 0

    def record_delta(self, delta: float):
        self._delta_sum += delta
        self._delta_count += 1
        self.avg_delta = self._delta_sum / self._delta_count
        self.min_delta = min(self.min_delta, delta)
        self.max_delta = max(self.max_delta, delta)

    def to_dict(self) -> Dict[str, Any]:
        skip_rate = self.delta_skipped / max(self.total_received, 1)
        return {
            "total_received": self.total_received,
            "delta_skipped": self.delta_skipped,
            "sparse_passed": self.sparse_passed,
            "full_passed": self.full_passed,
            "forced_full": self.forced_full,
            "warmup_passed": self.warmup_passed,
            "skip_rate": round(skip_rate, 4),
            "avg_delta": round(self.avg_delta, 6),
            "min_delta": round(self.min_delta, 6),
            "max_delta": round(self.max_delta, 6),
        }


class KISSFilter:
    """Phase 1 KISS: Delta Gate + Sparse Extract.

    Sits between GraphSnapshot creation and brain socket processing.
    Tracks the last snapshot's features and decides:
      - SKIP: input is redundant (delta below threshold)
      - SPARSE: input has some changes (pass only the diffs)
      - FULL: input is significantly different or forced refresh

    Usage:
        kiss = KISSFilter()
        result = kiss.filter(snapshot_features)
        if result is None:
            # Skip — don't process this cycle
        else:
            # result contains filtered features + metadata
            brain_socket.process(result['snapshot'], context)
    """

    def __init__(self, config: KISSConfig = None):
        self._config = config or KISSConfig()
        self._last_features: Optional[np.ndarray] = None
        self._last_raw: Optional[Dict[str, Any]] = None
        self._messages_since_full: int = 0
        self.stats = KISSStats()

    def filter(
        self,
        snapshot_features: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Apply KISS filtering to a snapshot's extracted features.

        Args:
            snapshot_features: Dict with keys matching brain socket input:
                node_features, synapse_features, topo_features,
                temporal_features, identity_embedding

        Returns:
            None if input should be skipped.
            Dict with 'snapshot' (features to process), 'kiss_mode' ('full'|'sparse'),
            and 'kiss_meta' (delta info, changed features) if input should be processed.
        """
        self.stats.total_received += 1

        # Flatten features to a single vector for delta comparison
        current_vec = self._flatten(snapshot_features)

        # Warmup: pass everything until we've seen enough
        if self.stats.total_received <= self._config.warmup_messages:
            self._last_features = current_vec
            self._last_raw = snapshot_features
            self._messages_since_full = 0
            self.stats.warmup_passed += 1
            self.stats.full_passed += 1
            return {
                "snapshot": snapshot_features,
                "kiss_mode": "full",
                "kiss_meta": {
                    "reason": "warmup",
                    "message_num": self.stats.total_received,
                },
            }

        # Force full snapshot periodically to prevent drift
        self._messages_since_full += 1
        if self._messages_since_full >= self._config.force_full_every:
            self._last_features = current_vec
            self._last_raw = snapshot_features
            self._messages_since_full = 0
            self.stats.forced_full += 1
            self.stats.full_passed += 1
            return {
                "snapshot": snapshot_features,
                "kiss_mode": "full",
                "kiss_meta": {
                    "reason": "forced_refresh",
                    "cycles_since_last_full": self._config.force_full_every,
                },
            }

        # Delta Gate: cosine similarity between current and last
        if self._last_features is not None:
            similarity = self._cosine_similarity(current_vec, self._last_features)
            delta = 1.0 - similarity  # 0 = identical, 1 = completely different
            self.stats.record_delta(delta)

            if similarity >= self._config.delta_threshold:
                # Too similar — skip
                self.stats.delta_skipped += 1
                logger.debug(
                    "KISS: skipped (similarity=%.4f, threshold=%.4f, delta=%.6f)",
                    similarity, self._config.delta_threshold, delta,
                )
                return None
        else:
            delta = 1.0  # First real message after warmup

        # Sparse Extract: identify what changed
        if self._last_raw is not None:
            sparse_result = self._sparse_extract(snapshot_features, self._last_raw)
            changed_count = sparse_result["changed_count"]
            total_count = sparse_result["total_count"]

            # If most features changed, send full snapshot
            if changed_count > total_count * 0.5:
                self._last_features = current_vec
                self._last_raw = snapshot_features
                self._messages_since_full = 0
                self.stats.full_passed += 1
                return {
                    "snapshot": snapshot_features,
                    "kiss_mode": "full",
                    "kiss_meta": {
                        "reason": "major_change",
                        "delta": delta,
                        "changed": changed_count,
                        "total": total_count,
                    },
                }
            else:
                # Sparse pass — only changed features
                self._last_features = current_vec
                self._last_raw = snapshot_features
                self.stats.sparse_passed += 1
                return {
                    "snapshot": snapshot_features,  # Full data available
                    "kiss_mode": "sparse",
                    "kiss_meta": {
                        "reason": "sparse_change",
                        "delta": delta,
                        "changed": changed_count,
                        "total": total_count,
                        "changed_features": sparse_result["changed_features"],
                    },
                }

        # Fallback: first message, pass full
        self._last_features = current_vec
        self._last_raw = snapshot_features
        self._messages_since_full = 0
        self.stats.full_passed += 1
        return {
            "snapshot": snapshot_features,
            "kiss_mode": "full",
            "kiss_meta": {"reason": "first_message", "delta": delta},
        }

    def _flatten(self, features: Dict[str, Any]) -> np.ndarray:
        """Flatten feature dict to a single vector for similarity comparison."""
        parts = []
        for key in sorted(features.keys()):
            val = features[key]
            if hasattr(val, 'numpy'):
                parts.append(val.detach().numpy().flatten())
            elif isinstance(val, np.ndarray):
                parts.append(val.flatten())
            elif isinstance(val, (list, tuple)):
                parts.append(np.array(val, dtype=np.float32).flatten())
            elif isinstance(val, (int, float)):
                parts.append(np.array([val], dtype=np.float32))
        if not parts:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        if len(a) != len(b):
            # Dimension mismatch — can't compare, pass through
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _sparse_extract(
        self,
        current: Dict[str, Any],
        previous: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Identify which features changed between snapshots."""
        changed_features = []
        total_features = 0

        for key in sorted(current.keys()):
            if key not in previous:
                changed_features.append(key)
                total_features += 1
                continue

            cur_val = self._to_numpy(current[key])
            prev_val = self._to_numpy(previous[key])

            if cur_val.shape != prev_val.shape:
                changed_features.append(key)
                total_features += 1
                continue

            # Per-element delta
            diff = np.abs(cur_val - prev_val)
            max_diff = float(np.max(diff)) if diff.size > 0 else 0.0
            total_features += 1

            if max_diff > self._config.sparse_min_delta:
                changed_features.append(key)

        return {
            "changed_features": changed_features,
            "changed_count": len(changed_features),
            "total_count": total_features,
        }

    @staticmethod
    def _to_numpy(val) -> np.ndarray:
        if hasattr(val, 'numpy'):
            return val.detach().numpy().flatten()
        elif isinstance(val, np.ndarray):
            return val.flatten()
        elif isinstance(val, (list, tuple)):
            return np.array(val, dtype=np.float32).flatten()
        elif isinstance(val, (int, float)):
            return np.array([val], dtype=np.float32)
        return np.zeros(1, dtype=np.float32)
