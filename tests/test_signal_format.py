"""Tests for SubstrateSignal dataclass (PRD §6.1) and threshold constants (§14)."""

import json
import time

import pytest

from ng_ecosystem import (
    COHERENCE_CRITICAL,
    COHERENCE_DEGRADED,
    COHERENCE_HEALTHY,
    SIGNAL_TYPES,
    SubstrateSignal,
)


class TestSignalTypes:
    def test_signal_types_tuple(self):
        assert SIGNAL_TYPES == ("observation", "anomaly", "coherence", "health")

    def test_invalid_signal_type(self):
        with pytest.raises(ValueError, match="signal_type must be one of"):
            SubstrateSignal.create(
                signal_type="invalid",
                description="bad type",
            )


class TestThresholds:
    def test_threshold_values(self):
        assert COHERENCE_HEALTHY == 0.70
        assert COHERENCE_DEGRADED == 0.40
        assert COHERENCE_CRITICAL == 0.15

    def test_threshold_ordering(self):
        assert COHERENCE_HEALTHY > COHERENCE_DEGRADED > COHERENCE_CRITICAL > 0


class TestSubstrateSignal:
    def test_create_factory(self):
        sig = SubstrateSignal.create(
            signal_type="observation",
            description="test signal",
        )
        assert sig.module_id == "elmer"
        assert sig.signal_type == "observation"
        assert sig.description == "test signal"
        assert sig.coherence_score == 1.0
        assert sig.health_score == 1.0
        assert sig.anomaly_level == 0.0
        assert sig.confidence == 1.0
        assert sig.severity == 0.0
        assert sig.identity_coherence == 1.0
        assert sig.pruning_pressure == 0.0
        assert sig.topology_health == 1.0
        assert len(sig.signal_id) == 36

    def test_create_with_all_fields(self):
        sig = SubstrateSignal.create(
            signal_type="anomaly",
            description="high anomaly",
            coherence_score=0.3,
            health_score=0.5,
            anomaly_level=0.9,
            novelty=0.8,
            confidence=0.7,
            severity=0.85,
            temporal_window=10.0,
            metadata={"source": "test"},
            identity_coherence=0.6,
            pruning_pressure=0.4,
            topology_health=0.5,
        )
        assert sig.coherence_score == 0.3
        assert sig.anomaly_level == 0.9
        assert sig.severity == 0.85
        assert sig.identity_coherence == 0.6
        assert sig.pruning_pressure == 0.4
        assert sig.topology_health == 0.5
        assert sig.metadata == {"source": "test"}

    def test_immutable(self):
        sig = SubstrateSignal.create(signal_type="health", description="frozen")
        with pytest.raises(AttributeError):
            sig.confidence = 0.5  # type: ignore[misc]

    def test_to_dict(self):
        sig = SubstrateSignal.create(
            signal_type="coherence",
            description="coherence test",
            coherence_score=0.8,
        )
        d = sig.to_dict()
        assert d["module_id"] == "elmer"
        assert d["signal_type"] == "coherence"
        assert d["coherence_score"] == 0.8
        assert isinstance(d["timestamp"], float)

    def test_json_serializable(self):
        sig = SubstrateSignal.create(
            signal_type="observation",
            description="json test",
            metadata={"nested": {"a": 1}},
        )
        json_str = json.dumps(sig.to_dict())
        restored = json.loads(json_str)
        assert restored["module_id"] == "elmer"
        assert restored["metadata"]["nested"]["a"] == 1

    def test_from_dict(self):
        original = SubstrateSignal.create(
            signal_type="health",
            description="roundtrip",
            confidence=0.9,
            anomaly_level=0.2,
        )
        d = original.to_dict()
        restored = SubstrateSignal.from_dict(d)
        assert restored.signal_type == original.signal_type
        assert restored.confidence == original.confidence
        assert restored.anomaly_level == original.anomaly_level

    def test_with_updates(self):
        sig = SubstrateSignal.create(
            signal_type="observation",
            description="original",
            confidence=0.5,
        )
        updated = sig.with_updates(description="updated", confidence=0.9)
        assert sig.description == "original"
        assert sig.confidence == 0.5
        assert updated.description == "updated"
        assert updated.confidence == 0.9
        assert updated.signal_type == "observation"

    def test_timestamp_auto_set(self):
        before = time.time()
        sig = SubstrateSignal.create(signal_type="health", description="time test")
        after = time.time()
        assert before <= sig.timestamp <= after

    def test_unique_ids(self):
        sig1 = SubstrateSignal.create(signal_type="observation", description="1")
        sig2 = SubstrateSignal.create(signal_type="observation", description="2")
        assert sig1.signal_id != sig2.signal_id

    def test_coherence_status_healthy(self):
        sig = SubstrateSignal.create(
            signal_type="health", description="ok", coherence_score=0.85
        )
        assert sig.coherence_status == "healthy"

    def test_coherence_status_degraded(self):
        sig = SubstrateSignal.create(
            signal_type="health", description="deg", coherence_score=0.55
        )
        assert sig.coherence_status == "degraded"

    def test_coherence_status_warning(self):
        sig = SubstrateSignal.create(
            signal_type="health", description="warn", coherence_score=0.20
        )
        assert sig.coherence_status == "warning"

    def test_coherence_status_critical(self):
        sig = SubstrateSignal.create(
            signal_type="health", description="crit", coherence_score=0.10
        )
        assert sig.coherence_status == "critical"
