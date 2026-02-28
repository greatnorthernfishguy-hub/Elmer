"""Tests for SubstrateSignal dataclass (Grok priority #1)."""

import json
import time

import pytest

from ng_ecosystem import SubstrateSignal, SignalType


class TestSignalType:
    def test_enum_values(self):
        assert SignalType.SENSORY.value == "sensory"
        assert SignalType.INFERENCE.value == "inference"
        assert SignalType.HEALTH.value == "health"
        assert SignalType.MEMORY.value == "memory"
        assert SignalType.IDENTITY.value == "identity"

    def test_all_types_exist(self):
        assert len(SignalType) == 5


class TestSubstrateSignal:
    def test_create_factory(self):
        sig = SubstrateSignal.create(
            source_socket="test:socket",
            signal_type=SignalType.SENSORY,
            payload={"text": "hello"},
        )
        assert sig.source_socket == "test:socket"
        assert sig.signal_type == SignalType.SENSORY
        assert sig.payload == {"text": "hello"}
        assert sig.confidence == 1.0
        assert sig.priority == 5
        assert sig.graph_encoding is None
        assert isinstance(sig.metadata, dict)
        assert len(sig.signal_id) == 36  # UUID4 format

    def test_create_with_all_fields(self):
        sig = SubstrateSignal.create(
            source_socket="test:full",
            signal_type=SignalType.INFERENCE,
            payload={"result": 42},
            confidence=0.85,
            priority=8,
            graph_encoding={"embedding": [1.0, 2.0]},
            metadata={"source": "test"},
        )
        assert sig.confidence == 0.85
        assert sig.priority == 8
        assert sig.graph_encoding == {"embedding": [1.0, 2.0]}
        assert sig.metadata == {"source": "test"}

    def test_immutable(self):
        sig = SubstrateSignal.create(
            source_socket="test:immutable",
            signal_type=SignalType.HEALTH,
            payload={},
        )
        with pytest.raises(AttributeError):
            sig.confidence = 0.5  # type: ignore[misc]

    def test_to_dict(self):
        sig = SubstrateSignal.create(
            source_socket="test:dict",
            signal_type=SignalType.MEMORY,
            payload={"key": "value"},
        )
        d = sig.to_dict()
        assert d["source_socket"] == "test:dict"
        assert d["signal_type"] == "memory"
        assert d["payload"] == {"key": "value"}
        assert isinstance(d["timestamp"], float)

    def test_json_serializable(self):
        sig = SubstrateSignal.create(
            source_socket="test:json",
            signal_type=SignalType.SENSORY,
            payload={"text": "hello", "count": 42, "nested": {"a": 1}},
            metadata={"source": "test"},
        )
        d = sig.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["source_socket"] == "test:json"
        assert restored["payload"]["nested"]["a"] == 1

    def test_from_dict(self):
        original = SubstrateSignal.create(
            source_socket="test:roundtrip",
            signal_type=SignalType.IDENTITY,
            payload={"name": "elmer"},
            confidence=0.9,
        )
        d = original.to_dict()
        restored = SubstrateSignal.from_dict(d)
        assert restored.source_socket == original.source_socket
        assert restored.signal_type == original.signal_type
        assert restored.payload == original.payload
        assert restored.confidence == original.confidence

    def test_with_updates(self):
        sig = SubstrateSignal.create(
            source_socket="test:original",
            signal_type=SignalType.SENSORY,
            payload={"text": "hello"},
            confidence=0.5,
        )
        updated = sig.with_updates(
            source_socket="test:updated",
            confidence=0.9,
        )
        # Original unchanged
        assert sig.source_socket == "test:original"
        assert sig.confidence == 0.5
        # Updated has new values
        assert updated.source_socket == "test:updated"
        assert updated.confidence == 0.9
        # Preserved fields
        assert updated.payload == {"text": "hello"}
        assert updated.signal_type == SignalType.SENSORY

    def test_timestamp_auto_set(self):
        before = time.time()
        sig = SubstrateSignal.create(
            source_socket="test:time",
            signal_type=SignalType.HEALTH,
            payload={},
        )
        after = time.time()
        assert before <= sig.timestamp <= after

    def test_unique_ids(self):
        sig1 = SubstrateSignal.create(
            source_socket="test:id1",
            signal_type=SignalType.SENSORY,
            payload={},
        )
        sig2 = SubstrateSignal.create(
            source_socket="test:id2",
            signal_type=SignalType.SENSORY,
            payload={},
        )
        assert sig1.signal_id != sig2.signal_id
