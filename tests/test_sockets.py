"""Tests for ElmerSocket implementations (PRD §5.2.1)."""

import pytest

from core.base_socket import ElmerSocket, GraphSnapshot, HardwareRequirements, SocketHealth
from core.comprehension import ComprehensionSocket
from core.monitoring import MonitoringSocket


def _empty_snapshot() -> GraphSnapshot:
    return GraphSnapshot.empty()


def _populated_snapshot() -> GraphSnapshot:
    return GraphSnapshot(
        nodes=[
            {"id": "a", "type": "concept"},
            {"id": "b", "type": "concept"},
            {"id": "c", "type": "concept"},
        ],
        edges=[
            {"source": "a", "target": "b", "weight": 0.8},
            {"source": "b", "target": "c", "weight": 0.5},
        ],
        metadata={"test": True},
    )


class TestComprehensionSocket:
    def test_socket_id(self):
        s = ComprehensionSocket()
        assert s.socket_id == "elmer:comprehension"
        assert s.socket_type == "comprehension"

    def test_declare_requirements(self):
        s = ComprehensionSocket()
        req = s.declare_requirements()
        assert isinstance(req, HardwareRequirements)
        assert req.min_memory_mb == 256
        assert req.gpu_required is False

    def test_load_unload(self):
        s = ComprehensionSocket()
        assert not s.is_loaded
        assert s.load("models/comprehension") is True
        assert s.is_loaded
        s.unload()
        assert not s.is_loaded

    def test_idempotent_load(self):
        s = ComprehensionSocket()
        assert s.load("models/comprehension") is True
        assert s.load("models/comprehension") is True  # Should not error
        assert s.is_loaded

    def test_process(self):
        s = ComprehensionSocket()
        s.load("models/comprehension")

        snap = _populated_snapshot()
        out = s.process(snap, {})

        assert out.signal.signal_type == "observation"
        assert out.signal.module_id == "elmer"
        assert out.confidence > 0
        assert out.processing_time >= 0
        assert "socket" in out.signal.metadata

    def test_process_empty_snapshot(self):
        s = ComprehensionSocket()
        s.load("models/comprehension")
        out = s.process(_empty_snapshot(), {})
        assert out.signal.coherence_score == 1.0  # no nodes → full coherence

    def test_process_not_loaded(self):
        s = ComprehensionSocket()
        with pytest.raises(RuntimeError, match="not loaded"):
            s.process(_empty_snapshot(), {})

    def test_health_loaded(self):
        s = ComprehensionSocket()
        s.load("models/comprehension")
        h = s.health()
        assert isinstance(h, SocketHealth)
        assert h.status == "healthy"

    def test_health_not_loaded(self):
        s = ComprehensionSocket()
        h = s.health()
        assert h.status == "offline"

    def test_is_elmer_socket(self):
        assert isinstance(ComprehensionSocket(), ElmerSocket)


class TestMonitoringSocket:
    def test_socket_id(self):
        s = MonitoringSocket()
        assert s.socket_id == "elmer:monitoring"
        assert s.socket_type == "monitoring"

    def test_declare_requirements(self):
        req = MonitoringSocket().declare_requirements()
        assert req.min_memory_mb == 128

    def test_load_unload(self):
        s = MonitoringSocket()
        s.load("models/monitoring")
        assert s.is_loaded
        s.unload()
        assert not s.is_loaded

    def test_process_healthy_graph(self):
        s = MonitoringSocket()
        s.load("models/monitoring")
        out = s.process(_populated_snapshot(), {})
        assert out.signal.signal_type in ("health", "anomaly")
        assert out.signal.health_score >= 0
        assert out.signal.anomaly_level >= 0

    def test_process_disconnected_nodes(self):
        """Graph with disconnected nodes should report anomaly."""
        s = MonitoringSocket()
        s.load("models/monitoring")
        snap = GraphSnapshot(
            nodes=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
            edges=[{"source": "a", "target": "b"}],
        )
        out = s.process(snap, {})
        assert out.signal.anomaly_level > 0  # node "c" is disconnected

    def test_health(self):
        s = MonitoringSocket()
        s.load("models/monitoring")
        h = s.health()
        assert h.status == "healthy"
