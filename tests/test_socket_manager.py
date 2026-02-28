"""Tests for the SocketManager (PRD §5.2)."""

import pytest

from core.base_socket import GraphSnapshot
from core.socket_manager import SocketManager
from core.comprehension import ComprehensionSocket
from core.monitoring import MonitoringSocket


def _snap() -> GraphSnapshot:
    return GraphSnapshot(
        nodes=[{"id": "a"}, {"id": "b"}],
        edges=[{"source": "a", "target": "b"}],
    )


class TestSocketManager:
    def test_register(self):
        mgr = SocketManager()
        sock = ComprehensionSocket()
        mgr.register(sock)
        assert mgr.get_socket("elmer:comprehension") is sock

    def test_register_duplicate(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        with pytest.raises(ValueError, match="already registered"):
            mgr.register(ComprehensionSocket())

    def test_register_at_capacity(self):
        mgr = SocketManager(max_sockets=1)
        mgr.register(ComprehensionSocket())
        with pytest.raises(RuntimeError, match="Max sockets"):
            mgr.register(MonitoringSocket())

    def test_load_all(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.register(MonitoringSocket())
        results = mgr.load_all()
        assert results["elmer:comprehension"] is True
        assert results["elmer:monitoring"] is True

    def test_unload_all(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.load_all()
        mgr.unload_all()
        sock = mgr.get_socket("elmer:comprehension")
        assert sock is not None
        assert not sock.is_loaded

    def test_route_comprehension(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.load_all()

        outputs = mgr.route(_snap(), {}, socket_type="comprehension")
        assert len(outputs) == 1
        assert outputs[0].signal.signal_type == "observation"

    def test_route_monitoring(self):
        mgr = SocketManager()
        mgr.register(MonitoringSocket())
        mgr.load_all()

        outputs = mgr.route(_snap(), {}, socket_type="monitoring")
        assert len(outputs) == 1
        assert outputs[0].signal.signal_type in ("health", "anomaly")

    def test_route_all(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.register(MonitoringSocket())
        mgr.load_all()

        outputs = mgr.route(_snap(), {})
        assert len(outputs) == 2

    def test_route_no_match(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.load_all()

        outputs = mgr.route(_snap(), {}, socket_type="nonexistent")
        assert len(outputs) == 0

    def test_health_report(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.register(MonitoringSocket())
        mgr.load_all()

        report = mgr.health_report()
        assert "elmer:comprehension" in report
        assert "elmer:monitoring" in report
        assert report["elmer:comprehension"]["status"] == "healthy"

    def test_health_report_empty(self):
        mgr = SocketManager()
        report = mgr.health_report()
        assert len(report) == 0

    def test_list_sockets(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        sockets = mgr.list_sockets()
        assert len(sockets) == 1
        assert sockets[0]["socket_id"] == "elmer:comprehension"
        assert "requirements" in sockets[0]

    def test_unregister(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.load_all()
        mgr.unregister("elmer:comprehension")
        assert mgr.get_socket("elmer:comprehension") is None

    def test_detect_hardware(self):
        hw = SocketManager.detect_hardware()
        assert hw["cpu_cores"] >= 1
        assert isinstance(hw["gpu_available"], bool)
