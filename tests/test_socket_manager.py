"""Tests for the SocketManager."""

import pytest

from core.socket_manager import SocketManager
from core.comprehension import ComprehensionSocket
from core.monitoring import MonitoringSocket
from ng_ecosystem import SubstrateSignal, SignalType


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
        with pytest.raises(ValueError, match="limit reached"):
            mgr.register(MonitoringSocket())

    def test_connect_all(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.register(MonitoringSocket())
        results = mgr.connect_all()
        assert results["elmer:comprehension"] is True
        assert results["elmer:monitoring"] is True

    def test_disconnect_all(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.connect_all()
        mgr.disconnect_all()
        sock = mgr.get_socket("elmer:comprehension")
        assert sock is not None
        assert not sock.is_connected

    def test_route_signal_sensory(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.connect_all()

        signal = SubstrateSignal.create(
            source_socket="test:input",
            signal_type=SignalType.SENSORY,
            payload={"text": "route me"},
        )
        result = mgr.route_signal(signal)
        assert result.source_socket == "elmer:comprehension"

    def test_route_signal_health(self):
        mgr = SocketManager()
        mgr.register(MonitoringSocket())
        mgr.connect_all()

        signal = SubstrateSignal.create(
            source_socket="test:input",
            signal_type=SignalType.HEALTH,
            payload={"status": "check"},
        )
        result = mgr.route_signal(signal)
        assert result.source_socket == "elmer:monitoring"

    def test_route_signal_no_match(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.connect_all()

        signal = SubstrateSignal.create(
            source_socket="test:input",
            signal_type=SignalType.MEMORY,  # No memory socket registered
            payload={"query": "something"},
        )
        result = mgr.route_signal(signal)
        # Pass-through: original signal returned unchanged
        assert result.source_socket == "test:input"

    def test_health_report(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.register(MonitoringSocket())
        mgr.connect_all()

        report = mgr.health_report()
        assert report["status"] == "healthy"
        assert report["socket_count"] == 2
        assert report["connected_count"] == 2
        assert "elmer:comprehension" in report["sockets"]
        assert "elmer:monitoring" in report["sockets"]

    def test_health_report_empty(self):
        mgr = SocketManager()
        report = mgr.health_report()
        assert report["status"] == "no_sockets"
        assert report["socket_count"] == 0

    def test_list_sockets(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        sockets = mgr.list_sockets()
        assert len(sockets) == 1
        assert sockets[0]["socket_id"] == "elmer:comprehension"

    def test_unregister(self):
        mgr = SocketManager()
        mgr.register(ComprehensionSocket())
        mgr.connect_all()
        mgr.unregister("elmer:comprehension")
        assert mgr.get_socket("elmer:comprehension") is None

    def test_detect_hardware(self):
        hw = SocketManager.detect_hardware()
        assert hw["cpu"]["available"] is True
        assert hw["cpu"]["cores"] >= 1
