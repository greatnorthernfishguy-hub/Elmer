"""Tests for ElmerSocket implementations (ComprehensionSocket, MonitoringSocket)."""

import pytest

from core.base_socket import ElmerSocket
from core.comprehension import ComprehensionSocket
from core.monitoring import MonitoringSocket
from ng_ecosystem import SubstrateSignal, SignalType


class TestComprehensionSocket:
    def test_socket_id(self):
        s = ComprehensionSocket()
        assert s.socket_id == "elmer:comprehension"
        assert s.socket_type == "sensory"

    def test_connect_disconnect(self):
        s = ComprehensionSocket()
        assert not s.is_connected
        s.connect()
        assert s.is_connected
        s.disconnect()
        assert not s.is_connected

    def test_idempotent_connect(self):
        s = ComprehensionSocket()
        s.connect()
        s.connect()  # Should not raise
        assert s.is_connected

    def test_process(self):
        s = ComprehensionSocket()
        s.connect()

        signal = SubstrateSignal.create(
            source_socket="test:input",
            signal_type=SignalType.SENSORY,
            payload={"text": "hello world"},
        )
        result = s.process(signal)

        assert result.source_socket == "elmer:comprehension"
        assert result.metadata.get("comprehension_processed") is True
        assert result.payload == signal.payload

    def test_process_not_connected(self):
        s = ComprehensionSocket()
        signal = SubstrateSignal.create(
            source_socket="test:input",
            signal_type=SignalType.SENSORY,
            payload={},
        )
        with pytest.raises(RuntimeError, match="not connected"):
            s.process(signal)

    def test_health_check_connected(self):
        s = ComprehensionSocket()
        s.connect()
        health = s.health_check()
        assert health["status"] == "healthy"
        assert health["socket_id"] == "elmer:comprehension"
        assert health["connected"] is True

    def test_health_check_disconnected(self):
        s = ComprehensionSocket()
        health = s.health_check()
        assert health["status"] == "offline"

    def test_hardware_affinity(self):
        s = ComprehensionSocket()
        assert s.hardware_affinity == "cpu"

    def test_is_elmer_socket(self):
        s = ComprehensionSocket()
        assert isinstance(s, ElmerSocket)


class TestMonitoringSocket:
    def test_socket_id(self):
        s = MonitoringSocket()
        assert s.socket_id == "elmer:monitoring"
        assert s.socket_type == "health"

    def test_connect_disconnect(self):
        s = MonitoringSocket()
        s.connect()
        assert s.is_connected
        s.disconnect()
        assert not s.is_connected

    def test_process(self):
        s = MonitoringSocket()
        s.connect()

        signal = SubstrateSignal.create(
            source_socket="test:health",
            signal_type=SignalType.HEALTH,
            payload={"status": "ok"},
        )
        result = s.process(signal)

        assert result.source_socket == "elmer:monitoring"
        assert result.metadata.get("monitoring_processed") is True

    def test_health_check(self):
        s = MonitoringSocket()
        s.connect()
        health = s.health_check()
        assert health["status"] == "healthy"
        assert health["socket_type"] == "health"
