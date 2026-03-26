"""
Tests for MyelinationSocket — Elmer's oligodendrocyte (punchlist #53 v0.4).
"""

import os
import sys
import time
import unittest

# Ensure Elmer repo is importable
_elmer_dir = os.path.expanduser("~/Elmer")
if _elmer_dir not in sys.path:
    sys.path.insert(0, _elmer_dir)

from core.myelination import MyelinationSocket
from core.base_socket import (
    ElmerSocket,
    GraphSnapshot,
    HardwareRequirements,
    SocketHealth,
    SocketOutput,
)
from core.substrate_signal import SubstrateSignal


class TestMyelinationSocketInterface(unittest.TestCase):
    """Verify MyelinationSocket implements ElmerSocket correctly."""

    def setUp(self):
        self.socket = MyelinationSocket()

    def test_is_subclass_of_elmer_socket(self):
        self.assertIsInstance(self.socket, ElmerSocket)

    def test_socket_id(self):
        self.assertEqual(self.socket.socket_id, "elmer:myelination")

    def test_socket_type(self):
        self.assertEqual(self.socket.socket_type, "myelination")

    def test_declare_requirements(self):
        reqs = self.socket.declare_requirements()
        self.assertIsInstance(reqs, HardwareRequirements)
        self.assertFalse(reqs.gpu_required)

    def test_load_unload(self):
        self.assertFalse(self.socket.is_loaded)
        self.assertTrue(self.socket.load("dummy"))
        self.assertTrue(self.socket.is_loaded)
        self.socket.unload()
        self.assertFalse(self.socket.is_loaded)

    def test_health(self):
        self.socket.load("dummy")
        h = self.socket.health()
        self.assertIsInstance(h, SocketHealth)
        self.assertEqual(h.status, "healthy")


class TestMyelinationProcessing(unittest.TestCase):
    """Verify process() returns valid SubstrateSignal with recommendations."""

    def setUp(self):
        self.socket = MyelinationSocket()
        self.socket.load("dummy")

    def test_process_returns_socket_output(self):
        snapshot = GraphSnapshot.empty()
        output = self.socket.process(snapshot, {"autonomic_state": "PARASYMPATHETIC"})
        self.assertIsInstance(output, SocketOutput)
        self.assertIsInstance(output.signal, SubstrateSignal)

    def test_process_metadata_has_socket_id(self):
        snapshot = GraphSnapshot.empty()
        output = self.socket.process(snapshot, {})
        self.assertEqual(output.signal.metadata["socket"], "elmer:myelination")

    def test_process_metadata_has_recommendations(self):
        snapshot = GraphSnapshot.empty()
        output = self.socket.process(snapshot, {})
        recs = output.signal.metadata.get("myelination_recommendations", {})
        self.assertIn("myelinate", recs)
        self.assertIn("demyelinate", recs)

    def test_empty_snapshot_produces_no_recommendations(self):
        """With no bridge ref and no data, recommendations should be empty."""
        snapshot = GraphSnapshot.empty()
        output = self.socket.process(snapshot, {})
        recs = output.signal.metadata["myelination_recommendations"]
        self.assertEqual(recs["myelinate"], [])
        self.assertEqual(recs["demyelinate"], [])

    def test_raises_when_not_loaded(self):
        unloaded = MyelinationSocket()
        with self.assertRaises(RuntimeError):
            unloaded.process(GraphSnapshot.empty(), {})


class TestApprenticeHeuristic(unittest.TestCase):
    """Verify Apprentice-tier heuristic produces sensible recommendations."""

    def setUp(self):
        self.socket = MyelinationSocket()
        self.socket.load("dummy")

    def _make_mock_bridge(self, peer_events):
        """Create a mock bridge with _peer_events attribute."""
        class MockBridge:
            pass
        b = MockBridge()
        b._peer_events = peer_events
        return b

    def test_high_activity_peer_recommended_for_myelination(self):
        """Peer with many events should be recommended for myelination."""
        events = []
        # neurograph: 50 events (high activity)
        for i in range(50):
            events.append({"module_id": "neurograph", "target_id": f"t:{i}"})
        # bunyan: 3 events (low activity)
        for i in range(3):
            events.append({"module_id": "bunyan", "target_id": f"b:{i}"})

        bridge = self._make_mock_bridge(events)
        self.socket.set_bridge_ref(bridge)

        snapshot = GraphSnapshot.empty()
        output = self.socket.process(snapshot, {})
        recs = output.signal.metadata["myelination_recommendations"]

        self.assertIn("neurograph", recs["myelinate"])

    def test_low_activity_peer_recommended_for_demyelination(self):
        """Peer with very few events should be a demyelination candidate."""
        events = []
        # immunis: 40 events
        for i in range(40):
            events.append({"module_id": "immunis", "target_id": f"i:{i}"})
        # bunyan: 2 events (very low)
        for i in range(2):
            events.append({"module_id": "bunyan", "target_id": f"b:{i}"})

        bridge = self._make_mock_bridge(events)
        self.socket.set_bridge_ref(bridge)

        snapshot = GraphSnapshot.empty()
        output = self.socket.process(snapshot, {})
        recs = output.signal.metadata["myelination_recommendations"]
        scores = output.signal.metadata["pathway_scores"]

        # bunyan should have a low score
        self.assertLess(scores.get("bunyan", 1.0), 0.2)
        self.assertIn("bunyan", recs["demyelinate"])

    def test_insufficient_data_produces_empty_recommendations(self):
        """With fewer than MIN_EVENTS events, no recommendations."""
        events = [{"module_id": "neurograph", "target_id": "t:1"}]  # Just 1

        bridge = self._make_mock_bridge(events)
        self.socket.set_bridge_ref(bridge)

        snapshot = GraphSnapshot.empty()
        output = self.socket.process(snapshot, {})
        scores = output.signal.metadata["pathway_scores"]
        self.assertEqual(scores, {})


if __name__ == "__main__":
    unittest.main()
