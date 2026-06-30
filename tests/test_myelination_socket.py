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
    """Verify Apprentice-tier heuristic produces sensible recommendations.

    Uses set_commons_ref() to inject a mock Commons — no process-singleton
    needed, no ng_peer_bridge._peer_events (deleted 2026-06-03, #326 fix).
    """

    def setUp(self):
        self.socket = MyelinationSocket()
        self.socket.load("dummy")

    def _wire_commons(self, entries):
        """Inject a mock Commons returning the given (target_id, weight, reasoning) entries."""
        class MockCommons:
            def __init__(self, data):
                self._data = data
            def bucket_recent(self, limit=200, **_kwargs):
                return self._data[:limit]
        self.socket.set_commons_ref(MockCommons(entries))

    def test_high_activity_peer_recommended_for_myelination(self):
        """Module with many recent deposits should be recommended for myelination."""
        entries = []
        # neurograph: 50 deposits (3-segment → segment[1] = "neurograph")
        for i in range(50):
            entries.append((f"metrics:neurograph:step{i}", 1.0, ""))
        # bunyan: 3 deposits (low activity)
        for i in range(3):
            entries.append((f"metrics:bunyan:narrate{i}", 1.0, ""))

        self._wire_commons(entries)
        output = self.socket.process(GraphSnapshot.empty(), {})
        recs = output.signal.metadata["myelination_recommendations"]

        self.assertIn("neurograph", recs["myelinate"])

    def test_low_activity_peer_recommended_for_demyelination(self):
        """Module with very few recent deposits should be a demyelination candidate."""
        entries = []
        # immunis: 40 deposits via autonomic namespace (fixed-source)
        for i in range(40):
            entries.append((f"threat:immunis:scan{i}", 1.0, ""))
        # bunyan: 2 deposits (very low — below _DEMYELINATE_PERCENTILE)
        for i in range(2):
            entries.append((f"metrics:bunyan:log{i}", 1.0, ""))

        self._wire_commons(entries)
        output = self.socket.process(GraphSnapshot.empty(), {})
        recs = output.signal.metadata["myelination_recommendations"]
        scores = output.signal.metadata["pathway_scores"]

        self.assertLess(scores.get("bunyan", 1.0), 0.2)
        self.assertIn("bunyan", recs["demyelinate"])

    def test_insufficient_data_produces_empty_recommendations(self):
        """With fewer than MIN_EVENTS deposits, no recommendations returned."""
        # 5 entries — below _MIN_EVENTS_TO_RECOMMEND (10)
        entries = [(f"metrics:neurograph:step{i}", 1.0, "") for i in range(5)]

        self._wire_commons(entries)
        output = self.socket.process(GraphSnapshot.empty(), {})
        scores = output.signal.metadata["pathway_scores"]
        self.assertEqual(scores, {})

    def test_fixed_source_namespace_attribution(self):
        """topology/experience/autonomic namespaces are attributed to the correct module."""
        entries = []
        for i in range(8):
            entries.append((f"topology:hash{i}", 1.0, ""))   # → neurograph
        for i in range(6):
            entries.append((f"autonomic:arousal", 1.0, ""))  # → immunis
        for i in range(4):
            entries.append((f"repair:entry{i}", 1.0, ""))    # → thc

        self._wire_commons(entries)
        output = self.socket.process(GraphSnapshot.empty(), {})
        scores = output.signal.metadata["pathway_scores"]

        self.assertIn("neurograph", scores)
        self.assertIn("immunis", scores)
        self.assertIn("thc", scores)
        # neurograph (8) should outscore thc (4)
        self.assertGreater(scores["neurograph"], scores["thc"])


if __name__ == "__main__":
    unittest.main()
