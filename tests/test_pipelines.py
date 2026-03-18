"""Tests for all pipelines (PRD §8)."""

from core.substrate_signal import SubstrateSignal
from pipelines.sensory import SensoryPipeline
from pipelines.inference import InferencePipeline
from pipelines.health import HealthPipeline
from pipelines.memory import MemoryPipeline
from pipelines.identity import IdentityPipeline


def _obs_signal() -> SubstrateSignal:
    return SubstrateSignal.create(
        signal_type="observation",
        description="test observation",
        coherence_score=0.8,
        novelty=0.3,
    )


class TestSensoryPipeline:
    def test_process(self):
        p = SensoryPipeline()
        sig = p.process("hello world")
        assert sig.signal_type == "observation"
        assert sig.module_id == "elmer"
        assert sig.metadata["text_length"] == 11
        assert sig.metadata["text_preview"] == "hello world"

    def test_novelty_scaling(self):
        p = SensoryPipeline()
        short = p.process("hi")
        long = p.process("x" * 2000)
        assert short.novelty < long.novelty
        assert long.novelty == 1.0  # capped at 1.0

    def test_stats(self):
        p = SensoryPipeline()
        p.process("a")
        p.process("b")
        assert p.stats()["process_count"] == 2


class TestInferencePipeline:
    def test_process(self):
        p = InferencePipeline()
        obs = _obs_signal()
        result = p.process(obs)
        assert result.signal_type == "coherence"
        assert result.metadata["parent_signal"] == obs.signal_id
        assert result.coherence_score == obs.coherence_score

    def test_confidence_calibration(self):
        p = InferencePipeline()
        obs = _obs_signal()
        result = p.process(obs)
        # Slight calibration loss
        assert result.confidence < obs.confidence

    def test_stats(self):
        p = InferencePipeline()
        p.process(_obs_signal())
        assert p.stats()["process_count"] == 1


class TestHealthPipeline:
    def test_check_healthy(self):
        p = HealthPipeline()
        sig = p.check(coherence=0.85)
        assert sig.signal_type == "health"
        assert sig.severity == 0.0
        assert sig.coherence_status == "healthy"

    def test_check_degraded(self):
        p = HealthPipeline()
        sig = p.check(coherence=0.50)
        assert sig.severity == 0.3
        assert sig.coherence_status == "degraded"

    def test_check_critical(self):
        p = HealthPipeline()
        sig = p.check(coherence=0.10)
        assert sig.severity == 1.0
        assert sig.coherence_status == "critical"

    def test_process(self):
        p = HealthPipeline()
        obs = _obs_signal()
        result = p.process(obs)
        assert result.metadata.get("health_processed") is True
        assert result.metadata.get("coherence_status") is not None

    def test_stats(self):
        p = HealthPipeline()
        p.check()
        stats = p.stats()
        assert stats["check_count"] == 1
        assert stats["uptime"] >= 0


class TestMemoryPipeline:
    def test_store(self):
        p = MemoryPipeline()
        obs = _obs_signal()
        result = p.store(obs)
        assert result.signal_type == "observation"
        assert result.metadata["action"] == "stored"
        assert result.metadata["buffer_size"] == 1

    def test_recall(self):
        p = MemoryPipeline()
        obs = _obs_signal()
        p.store(obs)
        result = p.recall("test query")
        assert result.signal_type == "observation"
        assert result.metadata["action"] == "recalled"
        assert result.metadata["recalled_count"] == 1

    def test_bounded_buffer(self):
        p = MemoryPipeline(max_signals=3)
        for i in range(5):
            p.store(_obs_signal())
        result = p.recall("any", k=10)
        assert result.metadata["recalled_count"] == 3

    def test_stats(self):
        p = MemoryPipeline()
        p.store(_obs_signal())
        p.recall("x")
        stats = p.stats()
        assert stats["store_count"] == 1
        assert stats["recall_count"] == 1
        assert stats["buffer_size"] == 1


class TestIdentityPipeline:
    def test_query(self):
        p = IdentityPipeline()
        sig = p.query()
        assert sig.signal_type == "coherence"
        assert sig.identity_coherence == 1.0
        assert sig.metadata["identity"]["name"] == "Elmer"

    def test_process(self):
        p = IdentityPipeline()
        obs = _obs_signal()
        result = p.process(obs)
        assert result.metadata["identity"] == "Elmer"
        assert result.metadata["identity_version"] == "0.2.0"

    def test_stats(self):
        p = IdentityPipeline()
        p.query()
        stats = p.stats()
        assert stats["query_count"] == 1
        assert stats["identity"]["name"] == "Elmer"
