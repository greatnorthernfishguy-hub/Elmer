"""Tests for all pipeline stubs."""

from ng_ecosystem import SubstrateSignal, SignalType
from pipelines.sensory import SensoryPipeline
from pipelines.inference import InferencePipeline
from pipelines.health import HealthPipeline
from pipelines.memory import MemoryPipeline
from pipelines.identity import IdentityPipeline


class TestSensoryPipeline:
    def test_process(self):
        p = SensoryPipeline()
        sig = p.process("hello world")
        assert sig.signal_type == SignalType.SENSORY
        assert sig.payload["text"] == "hello world"
        assert sig.payload["input_length"] == 11
        assert sig.source_socket == "pipeline:sensory"

    def test_stats(self):
        p = SensoryPipeline()
        p.process("a")
        p.process("b")
        stats = p.stats()
        assert stats["process_count"] == 2


class TestInferencePipeline:
    def test_process(self):
        p = InferencePipeline()
        input_sig = SubstrateSignal.create(
            source_socket="test",
            signal_type=SignalType.SENSORY,
            payload={"text": "test"},
        )
        result = p.process(input_sig)
        assert result.signal_type == SignalType.INFERENCE
        assert result.payload["text"] == "test"
        assert result.metadata["parent_signal"] == input_sig.signal_id

    def test_stats(self):
        p = InferencePipeline()
        sig = SubstrateSignal.create(
            source_socket="test",
            signal_type=SignalType.SENSORY,
            payload={},
        )
        p.process(sig)
        assert p.stats()["process_count"] == 1


class TestHealthPipeline:
    def test_check(self):
        p = HealthPipeline()
        sig = p.check()
        assert sig.signal_type == SignalType.HEALTH
        assert sig.payload["status"] == "healthy"
        assert sig.payload["check_count"] == 1

    def test_process(self):
        p = HealthPipeline()
        input_sig = SubstrateSignal.create(
            source_socket="test",
            signal_type=SignalType.HEALTH,
            payload={"alert": "none"},
        )
        result = p.process(input_sig)
        assert result.metadata.get("health_processed") is True

    def test_stats(self):
        p = HealthPipeline()
        p.check()
        stats = p.stats()
        assert stats["check_count"] == 1
        assert stats["uptime"] >= 0


class TestMemoryPipeline:
    def test_store(self):
        p = MemoryPipeline()
        sig = SubstrateSignal.create(
            source_socket="test",
            signal_type=SignalType.SENSORY,
            payload={"text": "remember this"},
        )
        result = p.store(sig)
        assert result.signal_type == SignalType.MEMORY
        assert result.payload["action"] == "stored"
        assert result.payload["buffer_size"] == 1

    def test_recall(self):
        p = MemoryPipeline()
        sig = SubstrateSignal.create(
            source_socket="test",
            signal_type=SignalType.SENSORY,
            payload={"text": "data"},
        )
        p.store(sig)
        result = p.recall("data")
        assert result.signal_type == SignalType.MEMORY
        assert result.payload["action"] == "recalled"
        assert result.payload["recalled_count"] == 1

    def test_bounded_buffer(self):
        p = MemoryPipeline(max_signals=3)
        for i in range(5):
            sig = SubstrateSignal.create(
                source_socket="test",
                signal_type=SignalType.SENSORY,
                payload={"i": i},
            )
            p.store(sig)
        result = p.recall("any", k=10)
        assert result.payload["recalled_count"] == 3

    def test_stats(self):
        p = MemoryPipeline()
        sig = SubstrateSignal.create(
            source_socket="test",
            signal_type=SignalType.SENSORY,
            payload={},
        )
        p.store(sig)
        p.recall("x")
        stats = p.stats()
        assert stats["store_count"] == 1
        assert stats["recall_count"] == 1
        assert stats["buffer_size"] == 1


class TestIdentityPipeline:
    def test_query(self):
        p = IdentityPipeline()
        sig = p.query()
        assert sig.signal_type == SignalType.IDENTITY
        assert sig.payload["name"] == "Elmer"
        assert sig.payload["module_id"] == "elmer"
        assert sig.payload["version"] == "0.1.0"

    def test_process(self):
        p = IdentityPipeline()
        input_sig = SubstrateSignal.create(
            source_socket="test",
            signal_type=SignalType.SENSORY,
            payload={"text": "who are you?"},
        )
        result = p.process(input_sig)
        assert result.metadata["identity"] == "Elmer"

    def test_stats(self):
        p = IdentityPipeline()
        p.query()
        stats = p.stats()
        assert stats["query_count"] == 1
        assert stats["identity"]["name"] == "Elmer"
