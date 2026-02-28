"""Tests for ElmerHook (PRD §7, §9)."""

import pytest

from ng_ecosystem import NGEcosystem
from elmer_hook import ElmerHook
from core.config import ElmerConfig


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons before each test."""
    ElmerHook.reset_instance()
    NGEcosystem.reset_instance("elmer")
    yield
    ElmerHook.reset_instance()
    NGEcosystem.reset_instance("elmer")


class TestElmerHook:
    def test_singleton(self):
        h1 = ElmerHook.get_instance()
        h2 = ElmerHook.get_instance()
        assert h1 is h2

    def test_on_message(self):
        hook = ElmerHook()
        result = hook.on_message("Hello, Elmer!")
        assert result["status"] == "ingested"
        assert result["tier"] >= 1
        assert "elmer" in result
        assert result["elmer"]["pipelines_active"] is True
        assert "autonomic_state" in result["elmer"]
        assert "coherence_score" in result["elmer"]
        assert "coherence_status" in result["elmer"]

    def test_on_message_empty(self):
        hook = ElmerHook()
        result = hook.on_message("")
        assert result["status"] == "skipped"

    def test_get_context(self):
        hook = ElmerHook()
        context = hook.get_context("What do you know?")
        assert "tier" in context
        assert "recommendations" in context
        assert "novelty" in context
        assert "memory" in context
        assert "identity" in context
        assert "autonomic_state" in context

    def test_get_context_empty(self):
        hook = ElmerHook()
        context = hook.get_context("")
        assert context["tier"] >= 1

    def test_stats(self):
        hook = ElmerHook()
        stats = hook.stats()
        assert stats["module_id"] == "elmer"
        assert "adapter_version" in stats
        assert "tier" in stats

    def test_health(self):
        hook = ElmerHook()
        health = hook.health()
        assert health["module"] == "elmer"
        assert health["version"] == "0.2.0"
        assert "pipelines" in health
        assert "sensory" in health["pipelines"]
        assert "autonomic_state" in health
        assert "coherence_status" in health

    def test_start_stop(self):
        hook = ElmerHook()
        result = hook.start()
        assert result["status"] == "started"
        assert "sockets" in result
        hook.stop()

    def test_derive_target(self):
        hook = ElmerHook()
        target = hook._derive_target("any text")
        assert target == "elmer:substrate_input"

    def test_custom_config(self):
        cfg = ElmerConfig()
        cfg.version = "9.9.9"
        hook = ElmerHook(config=cfg)
        health = hook.health()
        assert health["version"] == "9.9.9"
